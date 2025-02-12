use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, Dropout, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap,
};
use log::info;
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::building_blocks::building_blocks::{
    DecoderLinear, HiddenHfaceTransformer, Input26aaModPositionalEncoding, MetaEmbedding,
    ModLossNN, AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE,
};
use crate::building_blocks::featurize::{aa_one_hot, get_aa_indices, get_mod_features};
use crate::utils::data_handling::PeptideData;
use crate::utils::logging::Progress;
use crate::utils::peptdeep_utils::{extract_masses_and_indices, get_modification_indices, remove_mass_shift};
use crate::{
    models::model_interface::{ModelInterface, PredictionResult, create_var_map},
    utils::peptdeep_utils::{
        load_mod_to_feature, parse_instrument_index, parse_model_constants, ModelConstants,
    },
};

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;

// Main Model Struct
#[derive(Clone)]
/// Represents an AlphaPeptDeep MS2BERT model.
pub struct MS2BertModel<'a> {
    var_store: VarBuilder<'a>,
    varmap: VarMap,
    constants: ModelConstants,
    mod_to_feature: HashMap<String, Vec<f32>>,
    fixed_sequence_len: usize,
    // Total number of fragment types of a fragmentation position to predict
    num_frag_types: usize,
    // Number of fragment types of a fragmentation position to predict, by default 0
    num_modloss_types: usize,
    // If True, the modloss layer will be disabled, by default True
    mask_modloss: bool,
    min_inten: f32,
    device: Device,
    dropout: Dropout,
    input_nn: Input26aaModPositionalEncoding,
    meta_nn: MetaEmbedding,
    hidden_nn: HiddenHfaceTransformer,
    output_nn: DecoderLinear,
    modloss_nn: ModLossNN,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl<'a> Send for MS2BertModel<'a> {}
unsafe impl<'a> Sync for MS2BertModel<'a> {}

// Code Model Implementation
impl<'a> ModelInterface for MS2BertModel<'a> {
    /// Create a new MS2BERT model from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self> {
        // let var_store = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;
        let tensor_data = candle_core::pickle::read_all(model_path.as_ref())?;

        let mut varmap = VarMap::new();
        create_var_map(&mut varmap, tensor_data, &device)?;

        let var_store = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants: ModelConstants =
            parse_model_constants(constants_path.as_ref().to_str().unwrap())?;

        // Load the mod_to_feature mapping
        let mod_to_feature = load_mod_to_feature(&constants)?;

        let dropout = Dropout::new(0.1);

        let meta_dim = 8;
        let input_nn = Input26aaModPositionalEncoding::from_varstore(
            &var_store,
            256 - 8,
            200,
            vec![
                "input_nn.mod_nn.nn.weight",
                "input_nn.aa_emb.weight",
                "input_nn.pos_encoder.pe",
            ],
        )
        .unwrap();

        let meta_nn = MetaEmbedding::from_varstore(
            &var_store,
            8,
            vec!["meta_nn.nn.weight", "meta_nn.nn.bias"],
        )
        .unwrap();

        let hidden_nn = HiddenHfaceTransformer::from_varstore(
            var_store.pp("hidden_nn.bert"),
            256,
            4,
            8,
            4,
            0.1,
            false,
        )
        .unwrap();

        let output_nn = DecoderLinear::from_varstore(
            &var_store,
            256,
            4,
            vec![
                "output_nn.nn.0.weight",
                "output_nn.nn.1.weight",
                "output_nn.nn.2.weight",
            ],
            vec!["output_nn.nn.0.bias", "output_nn.nn.2.bias"],
        )
        .unwrap();

        let modloss_nn = ModLossNN::from_varstore(
            var_store.clone(),
            256,
            4,
            8,
            1,
            0.1,
            false,
            4,
            "modloss_nn.0.bert",
            vec![
                "modloss_nn.1.nn.0.weight",
                "modloss_nn.1.nn.1.weight",
                "modloss_nn.1.nn.2.weight",
            ],
            vec!["modloss_nn.1.nn.0.bias", "modloss_nn.1.nn.2.bias"],
        )
        .unwrap();

        Ok(Self {
            var_store: var_store,
            varmap: varmap,
            constants: constants,
            mod_to_feature: mod_to_feature,
            fixed_sequence_len: fixed_sequence_len,
            num_frag_types: num_frag_types,
            num_modloss_types: num_modloss_types,
            mask_modloss: mask_modloss,
            min_inten: 1e-4,
            device,
            dropout: dropout,
            input_nn: input_nn,
            meta_nn: meta_nn,
            hidden_nn: hidden_nn,
            output_nn: output_nn,
            modloss_nn: modloss_nn,
        })
    }

    fn predict(
        &self,
        peptide_sequence: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<PredictionResult> {
        let input_tensor =
            self.encode_peptides(peptide_sequence, mods, mod_sites, charge, nce, intsrument)?;
        let output = self.forward(&input_tensor)?;

        let out = self.process_predictions(&output, self.min_inten)?;

        let predictions = PredictionResult::MS2Result(out.squeeze(0)?.to_vec2()?);


        Ok(predictions)
    }

    fn encode_peptides(
        &self,
        peptide_sequences: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<Tensor> {
        let aa_indices = get_aa_indices(peptide_sequences)?;

        // Convert ndarray to Tensor and ensure it's F32
        let aa_indices_tensor = Tensor::from_slice(
            &aa_indices.as_slice().unwrap(),
            (aa_indices.shape()[0], aa_indices.shape()[1]),
            &self.device,
        )?
        .to_dtype(DType::F32)?;

        let (batch_size, seq_len) = aa_indices_tensor.shape().dims2()?;

        // unsqueeze aa_indices_tensor to match the shape of mod_x, which is batch_size x seq_len x mod_feature_size
        let aa_indices_tensor = aa_indices_tensor.unsqueeze(2)?;

        // Get modification features
        let mod_x = get_mod_features(
            mods,
            mod_sites,
            seq_len,
            self.constants.mod_elements.len(),
            self.mod_to_feature.clone(),
            self.device.clone(),
        )?;

        // Charges
        let charge = Tensor::from_slice(
            &vec![charge.unwrap() as f64 * CHARGE_FACTOR; seq_len],
            &[batch_size, seq_len, 1],
            &self.device,
        )?
        .to_dtype(DType::F32)?;

        // NCE
        let nce = Tensor::from_slice(
            &vec![nce.unwrap() as f64 * NCE_FACTOR; seq_len],
            &[batch_size, seq_len, 1],
            &self.device,
        )?
        .to_dtype(DType::F32)?;

        // Instrument
        let instrument_indices = Tensor::from_slice(
            &vec![parse_instrument_index(intsrument.unwrap()) as u32; seq_len],
            &[batch_size, seq_len, 1],
            &self.device,
        )?
        .to_dtype(DType::F32)?;

        // Combine aa_one_hot, mod_x, charge, nce, and instrument
        let combined = Tensor::cat(
            &[aa_indices_tensor, mod_x, charge, nce, instrument_indices],
            2,
        )?;

        Ok(combined)
    }

    fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        info!("Fine-tuning model with {} epochs and learning rate {}", epochs, learning_rate);

        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.varmap.all_vars(), params)?;

        for epoch in 0..epochs {
            let progress = Progress::new(training_data.len(), &format!("[fine-tuning] Epoch {}: ", epoch));
            let mut total_loss = 0.0;
            for peptide in training_data {
                let naked_peptide = remove_mass_shift(&peptide.sequence.to_string());

                // Collect indices of non-zero modifications
                let modified_indices = get_modification_indices(&peptide.sequence);

                // Extract masses and indices
                let extracted_masses_and_indices = extract_masses_and_indices(&peptide.sequence.to_string());

                let mut found_modifications = Vec::new();

                // Map modifications based on extracted masses and indices
                for (mass, index) in extracted_masses_and_indices {
                    let amino_acid = peptide.sequence.to_string().chars().nth(index).unwrap_or('\0');
                    if let Some(modification) =
                        modifications.get(&(format!("{:.4}", mass), Some(amino_acid)))
                    {
                        found_modifications.push(modification.name.clone());
                    } else if let Some(modification) =
                        modifications.get(&(format!("{:.4}", mass), None))
                    {
                        found_modifications.push(modification.name.clone());
                    }
                }

                // Prepare strings for prediction
                let peptides_str = &vec![naked_peptide.to_string()];
                let mod_str = &found_modifications.join("; ");
                let mod_site_str = &modified_indices;
                let charge = Some(peptide.charge.unwrap());
                let nce = Some(peptide.nce.unwrap());
                let instrument = &peptide.instrument.as_deref();

                // Forward pass
                let input =
                    self.encode_peptides(peptides_str, mod_str, mod_site_str, charge, nce, *instrument)?;
                let predicted = self.forward(&input)?;

                let target = Tensor::new(peptide.ms2_intensities.clone().unwrap(), &self.device)?;

                // Unsqueeze target for batch dimension
                let target = target.unsqueeze(0)?;

                // Compute loss
                let loss = candle_nn::loss::mse(
                    &predicted,
                    &target,
                )?;

                // Backward pass
                opt.backward_step(&loss)?;

                total_loss += loss.to_vec0::<f32>()?;

                progress.inc();
                progress.update_description(&format!("[fine-tuning] Epoch {}: Loss: {}", epoch, loss.to_vec0::<f32>()?));
            }
            progress.update_description(&format!("[fine-tuning] Epoch {}: Avg. Loss: {}", epoch, total_loss / training_data.len() as f32));
            progress.finish();
        }
        
        Ok(())
    }

    fn set_evaluation_mode(&mut self) {
        todo!()
    }

    fn set_training_mode(&mut self) {
        todo!()
    }

    fn print_summary(&self) {
        todo!()
    }

    fn print_weights(&self) {
        todo!()
    }

    fn save(&self, path: &Path) -> Result<()> {
        todo!()
    }
}

// Module Trait Implementation
impl<'a> Module for MS2BertModel<'a> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _) = xs.shape().dims3()?;

        // Separate the input tensor into the different parts

        // Calculate starting indices
        let start_mod_x = 1;
        let start_charge = start_mod_x + MOD_FEATURE_SIZE;
        let start_nce = start_charge + 1;
        let start_instrument = start_nce + 1;

        // Extract tensors using indexing
        let aa_indices_out = xs.i((.., .., 0))?;
        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;
        let charge_out = xs.i((.., 0..1, start_charge..start_charge + 1))?;
        let nce_out = xs.i((.., 0..1, start_nce..start_nce + 1))?;
        let instrument_out = xs.i((.., 0..1, start_instrument..start_instrument + 1))?;

        // Adjust shapes after extraction if necessary
        // let aa_indices_out = aa_indices_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let charge_out = charge_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let nce_out = nce_out.squeeze(2)?; // Squeeze to remove dimensions of size 1 if needed
        let instrument_out = instrument_out.squeeze(2)?.squeeze(1)?; // Squeeze to remove dimensions of size 1 if needed

        // Forward pass through input_nn with dropout
        let in_x = self
            .dropout
            .forward(&self.input_nn.forward(&aa_indices_out, &mod_x_out)?, true)?;

        // Prepare metadata for meta_nn
        let meta_x = self
            .meta_nn
            .forward(&charge_out, &nce_out, &instrument_out)?
            .unsqueeze(1)?
            .repeat(vec![1, seq_len as usize, 1])?;

        // Concatenate in_x and meta_x along dimension 2
        let combined_input = Tensor::cat(&[in_x.clone(), meta_x], 2)?;

        // Forward pass through hidden_nn
        let hidden_x = self.hidden_nn.forward(&combined_input.clone(), None)?;

        // // Handle attentions if needed (similar to PyTorch)
        // if self.output_attentions {
        //     self.attentions = hidden_x[1];
        // } else {
        //     self.attentions = None;
        // }

        // Apply dropout and combine with input
        let x_tmp = (hidden_x + combined_input * 0.2)?;
        let hidden_output = self.dropout.forward(&x_tmp, true)?;

        // Forward pass through output_nn
        let mut out_x = self.output_nn.forward(&hidden_output)?;

        // Handle modloss if applicable (similar logic as PyTorch)
        if self.num_modloss_types > 0 {
            if self.mask_modloss {
                // Create a tensor of zeros with the appropriate shape
                let zeros_shape = (out_x.shape().dims()[0], out_x.shape().dims()[1], self.num_modloss_types);
                let zeros_tensor = Tensor::zeros(zeros_shape, DType::F32, &self.device)?; // Adjust device as necessary

                // Concatenate along the last dimension
                out_x = Tensor::cat(&[out_x, zeros_tensor], 2)?;
            } else {
                // // Forward pass through the first modloss neural network
                // let modloss_output = self.modloss_nn[0].forward(in_x)?;

                // // // If output attentions is enabled, save them
                // // if self.output_attentions {
                // //     self.modloss_attentions = Some(modloss_output.clone()); // Assuming you want to store a clone
                // // }

                // // Add hidden_x to the first output
                // let modloss_combined = &modloss_output + hidden_x;

                // // Forward pass through the last modloss neural network
                // let modloss_final = self.modloss_nn.last().unwrap().forward(&modloss_combined)?;

                // // Concatenate the outputs along the last dimension
                // out_x = Tensor::cat(&[out_x, modloss_final], 2)?;
                todo!();
            }
        }

        Ok(out_x.i((.., 3.., ..))?)
    }
}

impl<'a> fmt::Debug for MS2BertModel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MS2BertModel(")?;
        writeln!(f, "  (dropout): Dropout(p={})", 0.1)?;
        writeln!(f, "  (input_nn): Input26aaModPositionalEncoding(")?;
        writeln!(f, "    (mod_nn): Mod_Embedding_FixFirstK(")?;
        writeln!(
            f,
            "      (nn): Linear(in_features=103, out_features=2, bias=False)"
        )?;
        writeln!(f, "    )")?;
        writeln!(f, "    (aa_emb): Embedding(27, 240, padding_idx=0)")?;
        writeln!(f, "    (pos_encoder): PositionalEncoding()")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (meta_nn): MetaEmbedding(")?;
        writeln!(
            f,
            "    (nn): Linear(in_features=9, out_features=7, bias=True)"
        )?;
        writeln!(f, "  )")?;
        writeln!(f, "  (hidden_nn): HiddenHfaceTransformer(")?;
        writeln!(f, "    (bert): BertEncoder(")?;
        writeln!(f, "      (layer): ModuleList(")?;
        for i in 0..self.hidden_nn.bert.layers.len() {
            writeln!(f, "        ({}): BertLayer(...)", i)?;
        }
        writeln!(f, "      )")?;
        writeln!(f, "    )")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (output_nn): DecoderLinear(")?;
        writeln!(f, "    (nn): Sequential(")?;
        writeln!(
            f,
            "      (0): Linear(in_features=256, out_features=64, bias=True)"
        )?;
        writeln!(f, "      (1): PReLU(num_parameters=1)")?;
        writeln!(
            f,
            "      (2): Linear(in_features=64, out_features=4, bias=True)"
        )?;
        writeln!(f, "    )")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (modloss_nn): ModLossNN(")?;
        writeln!(f, "    (0): HiddenHfaceTransformer(...)")?;
        writeln!(f, "    (1): DecoderLinear(...)")?;
        writeln!(f, "  )")?;
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_interface::ModelInterface;
    use crate::models::ms2_bert_model::MS2BertModel;
    use crate::utils::peptdeep_utils::load_modifications;
    use candle_core::Device;
    use std::path::PathBuf;
    use csv::Reader;
    use std::collections::HashMap;
    use std::fs::File;

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size.unwrap(), 27);
        // assert_eq!(constants.charge_factor, 0.1);
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        // assert_eq!(constants.nce_factor, 0.01);
    }

    #[test]
    fn test_load_pretrained_ms2_bert_model() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        println!("{:?}", model);
    }

    #[test]
    fn test_encode_peptides() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let result =
            model.encode_peptides(&peptide_sequences, mods, mod_sites, charge, nce, instrument);

        println!("{:?}", result);

        // assert!(result.is_ok());
        // let encoded_peptides = result.unwrap();
        // assert_eq!(encoded_peptides.shape().dims2().unwrap(), (1, 27 + 109 + 1 + 1 + 1));
    }

    #[test]
    fn test_forward() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let input_tensor = model
            .encode_peptides(&peptide_sequences, mods, mod_sites, charge, nce, instrument)
            .unwrap();
        let output = model.forward(&input_tensor).unwrap();
        println!("{:?}", output);
    }

    #[test]
    fn test_predict(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let result = model.predict(&peptide_sequences, mods, mod_sites, charge, nce, instrument);
        // Result is a PredictionResult of Vec<Vec<f32>>
        // ordered as b_z1	b_z2	y_z1	y_z2	b_modloss_z1	b_modloss_z2	y_modloss_z1	y_modloss_z2
        println!("{:?}", result);
        println!("Length peptide_sequences: {:?}", peptide_sequences[0].len());
        println!("Length of result: {:?}", result.unwrap().len());
    }

    #[test]
    fn test_fine_tuning(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let mut model = MS2BertModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();


        // Open the CSV file
        let file_path = "/home/singjc/Documents/github/sage_bruker/predicted_fragment_intensities.csv";
        let file = File::open(file_path).unwrap();

        // Create a CSV reader
        let mut rdr = Reader::from_reader(file);

        // Group fragment intensities by peptide sequence
        let mut peptide_data_map: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut peptide_charges: HashMap<String, i32> = HashMap::new();

        for result in rdr.records() {
            let record = result.unwrap();
            let peptide_sequence = &record[0];
            let precursor_charge: i32 = record[1].parse().unwrap();
            let fragment_type = &record[2];
            let fragment_ordinal: usize = record[3].parse().unwrap();
            let fragment_charge: i32 = record[4].parse().unwrap();
            let experimental_intensity: f32 = record[6].parse().unwrap();

            // Get naked peptide sequence
            let naked_peptide = remove_mass_shift(peptide_sequence);

            // Get length of the peptide sequence
            let peptide_len = naked_peptide.len() - 1;

            // Initialize the peptide's intensity matrix if it doesn't exist
            peptide_data_map
                .entry(peptide_sequence.to_string())
                .or_insert_with(|| vec![vec![0.0; 8]; peptide_len]); // Initialize with enough rows

            // Update the peptide's charge
            peptide_charges.insert(peptide_sequence.to_string(), precursor_charge);

            // Determine the column index based on fragment type and charge
            let col = match (fragment_type, fragment_charge) {
                ("B", 1) => 0, // b_z1
                ("B", 2) => 1, // b_z2
                ("Y", 1) => 2, // y_z1
                ("Y", 2) => 3, // y_z2
                _ => continue, // Skip unsupported fragment types or charges
            };

            // Update the MS2 intensities matrix
            let row = peptide_len - 1; // Convert to zero-based index
            peptide_data_map
                .get_mut(peptide_sequence)
                .unwrap()
                .resize(row + 1, vec![0.0; 8]); // Ensure the matrix has enough rows
            peptide_data_map.get_mut(peptide_sequence).unwrap()[row][col] = experimental_intensity;
        }

        // Create PeptideData instances for each peptide
        let mut training_data: Vec<PeptideData> = Vec::new();

        for (sequence, ms2_intensities) in peptide_data_map {
            let charge = peptide_charges.get(&sequence).copied();
            let peptide_data = PeptideData::new(
                &sequence,
                charge,
                Some(20), // Example NCE
                Some("QE"), // Example instrument
                None, // Retention time
                None, // Ion mobility
                Some(ms2_intensities), // MS2 intensities
            );
            training_data.push(peptide_data);
        }


        // let test_peptides = vec![
        //     ("SKEEETSIDVAGKP", "", "", 2, 0.998),
        //     ("LPILVPSAKKAIYM", "", "", 2, 1.12),
        //     ("RTPKIQVYSRHPAE", "", "", 3, 0.838),
        //     ("EEVQIDILDTAGQE", "", "", 2, 1.02),
        //     ("GAPLVKPLPVNPTDPA", "", "", 2, 1.01),
        //     ("FEDENFILK", "", "", 2, 0.897),
        //     ("YPSLPAQQV", "", "", 1, 1.45),
        //     ("YLPPATQVV", "", "", 2, 0.846),
        //     ("YISPDQLADLYK", "", "", 2, 0.979),
        //     ("PSIVRLLQCDPSSAGQF", "", "", 2, 1.10),
        // ];


        // // model.set_evaluation_mode();

        // let mut total_error = 0.0;
        // let mut count = 0;
        // for (peptide, mods, mod_sites, charge, observed) in &test_peptides {
        //     match model.predict(&[peptide.to_string()], mods, mod_sites, Some(*charge), Some(20), Some("QE")) {
        //         Ok(predictions) => {
        //             let predicted = predictions;
        //             let error = (predicted - observed).abs();
        //             total_error += error;
        //             count += 1;
        //         }
        //         Err(e) => {
        //             println!("Error during prediction for {}: {:?}", peptide, e);
        //         }
        //     }
        // }

        // let mean_absolute_error = total_error / count as f32;
        // println!("Mean Absolute Error prior to fine-tuning: {:.6}", mean_absolute_error);

        // Fine-tune the model
        let modifications = match load_modifications() {
            Ok(mods) => mods,
            Err(e) => {
                panic!("Failed to load modifications: {:?}", e);
            }
        };
        let learning_rate = 0.001;
        let epochs = 5;

        let result = model.fine_tune(&training_data, modifications, 10, learning_rate, epochs);
        assert!(
            result.is_ok(),
            "Failed to fine-tune model: {:?}",
            result.err()
        );

        // model.set_evaluation_mode();

        // let mut total_error = 0.0;
        // let mut count = 0;
        // for (peptide, mods, mod_sites, charge, observed) in &test_peptides {
        //     match model.predict(&[peptide.to_string()], mods, mod_sites, Some(*charge), None, None) {
        //         Ok(predictions) => {
        //             assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
        //             let predicted = predictions[0];
        //             let error = (predicted - observed).abs();
        //             total_error += error;
        //             count += 1;
        //         }
        //         Err(e) => {
        //             println!("Error during prediction for {}: {:?}", peptide, e);
        //         }
        //     }
        // }

        // let mean_absolute_error = total_error / count as f32;
        // println!("Mean Absolute Error post fine-tuning: {:.6}", mean_absolute_error);
    }

}
