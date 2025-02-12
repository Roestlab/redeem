use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{
    ops, Dropout, Module, Optimizer, VarBuilder, VarMap,
};
use log::info;
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::process::Output;
use std::{char, fmt, vec};
use std::path::Path;

use crate::building_blocks::building_blocks::{
    DecoderLinear, Encoder26aaModChargeCnnLstmAttnSum, AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE,
};
use crate::building_blocks::featurize::{aa_one_hot, get_aa_indices, get_mod_features};
use crate::utils::logging::Progress;
use crate::utils::data_handling::PeptideData;
use crate::utils::peptdeep_utils::{extract_masses_and_indices, get_modification_indices, remove_mass_shift};
use crate::{
    models::model_interface::{ModelInterface, PredictionResult,create_var_map},
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
pub struct CCSCNNLSTMModel<'a> {
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
    device: Device,
    dropout: Dropout,
    ccs_encoder: Encoder26aaModChargeCnnLstmAttnSum,
    ccs_decoder: DecoderLinear,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl<'a> Send for CCSCNNLSTMModel<'a> {}
unsafe impl<'a> Sync for CCSCNNLSTMModel<'a> {}

// Code Model Implementation
impl<'a> ModelInterface for CCSCNNLSTMModel<'a> {
    /// Create a new CCSCNNLSTMModel instance model from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self> {
        let tensor_data = candle_core::pickle::read_all(model_path.as_ref())?;

        let mut varmap = candle_nn::VarMap::new();
        create_var_map(&mut varmap, tensor_data, &device)?;

        let var_store = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants: ModelConstants =
            parse_model_constants(constants_path.as_ref().to_str().unwrap())?;

        // Load the mod_to_feature mapping
        let mod_to_feature = load_mod_to_feature(&constants)?;

        let dropout = Dropout::new(0.1);

        let ccs_encoder = Encoder26aaModChargeCnnLstmAttnSum::from_varstore(
            &var_store,
            8,
            128,
            2,
            vec![
                "ccs_encoder.mod_nn.nn.weight"
            ],
            vec![
                "ccs_encoder.input_cnn.cnn_short.weight",
                "ccs_encoder.input_cnn.cnn_medium.weight",
                "ccs_encoder.input_cnn.cnn_long.weight",
            ],
            vec![
                "ccs_encoder.input_cnn.cnn_short.bias",
                "ccs_encoder.input_cnn.cnn_medium.bias",
                "ccs_encoder.input_cnn.cnn_long.bias"
            ],
            "ccs_encoder.hidden_nn",
            vec!["ccs_encoder.attn_sum.attn.0.weight"]
        ).unwrap();

        let ccs_decoder = DecoderLinear::from_varstore(
            &var_store,
            257,
            1,
            vec![
                "ccs_decoder.nn.0.weight",
                "ccs_decoder.nn.1.weight",
                "ccs_decoder.nn.2.weight",
            ],
            vec!["ccs_decoder.nn.0.bias", "ccs_decoder.nn.2.bias"],
        )
        .unwrap();

        Ok(CCSCNNLSTMModel {
            var_store,
            varmap,
            constants,
            mod_to_feature,
            fixed_sequence_len,
            num_frag_types,
            num_modloss_types,
            mask_modloss,
            device,
            dropout,
            ccs_encoder,
            ccs_decoder,
        })
    }
    
    fn predict(
        &self,
        peptide_sequence: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        _nce: Option<i32>,
        _intsrument: Option<&str>,
    ) -> Result<PredictionResult> {
        let input_tesnor = self.encode_peptides(peptide_sequence, mods, mod_sites, charge, _nce, _intsrument)?;
        let output = self.forward(&input_tesnor)?;
        let predictions = PredictionResult::IMResult(output.to_vec1()?);

        Ok(predictions)

    }
    
    fn encode_peptides(
        &self,
        peptide_sequences: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        _nce: Option<i32>,
        _intsrument: Option<&str>,
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

        // Combine aa_one_hot, mod_x, charge, nce, and instrument
        let combined = Tensor::cat(
            &[aa_indices_tensor, mod_x, charge],
            2,
        )?;

        Ok(combined)
    }
    
    fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<(String, Option<char>), crate::utils::peptdeep_utils::ModificationMap>,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        let num_batches = (training_data.len() as f64 / batch_size as f64).ceil() as usize;
        info!(
            "Fine-tuning model on {} batches with batch size {} and learning rate {} for {} epochs",
            num_batches, batch_size, learning_rate, epochs
        );
    
        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.varmap.all_vars(), params)?;
    
        for epoch in 0..epochs {
            let progress = Progress::new(num_batches, &format!("[fine-tuning] Epoch {}: ", epoch));
            let mut total_loss = 0.0;
            let mut batch_inputs = Vec::new();
            let mut batch_targets = Vec::new();
    
            for (i, peptide) in training_data.iter().enumerate() {
                let naked_peptide = remove_mass_shift(&peptide.sequence);
                let modified_indices = get_modification_indices(&peptide.sequence);
                let extracted_masses_and_indices = extract_masses_and_indices(&peptide.sequence);
    
                let mut found_modifications = Vec::new();
                for (mass, index) in extracted_masses_and_indices {
                    let amino_acid = peptide.sequence.chars().nth(index).unwrap_or('\0');
                    if let Some(modification) = modifications.get(&(format!("{:.4}", mass), Some(amino_acid))) {
                        found_modifications.push(modification.name.clone());
                    } else if let Some(modification) = modifications.get(&(format!("{:.4}", mass), None)) {
                        found_modifications.push(modification.name.clone());
                    }
                }
    
                let peptides_str = &vec![naked_peptide.to_string()];
                let mod_str = &found_modifications.join("; ");
                let mod_site_str = &modified_indices;
                let charge = Some(peptide.charge.unwrap());
                
                let input = self.encode_peptides(peptides_str, mod_str, mod_site_str, charge, None, None)?;
                batch_inputs.push(input);
                batch_targets.push(peptide.ion_mobility.unwrap());
    
                if batch_inputs.len() == batch_size || i == training_data.len() - 1 {
                    let max_seq_len = batch_inputs.iter().map(|t| t.shape().dims3().unwrap().1).max().unwrap();
                    let padded_inputs: Result<Vec<_>> = batch_inputs
                        .iter()
                        .map(|t| {
                            let (_, seq_len, _) = t.shape().dims3()?;
                            if seq_len < max_seq_len {
                                t.pad_with_zeros(1, 0, max_seq_len - seq_len)
                            } else {
                                Ok(t.clone())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>() 
                        .map_err(Into::into);
                    
                    let padded_inputs = padded_inputs?;
                    let input_batch = Tensor::cat(&padded_inputs, 0)?;
                    let target_batch = Tensor::new(batch_targets.as_slice(), &self.device)?;
    
                    let predicted = self.forward(&input_batch)?;
                    let loss = candle_nn::loss::mse(&predicted, &target_batch)?;
                    opt.backward_step(&loss)?;
    
                    total_loss += loss.to_vec0::<f32>()?;
                    batch_inputs.clear();
                    batch_targets.clear();
    
                    progress.inc();
                    progress.update_description(&format!("[fine-tuning] Epoch {}: Loss: {}", epoch, loss.to_vec0::<f32>()?));
                }
            }
            progress.update_description(&format!("[fine-tuning] Epoch {}: Avg. Batch Loss: {}", epoch, total_loss / num_batches as f32));
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


// Forward Module Trait Implementation
impl <'a> Module for CCSCNNLSTMModel<'a> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _) = xs.shape().dims3()?;

        // Separate input into aa_indices, mod_x, charge
        let start_mod_x = 1;
        let start_charge = start_mod_x + MOD_FEATURE_SIZE;

        let aa_indices_out = xs.i((.., .., 0))?;
        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;
        let charge_out = xs.i((.., 0..1, start_charge..start_charge + 1))?;
        let charge_out = charge_out.squeeze(2)?;
        
        let x = self.ccs_encoder.forward(&aa_indices_out, &mod_x_out, &charge_out)?;
        let x = self.dropout.forward(&x, true)?;
        let x = Tensor::cat(&[x, charge_out], 1)?;
        let x = self.ccs_decoder.forward(&x)?;

        Ok(x.squeeze(1)?)
    }
    
}

impl<'a> fmt::Debug for CCSCNNLSTMModel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ModelCCS_LSTM(")?;
        writeln!(f, "  (dropout): Dropout(p={}, inplace={})", 0.1, false)?;
        writeln!(f, "  (ccs_encoder): Input_AA_CNN_LSTM_cat_Charge_Encoder(")?;

        // Mod Net
        writeln!(f, "    (mod_nn): InputModNetFixFirstK(")?;
        writeln!(
            f,
            "      (nn): Linear(in_features=103, out_features=2, bias=False)"
        )?;
        writeln!(f, "    )")?;

        // CNN
        writeln!(f, "    (input_cnn): SeqCNN(")?;
        writeln!(f, "      (cnn_short): Conv1d(36, 36, kernel_size=(3,), stride=(1,), padding=(1,))")?;
        writeln!(f, "      (cnn_medium): Conv1d(36, 36, kernel_size=(5,), stride=(1,), padding=(2,))")?;
        writeln!(f, "      (cnn_long): Conv1d(36, 36, kernel_size=(7,), stride=(1,), padding=(3,))")?;
        writeln!(f, "    )")?;

        // Hidden LSTM
        writeln!(f, "    (hidden_nn): SeqLSTM(")?;
        writeln!(f, "      (rnn): LSTM(144, 128, num_layers=2, batch_first=True, bidirectional=True)")?;
        writeln!(f, "    )")?;

        // Attention Sum
        writeln!(f, "    (attn_sum): SeqAttentionSum(")?;
        writeln!(f, "      (attn): Sequential(")?;
        writeln!(
            f,
            "        (0): Linear(in_features=256, out_features=1, bias=False)"
        )?;
        writeln!(f, "        (1): Softmax(dim=1)")?;
        writeln!(f, "      )")?;
        writeln!(f, "    )")?;

        writeln!(f, "  )")?;

        // Decoder
        writeln!(f, "  (ccs_decoder): LinearDecoder(")?;
        writeln!(f, "    (nn): Sequential(")?;
        writeln!(
            f,
            "      (0): Linear(in_features=257, out_features=64, bias=True)"
        )?;
        writeln!(f, "      (1): PReLU(num_parameters=1)")?;
        writeln!(
            f,
            "      (2): Linear(in_features=64, out_features=1, bias=True)"
        )?;
        writeln!(f, "    )")?;
        
        writeln!(f, "  )")?;
        
        write!(f, ")")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::model_interface::ModelInterface;
    use crate::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;
    use crate::utils::peptdeep_utils::load_modifications;
    use crate::utils::data_handling::PeptideData;
    use candle_core::Device;
    use std::path::PathBuf;

    #[test]
    fn test_load_pretrained_ccs_cnn_lstm_model() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = CCSCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        println!("{:?}", model);
    }

    #[test]
    fn test_encode_peptides() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = CCSCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

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
    fn test_predict(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = CCSCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);

        let result = model.predict(&peptide_sequences, mods, mod_sites, charge, None, None);
        println!("{:?}", result);
    }

    #[test]
    fn test_fine_tuning(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ccs.pth.model_const.yaml");
        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let mut model = CCSCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, device).unwrap();

        let training_data = vec![
            PeptideData::new("EHVIIQAEFYLNPDQ", Some(2), None, None, None, Some(1.10), None),
            PeptideData::new("KTLTGKTITLEVEPS", Some(2), None, None, None, Some(1.04), None),
            PeptideData::new("SLLAQNTSWLL", Some(1), None, None, None, Some(1.67), None),
            PeptideData::new("SLQEVAM[+15.9949]FL", Some(1), None, None, None, Some(1.53), None),
            PeptideData::new("VLADQVWTL", Some(2), None, None, None, Some(0.839), None),
            PeptideData::new("LLMEPGAMRFL", Some(2), None, None, None, Some(0.949), None),
            PeptideData::new("SGEIKIAYTYSVS", Some(2), None, None, None, Some(0.974), None),
            PeptideData::new("HTEIVFARTSPQQKL", Some(2), None, None, None, Some(1.13), None),
            PeptideData::new("SM[+15.9949]ADIPLGFGV", Some(1), None, None, None, Some(1.59), None),
            PeptideData::new("KLIDHQGLYL", Some(2), None, None, None, Some(0.937), None),
        ];


        let test_peptides = vec![
            ("SKEEETSIDVAGKP", "", "", 2, 0.998),
            ("LPILVPSAKKAIYM", "", "", 2, 1.12),
            ("RTPKIQVYSRHPAE", "", "", 3, 0.838),
            ("EEVQIDILDTAGQE", "", "", 2, 1.02),
            ("GAPLVKPLPVNPTDPA", "", "", 2, 1.01),
            ("FEDENFILK", "", "", 2, 0.897),
            ("YPSLPAQQV", "", "", 1, 1.45),
            ("YLPPATQVV", "", "", 2, 0.846),
            ("YISPDQLADLYK", "", "", 2, 0.979),
            ("PSIVRLLQCDPSSAGQF", "", "", 2, 1.10),
        ];


        // model.set_evaluation_mode();

        let mut total_error = 0.0;
        let mut count = 0;
        for (peptide, mods, mod_sites, charge, observed) in &test_peptides {
            match model.predict(&[peptide.to_string()], mods, mod_sites, Some(*charge), None, None) {
                Ok(predictions) => {
                    assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                    let predicted = predictions[0];
                    let error = (predicted - observed).abs();
                    total_error += error;
                    count += 1;
                }
                Err(e) => {
                    println!("Error during prediction for {}: {:?}", peptide, e);
                }
            }
        }

        let mean_absolute_error = total_error / count as f32;
        println!("Mean Absolute Error prior to fine-tuning: {:.6}", mean_absolute_error);

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

        let mut total_error = 0.0;
        let mut count = 0;
        for (peptide, mods, mod_sites, charge, observed) in &test_peptides {
            match model.predict(&[peptide.to_string()], mods, mod_sites, Some(*charge), None, None) {
                Ok(predictions) => {
                    assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                    let predicted = predictions[0];
                    let error = (predicted - observed).abs();
                    total_error += error;
                    count += 1;
                }
                Err(e) => {
                    println!("Error during prediction for {}: {:?}", peptide, e);
                }
            }
        }

        let mean_absolute_error = total_error / count as f32;
        println!("Mean Absolute Error post fine-tuning: {:.6}", mean_absolute_error);
    }

}