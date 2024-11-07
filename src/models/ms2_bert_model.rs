use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, Var, D, IndexOp};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, Dropout, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap,
};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::building_blocks::building_blocks::{AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE,
    DecoderLinear, HiddenHfaceTransformer, Input26aaModPositionalEncoding, MetaEmbedding, ModLossNN,
};
use crate::building_blocks::featurize::{aa_one_hot, get_aa_indices, get_mod_features};
use crate::{
    model_interface::ModelInterface,
    utils::peptdeep_utils::{ModelConstants, load_mod_to_feature, parse_model_constants, parse_instrument_index},
};

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;

// Main Model Struct
#[derive(Clone)]
/// Represents an AlphaPeptDeep MS2BERT model.
pub struct MS2BertModel<'a> {
    var_store: VarBuilder<'a>,
    constants: ModelConstants,
    mod_to_feature: HashMap<String, Vec<f32>>,
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
    fn new<P: AsRef<Path>>(model_path: P, constants_path: P, device: Device) -> Result<Self> {
        let var_store = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;

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
            constants: constants,
            mod_to_feature: mod_to_feature,
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
    ) -> Result<Vec<f32>> {
        let input_tensor = self.encode_peptides(peptide_sequence, mods, mod_sites, charge, nce, intsrument)?;
        let output = self.forward(&input_tensor)?;
        Ok(output.to_vec1()?)
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
        let charge = Tensor::from_slice(&vec![charge.unwrap() as f64* CHARGE_FACTOR; seq_len], &[batch_size, seq_len, 1], &self.device)?.to_dtype(DType::F32)?;

        // NCE
        let nce = Tensor::from_slice(&vec![nce.unwrap() as f64 * NCE_FACTOR; seq_len], &[batch_size, seq_len, 1], &self.device)?.to_dtype(DType::F32)?;

        // Instrument
        let instrument_indices = Tensor::from_slice(&vec![parse_instrument_index(intsrument.unwrap()) as u32; seq_len], &[batch_size, seq_len, 1], &self.device)?.to_dtype(DType::F32)?;

        // println!("Encoding Peptides");
        // println!("aa_indices_tensor: {:?}", aa_indices_tensor.shape());
        // println!("mod_x: {:?}", mod_x.shape());
        // println!("charge: {:?}", charge.shape());
        // println!("nce: {:?}", nce.shape());
        // println!("instrument_indices: {:?}", instrument_indices.shape());

        // Combine aa_one_hot, mod_x, charge, nce, and instrument
        let combined = Tensor::cat(&[aa_indices_tensor, mod_x, charge, nce, instrument_indices], 2)?;

        Ok(combined)
    }

    fn fine_tune(
        &mut self,
        training_data: &[(String, f32)],
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        todo!()
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
        println!("xs shape: {:?}", xs.shape());

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

        // // print shapes
        println!("Separating input tensor into different parts");
        println!("aa_indices_out: {:?}", aa_indices_out.shape());
        println!("mod_x_out: {:?}", mod_x_out.shape());
        // println!("charge_out: {:?}", charge_out.shape());
        // println!("nce_out: {:?}", nce_out.shape());
        // println!("instrument_out: {:?}", instrument_out.shape());
        // println!("charge_out values: {:?}", charge_out.to_vec2::<f32>()?[0..1].to_vec());
        // println!("nce_out values: {:?}", nce_out.to_vec2::<f32>()?[0..1].to_vec());
        // println!("instrument_out values: {:?}", instrument_out.to_vec1::<f32>()?[0..1].to_vec());

        // Forward pass through input_nn with dropout
        println!("Forward pass through input_nn with dropout");
        let in_x = self.dropout.forward(&self.input_nn.forward(&aa_indices_out, &mod_x_out)?, true)?;
        println!("in_x shape: {:?}", in_x.shape());

        // Prepare metadata for meta_nn
        println!("Prepare metadata for meta_nn");
        let meta_x = self.meta_nn.forward(&charge_out, &nce_out, &instrument_out)?
            .unsqueeze(1)?
            .repeat(vec![1, seq_len as usize, 1])?;

        // Concatenate in_x and meta_x along dimension 2
        println!("Concatenate in_x and meta_x along dimension 2");
        let combined_input = Tensor::cat(&[in_x.clone(), meta_x], 2)?;

        // Forward pass through hidden_nn
        println!("Forward pass through hidden_nn");
        println!("combined_input shape: {:?}", combined_input.shape());
        let hidden_x = self.hidden_nn.forward(&combined_input.clone(), None)?;

        // // Handle attentions if needed (similar to PyTorch)
        // if self.output_attentions {
        //     self.attentions = hidden_x[1];
        // } else {
        //     self.attentions = None;
        // }

        // Apply dropout and combine with input
        println!("Apply dropout and combine with input");
        println!("hidden_x shape: {:?}", hidden_x.shape());
        println!("combined_input shape: {:?}", combined_input.shape());
        let x_tmp = (hidden_x + combined_input * 0.2)?;
        println!("x_tmp shape: {:?}", x_tmp.shape());
        let hidden_output = self.dropout.forward(&x_tmp, true)?;

        // Forward pass through output_nn
        println!("Forward pass through output_nn");
        println!("hidden_output (dropout) shape: {:?}", hidden_output.shape());
        let out_x = self.output_nn.forward(&hidden_output)?;

        // Handle modloss if applicable (similar logic as PyTorch)

        println!("out_x: {:?}", out_x.shape());
        // first few values of out_x
        // println!("out_x: {:?}", out_x.to_vec3::<f32>()?.to_vec());

        // print out_x[:,3:,:] values
        println!("out_x[:,3:,:]: {:?}", out_x.i((.., 3.., ..))?.to_vec3::<f32>()?.to_vec());
        
        Ok(out_x)

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
    use crate::model_interface::ModelInterface;
    use crate::models::ms2_bert_model::MS2BertModel;
    use candle_core::Device;
    use std::path::PathBuf;

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size, 27);
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
        let model = MS2BertModel::new(model_path, constants_path, device).unwrap();

        println!("{:?}", model);
    }

    #[test]
    fn test_encode_peptides() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let result = model.encode_peptides(&peptide_sequences, mods, mod_sites, charge, nce, instrument);

        println!("{:?}", result);

        // assert!(result.is_ok());
        // let encoded_peptides = result.unwrap();
        // assert_eq!(encoded_peptides.shape().dims2().unwrap(), (1, 27 + 109 + 1 + 1 + 1));
    }

    #[test]
    fn test_forward(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, device).unwrap();

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string()];
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let input_tensor = model.encode_peptides(&peptide_sequences, mods, mod_sites, charge, nce, instrument).unwrap();
        let output = model.forward(&input_tensor).unwrap();
        println!("{:?}", output);
    }

}
