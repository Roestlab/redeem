use std::fmt;
use std::path::Path;
use candle_core::{Var, Device, Tensor, DType, D};
use candle_nn::{ops, Conv1d, Conv1dConfig, Dropout, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::collections::HashMap;
use ndarray::Array2;

use crate::{model_interface::ModelInterface, utils::peptdeep_utils::{parse_model_constants, ModelConstants, load_mod_to_feature}};
use crate::building_blocks::building_blocks::{Input26aaModPositionalEncoding, MetaEmbedding, HiddenHfaceTransformer, DecoderLinear, ModLossNN};
use crate::building_blocks::featurize::{get_aa_indices, aa_one_hot, get_mod_features};


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
    fn new<P: AsRef<Path>>(model_path: P, constants_path: P, device: Device) -> Result<Self>{

        let var_store = VarBuilder::from_pth(
            model_path,
            candle_core::DType::F32,
            &device
        )?;

        let constants: ModelConstants = parse_model_constants(constants_path.as_ref().to_str().unwrap())?;

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
            vec!["output_nn.nn.0.weight", "output_nn.nn.1.weight", "output_nn.nn.2.weight"],
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
            vec!["modloss_nn.1.nn.0.weight", "modloss_nn.1.nn.1.weight", "modloss_nn.1.nn.2.weight"],
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
    
    fn predict(&self, peptide_sequence: &[String], mods: &str, mod_sites: &str) -> Result<Vec<f32>> {
        todo!()
    }
    
    fn encode_peptides(&self, peptide_sequences: &[String], mods: &str, mod_sites: &str) -> Result<Tensor> {
        // println!("Peptide sequences to encode: {:?}", peptide_sequences);
        
        let aa_indices = get_aa_indices(peptide_sequences)?;
        // println!("AA indices: {:?}", aa_indices);

        // Convert ndarray to Tensor and ensure it's F32
        let aa_indices_tensor = Tensor::from_slice(
            &aa_indices.as_slice().unwrap(),
            (aa_indices.shape()[0], aa_indices.shape()[1]),
            &self.device
        )?.to_dtype(DType::F32)?;

        let (batch_size, seq_len) = aa_indices_tensor.shape().dims2()?;

        // One-hot encode amino acids
        let aa_one_hot = aa_one_hot(&aa_indices_tensor, self.constants.aa_embedding_size)?;
        // println!("AA one hot shape: {:?}", aa_one_hot.shape());

        // Get modification features
        let mut mod_x = get_mod_features(mods, mod_sites, seq_len, self.constants.mod_elements.len(), self.mod_to_feature, self.device)?;
        // println!("Mod features shape: {:?}", mod_x.shape());

        // Preprocess mod_x: keep first 6 features unchanged
        let mod_x_first_6 = mod_x.narrow(2, 0, 6)?;
        let mod_x_rest = mod_x.narrow(2, 6, 103)?;

        // Apply mod_nn to the rest of the features
        // let mod_x_processed = mod_x_rest.apply(&self.mod_nn)?;

        // Concatenate the first 6 features with the processed features
        mod_x = Tensor::cat(&[mod_x_first_6, mod_x_processed], 2)?;

        // println!("Mod features shape post nn: {:?}", mod_x.shape());

        // Combine aa_one_hot and mod_x
        let combined = Tensor::cat(&[aa_one_hot, mod_x], 2)?;
        // println!("Combined shape: {:?}", combined.shape());
        
        Ok(combined)
    }
    
    fn fine_tune(&mut self, training_data: &[(String, f32)], modifications: HashMap<(String, Option<char>), crate::utils::peptdeep_utils::ModificationMap>, learning_rate: f64, epochs: usize) -> Result<()> {
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
        let mut xs = xs.clone();
        xs = self.input_nn.forward(&xs)?;
        xs = self.meta_nn.forward(&xs)?;
        xs = self.hidden_nn.forward(&xs)?;
        xs = self.output_nn.forward(&xs)?;
        Ok(xs)
    }
}

impl<'a> fmt::Debug for MS2BertModel<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MS2BertModel(")?;
        writeln!(f, "  (dropout): Dropout(p={})", 0.1)?;
        writeln!(f, "  (input_nn): Input26aaModPositionalEncoding(")?;
        writeln!(f, "    (mod_nn): Mod_Embedding_FixFirstK(")?;
        writeln!(f, "      (nn): Linear(in_features=103, out_features=2, bias=False)")?;
        writeln!(f, "    )")?;
        writeln!(f, "    (aa_emb): Embedding(27, 240, padding_idx=0)")?;
        writeln!(f, "    (pos_encoder): PositionalEncoding()")?;
        writeln!(f, "  )")?;
        writeln!(f, "  (meta_nn): MetaEmbedding(")?;
        writeln!(f, "    (nn): Linear(in_features=9, out_features=7, bias=True)")?;
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
        writeln!(f, "      (0): Linear(in_features=256, out_features=64, bias=True)")?;
        writeln!(f, "      (1): PReLU(num_parameters=1)")?;
        writeln!(f, "      (2): Linear(in_features=64, out_features=4, bias=True)")?;
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
    use std::path::PathBuf;
    use candle_core::Device;
    use crate::model_interface::ModelInterface;
    use crate::models::ms2_bert_model::MS2BertModel;

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
        let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/ms2.pth.model_const.yaml");
        let device = Device::Cpu;
        let model = MS2BertModel::new(model_path, constants_path, device).unwrap();

        println!("{:?}", model);
    }
}