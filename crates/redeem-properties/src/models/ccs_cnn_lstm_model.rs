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
    models::model_interface::{ModelInterface, PropertyType, PredictionResult,create_var_map},
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
    is_training: bool,
    dropout: Dropout,
    ccs_encoder: Encoder26aaModChargeCnnLstmAttnSum,
    ccs_decoder: DecoderLinear,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl<'a> Send for CCSCNNLSTMModel<'a> {}
unsafe impl<'a> Sync for CCSCNNLSTMModel<'a> {}

// Code Model Implementation
impl<'a> ModelInterface for CCSCNNLSTMModel<'a> {
    fn property_type(&self) -> PropertyType {
        PropertyType::CCS
    }

    fn model_arch(&self) -> &'static str {
        "ccs_cnn_lstm"   
    }

    /// Create a new CCSCNNLSTMModel instance model from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device
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
            is_training: false,
            dropout,
            ccs_encoder,
            ccs_decoder
        })
    }
    

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
    
    /// Set model to evaluation mode for inference
    /// This disables dropout and other training-specific layers.
    fn set_evaluation_mode(&mut self) {
        // println!("Setting evaluation mode");
        self.is_training = false;
    }

    /// Set model to training mode for training
    /// This enables dropout and other training-specific layers.
    fn set_training_mode(&mut self) {
        self.is_training = true;
    }

    fn get_property_type(&self) -> String {
        self.property_type().clone().as_str().to_string()
    }

    fn get_model_arch(&self) -> String {
        self.model_arch().to_string()
    }

    fn get_device(&self) -> &Device {
        &self.device
    }

    fn get_mod_element_count(&self) -> usize {
        self.constants.mod_elements.len()
    }

    fn get_mod_to_feature(&self) -> &HashMap<String, Vec<f32>> {
        &self.mod_to_feature
    }

    fn get_min_pred_intensity(&self) -> f32 {
        unimplemented!("Method not implemented for architecture: {}", self.model_arch())
    }


    fn get_mut_varmap(&mut self) -> &mut VarMap {
        &mut self.varmap
    }
    
    fn print_summary(&self) {
        todo!()
    }
    
    fn print_weights(&self) {
        todo!()
    }
    
}


// // Forward Module Trait Implementation
// impl <'a> Module for CCSCNNLSTMModel<'a> {
//     fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
//         ModelInterface::forward(self, input)
//     }    
// }

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

        let peptide_sequences = "AGHCEWQMKYR";
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let charge = Some(2);
        let nce = Some(20);
        let instrument = Some("QE");

        let result =
            model.encode_peptide(&peptide_sequences, mods, mod_sites, charge, nce, instrument);

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

        let peptide_sequences = vec!["AGHCEWQMKYR".to_string(), "AGHCEWQMKYR".to_string()];
        let mods = vec!["Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M".to_string(), "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M".to_string()];
        let mod_sites = vec!["0;4;8".to_string(), "0;4;8".to_string()];
        let charge = Some(vec![2, 2]);

        let result = model.predict(&peptide_sequences, &mods, &mod_sites, charge, None, None);
        println!("{:?}", result);
    }

    

}