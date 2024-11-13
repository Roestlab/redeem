use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, Dropout, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap,
};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::process::Output;
use std::{fmt, vec};
use std::path::Path;

use crate::building_blocks::building_blocks::{
    DecoderLinear, Encoder26aaModChargeCnnLstmAttnSum, AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE,
};
use crate::building_blocks::featurize::{aa_one_hot, get_aa_indices, get_mod_features};
use crate::{
    model_interface::{ModelInterface, PredictionResult},
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
        let var_store = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;

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
        training_data: &[(String, f32)],
        modifications: HashMap<(String, Option<char>), crate::utils::peptdeep_utils::ModificationMap>,
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

        println!("aa_indices_out: {:?}", aa_indices_out.shape());
        println!("mod_x_out: {:?}", mod_x_out.shape());
        println!("charge_out: {:?}", charge_out.shape());
        
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
    use crate::model_interface::ModelInterface;
    use crate::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;
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

}