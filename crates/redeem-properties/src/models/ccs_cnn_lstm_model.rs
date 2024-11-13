use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, Dropout, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap,
};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
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
            256,
            4,
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
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<PredictionResult> {
        todo!()
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
        todo!()
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

}