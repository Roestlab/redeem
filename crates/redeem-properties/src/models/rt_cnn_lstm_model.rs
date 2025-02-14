use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{ops, Dropout, Module, Optimizer, VarBuilder, VarMap};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use log::info;

// use crate::models::rt_model::RTModel;
use crate::building_blocks::bilstm::BidirectionalLSTM;
use crate::building_blocks::building_blocks::{
    DecoderLinear, Encoder26aaModCnnLstmAttnSum, AA_EMBEDDING_SIZE, MOD_FEATURE_SIZE,
};
use crate::building_blocks::featurize::{aa_one_hot, get_aa_indices, get_mod_features};
use crate::models::model_interface::{ModelInterface, PropertyType, PredictionResult, create_var_map};
use crate::utils::data_handling::PeptideData;
use crate::utils::peptdeep_utils::{
    extract_masses_and_indices, get_modification_indices, load_mod_to_feature, load_modifications,
    parse_model_constants, remove_mass_shift, ModelConstants, ModificationMap,
};
use crate::utils::logging::Progress;

// Main Model Struct

#[derive(Clone)]
/// Represents an AlphaPeptDeep CNN-LSTM Retention Time model.
pub struct RTCNNLSTMModel<'a> {
    var_store: VarBuilder<'a>,
    varmap: VarMap,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<String, Vec<f32>>,
    dropout: Dropout,
    rt_encoder: Encoder26aaModCnnLstmAttnSum,
    rt_decoder: DecoderLinear,
    is_training: bool,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl<'a> Send for RTCNNLSTMModel<'a> {}
unsafe impl<'a> Sync for RTCNNLSTMModel<'a> {}

// Core Model Implementation

impl<'a> ModelInterface for RTCNNLSTMModel<'a> {
    fn property_type(&self) -> PropertyType {
        PropertyType::RT
    }

    fn model_arch(&self) -> &'static str {
        "rt_cnn_lstm"   
    }

    /// Create a new RTCNNLSTMModel from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        _fixed_sequence_len: usize,
        _num_frag_types: usize,
        _num_modloss_types: usize,
        _mask_modloss: bool,
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

        // Encoder
        let dropout = Dropout::new(0.1);

        let rt_encoder = Encoder26aaModCnnLstmAttnSum::from_varstore(
            &var_store,
            8,
            128,
            2,
            vec!["rt_encoder.mod_nn.nn.weight"],
            vec!["rt_encoder.input_cnn.cnn_short.weight", "rt_encoder.input_cnn.cnn_medium.weight", "rt_encoder.input_cnn.cnn_long.weight"],
            vec!["rt_encoder.input_cnn.cnn_short.bias", "rt_encoder.input_cnn.cnn_medium.bias", "rt_encoder.input_cnn.cnn_long.bias"],
            "rt_encoder.hidden_nn",
            vec![
                "rt_encoder.attn_sum.attn.0.weight",
            ],
        ).unwrap();

        let rt_decoder = DecoderLinear::from_varstore(
            &var_store,
            256,
            1,
            vec!["rt_decoder.nn.0.weight", "rt_decoder.nn.1.weight", "rt_decoder.nn.2.weight"],
            vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"]
        ).unwrap();

        Ok(Self {
            var_store,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout,
            rt_encoder,
            rt_decoder,
            is_training: true,
        })
    }


    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _) = xs.shape().dims3()?;

        let start_mod_x = 1;
        let aa_indices_out = xs.i((.., .., 0))?;
        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;

        let x = self.rt_encoder.forward(&aa_indices_out, &mod_x_out)?;
        let x = self.dropout.forward(&x, self.is_training)?;
        let x = self.rt_decoder.forward(&x)?;

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

    /// Print a summary of the model's constants.
    fn print_summary(&self) {
        println!("RTModel Summary:");
        println!("AA Embedding Size: {}", self.constants.aa_embedding_size.unwrap());
        println!("Charge Factor: {:?}", self.constants.charge_factor);
        println!("Instruments: {:?}", self.constants.instruments);
        println!("Max Instrument Num: {}", self.constants.max_instrument_num);
        println!("Mod Elements: {:?}", self.constants.mod_elements);
        println!("NCE Factor: {:?}", self.constants.nce_factor);
    }

    /// Print the model's weights.
    fn print_weights(&self) {
        println!("RTModel Weights:");
    
        // Helper function to print the first 5 values of a tensor
        fn print_first_5_values(tensor: &Tensor, name: &str) {
            let shape = tensor.shape();
            if shape.dims().len() == 2 {
                // Extract the first row
                if let Ok(row) = tensor.i((0, ..)) {
                    match row.to_vec1::<f32>() {
                        Ok(values) => println!("{} (first 5 values of first row): {:?}", name, &values[..5.min(values.len())]),
                        Err(e) => eprintln!("Error printing {}: {:?}", name, e),
                    }
                } else {
                    eprintln!("Error extracting first row for {}", name);
                }
            } else {
                match tensor.to_vec1::<f32>() {
                    Ok(values) => println!("{} (first 5 values): {:?}", name, &values[..5.min(values.len())]),
                    Err(e) => eprintln!("Error printing {}: {:?}", name, e),
                }
            }
        }
        
    
        // Print the first 5 values of each weight tensor
        if let Ok(tensor) = self.var_store.get((2, 103), "rt_encoder.mod_nn.nn.weight") {
            print_first_5_values(&tensor, "rt_encoder.mod_nn.nn.weight");
        }
        // if let Ok(tensor) = self.var_store.get((35, 35, 3), "rt_encoder.input_cnn.cnn_short.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_short.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_medium.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight") {
        //     print_first_5_values(&tensor, "rt_encoder.input_cnn.cnn_long.weight");
        // }
        // if let Ok(tensor) = self.var_store.get((4, 1, 128), "rt_encoder.hidden_nn.rnn_h0") {
        //     print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn_h0");
        // }
        // if let Ok(tensor) = self.var_store.get((4, 1, 128), "rt_encoder.hidden_nn.rnn_c0") {
        //     print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn_c0");
        // }
        if let Ok(tensor) = self.var_store.get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l0");
        }
        if let Ok(tensor) = self.var_store.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l0");
        }
        if let Ok(tensor) = self.var_store.get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse");
        }
        if let Ok(tensor) = self.var_store.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse");
        }
        if let Ok(tensor) = self.var_store.get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l1");
        }
        if let Ok(tensor) = self.var_store.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l1");
        }
        if let Ok(tensor) = self.var_store.get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse");
        }
        if let Ok(tensor) = self.var_store.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse") {
            print_first_5_values(&tensor, "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse");
        }
        if let Ok(tensor) = self.var_store.get((1, 256), "rt_encoder.attn_sum.attn.0.weight") {
            print_first_5_values(&tensor, "rt_encoder.attn_sum.attn.0.weight");
        }
        if let Ok(tensor) = self.var_store.get((256, 256), "rt_decoder.nn.0.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.0.weight");
        }
        if let Ok(tensor) = self.var_store.get((256, 256), "rt_decoder.nn.1.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.1.weight");
        }
        if let Ok(tensor) = self.var_store.get((1, 256), "rt_decoder.nn.2.weight") {
            print_first_5_values(&tensor, "rt_decoder.nn.2.weight");
        }
    }


}

// Module Trait Implementation

// impl<'a> Module for RTCNNLSTMModel<'a> {
//     fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
//         ModelInterface::forward(self, input)
//     }
// }


#[cfg(test)]
mod tests {
    use crate::models::model_interface::ModelInterface;
    use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
    use crate::utils::peptdeep_utils::load_modifications;
    use candle_core::Device;
    use std::path::PathBuf;
    use std::time::Instant;
    // use itertools::izip;

    use super::*;

    #[test]
    fn test_tensor_from_pth(){
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let tensor_data = candle_core::pickle::read_all(model_path).unwrap();
        println!("{:?}", tensor_data);
    }

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/rt.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size.unwrap(), 27);
        assert_eq!(constants.charge_factor, Some(0.1));
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        assert_eq!(constants.nce_factor, Some(0.01));
    }

    #[test]
    fn test_prediction() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let device = /* Assuming Device is defined */ Device::new_cuda(0).unwrap_or(/* assuming Device::Cpu is defined */ Device::Cpu); // Replace with actual Device code.
        let result = /* Assuming RTCNNLSTMModel is defined */ RTCNNLSTMModel::new(&model_path, &constants_path, 0, 8, 4, true, device); // Replace with actual RTCNNLSTMModel code
        let mut model = result.unwrap();
    
        // Test prediction with a few peptides after fine-tuning
        let test_peptides = vec![
            ("QPYAVSELAGHQTSAESWGTGR", "", "", 0.4328955),
            ("GMSVSDLADKLSTDDLNSLIAHAHR", "Oxidation@M", "1", 0.6536107),
            (
                "TVQHHVLFTDNMVLICR",
                "Oxidation@M;Carbamidomethyl@C",
                "11;15",
                0.7811949,
            ),
            ("EAELDVNEELDKK", "", "", 0.2934583),
            ("YTPVQQGPVGVNVTYGGDPIPK", "", "", 0.5863009),
            ("YYAIDFTLDEIK", "", "", 0.8048359),
            ("VSSLQAEPLPR", "", "", 0.3201348),
            (
                "NHAVVCQGCHNAIDPEVQR",
                "Carbamidomethyl@C;Carbamidomethyl@C",
                "5;8",
                0.1730425,
            ),
            ("IPNIYAIGDVVAGPMLAHK", "", "", 0.8220097),
            ("AELGIPLEEVPPEEINYLTR", "", "", 0.8956433),
            ("NESTPPSEELELDKWK", "", "", 0.4471560),
            ("SIQEIQELDKDDESLR", "", "", 0.4157068),
            ("EMEENFAVEAANYQDTIGR", "Oxidation@M", "1", 0.6388353),
            ("MDSFDEDLARPSGLLAQER", "Oxidation@M", "0", 0.5593624),
            ("SLLTEADAGHTEFTDEVYQNESR", "", "", 0.5538696),
            ("NQDLAPNSAEQASILSLVTK", "", "", 0.7682227),
            ("GKVEEVELPVEK", "", "", 0.2943246),
            ("IYVASVHQDLSDDDIK", "", "", 0.3847130),
            ("IKGDMDISVPK", "", "", 0.2844255),
            ("IIPVLLEHGLER", "", "", 0.5619017),
            ("AGYTDKVVIGMDVAASEFFR", "", "", 0.8972052),
            ("TDYNASVSVPDSSGPER", "", "", 0.3279318),
            ("DLKPQNLLINTEGAIK", "", "", 0.6046495),
            ("VAEAIAASFGSFADFK", "", "", 0.8935943),
            ("AMVSNAQLDNEK", "Oxidation@M", "1", 0.1724159),
            ("THINIVVIGHVDSGK", "", "", 0.4865058),
            ("LILPHVDIQLK", "", "", 0.6268850),
            ("LIAPVAEEEATVPNNK", "", "", 0.4162872),
            ("FTASAGIQVVGDDLTVTNPK", "", "", 0.7251064),
            ("HEDLKDMLEFPAQELR", "", "", 0.6529368),
            ("LLPDFLLER", "", "", 0.7852863),
        ];
    
        let batch_size = 16; // Set an appropriate batch size
        let peptides: Vec<String> = test_peptides.iter().map(|(pep, _, _, _)| pep.to_string()).collect();
        let mods: Vec<String> = test_peptides.iter().map(|(_, mod_, _, _)| mod_.to_string()).collect();
        let mod_sites: Vec<String> = test_peptides.iter().map(|(_, _, sites, _)| sites.to_string()).collect();
        let observed_rts: Vec<f32> = test_peptides.iter().map(|(_, _, _, rt)| *rt).collect();
    
        match model.predict(&peptides, &mods, &mod_sites, None, None, None) {
            Ok(predictions) => {
                if let /* Assuming PredictionResult and RTResult are defined */ PredictionResult::RTResult(rt_preds) = predictions {  // Replace with actual PredictionResult and RTResult code
                    let total_error: f32 = rt_preds.iter().zip(observed_rts.iter())
                        .map(|(pred, obs)| (pred - obs).abs())
                        .sum();
    
                    // PRINT PREDICTIONS AND OBSERVED RTs WITHOUT IZIP
                    let mut peptides_iter = peptides.iter();
                    let mut rt_preds_iter = rt_preds.iter();
                    let mut observed_rts_iter = observed_rts.iter();
    
                    loop {
                        match (peptides_iter.next(), rt_preds_iter.next(), observed_rts_iter.next()) {
                            (Some(pep), Some(pred), Some(obs)) => {
                                println!("Peptide: {}, Predicted RT: {}, Observed RT: {}", pep, pred, obs);
                            }
                            _ => break, // Exit the loop if any iterator is exhausted
                        }
                    }
    
                    let mean_absolute_error = total_error / rt_preds.len() as f32;
                    println!("Mean Absolute Error: {:.6}", mean_absolute_error);
                } else {
                    println!("Unexpected prediction result type.");
                }
            }
            Err(e) => {
                println!("Error during batch prediction: {:?}", e);
            }
        }
    }
    
}
