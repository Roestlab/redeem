use std::path::Path;
use candle_core::{Device, Tensor, DType, D};
use candle_nn::{VarBuilder, Conv1d, Conv1dConfig, Linear, PReLU, ops, Module};
use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::collections::HashMap;
use ndarray::Array2;

// use crate::models::rt_model::RTModel;
use crate::model_interface::ModelInterface;
use crate::building_blocks::bilstm::BidirectionalLSTM;
use crate::utils::peptdeep_utils::{load_mod_to_feature, parse_model_constants};

// Constants and Utility Structs

#[derive(Clone, Debug, Deserialize)]
/// Represents the constants used in a model.
pub struct ModelConstants {
    /// The size of the amino acid embedding.
    aa_embedding_size: usize,
    /// The charge factor used in the model.
    charge_factor: f32,
    /// The list of instruments used in the model.
    instruments: Vec<String>,
    /// The maximum number of instruments allowed in the model.
    max_instrument_num: usize,
    /// The list of modification elements used in the model.
    pub mod_elements: Vec<String>,
    /// The NCE (Normalized Collision Energy) factor used in the model.
    nce_factor: f32,
}

// Main Model Struct

#[derive(Clone)]
/// Represents an AlphaPeptDeep CNN-LSTM Retention Time model.
pub struct RTCNNLSTMModel<'a> {
    var_store: VarBuilder<'a>,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<String, Vec<f32>>,
    mod_nn: Linear,
    cnn_short: Conv1d,
    cnn_medium: Conv1d,
    cnn_long: Conv1d,
    bilstm: BidirectionalLSTM,
    attn: Linear,
    decoder: [Linear; 2],
    prelu: PReLU,
    dropout: f32,
    is_training: bool,
}

// Core Model Implementation

impl<'a> ModelInterface for RTCNNLSTMModel<'a> {
    /// Create a new RTCNNLSTMModel from the given model and constants files.
    fn new<P: AsRef<Path>>(model_path: P, constants_path: P, device: Device) -> Result<Self> {

        let var_store = VarBuilder::from_pth(
            model_path,
            candle_core::DType::F32,
            &device
        )?;

        let constants: ModelConstants = parse_model_constants(constants_path.as_ref().to_str().unwrap())?;

        // Load the mod_to_feature mapping
        let mod_to_feature = load_mod_to_feature(&constants)?;
        
        // Encoder 

        let mod_nn = Linear::new(
            var_store.get((2, 103), "rt_encoder.mod_nn.nn.weight")?,
            None,
        );
        
        let cnn_short = Conv1d::new(
            var_store.get((35, 35, 3), "rt_encoder.input_cnn.cnn_short.weight")?,
            Some(var_store.get(35, "rt_encoder.input_cnn.cnn_short.bias")?),
            Conv1dConfig { padding: 1, ..Default::default() },
        );

        let cnn_medium = Conv1d::new(
            var_store.get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight")?,
            Some(var_store.get(35, "rt_encoder.input_cnn.cnn_medium.bias")?),
            Conv1dConfig { padding: 2, ..Default::default() },
        );

        let cnn_long = Conv1d::new(
            var_store.get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight")?,
            Some(var_store.get(35, "rt_encoder.input_cnn.cnn_long.bias")?),
            Conv1dConfig { padding: 3, ..Default::default() },
        );

        let bilstm: BidirectionalLSTM = BidirectionalLSTM::new(
            140,
            128,
            2,
            &var_store,
        )?;

        let attn = Linear::new(
            var_store.get((1, 256), "rt_encoder.attn_sum.attn.0.weight")?,
            None,
        );

        // Decoder
        let decoder = [
            Linear::new(
                var_store.get((64, 256), "rt_decoder.nn.0.weight")?,
                Some(var_store.get(64, "rt_decoder.nn.0.bias")?),
            ),
            Linear::new(
                var_store.get((1, 64), "rt_decoder.nn.2.weight")?,
                Some(var_store.get(1, "rt_decoder.nn.2.bias")?),
            ),
        ];

        let prelu = PReLU::new(var_store.get(1, "rt_decoder.nn.1.weight")?, true);

        let dropout = 0.1;

        Ok(Self {
            var_store,
            constants,
            device,
            mod_to_feature,
            mod_nn,
            cnn_short,
            cnn_medium,
            cnn_long,
            bilstm,
            attn,
            decoder,
            prelu,
            dropout,
            // aa_to_index,
            // mod_to_index,
            is_training: true,
        })
    }

    /// Predict the retention times for a peptide sequence.
    /// 
    /// # Arguments
    ///   * `peptide_sequence` - A vector of peptide sequences to predict retention times for.
    ///   * `mods` - A string representing the modifications for each peptide.
    ///   * `mod_sites` - A string representing the modification sites for each peptide.
    /// 
    /// # Returns
    ///    A vector of predicted retention times.
    fn predict(&self, peptide_sequence: &[String], mods: &str, mod_sites: &str) -> Result<Vec<f32>> {
        // Preprocess the peptide sequences and modifications
        let input_tensor = self.encode_peptides(peptide_sequence, mods, mod_sites)?;

        // Pass the data through the model
        let output = self.forward(&input_tensor)?;

        // Convert the output tensor to a Vec<f32>
        let predictions = output.to_vec1()?;

        Ok(predictions)
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

    /// Encode peptide sequences (plus modifications) into a tensor.
    fn encode_peptides(&self, peptide_sequences: &[String], mods: &str, mod_sites: &str) -> Result<Tensor> {
        // println!("Peptide sequences to encode: {:?}", peptide_sequences);
        
        let aa_indices = Self::get_aa_indices(peptide_sequences)?;
        // println!("AA indices: {:?}", aa_indices);

        // Convert ndarray to Tensor and ensure it's F32
        let aa_indices_tensor = Tensor::from_slice(
            &aa_indices.as_slice().unwrap(),
            (aa_indices.shape()[0], aa_indices.shape()[1]),
            &self.device
        )?.to_dtype(DType::F32)?;

        let (batch_size, seq_len) = aa_indices_tensor.shape().dims2()?;

        // One-hot encode amino acids
        let aa_one_hot = self.aa_one_hot(&aa_indices_tensor)?;
        // println!("AA one hot shape: {:?}", aa_one_hot.shape());

        // Get modification features
        let mut mod_x = self.get_mod_features(mods, mod_sites, seq_len)?;
        // println!("Mod features shape: {:?}", mod_x.shape());

        // Preprocess mod_x: keep first 6 features unchanged
        let mod_x_first_6 = mod_x.narrow(2, 0, 6)?;
        let mod_x_rest = mod_x.narrow(2, 6, 103)?;

        // Apply mod_nn to the rest of the features
        let mod_x_processed = mod_x_rest.apply(&self.mod_nn)?;

        // Concatenate the first 6 features with the processed features
        mod_x = Tensor::cat(&[mod_x_first_6, mod_x_processed], 2)?;

        // println!("Mod features shape post nn: {:?}", mod_x.shape());

        // Combine aa_one_hot and mod_x
        let combined = Tensor::cat(&[aa_one_hot, mod_x], 2)?;
        // println!("Combined shape: {:?}", combined.shape());
        
        Ok(combined)
    }
    
    /// Print a summary of the model's constants.
    fn print_summary(&self) {
        println!("RTModel Summary:");
        println!("AA Embedding Size: {}", self.constants.aa_embedding_size);
        println!("Charge Factor: {}", self.constants.charge_factor);
        println!("Instruments: {:?}", self.constants.instruments);
        println!("Max Instrument Num: {}", self.constants.max_instrument_num);
        println!("Mod Elements: {:?}", self.constants.mod_elements);
        println!("NCE Factor: {}", self.constants.nce_factor);
    }

    /// Print the model's weights.
    fn print_weights(&self) {
        println!("Model weights:");

        // Helper function to print first few values of a tensor
        fn print_first_few(tensor: &Tensor, name: &str) -> Result<()> {
            let flattened = tensor.flatten_all()?;
            let num_elements = flattened.dim(0)?;
            let num_to_print = 5.min(num_elements);
            println!("{} shape: {:?}", name, tensor.shape());
            println!("{} (first few values): {:?}", name, flattened.narrow(0, 0, num_to_print)?.to_vec1::<f32>()?);
            Ok(())
        }

        // Print weights for each layer
        let _ = print_first_few(self.mod_nn.weight(), "rt_encoder.mod_nn.nn weights");
        let _ = print_first_few(self.cnn_short.weight(), "rt_encoder.input_cnn.cnn_short weights");
        let _ = print_first_few(self.cnn_medium.weight(), "rt_encoder.input_cnn.cnn_medium weights");
        let _ = print_first_few(self.cnn_long.weight(), "rt_encoder.input_cnn.cnn_long weights");

        // Print bilstm weights
        let _ = self.bilstm.print_weights(&self.var_store);

        // Decoder
        let _ = print_first_few(self.attn.weight(), "rt_encoder.attn_sum.attn weights");
        let _ = print_first_few(self.decoder[0].weight(), "rt_decoder.nn.0 weights");
        // Print prelu weights
        let _ = println!("rt_decoder.nn.1 (prelu weights) shape: {:?}", self.prelu.weight().shape());
        let _ = println!("rt_decoder.nn.1 (prelu weights) weights: {:?}", self.prelu.weight().to_vec1::<f32>());

        let _ = print_first_few(self.decoder[1].weight(), "rt_decoder.nn.2 weights");

    }
}

// Helper Methods

impl<'a> RTCNNLSTMModel<'a> {
    

    /// Convert peptide sequences into AA ID array.
    /// 
    /// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
    pub fn get_aa_indices(seq_array: &[String]) -> Result<Array2<i64>> {
        let seq_len = seq_array[0].len();
        let mut result = Array2::<i64>::zeros((seq_array.len(), seq_len + 2));
    
        for (i, seq) in seq_array.iter().enumerate() {
            for (j, c) in seq.chars().enumerate() {
                let aa_index = (c as i64) - ('A' as i64) + 1;
                result[[i, j + 1]] = aa_index;
            }
        }
        
        Ok(result)
    }

    /// Convert peptide sequences into ASCII code array.
    /// 
    /// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L115
    pub fn get_ascii_indices(&self, peptide_sequences: &[String]) -> Result<Tensor> {
        // println!("Peptide sequences to encode: {:?}", peptide_sequences);
        let max_len = peptide_sequences.iter().map(|s| s.len()).max().unwrap_or(0) + 2; // +2 for padding
        let batch_size = peptide_sequences.len();
        
        let mut aa_indices = vec![0u32; batch_size * max_len];
        
        for (i, peptide) in peptide_sequences.iter().enumerate() {
            for (j, c) in peptide.chars().enumerate() {
                aa_indices[i * max_len + j + 1] = c as u32; // +1 to skip the first padding
            }
        }
        
        // println!("AA indices: {:?}", aa_indices);
        
        let aa_indices_tensor = Tensor::from_slice(&aa_indices, (batch_size, max_len), &self.device)?;
        Ok(aa_indices_tensor)
    }

    /// One-hot encode amino acid indices.
    fn aa_one_hot(&self, aa_indices: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = aa_indices.shape().dims2()?;
        let num_classes = self.constants.aa_embedding_size;
    
        let mut one_hot_data = vec![0.0f32; batch_size * seq_len * num_classes];
    
        // Iterate over the 2D tensor directly
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let index = aa_indices.get(batch_idx)?.get(seq_idx)?.to_scalar::<f32>()?;
                let class_idx = index.round() as usize; // Round to nearest integer and convert to usize
                if class_idx < num_classes {
                    one_hot_data[batch_idx * seq_len * num_classes + seq_idx * num_classes + class_idx] = 1.0;
                }
            }
        }
    
        // Convert the one_hot_data vector directly to a tensor
        Tensor::from_slice(&one_hot_data, (batch_size, seq_len, num_classes), aa_indices.device())
            .map_err(|e| anyhow!("{}", e))
    }

    /// Get the modification features for a given set of modifications and modification sites.
    /// 
    /// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L47
    fn get_mod_features(&self, mods: &str, mod_sites: &str, seq_len: usize) -> Result<Tensor> {
        let mod_names: Vec<&str> = mods.split(';').filter(|&s| !s.is_empty()).collect();
        let mod_sites: Vec<usize> = mod_sites
            .split(';')
            .filter(|&s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .collect();

        let mod_feature_size = self.constants.mod_elements.len();
    
        let mut mod_x = vec![0.0f32; seq_len * mod_feature_size];
    
        for (mod_name, &site) in mod_names.iter().zip(mod_sites.iter()) {
            if let Some(feat) = self.mod_to_feature.get(*mod_name) {
                for (i, &value) in feat.iter().enumerate() {
                    if site < seq_len {
                        mod_x[site * mod_feature_size + i] += value;
                    }
                }
                // println!("Site: {}, feat: {:?}", site, feat);
            }
        }
    
        Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), &self.device)
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))
    }
}

// Module Trait Implementation

impl<'a> Module for RTCNNLSTMModel<'a> {
    /// Forward pass for the model.
    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    
        let (batch_size, seq_len, _) = x.shape().dims3()?;
        // println!("Input shape: {:?}", x.shape());
        // print_tensor(&x)?;

        // Transpose the input tensor to match the CNN layer expectations
        let x_transposed = x.transpose(1, 2)?;
        // println!("Input transposed shape: {:?}", x_transposed.shape());
        // print_tensor(&x_transposed)?;

        // CNN layers
        let cnn_short = self.cnn_short.forward(&x_transposed)?;
        let cnn_medium = self.cnn_medium.forward(&x_transposed)?;
        let cnn_long = self.cnn_long.forward(&x_transposed)?;

        // Concatenate CNN outputs, including the original input (residual connection)
        let cnn_out = Tensor::cat(&[x_transposed, cnn_short, cnn_medium, cnn_long], 1)?;

        // Transpose back to (batch_size, seq_len, features)
        let cnn_out_transposed = cnn_out.transpose(1, 2)?;

        // Convert to float32 if necessary (should already be float32)
        let cnn_out_f32 = cnn_out_transposed.to_dtype(candle_core::DType::F32)?;

        // Ensure LSTM input has the correct shape (140 features)
        let lstm_input = if cnn_out_f32.dim(2)? > 140 {
            cnn_out_f32.narrow(2, 0, 140)?
        } else if cnn_out_f32.dim(2)? < 140 {
            let padding = Tensor::zeros((batch_size, seq_len, 140 - cnn_out_f32.dim(2)?), cnn_out_f32.dtype(), cnn_out_f32.device())?;
            Tensor::cat(&[cnn_out_f32, padding], 2)?
        } else {
            cnn_out_f32
        };

        // print_tensor(&lstm_input, 4)?;

        // Pass the correctly shaped input to the LSTM
        let lstm_out = self.bilstm.forward(&lstm_input)?;

        // print_tensor(&lstm_out, 4, None, Some(3))?;

        // Attention
        let attn_scores = self.attn.forward(&lstm_out)?;
    
        // Ensure attn_scores has the correct shape before softmax
        let attn_scores_reshaped = attn_scores.squeeze(2)?; // Remove last dimension if it's 1
    
        let attn_weights = ops::softmax(&attn_scores_reshaped, D::Minus1)?; // Apply softmax over the sequence dimension

        // Ensure attn_weights has the correct shape for multiplication
        let attn_weights_expanded = attn_weights.unsqueeze(2)?;
    
        // Use broadcasting for multiplication
        let weighted_lstm_out = lstm_out.broadcast_mul(&attn_weights_expanded)?;

        let context = weighted_lstm_out.sum(1)?; // Sum over the sequence dimension (second dimension)

        // Decoder
        let mut x = self.decoder[0].forward(&context)?;

        x = self.prelu.forward(&x)?;

        x = self.decoder[1].forward(&x)?;

        // Dropout
        if self.is_training {
            x = ops::dropout(&x, self.dropout)?;
        }
    
        // Ensure the output has the correct shape (batch_size,)
        let output = x.squeeze(1)?;  // Remove the last dimension which should be 1

        Ok(output)
    }
    
}


#[cfg(test)]
mod tests {
    use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
    use crate::model_interface::ModelInterface;
    use candle_core::Device;
    use std::path::PathBuf;
    // use itertools::izip;

    use super::*;

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/rt.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size, 27);
        assert_eq!(constants.charge_factor, 0.1);
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        assert_eq!(constants.nce_factor, 0.01);
    }

    #[test]
    fn test_aa_one_hot() {
        let device = Device::Cpu;
        
        // Create aa_indices similar to how you're doing it in encode_peptides
        let aa_indices_vec: Vec<i64> = vec![0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0].into_iter().map(|x| x as i64).collect();
        let aa_indices_tensor = Tensor::from_slice(&aa_indices_vec, (1, aa_indices_vec.len()), &device).unwrap();
        let aa_indices_tensor = aa_indices_tensor.to_dtype(DType::F32).unwrap();

        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        
        assert!(model_path.exists(), "Test model file does not exist");
        assert!(constants_path.exists(), "Test constants file does not exist");

        let model = RTCNNLSTMModel::new(model_path, constants_path, Device::Cpu).unwrap();

        let aa_one_hot = model.aa_one_hot(&aa_indices_tensor).unwrap();
        
        // Check the shape
        let (batch_size, seq_len, num_classes) = aa_one_hot.shape().dims3().unwrap();
        assert_eq!((batch_size, seq_len, num_classes), (1, 13, model.constants.aa_embedding_size));
    }

        
}