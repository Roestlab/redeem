use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{ops, Conv1d, Conv1dConfig, Linear, Module, Optimizer, PReLU, VarBuilder, VarMap};
use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// use crate::models::rt_model::RTModel;
use crate::building_blocks::bilstm::BidirectionalLSTM;
use crate::model_interface::{ModelInterface, PredictionResult};
use crate::utils::peptdeep_utils::{
    extract_masses_and_indices, get_modification_indices, load_mod_to_feature, load_modifications,
    parse_model_constants, remove_mass_shift, ModelConstants, ModificationMap,
};

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

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl<'a> Send for RTCNNLSTMModel<'a> {}
unsafe impl<'a> Sync for RTCNNLSTMModel<'a> {}

// Core Model Implementation

impl<'a> ModelInterface for RTCNNLSTMModel<'a> {
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
        let var_store = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;

        let constants: ModelConstants =
            parse_model_constants(constants_path.as_ref().to_str().unwrap())?;

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
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        );

        let cnn_medium = Conv1d::new(
            var_store.get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight")?,
            Some(var_store.get(35, "rt_encoder.input_cnn.cnn_medium.bias")?),
            Conv1dConfig {
                padding: 2,
                ..Default::default()
            },
        );

        let cnn_long = Conv1d::new(
            var_store.get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight")?,
            Some(var_store.get(35, "rt_encoder.input_cnn.cnn_long.bias")?),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );

        let bilstm: BidirectionalLSTM = BidirectionalLSTM::new(140, 128, 2, &var_store.pp("rt_encoder.hidden_nn"))?;

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
    fn predict(
        &self,
        peptide_sequence: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<PredictionResult> {
        // Preprocess the peptide sequences and modifications
        let input_tensor =
            self.encode_peptides(peptide_sequence, mods, mod_sites, None, None, None)?;

        // Pass the data through the model
        let output = self.forward(&input_tensor)?;

        // Convert the output tensor to a Vec<f32>
        let predictions = PredictionResult::RTResult(output.to_vec1()?);

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
    fn encode_peptides(
        &self,
        peptide_sequences: &[String],
        mods: &str,
        mod_sites: &str,
        _charge: Option<i32>,
        _nce: Option<i32>,
        _intsrument: Option<&str>,
    ) -> Result<Tensor> {
        // println!("Peptide sequences to encode: {:?}", peptide_sequences);

        let aa_indices = Self::get_aa_indices(peptide_sequences)?;
        // println!("AA indices: {:?}", aa_indices);

        // Convert ndarray to Tensor and ensure it's F32
        let aa_indices_tensor = Tensor::from_slice(
            &aa_indices.as_slice().unwrap(),
            (aa_indices.shape()[0], aa_indices.shape()[1]),
            &self.device,
        )?
        .to_dtype(DType::F32)?;

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

    // fn fine_tune(&mut self, training_data: &[(String, f32)], modifications: HashMap <(String, Option<char>), ModificationMap>, learning_rate: f64, epochs: usize) -> Result<()> {
    //     unimplemented!()
    // }

    fn fine_tune(
        &mut self,
        training_data: &[(String, f32)],
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        let var_map = self.create_var_map()?;
        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(var_map.all_vars(), params)?;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (peptide, target_rt) in training_data {
                let naked_peptide = remove_mass_shift(&peptide.to_string());

                // Collect indices of non-zero modifications
                let modified_indices = get_modification_indices(&peptide.to_string());

                // Extract masses and indices
                let extracted_masses_and_indices = extract_masses_and_indices(&peptide.to_string());

                let mut found_modifications = Vec::new();

                // Map modifications based on extracted masses and indices
                for (mass, index) in extracted_masses_and_indices {
                    let amino_acid = peptide.to_string().chars().nth(index).unwrap_or('\0');
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

                // Forward pass
                let input =
                    self.encode_peptides(peptides_str, mod_str, mod_site_str, None, None, None)?;
                let predicted_rt = self.forward(&input)?;

                // Compute loss
                let loss = candle_nn::loss::mse(
                    &predicted_rt,
                    &Tensor::new(&[*target_rt], &self.device)?,
                )?;

                // Backward pass
                opt.backward_step(&loss)?;

                total_loss += loss.to_vec0::<f32>()?;
            }
            println!(
                "Epoch {}: Avg Loss: {}",
                epoch,
                total_loss / training_data.len() as f32
            );
        }
        Ok(())
    }

    /// Print a summary of the model's constants.
    fn print_summary(&self) {
        println!("RTModel Summary:");
        println!("AA Embedding Size: {}", self.constants.aa_embedding_size);
        println!("Charge Factor: {:?}", self.constants.charge_factor);
        println!("Instruments: {:?}", self.constants.instruments);
        println!("Max Instrument Num: {}", self.constants.max_instrument_num);
        println!("Mod Elements: {:?}", self.constants.mod_elements);
        println!("NCE Factor: {:?}", self.constants.nce_factor);
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
            println!(
                "{} (first few values): {:?}",
                name,
                flattened.narrow(0, 0, num_to_print)?.to_vec1::<f32>()?
            );
            Ok(())
        }

        // Print weights for each layer
        let _ = print_first_few(self.mod_nn.weight(), "rt_encoder.mod_nn.nn weights");
        let _ = print_first_few(
            self.cnn_short.weight(),
            "rt_encoder.input_cnn.cnn_short weights",
        );
        let _ = print_first_few(
            self.cnn_medium.weight(),
            "rt_encoder.input_cnn.cnn_medium weights",
        );
        let _ = print_first_few(
            self.cnn_long.weight(),
            "rt_encoder.input_cnn.cnn_long weights",
        );

        // Print bilstm weights
        let _ = self.bilstm.print_weights(&self.var_store);

        // Decoder
        let _ = print_first_few(self.attn.weight(), "rt_encoder.attn_sum.attn weights");
        let _ = print_first_few(self.decoder[0].weight(), "rt_decoder.nn.0 weights");
        // Print prelu weights
        let _ = println!(
            "rt_decoder.nn.1 (prelu weights) shape: {:?}",
            self.prelu.weight().shape()
        );
        let _ = println!(
            "rt_decoder.nn.1 (prelu weights) weights: {:?}",
            self.prelu.weight().to_vec1::<f32>()
        );

        let _ = print_first_few(self.decoder[1].weight(), "rt_decoder.nn.2 weights");
    }

    fn save(&self, path: &Path) -> Result<()> {
        //self.var_store.save(path)
        unimplemented!()
    }
}

// Helper Methods

impl<'a> RTCNNLSTMModel<'a> {
    // Method to create VarMap from loaded weights
    pub fn create_var_map(&self) -> Result<VarMap> {
        let mut var_map = VarMap::new();

        // Lock the internal data of VarMap for thread-safe access
        {
            let mut ws = var_map.data().lock().unwrap();

            // Populate VarMap with encoder parameters
            ws.insert(
                "rt_encoder.mod_nn.nn.weight".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((2, 103), "rt_encoder.mod_nn.nn.weight")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_short.weight".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((35, 35, 3), "rt_encoder.input_cnn.cnn_short.weight")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_short.bias".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(35, "rt_encoder.input_cnn.cnn_short.bias")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_medium.weight".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_medium.bias".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(35, "rt_encoder.input_cnn.cnn_medium.bias")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_long.weight".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight")?,
                )?,
            );
            ws.insert(
                "rt_encoder.input_cnn.cnn_long.bias".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(35, "rt_encoder.input_cnn.cnn_long.bias")?,
                )?,
            );

            // Add Bidirectional LSTM parameters
            let num_layers = 2; // Number of layers
            let hidden_size = 128; // Hidden size

            // Initial hidden and cell states
            ws.insert(
                "rt_encoder.hidden_nn.rnn_h0".to_string(),
                Var::from_tensor(&self.var_store.get(
                    (num_layers * 2, 1, hidden_size),
                    "rt_encoder.hidden_nn.rnn_h0",
                )?)?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn_c0".to_string(),
                Var::from_tensor(&self.var_store.get(
                    (num_layers * 2, 1, hidden_size),
                    "rt_encoder.hidden_nn.rnn_c0",
                )?)?,
            );

            // LSTM layer weights and biases for both layers and directions (hardcoded)

            // Layer 0 (Forward)
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_ih_l0".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_hh_l0".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_ih_l0".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l0")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_hh_l0".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l0")?,
                )?,
            );

            // Layer 0 (Backward)
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_ih_l0_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l0_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_hh_l0_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l0_reverse")?,
                )?,
            );

            // Layer 1 (Forward)
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_ih_l1".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_hh_l1".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_ih_l1".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l1")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_hh_l1".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l1")?,
                )?,
            );

            // Layer 1 (Backward)
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_ih_l1_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l1_reverse")?,
                )?,
            );
            ws.insert(
                "rt_encoder.hidden_nn.rnn.bias_hh_l1_reverse".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l1_reverse")?,
                )?,
            );

            // Add attention parameters
            ws.insert(
                "rt_encoder.attn_sum.attn.0.weight".to_string(),
                Var::from_tensor(
                    &self
                        .var_store
                        .get((1, 256), "rt_encoder.attn_sum.attn.0.weight")?,
                )?,
            );

            // Add decoder parameters
            ws.insert(
                "rt_decoder.nn.0.weight".to_string(),
                Var::from_tensor(&self.var_store.get((64, 256), "rt_decoder.nn.0.weight")?)?,
            );
            ws.insert(
                "rt_decoder.nn.0.bias".to_string(),
                Var::from_tensor(&self.var_store.get(64, "rt_decoder.nn.0.bias")?)?,
            );
            ws.insert(
                "rt_decoder.nn.1.weight".to_string(),
                Var::from_tensor(&self.var_store.get(1, "rt_decoder.nn.1.weight")?)?,
            );
            ws.insert(
                "rt_decoder.nn.2.weight".to_string(),
                Var::from_tensor(&self.var_store.get((1, 64), "rt_decoder.nn.2.weight")?)?,
            );
            ws.insert(
                "rt_decoder.nn.2.bias".to_string(),
                Var::from_tensor(&self.var_store.get(1, "rt_decoder.nn.2.bias")?)?,
            );
        }

        Ok(var_map)
    }

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

        let aa_indices_tensor =
            Tensor::from_slice(&aa_indices, (batch_size, max_len), &self.device)?;
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
                let index = aa_indices
                    .get(batch_idx)?
                    .get(seq_idx)?
                    .to_scalar::<f32>()?;
                let class_idx = index.round() as usize; // Round to nearest integer and convert to usize
                if class_idx < num_classes {
                    one_hot_data
                        [batch_idx * seq_len * num_classes + seq_idx * num_classes + class_idx] =
                        1.0;
                }
            }
        }

        // Convert the one_hot_data vector directly to a tensor
        Tensor::from_slice(
            &one_hot_data,
            (batch_size, seq_len, num_classes),
            aa_indices.device(),
        )
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

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        //self.var_store.save(path)
        unimplemented!()
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
            let padding = Tensor::zeros(
                (batch_size, seq_len, 140 - cnn_out_f32.dim(2)?),
                cnn_out_f32.dtype(),
                cnn_out_f32.device(),
            )?;
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
        let output = x.squeeze(1)?; // Remove the last dimension which should be 1

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::model_interface::ModelInterface;
    use crate::models::rt_cnn_lstm_model::RTCNNLSTMModel;
    use crate::utils::peptdeep_utils::load_modifications;
    use candle_core::Device;
    use std::path::PathBuf;
    use std::time::Instant;
    // use itertools::izip;

    use super::*;

    #[test]
    fn test_parse_model_constants() {
        let path = "data/models/alphapeptdeep/generic/rt.pth.model_const.yaml";
        let result = parse_model_constants(path);
        assert!(result.is_ok());
        let constants = result.unwrap();
        assert_eq!(constants.aa_embedding_size, 27);
        assert_eq!(constants.charge_factor, Some(0.1));
        assert_eq!(constants.instruments.len(), 4);
        assert_eq!(constants.max_instrument_num, 8);
        assert_eq!(constants.mod_elements.len(), 109);
        assert_eq!(constants.nce_factor, Some(0.01));
    }

    #[test]
    fn test_aa_one_hot() {
        let device = Device::Cpu;

        // Create aa_indices similar to how you're doing it in encode_peptides
        let aa_indices_vec: Vec<i64> = vec![0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]
            .into_iter()
            .map(|x| x as i64)
            .collect();
        let aa_indices_tensor =
            Tensor::from_slice(&aa_indices_vec, (1, aa_indices_vec.len()), &device).unwrap();
        let aa_indices_tensor = aa_indices_tensor.to_dtype(DType::F32).unwrap();

        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let model =
            RTCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, Device::Cpu).unwrap();

        let aa_one_hot = model.aa_one_hot(&aa_indices_tensor).unwrap();

        // Check the shape
        let (batch_size, seq_len, num_classes) = aa_one_hot.shape().dims3().unwrap();
        assert_eq!(
            (batch_size, seq_len, num_classes),
            (1, 13, model.constants.aa_embedding_size)
        );
    }

    #[test]
    fn test_variable_map_creation() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

        assert!(model_path.exists(), "Test model file does not exist");
        assert!(
            constants_path.exists(),
            "Test constants file does not exist"
        );

        let model =
            RTCNNLSTMModel::new(model_path, constants_path, 0, 8, 4, true, Device::Cpu).unwrap();

        let var_map = match model.create_var_map() {
            Ok(vars) => vars,
            Err(e) => {
                panic!("Failed to create var map: {:?}", e);
            }
        };

        println!("VarMap created successfully");
        // // Check that the VarMap contains the expected number of variables
        // assert_eq!(var_map.len(), 21);
    }

    #[test]
    fn peptide_retention_time_prediction() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        // let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt_resaved_model.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

        assert!(
            model_path.exists(),
            "\n╔══════════════════════════════════════════════════════════════════╗\n\
             ║                     *** ERROR: FILE NOT FOUND ***                ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ Test model file does not exist:                                  ║\n\
             ║ {:?}\n\
             ║ \n\
             ║ Visit AlphaPeptDeeps github repo on how to download their \n\
             ║ pretrained model files: https://github.com/MannLabs/\n\
             ║ alphapeptdeep?tab=readme-ov-file#install-models\n\
             ╚══════════════════════════════════════════════════════════════════╝\n",
            model_path
        );

        assert!(
            constants_path.exists(),
            "\n╔══════════════════════════════════════════════════════════════════╗\n\
             ║                     *** ERROR: FILE NOT FOUND ***                  ║\n\
             ╠══════════════════════════════════════════════════════════════════╣\n\
             ║ Test constants file does not exist:                                ║\n\
             ║ {:?}\n\
             ║ \n\
             ║ Visit AlphaPeptDeeps github repo on how to download their \n\
             ║ pretrained model files: https://github.com/MannLabs/\n\
             ║ alphapeptdeep?tab=readme-ov-file#install-models\n\
             ╚══════════════════════════════════════════════════════════════════╝\n",
            constants_path
        );

        let result = RTCNNLSTMModel::new(&model_path, &constants_path, 0, 8, 4, true, Device::Cpu);

        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

        let mut model = result.unwrap();
        model.print_summary();

        // Print the model's weights
        model.print_weights();

        // Test prediction with real peptides
        let peptide = "AGHCEWQMKYR".to_string();
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M".to_string();
        let mod_sites = "0;4;8".to_string();

        println!("Predicting retention time for peptide: {:?}", peptide);
        println!("Modifications: {:?}", mods);
        println!("Modification sites: {:?}", mod_sites);

        model.set_evaluation_mode();

        let start = Instant::now();
        match model.predict(&[peptide.clone()], &mods, &mod_sites, None, None, None) {
            Ok(predictions) => {
                let io_time = Instant::now() - start;
                assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                println!("Prediction for real peptide:");
                println!(
                    "Peptide: {} ({} @ {}), Predicted RT: {}:  {:8} ms",
                    peptide,
                    mods,
                    mod_sites,
                    predictions[0],
                    io_time.as_millis()
                );
            }
            Err(e) => {
                println!("Error during prediction: {:?}", e);
                println!("Attempting to encode peptide...");
                match model.encode_peptides(&[peptide.clone()], &mods, &mod_sites, None, None, None)
                {
                    Ok(encoded) => println!("Encoded peptide shape: {:?}", encoded.shape()),
                    Err(e) => println!("Error encoding peptide: {:?}", e),
                }
            }
        }
    }

    #[test]
    fn peptide_retention_time_prediction_and_fine_tuning() {
        let model_path = PathBuf::from("data/models/alphapeptdeep/generic/rt.pth");
        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");

        // Assert model and constants files exist (your existing assertions)

        let result = RTCNNLSTMModel::new(&model_path, &constants_path, 0, 8, 4, true, Device::Cpu);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());
        let mut model = result.unwrap();

        // Define training data
        let training_data = vec![
            ("AKPLMELIER".to_string(), 0.4231399),
            ("TEM[+15.9949]VTISDASQR".to_string(), 0.2192762),
            ("AGKFPSLLTHNENMVAK".to_string(), 0.3343900),
            ("LSELDDRADALQAGASQFETSAAK".to_string(), 0.5286755),
            ("FLLQDTVELR".to_string(), 0.6522490),
            ("SVTEQGAELSNEER".to_string(), 0.2388270),
            ("EHALLAYTLGVK".to_string(), 0.5360210),
            ("HGAGAEISTVNPEQYSK".to_string(), 0.2608499),
            ("LLHIEELR".to_string(), 0.3490962),
            ("QFFVAVER".to_string(), 0.4522124),
            ("VEVERDNLAEDIM[+15.9949]R".to_string(), 0.3864688),
            ("AMKDEEKMELQEMQLK".to_string(), 0.3475575),
            ("DVKPSNVLINK".to_string(), 0.2627346),
            ("DTSEDIEELVEPVAAHGPK".to_string(), 0.6682840),
            ("SLTNDWEDHLAVK".to_string(), 0.4844666),
            ("AAVPSGASTGIYEALELRDNDK".to_string(), 0.5861278),
            ("YWQQVIDM[+15.9949]NDYQR".to_string(), 0.4894387),
            ("GALTVGITNTVGSSISR".to_string(), 0.5353391),
            ("TALVANTSNMPVAAR".to_string(), 0.3763506),
            ("QLVEQVEQIQK".to_string(), 0.3484997),
            ("LALGQINIAK".to_string(), 0.5013100),
            ("GAVWGATLNK".to_string(), 0.3454413),
            ("MREEVITLIR".to_string(), 0.4111288),
            ("GGIM[+15.9949]LPEK".to_string(), 0.2743906),
            ("SQWSPALTVSK".to_string(), 0.4027683),
            ("GSNNVALGYDEGSIIVK".to_string(), 0.5345799),
            ("DIVENYFMR".to_string(), 0.6564170),
            ("VIHDNFGIVEGLM[+15.9949]TTVHAITATQK".to_string(), 0.8519899),
            ("SIVEEIEDLVAR".to_string(), 0.8994804),
            ("TIAEC[+57.0215]LADELINAAK".to_string(), 0.8946646),
            ("VALVYGQMNEPPGAR".to_string(), 0.4537694),
            ("HFTILDAPGHK".to_string(), 0.2430375),
            ("TMLITHMQDLQEVTQDLHYENFR".to_string(), 0.7410552),
            ("VAEEHAPSIVFIDEIDAIGTKR".to_string(), 0.6844217),
            ("LYVSNLGIGHTR".to_string(), 0.3350410),
            ("TPELNLDQFHDKTPYTIMFGPDK".to_string(), 0.7189609),
            ("VKLEAEIATYR".to_string(), 0.3441208),
            ("KVYAAIEAGDK".to_string(), 0.1530408),
            ("VLEALLPLK".to_string(), 0.6958856),
            ("SITVLVEGENTR".to_string(), 0.4079256),
            ("TVQSLEIDLDSM[+15.9949]R".to_string(), 0.5787880),
            ("SKGHYEVTGSDDETGKLQGSGVSLASK".to_string(), 0.2422714),
            ("ETNLDSLPLVDTHSK".to_string(), 0.4655342),
            ("LGPLSVFSANKR".to_string(), 0.4307850),
            ("DLMVGDEASELR".to_string(), 0.5141643),
            ("HSTPHAAFQPNSQIGEEMSQNSFIK".to_string(), 0.4549789),
            ("FSGEELDKLWR".to_string(), 0.5522011),
            ("FSSELEQIELHNSIR".to_string(), 0.4875712),
            ("NLDDGIDDERLR".to_string(), 0.3055077),
            ("TKPQDMISAGGESVAGITAISGKPGDK".to_string(), 0.4904854),
            ("KPLLPYTPGSDVAGVIEAVGDNASAFK".to_string(), 0.8709913),
            ("DALSDLALHFLNK".to_string(), 0.8977105),
            ("NYQQNYQNSESGEKNEGSESAPEGQAQQR".to_string(), 0.1815779),
            ("ITGEAFVQFASQELAEK".to_string(), 0.8284199),
            ("QLEDILVLAK".to_string(), 0.7016399),
            ("ATLWYVPLSLK".to_string(), 0.8360704),
            ("LNLEAINYMAADGDFK".to_string(), 0.8471803),
            ("TTGIVMDSGDGVTHTVPIYEGYALPHAILR".to_string(), 0.7528398),
            ("KIKDPDASKPEDWDER".to_string(), 0.1692563),
            ("TDQVIQSLIALVNDPQPEHPLR".to_string(), 0.8987013),
            ("EGAKDIDISSPEFK".to_string(), 0.3499846),
            ("KILATPPQEDAPSVDIANIR".to_string(), 0.5166478),
            ("VVSQYSSLLSPMSVNAVM[+15.9949]K".to_string(), 0.6726230),
            ("KQVVNIPSFIVR".to_string(), 0.5296390),
            ("NPVWYQALTHGLNEEQR".to_string(), 0.6143036),
            ("EDYKFHHTFSTEIAK".to_string(), 0.2450278),
            ("IEKLEEYITTSK".to_string(), 0.3518330),
            ("HFSVEGQLEFR".to_string(), 0.7494319),
            ("TFLALINQVFPAEEDSKK".to_string(), 0.8345350),
        ];

        // Fine-tune the model
        let modifications = match load_modifications() {
            Ok(mods) => mods,
            Err(e) => {
                panic!("Failed to load modifications: {:?}", e);
            }
        };
        let learning_rate = 0.001;
        let epochs = 5;
        println!(
            "Fine-tuning model with {} peptides and {} epochs with learning rate: {}",
            training_data.len(),
            epochs,
            learning_rate
        );
        let result = model.fine_tune(&training_data, modifications, learning_rate, epochs);
        assert!(
            result.is_ok(),
            "Failed to fine-tune model: {:?}",
            result.err()
        );

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

        model.set_evaluation_mode();

        let mut total_error = 0.0;
        let mut count = 0;

        for (peptide, mods, mod_sites, observed_rt) in test_peptides {
            let start = Instant::now();
            match model.predict(&[peptide.to_string()], mods, mod_sites, None, None, None) {
                Ok(predictions) => {
                    let io_time = Instant::now() - start;
                    assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                    let predicted_rt = predictions[0];
                    let error = (predicted_rt - observed_rt).abs();
                    total_error += error;
                    count += 1;
                    println!(
                        "Peptide: {} (Mods: {}, Sites: {}), Predicted RT: {:.6}, Observed RT: {:.6}, Error: {:.6}, Time: {:8} ms",
                        peptide, mods, mod_sites, predicted_rt, observed_rt, error, io_time.as_millis()
                    );
                }
                Err(e) => {
                    println!("Error during prediction for {}: {:?}", peptide, e);
                }
            }
        }

        let mean_absolute_error = total_error / count as f32;
        println!("Mean Absolute Error: {:.6}", mean_absolute_error);

        // // Optionally, save the fine-tuned model
        // let save_path = PathBuf::from("data/models/alphapeptdeep/generic/rt_fine_tuned.pth");
        // let save_result = model.save(&save_path);
        // assert!(save_result.is_ok(), "Failed to save fine-tuned model: {:?}", save_result.err());
    }
}
