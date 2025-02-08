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
use crate::models::model_interface::{ModelInterface, PredictionResult};
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

// Method to create VarMap from loaded weights
pub fn create_var_map(var_map: &mut VarMap, vb: &VarBuilder) -> Result<()> {
    // let mut var_map = VarMap::new();

    // Lock the internal data of VarMap for thread-safe access
    {
        let mut ws = var_map.data().lock().unwrap();

        // Populate VarMap with encoder parameters
        ws.insert(
            "rt_encoder.mod_nn.nn.weight".to_string(),
            Var::from_tensor(
                &vb.get((2, 103), "rt_encoder.mod_nn.nn.weight")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_short.weight".to_string(),
            Var::from_tensor(
                &vb.get((35, 35, 3), "rt_encoder.input_cnn.cnn_short.weight")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_short.bias".to_string(),
            Var::from_tensor(
                &vb.get(35, "rt_encoder.input_cnn.cnn_short.bias")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_medium.weight".to_string(),
            Var::from_tensor(
                &vb.get((35, 35, 5), "rt_encoder.input_cnn.cnn_medium.weight")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_medium.bias".to_string(),
            Var::from_tensor(
                &vb.get(35, "rt_encoder.input_cnn.cnn_medium.bias")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_long.weight".to_string(),
            Var::from_tensor(
                &vb.get((35, 35, 7), "rt_encoder.input_cnn.cnn_long.weight")?,
            )?,
        );
        ws.insert(
            "rt_encoder.input_cnn.cnn_long.bias".to_string(),
            Var::from_tensor(
                &vb.get(35, "rt_encoder.input_cnn.cnn_long.bias")?,
            )?,
        );

        // Add Bidirectional LSTM parameters
        let num_layers = 2; // Number of layers
        let hidden_size = 128; // Hidden size

        // Initial hidden and cell states
        ws.insert(
            "rt_encoder.hidden_nn.rnn_h0".to_string(),
            Var::from_tensor(&vb.get(
                (num_layers * 2, 1, hidden_size),
                "rt_encoder.hidden_nn.rnn_h0",
            )?)?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn_c0".to_string(),
            Var::from_tensor(&vb.get(
                (num_layers * 2, 1, hidden_size),
                "rt_encoder.hidden_nn.rnn_c0",
            )?)?,
        );

        // LSTM layer weights and biases for both layers and directions (hardcoded)

        // Layer 0 (Forward)
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_ih_l0".to_string(),
            Var::from_tensor(
                &vb.get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_hh_l0".to_string(),
            Var::from_tensor(
                &vb.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_ih_l0".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l0")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_hh_l0".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l0")?,
            )?,
        );

        // Layer 0 (Backward)
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse".to_string(),
            Var::from_tensor(
                &vb.get((512, 140), "rt_encoder.hidden_nn.rnn.weight_ih_l0_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse".to_string(),
            Var::from_tensor(
                &vb.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l0_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_ih_l0_reverse".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l0_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_hh_l0_reverse".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l0_reverse")?,
            )?,
        );

        // Layer 1 (Forward)
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_ih_l1".to_string(),
            Var::from_tensor(
                &vb.get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_hh_l1".to_string(),
            Var::from_tensor(
                &vb.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_ih_l1".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l1")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_hh_l1".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l1")?,
            )?,
        );

        // Layer 1 (Backward)
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse".to_string(),
            Var::from_tensor(
                &vb.get((512, 256), "rt_encoder.hidden_nn.rnn.weight_ih_l1_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse".to_string(),
            Var::from_tensor(
                &vb.get((512, 128), "rt_encoder.hidden_nn.rnn.weight_hh_l1_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_ih_l1_reverse".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_ih_l1_reverse")?,
            )?,
        );
        ws.insert(
            "rt_encoder.hidden_nn.rnn.bias_hh_l1_reverse".to_string(),
            Var::from_tensor(
                &vb.get(512, "rt_encoder.hidden_nn.rnn.bias_hh_l1_reverse")?,
            )?,
        );

        // Add attention parameters
        ws.insert(
            "rt_encoder.attn_sum.attn.0.weight".to_string(),
            Var::from_tensor(
                &vb.get((1, 256), "rt_encoder.attn_sum.attn.0.weight")?,
            )?,
        );

        // Add decoder parameters
        ws.insert(
            "rt_decoder.nn.0.weight".to_string(),
            Var::from_tensor(&vb.get((64, 256), "rt_decoder.nn.0.weight")?)?,
        );
        ws.insert(
            "rt_decoder.nn.0.bias".to_string(),
            Var::from_tensor(&vb.get(64, "rt_decoder.nn.0.bias")?)?,
        );
        ws.insert(
            "rt_decoder.nn.1.weight".to_string(),
            Var::from_tensor(&vb.get(1, "rt_decoder.nn.1.weight")?)?,
        );
        ws.insert(
            "rt_decoder.nn.2.weight".to_string(),
            Var::from_tensor(&vb.get((1, 64), "rt_decoder.nn.2.weight")?)?,
        );
        ws.insert(
            "rt_decoder.nn.2.bias".to_string(),
            Var::from_tensor(&vb.get(1, "rt_decoder.nn.2.bias")?)?,
        );
    }

    // Ok(var_map)
    Ok(())
}

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
        let vb = VarBuilder::from_pth(model_path, candle_core::DType::F32, &device)?;
 
        let mut varmap = candle_nn::VarMap::new();
        create_var_map(&mut varmap, &vb)?;
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

        // Combine aa_one_hot and mod_x
        let combined = Tensor::cat(
            &[aa_indices_tensor, mod_x],
            2,
        )?;

        Ok(combined)
    }

    /// Fine-tune the model on a dataset of peptide sequences and retention times.
    fn fine_tune(
        &mut self,
        training_data: &[(String, f32)],
        modifications: HashMap<(String, Option<char>), ModificationMap>,
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

                progress.inc();
                progress.update_description(&format!("[fine-tuning] Epoch {}: Loss: {}", epoch, loss.to_vec0::<f32>()?));
            }
            progress.update_description(&format!("[fine-tuning] Epoch {}: Avg. Loss: {}", epoch, total_loss / training_data.len() as f32));
            progress.finish();
        }
        
        Ok(())
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

    fn save(&self, path: &Path) -> Result<()> {
        //self.var_store.save(path)
        unimplemented!()
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

        let aa_indices_tensor =
            Tensor::from_slice(&aa_indices, (batch_size, max_len), &self.device)?;
        Ok(aa_indices_tensor)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        //self.var_store.save(path)
        unimplemented!()
    }
}

// Module Trait Implementation

impl<'a> Module for RTCNNLSTMModel<'a> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let (batch_size, seq_len, _) = xs.shape().dims3()?;
        
        // Separate input into aa_indices, mod_x
        let start_mod_x = 1;

        let aa_indices_out = xs.i((.., .., 0))?;
        let mod_x_out = xs.i((.., .., start_mod_x..start_mod_x + MOD_FEATURE_SIZE))?;

        let x = self.rt_encoder.forward(&aa_indices_out, &mod_x_out)?;
        let x = self.dropout.forward(&x, self.is_training)?;
        let x = self.rt_decoder.forward(&x)?;

        Ok(x.squeeze(1)?)
    }
}

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
        // model.print_weights();

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

        // Predictions prior to fine-tuning
        model.set_evaluation_mode();
        // model.print_weights();
        
        let mut total_error = 0.0;
        let mut count = 0;

        for (peptide, mods, mod_sites, observed_rt) in &test_peptides {
            // let start = Instant::now();
            match model.predict(&[peptide.to_string()], mods, mod_sites, None, None, None) {
                Ok(predictions) => {
                    // let io_time = Instant::now() - start;
                    assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                    let predicted_rt = predictions[0];
                    let error = (predicted_rt - observed_rt).abs();
                    total_error += error;
                    count += 1;
                    // println!(
                    //     "Peptide: {} (Mods: {}, Sites: {}), Predicted RT: {:.6}, Observed RT: {:.6}, Error: {:.6}, Time: {:8} ms",
                    //     peptide, mods, mod_sites, predicted_rt, observed_rt, error, io_time.as_millis()
                    // );
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
        // println!(
        //     "Fine-tuning model with {} peptides and {} epochs with learning rate: {}",
        //     training_data.len(),
        //     epochs,
        //     learning_rate
        // );
        let result = model.fine_tune(&training_data, modifications, learning_rate, epochs);
        assert!(
            result.is_ok(),
            "Failed to fine-tune model: {:?}",
            result.err()
        );

        model.set_evaluation_mode();

        let mut total_error = 0.0;
        let mut count = 0;

        for (peptide, mods, mod_sites, observed_rt) in test_peptides {
            // let start = Instant::now();
            match model.predict(&[peptide.to_string()], mods, mod_sites, None, None, None) {
                Ok(predictions) => {
                    // let io_time = Instant::now() - start;
                    assert_eq!(predictions.len(), 1, "Unexpected number of predictions");
                    let predicted_rt = predictions[0];
                    let error = (predicted_rt - observed_rt).abs();
                    total_error += error;
                    count += 1;
                    // println!(
                    //     "Peptide: {} (Mods: {}, Sites: {}), Predicted RT: {:.6}, Observed RT: {:.6}, Error: {:.6}, Time: {:8} ms",
                    //     peptide, mods, mod_sites, predicted_rt, observed_rt, error, io_time.as_millis()
                    // );
                }
                Err(e) => {
                    println!("Error during prediction for {}: {:?}", peptide, e);
                }
            }
        }

        let mean_absolute_error = total_error / count as f32;
        println!("Mean Absolute Error post fine-tuning: {:.6}", mean_absolute_error);
        // model.print_weights();

        // // Optionally, save the fine-tuned model
        // let save_path = PathBuf::from("data/models/alphapeptdeep/generic/rt_fine_tuned.pth");
        // let save_result = model.save(&save_path);
        // assert!(save_result.is_ok(), "Failed to save fine-tuned model: {:?}", save_result.err());
    }
}
