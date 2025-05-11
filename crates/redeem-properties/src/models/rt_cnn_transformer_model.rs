use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Dropout, Module, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

use crate::building_blocks::building_blocks::{
    DecoderLinear, Encoder26aaModCnnTransformerAttnSum, MOD_FEATURE_SIZE,
};
use crate::models::model_interface::{ModelInterface, PropertyType, load_tensors_from_model, create_var_map};
use crate::utils::peptdeep_utils::{
    load_mod_to_feature,
    parse_model_constants, ModelConstants,
};
use crate::utils::utils::get_tensor_stats;


// Main Model Struct

#[derive(Clone)]
/// Represents an CNN-TF Retention Time model.
pub struct RTCNNTFModel {
    var_store: VarBuilder<'static>,
    varmap: VarMap,
    constants: ModelConstants,
    device: Device,
    mod_to_feature: HashMap<String, Vec<f32>>,
    dropout: Dropout,
    rt_encoder: Encoder26aaModCnnTransformerAttnSum,
    rt_decoder: DecoderLinear,
    is_training: bool,
}

// Automatically implement Send and Sync if all fields are Send and Sync
unsafe impl Send for RTCNNTFModel {}
unsafe impl Sync for RTCNNTFModel {}

// Core Model Implementation

impl ModelInterface for RTCNNTFModel {
    fn property_type(&self) -> PropertyType {
        PropertyType::RT
    }

    fn model_arch(&self) -> &'static str {
        "rt_cnn_tf"   
    }

    fn new_untrained(device: Device) -> Result<Self> {
        let mut varmap = VarMap::new();
        let varbuilder = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        log::trace!("[RTCNNTFModel] Initializing rt_encoder");
        let rt_encoder = Encoder26aaModCnnTransformerAttnSum::new(
            &varbuilder.pp("rt_encoder"),
            8,     // mod_hidden_dim
            140,   // hidden_dim
            256,   // ff_dim
            4,     // num_heads
            2,     // num_layers
            100,   // max_len
            0.1,   // dropout_prob
            &device
        )?;

        log::trace!("[RTCNNTFModel] Initializing rt_decoder");
        let rt_decoder = DecoderLinear::new(140, 1, &varbuilder.pp("rt_decoder"))?;
        let constants = ModelConstants::default();
        let mod_to_feature = load_mod_to_feature(&constants)?;

        Ok(Self {
            var_store: varbuilder,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout: Dropout::new(0.1),
            rt_encoder,
            rt_decoder,
            is_training: true,
        })
    }

    /// Create a new RTCNNTFModel from the given model and constants files.
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        _fixed_sequence_len: usize,
        _num_frag_types: usize,
        _num_modloss_types: usize,
        _mask_modloss: bool,
        device: Device,
    ) -> Result<Self> {
        let tensor_data = load_tensors_from_model(model_path.as_ref(), &device)?;
        let mut varmap = candle_nn::VarMap::new();
        create_var_map(&mut varmap, tensor_data, &device)?;
        let var_store = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let constants = match constants_path {
            Some(path) => parse_model_constants(path.as_ref().to_str().unwrap())?,
            None => ModelConstants::default(),
        };

        let mod_to_feature = load_mod_to_feature(&constants)?;
        let dropout = Dropout::new(0.1);

        let rt_encoder = Encoder26aaModCnnTransformerAttnSum::from_varstore(
            &var_store,
            8,      // mod_hidden_dim
            140,    // hidden_dim
            256,    // ff_dim
            4,      // num_heads
            2,      // num_layers
            100,    // max_len (set appropriately for your sequence length)
            0.1,    // dropout_prob
            vec!["rt_encoder.mod_nn.nn.weight"],
            vec![
                "rt_encoder.input_cnn.cnn_short.weight",
                "rt_encoder.input_cnn.cnn_medium.weight",
                "rt_encoder.input_cnn.cnn_long.weight",
            ],
            vec![
                "rt_encoder.input_cnn.cnn_short.bias",
                "rt_encoder.input_cnn.cnn_medium.bias",
                "rt_encoder.input_cnn.cnn_long.bias",
            ],
            "rt_encoder.input_transformer",
            vec!["rt_encoder.attn_sum.attn.0.weight"],
            &device,
        )?;
        

        let rt_decoder = DecoderLinear::from_varstore(
            &var_store,
            140,
            1,
            vec!["rt_decoder.nn.0.weight", "rt_decoder.nn.1.weight", "rt_decoder.nn.2.weight"],
            vec!["rt_decoder.nn.0.bias", "rt_decoder.nn.2.bias"]
        )?;

        Ok(Self {
            var_store,
            varmap,
            constants,
            device,
            mod_to_feature,
            dropout,
            rt_encoder,
            rt_decoder,
            is_training: false,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        let aa_indices_out = xs.i((.., .., 0))?;
        let (mean, min, max) = get_tensor_stats(&aa_indices_out)?;
        log::debug!("[RTCNNTFModel] aa_indices_out stats - min: {min}, max: {max}, mean: {mean}");
        let mod_x_out = xs.i((.., .., 1..1 + MOD_FEATURE_SIZE))?;    
         
        if mod_x_out.shape().elem_count() == 0  {
            log::error!("[RTCNNTFModel] mod_x_out is empty! shape: {:?}", mod_x_out.shape());
        } else {
            match get_tensor_stats(&mod_x_out) {
                Ok((mean, min, max)) => {
                    log::debug!("[RTCNNTFModel] mod_x_out stats - min: {min}, max: {max}, mean: {mean}");
                }
                Err(e) => {
                    log::error!("[RTCNNTFModel] Failed to compute stats for mod_x_out: {:?}", e);
                }
            }
        }        
        
        log::trace!("[RTCNNTFModel] aa_indices_out: {:?}, mod_x_out: {:?}", aa_indices_out, mod_x_out);
        let x = self.rt_encoder.forward(&aa_indices_out, &mod_x_out)?;
        log::trace!("[RTCNNTFModel] x.shape after rt_encoder: {:?}", x.shape());
        let (mean, min, max) = get_tensor_stats(&x)?;
        log::debug!("[RTCNNTFModel] rt_encoder output stats - min: {min}, max: {max}, mean: {mean}");
        let x = self.dropout.forward(&x, self.is_training)?;
        log::trace!("[RTCNNTFModel] x.shape after dropout: {:?}", x.shape());
        let (mean, min, max) = get_tensor_stats(&x)?;
        log::debug!("[RTCNNTFModel] dropout output stats - min: {min}, max: {max}, mean: {mean}");
        let x = self.rt_decoder.forward(&x)?;
        log::trace!("[RTCNNTFModel] x.shape after rt_decoder: {:?}", x.shape());
        let (mean, min, max) = get_tensor_stats(&x)?;
        log::debug!("[RTCNNTFModel] rt_decoder output stats - min: {min}, max: {max}, mean: {mean}");
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
        todo!("Implement print_weights for RTCNNTFModel");
    }


}


