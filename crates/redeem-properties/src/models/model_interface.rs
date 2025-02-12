use crate::utils::data_handling::PeptideData;
// In model_interface.rs
use crate::utils::peptdeep_utils::ModificationMap;
use anyhow::Result;
use candle_core::cuda::DeviceId;
use candle_core::{Device, Tensor, Var};
use candle_nn::{VarMap, VarBuilder};
use std::collections::HashMap;
use std::path::Path;
use std::ops::{Index, IndexMut};
use std::sync::{Arc, Mutex};

use crate::models::{rt_model::RTModelWrapper, ccs_model::CCSModelWrapper, ms2_model::MS2ModelWrapper};

#[derive(Debug)]
pub enum PredictionResult {
    RTResult(Vec<f32>),
    IMResult(Vec<f32>),
    MS2Result(Vec<Vec<f32>>),
}

impl PredictionResult {
    pub fn len(&self) -> usize {
        match self {
            PredictionResult::RTResult(vec) => vec.len(),
            PredictionResult::IMResult(vec) => vec.len(),
            PredictionResult::MS2Result(vec) => vec.len(),
        }
    }
}

impl Index<usize> for PredictionResult {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            PredictionResult::RTResult(vec) => &vec[index],
            PredictionResult::IMResult(vec) => &vec[index],
            PredictionResult::MS2Result(vec) => &vec[index][0], // Assuming you want the first element of each inner vector
        }
    }
}


impl IndexMut<usize> for PredictionResult {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            PredictionResult::RTResult(vec) => &mut vec[index],
            PredictionResult::IMResult(vec) => &mut vec[index],
            PredictionResult::MS2Result(vec) => &mut vec[index][0], // Assuming you want the first element of each inner vector
        }
    }
}


impl PredictionResult {
    pub fn get(&self, i: usize, j: usize) -> Option<&f32> {
        match self {
            PredictionResult::RTResult(_) => None, 
            PredictionResult::IMResult(_) => None,
            PredictionResult::MS2Result(vec) => vec.get(i).and_then(|row| row.get(j)),
        }
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut f32> {
        match self {
            PredictionResult::RTResult(_) => None, 
            PredictionResult::IMResult(_) => None,
            PredictionResult::MS2Result(vec) => vec.get_mut(i).and_then(|row| row.get_mut(j)),
        }
    }
}

/// Creates a new `VarMap` and populates it with the given tensor data.
pub fn create_var_map(var_map: &mut VarMap, tensor_data: Vec<(String, Tensor)>, device: &Device) -> Result<()> {
    let mut ws = var_map.data().lock().unwrap();

    for (name, tensor) in tensor_data {
        // NOTE: This is a temporary hack-fix for LSTM weights, which use sigmoid, which currently throws an error on CUDA: `no cuda implementation for sigmoid`
        if name.contains("hidden") {
            ws.insert(name, Var::from_tensor(&tensor.to_device(&Device::Cpu)?)?);
        } else {
            ws.insert(name, Var::from_tensor(&tensor.to_device(device)?)?); 
        }
    }

    Ok(())
}


pub trait ModelInterface: Send + Sync {
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: P,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self>
    where
        Self: Sized;
    
    fn predict(
        &self,
        peptide_sequence: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<PredictionResult>;

    fn encode_peptides(
        &self,
        peptide_sequences: &[String],
        mods: &str,
        mod_sites: &str,
        charge: Option<i32>,
        nce: Option<i32>,
        intsrument: Option<&str>,
    ) -> Result<Tensor>;

    fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<(String, Option<char>), ModificationMap>,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()>;
    fn set_evaluation_mode(&mut self);
    fn set_training_mode(&mut self);
    fn print_summary(&self);
    fn print_weights(&self);
    fn save(&self, path: &Path) -> Result<()>;
    fn apply_min_pred_value(&self, tensor: &Tensor, min_pred_value: f32) -> Result<Tensor> {
        // Create a tensor with the same shape as the input, filled with min_pred_value
        let min_tensor = Tensor::full(min_pred_value, tensor.shape(), tensor.device())?;
        
        // Use the maximum operation to replace values less than min_pred_value
        Ok(tensor.maximum(&min_tensor)?)
    }

    // TODO: Maybe move to ms2_bert_model, since it's specific to that model
    fn process_predictions(&self, predicts: &Tensor, min_inten: f32) -> Result<Tensor> {
        // Reshape and get max
        let (batch_size, seq_len, feature_size) = predicts.shape().dims3()?;
        let reshaped = predicts.reshape((batch_size, ()))?;
        let apex_intens = reshaped.max(1)?;

        // Replace values <= 0 with 1
        // let ones = Tensor::ones_like(&apex_intens)?;
        let apex_intens = apex_intens.maximum(&apex_intens)?;

        // Reshape apex_intens for broadcasting
        let apex_intens_reshaped = apex_intens.reshape(((), 1, 1))?;
        
        // Explicitly broadcast apex_intens_reshaped to match predicts shape
        let broadcasted_apex_intens = apex_intens_reshaped.broadcast_as(predicts.shape())?;

        // Divide predicts by broadcasted apex_intens
        let normalized = predicts.div(&broadcasted_apex_intens)?;

        // Replace values < min_inten with 0.0
        let zeros = Tensor::zeros_like(&normalized)?;
        let min_inten_tensor = Tensor::full(min_inten, normalized.shape(), normalized.device())?;
        let mask = normalized.ge(&min_inten_tensor)?;
        let final_predicts = mask.where_cond(&normalized, &zeros)?;

        Ok(final_predicts)
    }
}


/// Parameters for the `predict` method of a `ModelInterface` implementation.
pub struct Parameters {
    /// The instrument data was acquired on. Refer to list of supported instruments in const yaml file.
    pub instrument: String,
    /// The nominal collision energy (NCE) used for data acquisition.
    pub nce: f32,
}

impl Parameters {
    /// Creates a new instance of `Parameters` with the given instrument and NCE.
    ///
    /// # Arguments
    ///
    /// * `instrument` - The instrument data was acquired on.
    /// * `nce` - The nominal collision energy (NCE) used for data acquisition.
    ///
    /// # Returns
    ///
    /// A new `Parameters` instance with the given instrument and NCE.
    ///
    /// # Example
    ///
    /// ```
    /// let params = Parameters::new("QE", 20);
    /// ```
    pub fn new(instrument: &str, nce: f32) -> Self {
        Parameters {
            instrument: instrument.to_string(),
            nce,
        }
    }
}

/// Represents a collection of deep learning models for various property prediction tasks.
///
/// This struct holds optional references to models for retention time (RT),
/// collision cross-section (CCS), and MS2 intensity predictions. Each model
/// is wrapped in an Arc<Mutex<>> for thread-safe shared ownership.
pub struct DLModels {
    /// Parameters for prediction models.
    pub params: Option<Parameters>,

    /// Optional retention time prediction model.
    pub rt_model: Option<Arc<Mutex<RTModelWrapper>>>,
    
    /// Optional collision cross-section prediction model.
    pub ccs_model: Option<Arc<Mutex<CCSModelWrapper>>>,
    
    /// Optional MS2 intensity prediction model.
    pub ms2_model: Option<Arc<Mutex<MS2ModelWrapper>>>,
}

impl DLModels {
    /// Creates a new instance of `DLModels` with all models set to `None`.
    ///
    /// # Returns
    ///
    /// A new `DLModels` instance with no models initialized.
    ///
    /// # Example
    ///
    /// ```
    /// let mut models = DLModels::new();
    /// 
    /// models.rt_model = Some(Arc::new(Mutex::new(RTModelWrapper::new())));
    /// 
    /// ```
    pub fn new() -> Self {
        DLModels {
            params: None,
            rt_model: None,
            ccs_model: None,
            ms2_model: None,
        }
    }

    /// Checks if any of the models are present (not None).
    ///
    /// # Returns
    ///
    /// `true` if at least one model is present, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// let mut models = DLModels::new();
    /// assert!(!models.is_not_empty());
    ///
    /// models.rt_model = Some(Arc::new(Mutex::new(RTModelWrapper::new())));
    /// assert!(models.is_not_empty());
    /// ```
    pub fn is_not_empty(&self) -> bool {
        self.rt_model.is_some() || self.ccs_model.is_some() || self.ms2_model.is_some()
    }
}
