// In model_interface.rs
use crate::utils::peptdeep_utils::ModificationMap;
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::ops::{Index, IndexMut};

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
        training_data: &[(String, f32)],
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

    fn process_predictions(&self, predicts: &Tensor, min_inten: f32) -> Result<Tensor> {
        println!("predicts shape: {:?}", predicts.shape());

        // Reshape and get max
        let (batch_size, seq_len, feature_size) = predicts.shape().dims3()?;
        let reshaped = predicts.reshape((batch_size, ()))?;
        let apex_intens = reshaped.max(1)?;
        println!("apex_intens shape: {:?}", apex_intens.shape());
        // println!("apex_intens: {:?}", apex_intens.to_vec1::<f32>()?);

        // Replace values <= 0 with 1
        // let ones = Tensor::ones_like(&apex_intens)?;
        let apex_intens = apex_intens.maximum(&apex_intens)?;
        println!("apex_intens: {:?}", apex_intens.to_vec1::<f32>()?);

        // Reshape apex_intens for broadcasting
        let apex_intens_reshaped = apex_intens.reshape(((), 1, 1))?;
        println!("apex_intens_reshaped shape: {:?}", apex_intens_reshaped.shape());
        println!("apex_intens_reshaped: {:?}", apex_intens_reshaped.to_vec3::<f32>()?);
        
        // Explicitly broadcast apex_intens_reshaped to match predicts shape
        let broadcasted_apex_intens = apex_intens_reshaped.broadcast_as(predicts.shape())?;

        // Divide predicts by broadcasted apex_intens
        let normalized = predicts.div(&broadcasted_apex_intens)?;

        // Replace values < min_inten with 0.0
        let zeros = Tensor::zeros_like(&normalized)?;
        let min_inten_tensor = Tensor::full(min_inten, normalized.shape(), normalized.device())?;
        let mask = normalized.ge(&min_inten_tensor)?;
        let final_predicts = mask.where_cond(&normalized, &zeros)?;

        println!("predicts: {:?}", final_predicts.to_vec3::<f32>()?);

        Ok(final_predicts)
    }
}
