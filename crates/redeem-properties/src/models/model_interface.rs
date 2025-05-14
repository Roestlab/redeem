use crate::{
    building_blocks::featurize::{
        self, aa_indices_tensor, aa_indices_tensor_from_arc, get_mod_features_from_parsed,
        get_mod_features_from_parsed_arc,
    },
    models::{ccs_model::CCSModelWrapper, ms2_model::MS2ModelWrapper, rt_model::RTModelWrapper},
    utils::{
        data_handling::{PeptideBatchData, PeptideData, TargetNormalization},
        logging::Progress,
        peptdeep_utils::{
            get_modification_indices, get_modification_string, parse_instrument_index,
            remove_mass_shift,
        },
        stats::{compute_loss_stats, Metrics, TrainingPhase, TrainingStepMetrics},
        utils::{get_tensor_stats, CosineWithWarmup, LRScheduler},
    },
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Optimizer, VarMap};
use log::info;
use rayon::prelude::*;
use std::path::Path;
use std::{collections::HashMap, path::PathBuf};
use std::{
    ops::{Deref, Index},
    process::Output,
    sync::Arc,
};

// Constants
const CHARGE_FACTOR: f64 = 0.1;
const NCE_FACTOR: f64 = 0.01;

/// Load tensors from a model file.
///
/// Supported model formats include:
/// - PyTorch (.pt, .pth, .pkl)
/// - SafeTensors (.safetensors)
///
/// # Arguments
/// * `model_path` - Path to the model file.
/// * `device` - Device to load the tensors on.
///
/// # Returns
/// A vector of tuples containing the tensor names and their corresponding tensors.
pub fn load_tensors_from_model<P: AsRef<Path>>(
    model_path: P,
    device: &Device,
) -> Result<Vec<(String, Tensor)>> {
    let path: &Path = model_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "pt" | "pth" | "pkl" => {
            log::trace!("Loading tensors from PyTorch model: {:?}", path);
            let tensor_data = candle_core::pickle::read_all(path)
                .with_context(|| format!("Failed to read PyTorch model from: {:?}", path))?;
            Ok(tensor_data)
        }
        "safetensors" => {
            log::trace!("Loading tensors from SafeTensors model: {:?}", path);
            let tensor_data = candle_core::safetensors::load(path, device)
                .with_context(|| format!("Failed to load SafeTensors from: {:?}", path))?;

            // Convert HashMap<String, Tensor> to Vec<(String, Tensor)>
            let tensors = tensor_data.into_iter().collect();

            Ok(tensors)
        }
        _ => Err(anyhow::anyhow!("Unsupported model format: {:?}", path)),
    }
}

/// Represents the type of property to predict.
#[derive(Clone)]
pub enum PropertyType {
    RT,
    CCS,
    MS2,
}

impl PropertyType {
    pub fn as_str(&self) -> &str {
        match self {
            PropertyType::RT => "RT",
            PropertyType::CCS => "CCS",
            PropertyType::MS2 => "MS2",
        }
    }
}

/// Represents a single prediction value or a matrix of prediction values.
///
/// This enum is used to store the output of a model prediction, which can be a single value or a matrix of values. For example, retention time (RT) and collision cross-section (CCS) predictions are single values, while MS2 intensity predictions are matrices.
#[derive(Clone)]
pub enum PredictionValue {
    Single(f32),
    Matrix(Vec<Vec<f32>>),
}

impl PredictionValue {
    // Returns a reference to the element at position (i, j) if it exists
    pub fn get(&self, i: usize, j: usize) -> Option<&f32> {
        match self {
            PredictionValue::Single(_) => None,
            PredictionValue::Matrix(vec) => vec.get(i).and_then(|row| row.get(j)),
        }
    }
}

impl Index<usize> for PredictionValue {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            PredictionValue::Single(val) => val,
            PredictionValue::Matrix(matrix) => &matrix[index][0], // Single index for Matrix (first element of each row)
        }
    }
}

impl Index<(usize, usize)> for PredictionValue {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        match self {
            PredictionValue::Single(val) => val, // Single variant does not support multi-indexing
            PredictionValue::Matrix(matrix) => &matrix[index.0][index.1], // Multi-index for Matrix
        }
    }
}

/// Represents the output of a model prediction.
///
/// This enum is used to store the output of a model prediction, which can be a vector of retention times (RT), collision cross-sections (CCS), or a vector matrices of MS2 intensities.
#[derive(Debug, Clone)]
pub enum PredictionResult {
    RTResult(Vec<f32>),
    CCSResult(Vec<f32>),
    MS2Result(Vec<Vec<Vec<f32>>>),
}

impl PredictionResult {
    pub fn len(&self) -> usize {
        match self {
            PredictionResult::RTResult(vec) => vec.len(),
            PredictionResult::CCSResult(vec) => vec.len(),
            PredictionResult::MS2Result(vec) => vec.len(),
        }
    }

    pub fn get_prediction_entry(&self, index: usize) -> PredictionValue {
        match self {
            PredictionResult::RTResult(vec) => PredictionValue::Single(vec[index].clone()),
            PredictionResult::CCSResult(vec) => PredictionValue::Single(vec[index].clone()),
            PredictionResult::MS2Result(vec) => PredictionValue::Matrix(vec[index].clone()),
        }
    }
}

/// Populates a mutable `VarMap` instance with tensors.
///
/// # Arguments
/// * `var_map` - A mutable reference to a `VarMap` instance.
/// * `tensor_data` - A vector of tuples containing the tensor names and their corresponding tensors.
/// * `device` - The device to load the tensors on.
pub fn create_var_map(
    var_map: &mut VarMap,
    tensor_data: Vec<(String, Tensor)>,
    device: &Device,
) -> Result<()> {
    let mut ws = var_map.data().lock().unwrap();

    for (name, tensor) in tensor_data {
        ws.insert(name, Var::from_tensor(&tensor.to_device(device)?)?);
    }

    Ok(())
}

pub trait ModelClone {
    fn clone_box(&self) -> Box<dyn ModelInterface + Send + Sync>;
}

impl<T> ModelClone for T
where
    T: 'static + ModelInterface + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn ModelInterface + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelInterface + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Represents an abstract deep learning model interface.
///
/// This trait defines the methods and properties that a deep learning model must implement to be used for property prediction tasks.
pub trait ModelInterface: Send + Sync + ModelClone {
    /// Get the property type of the model.
    fn property_type(&self) -> PropertyType;

    /// Get the model architecture name.
    fn model_arch(&self) -> &'static str;

    /// Create a new model instance from scratch (no pretrained weights).
    /// This is typically used when training a new model from scratch.
    fn new_untrained(device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Create a new instance of the model (given a pretrained model (.pth or .safetensors and constants file).
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file (.pth or .safetensors).
    /// * `constants_path` - Optional path to the model constants file (.yaml). If none, will use the default constants.
    ///
    fn new<P: AsRef<Path>>(
        model_path: P,
        constants_path: Option<P>,
        fixed_sequence_len: usize,
        num_frag_types: usize,
        num_modloss_types: usize,
        mask_modloss: bool,
        device: Device,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Forward pass through the model.
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error>;

    /// Predict the property for a batch of peptide sequences.
    ///
    /// # Arguments
    /// * `peptide_sequences` - A slice of `Arc<[u8]>` containing each peptide sequence.
    /// * `mods` - A slice of `Arc<[u8]>` with modifications for each peptide.
    /// * `mod_sites` - A slice of `Arc<[u8]>` representing modification sites per peptide.
    /// * `charges` - Optional vector of charge states.
    /// * `nces` - Optional vector of normalized collision energies.
    /// * `instruments` - Optional vector of instrument names as `Arc<[u8]>`.
    ///
    /// # Returns
    /// A `PredictionResult` containing either RT, CCS, or MS2 predictions.
    fn predict(
        &self,
        peptide_sequences: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charges: Option<Vec<i32>>,
        nces: Option<Vec<i32>>,
        instruments: Option<Vec<Option<Arc<[u8]>>>>,
    ) -> Result<PredictionResult> {
        let input_tensor = self
            .encode_peptides(
                peptide_sequences,
                mods,
                mod_sites,
                charges,
                nces,
                instruments,
            )?
            .to_device(self.get_device())?;

        let output = self.forward(&input_tensor)?;

        match self.property_type() {
            PropertyType::RT => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::RTResult(predictions))
            }
            PropertyType::CCS => {
                let predictions: Vec<f32> = output.to_vec1()?;
                Ok(PredictionResult::CCSResult(predictions))
            }
            PropertyType::MS2 => {
                let out = self.process_predictions(&output, self.get_min_pred_intensity())?;
                let predictions: Vec<Vec<Vec<f32>>> = out.to_vec3()?;
                Ok(PredictionResult::MS2Result(predictions))
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported property type: {:?}",
                self.get_property_type()
            )),
        }
    }

    /// Encode peptide sequence (plus modifications) into a tensor.
    fn encode_peptide(
        &self,
        peptide_sequence: &Arc<[u8]>,
        mods: &Arc<[u8]>,
        mod_sites: &Arc<[u8]>,
        charge: Option<i32>,
        nce: Option<i32>,
        instrument: Option<&Arc<[u8]>>,
    ) -> Result<Tensor> {
        let device = self.get_device();
        let mod_feature_size = self.get_mod_element_count();
        let mod_to_feature = self.get_mod_to_feature();

        log::trace!(
            "[ModelInterface::encode_peptide] peptide_sequence: {:?} | mods: {:?} | mod_sites: {:?} | charge: {:?} | nce: {:?} | instrument: {:?}",
            peptide_sequence, mods, mod_sites, charge, nce, instrument
        );

        let aa_tensor = aa_indices_tensor_from_arc(peptide_sequence, device)?;
        let (batch_size, seq_len, _) = aa_tensor.shape().dims3()?;

        let mod_names: Vec<Arc<[u8]>> = std::str::from_utf8(mods)
            .unwrap_or("")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| Arc::from(s.as_bytes().to_vec().into_boxed_slice()))
            .collect();

        let mod_indices: Vec<usize> = std::str::from_utf8(mod_sites)
            .unwrap_or("")
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .collect();

        let mod_tensor = get_mod_features_from_parsed_arc(
            &mod_names,
            &mod_indices,
            seq_len,
            mod_feature_size,
            mod_to_feature,
            device,
        )?;

        let mut features = vec![aa_tensor, mod_tensor];

        if let Some(c) = charge {
            let charge_tensor = Tensor::from_slice(
                &vec![c as f64 * CHARGE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                device,
            )?
            .to_dtype(DType::F32)?;
            features.push(charge_tensor);
        }

        if let Some(n) = nce {
            let nce_tensor = Tensor::from_slice(
                &vec![n as f64 * NCE_FACTOR; seq_len],
                &[batch_size, seq_len, 1],
                device,
            )?
            .to_dtype(DType::F32)?;
            features.push(nce_tensor);
        }

        if let Some(instr) = instrument {
            let instr_str = std::str::from_utf8(instr).unwrap_or("");
            let instr_idx = parse_instrument_index(instr_str) as u32;
            let instr_tensor =
                Tensor::from_slice(&vec![instr_idx; seq_len], &[batch_size, seq_len, 1], device)?
                    .to_dtype(DType::F32)?;
            features.push(instr_tensor);
        }

        if features.len() == 1 {
            let output = features.remove(0);
            let (mean, min, max) = get_tensor_stats(&output)?;
            if !mean.is_finite() || !min.is_finite() || !max.is_finite() {
                log::error!("For Peptide = {:?} encode_peptides produced non-finite tensor stats: mean={mean}, min={min}, max={max}", peptide_sequence);
                anyhow::bail!("Non-finite values found in peptide encoding output.");
            }
            Ok(output)
        } else {
            let output = Tensor::cat(&features, 2)?;
            let (mean, min, max) = get_tensor_stats(&output)?;
            if !mean.is_finite() || !min.is_finite() || !max.is_finite() {
                log::error!("For Peptide = {:?} encode_peptides produced non-finite tensor stats: mean={mean}, min={min}, max={max}", peptide_sequence);
                anyhow::bail!("Non-finite values found in peptide encoding output.");
            }
            Ok(output)
        }
    }

    /// Encode a batch of peptide sequences into a tensor
    fn encode_peptides(
        &self,
        peptide_sequences: &[Arc<[u8]>],
        mods: &[Arc<[u8]>],
        mod_sites: &[Arc<[u8]>],
        charges: Option<Vec<i32>>,
        nces: Option<Vec<i32>>,
        instruments: Option<Vec<Option<Arc<[u8]>>>>,
    ) -> Result<Tensor> {
        let len = peptide_sequences.len();

        let tensors: Vec<_> = (0..len)
            .into_par_iter()
            .map(|i| {
                self.encode_peptide(
                    &peptide_sequences[i],
                    &mods[i],
                    &mod_sites[i],
                    charges.as_ref().map(|v| v[i]),
                    nces.as_ref().map(|v| v[i]),
                    instruments.as_ref().and_then(|v| v[i].as_ref()),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        if tensors.is_empty() {
            return Err(anyhow::anyhow!(
                "Encoding batch of peptides failed, the resulting tensor batch is empty."
            ));
        }

        let max_len = tensors
            .iter()
            .map(|t| t.shape().dims3().unwrap().1)
            .max()
            .unwrap_or(0);

        let expected_feat_dim = tensors
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("Empty input batch"))?
            .shape()
            .dims3()?
            .2;

        let padded = tensors
            .into_par_iter()
            .map(|t| {
                let (_, seq_len, feat_dim) = t.shape().dims3()?;

                if feat_dim != expected_feat_dim {
                    return Err(anyhow::anyhow!(
                        "Inconsistent feature dim: expected {}, got {}",
                        expected_feat_dim,
                        feat_dim
                    ));
                }

                if seq_len < max_len {
                    let pad =
                        Tensor::zeros(&[1, max_len - seq_len, feat_dim], t.dtype(), t.device())?;
                    Ok(Tensor::cat(&[&t, &pad], 1)?)
                } else {
                    Ok(t)
                }
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        Ok(Tensor::cat(&padded, 0)?)
    }

    /// Train the model from scratch using a batch of training data.
    ///
    /// This method initializes model weights from scratch and trains over the given peptide feature data for a specified
    /// number of epochs. Optionally performs validation and tracks both training and validation loss statistics.
    /// Early stopping is applied if the validation loss does not improve for a consecutive number of epochs.
    ///
    /// # Arguments
    /// * `training_data` - Vector of peptide records used for training.
    /// * `validation_data` - Optional vector of peptide records used for validation at the end of each epoch.
    /// * `modifications` - A map of known modifications to encode modified peptides.
    /// * `batch_size` - Batch size used for training.
    /// * `validation_batch_size` - Batch size used during validation.
    /// * `learning_rate` - Learning rate for the AdamW optimizer.
    /// * `epochs` - Maximum number of training epochs.
    /// * `early_stopping_patience` - Number of epochs to wait before stopping if validation loss does not improve.
    ///
    /// # Returns
    /// A `Vec` of tuples where each tuple contains:
    /// * `epoch` - Epoch number.
    /// * `avg_train_loss` - Average training loss for the epoch.
    /// * `avg_val_loss` - Optional average validation loss for the epoch.
    /// * `train_std` - Standard deviation of training loss across batches.
    /// * `val_std` - Optional standard deviation of validation loss across batches.
    fn train(
        &mut self,
        training_data: &Vec<PeptideData>,
        validation_data: Option<&Vec<PeptideData>>,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        batch_size: usize,
        validation_batch_size: usize,
        learning_rate: f64,
        epochs: usize,
        early_stopping_patience: usize,
    ) -> Result<TrainingStepMetrics> {
        let num_batches = (training_data.len() + batch_size - 1) / batch_size;
        let total_steps = num_batches * epochs;
        let warmup_steps = total_steps / 10; // 10% of total steps

        info!(
            "Training {} model from on {} peptide features ({} batches) for {} epochs",
            self.get_model_arch(),
            training_data.len(),
            num_batches,
            epochs
        );

        let mut step_metrics = TrainingStepMetrics {
            epochs: vec![],
            steps: vec![],
            learning_rates: vec![],
            losses: vec![],
            phases: vec![],
            precisions: vec![],
            recalls: vec![],
            accuracies: vec![],
        };

        let mut step_idx = 0;
        let mut val_step_idx = 0;

        let params = candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?;
        let mut lr_scheduler = CosineWithWarmup::new(
            learning_rate,
            warmup_steps,
            total_steps,
            0.5, // one full cosine cycle
        );

        let mut best_val_loss = f32::INFINITY;
        let mut epochs_without_improvement = 0;
        let mut epoch_losses = vec![];

        for epoch in 0..epochs {
            let progress = Progress::new(num_batches, &format!("[training] Epoch {}: ", epoch));
            let mut batch_losses = vec![];

            training_data.chunks(batch_size).enumerate().try_for_each(
                |(_batch_idx, batch_data)| -> anyhow::Result<()> {
                    let (input_batch, target_batch) =
                        self.prepare_batch_inputs(batch_data, &modifications)?;

                    let predicted = self.forward(&input_batch)?;
                    let loss = candle_nn::loss::mse(&predicted, &target_batch)?;
                    opt.backward_step(&loss)?;

                    // Update learning rate after optimizer step
                    opt.set_learning_rate(lr_scheduler.get_last_lr());
                    lr_scheduler.step();

                    let loss_val = loss.to_vec0::<f32>().unwrap_or(999.0);
                    batch_losses.push(loss_val);

                    let predictions = predicted.to_vec1::<f32>()?;
                    let targets = target_batch.to_vec1::<f32>()?;

                    let acc = match self.property_type() {
                        PropertyType::RT => Some(Metrics::accuracy(&predictions, &targets, 0.5)), // is predicted RT within 0.5 min of target RT?
                        PropertyType::CCS => {
                            let tol: Vec<f32> = targets.iter().map(|t| t * 0.02).collect();
                            Some(Metrics::accuracy_dynamic(&predictions, &targets, &tol))
                        } // is predicted CCS within 2% of target CCS?
                        _ => None,
                    };

                    step_metrics.epochs.push(epoch);
                    step_metrics.steps.push(step_idx);
                    step_metrics
                        .learning_rates
                        .push(lr_scheduler.get_last_lr() as f64);
                    step_metrics.losses.push(loss_val);
                    step_metrics.phases.push(TrainingPhase::Train);
                    step_metrics.accuracies.push(acc);
                    step_metrics.precisions.push(None);
                    step_metrics.recalls.push(None);
                    step_idx += 1;

                    progress.update_description(&format!(
                        "[training] Epoch {}: Loss: {:.4}",
                        epoch, loss_val
                    ));
                    progress.inc();

                    // If the loss is NaN, stop training and throw an error
                    if loss_val.is_nan() {
                        log::error!("Loss is NaN, stopping training.");
                        return Err(anyhow::anyhow!("Loss is NaN, stopping training."));
                    }

                    Ok(())
                },
            )?;

            let (avg_loss, std_loss) = compute_loss_stats(&batch_losses);

            if let Some(val_data) = validation_data {
                let val_batches =
                    (val_data.len() + validation_batch_size - 1) / validation_batch_size;

                let val_results: Vec<(f32, usize, f64, Option<f32>)> = val_data
                    .par_chunks(validation_batch_size)
                    .enumerate()
                    .map(|(idx, batch_data)| {
                        let (input_val, target_val) =
                            self.prepare_batch_inputs(batch_data, &modifications)?;
                        let predicted = self.forward(&input_val)?;
                        let val_loss = candle_nn::loss::mse(&predicted, &target_val)?;
                        let loss_val = val_loss.to_vec0::<f32>()?;

                        let predictions = predicted.to_vec1::<f32>()?;
                        let targets = target_val.to_vec1::<f32>()?;

                        let acc = match self.property_type() {
                            PropertyType::RT => {
                                Some(Metrics::accuracy(&predictions, &targets, 0.5))
                            }
                            PropertyType::CCS => {
                                let tol: Vec<f32> = targets.iter().map(|t| t * 0.02).collect();
                                Some(Metrics::accuracy_dynamic(&predictions, &targets, &tol))
                            }
                            _ => None,
                        };

                        Ok((loss_val, idx, lr_scheduler.get_last_lr(), acc))
                    })
                    .collect::<Result<_>>()?;

                for (val_loss, idx, lr, acc) in &val_results {
                    step_metrics.epochs.push(epoch);
                    step_metrics.steps.push(val_step_idx + idx);
                    step_metrics.learning_rates.push(*lr);
                    step_metrics.losses.push(*val_loss);
                    step_metrics.phases.push(TrainingPhase::Validation);
                    step_metrics.accuracies.push(*acc);
                    step_metrics.precisions.push(None);
                    step_metrics.recalls.push(None);
                }
                val_step_idx += val_results.len();

                let val_losses: Vec<f32> =
                    val_results.iter().map(|(loss, _, _, _)| *loss).collect();
                let (avg_val_loss, std_val_loss): (f32, f32) = compute_loss_stats(&val_losses);

                epoch_losses.push((
                    epoch,
                    avg_loss,
                    Some(avg_val_loss),
                    std_loss,
                    Some(std_val_loss),
                ));

                progress.update_description(&format!(
                    "Epoch {}: Avg. Train Loss: {:.4} (±{:.4}) | Avg. Val. Loss: {:.4} (±{:.4})",
                    epoch, avg_loss, std_loss, avg_val_loss, std_val_loss
                ));
                progress.finish();

                if avg_val_loss < best_val_loss {
                    best_val_loss = avg_val_loss;
                    epochs_without_improvement = 0;

                    let checkpoint_path = format!(
                        "redeem_{}_best_val_ckpt_model_epoch_{}.safetensors",
                        self.get_model_arch(),
                        epoch
                    );
                    self.get_mut_varmap().save(&checkpoint_path)?;
                } else {
                    epochs_without_improvement += 1;
                    if epochs_without_improvement >= early_stopping_patience {
                        info!("Early stopping triggered after {} epochs without validation loss improvement.", early_stopping_patience);
                        return Ok(step_metrics);
                    }
                    let checkpoint_path = format!(
                        "redeem_{}_ckpt_model_epoch_{}.safetensors",
                        self.get_model_arch(),
                        epoch - 1
                    );
                    // Check if the prior checkpoint exists, if it does delete it
                    if PathBuf::from(&checkpoint_path).exists() {
                        std::fs::remove_file(&checkpoint_path)?;
                    }
                    // Save the current checkpoint
                    let checkpoint_path = format!(
                        "redeem_{}_ckpt_model_epoch_{}.safetensors",
                        self.get_model_arch(),
                        epoch
                    );
                    self.get_mut_varmap().save(&checkpoint_path)?;
                }
            } else {
                epoch_losses.push((epoch, avg_loss, None, std_loss, None));
                progress.update_description(&format!(
                    "Epoch {}: Avg. Train Loss: {:.4} (±{:.4})",
                    epoch, avg_loss, std_loss
                ));
                progress.finish();

                let checkpoint_path = format!(
                    "redeem_{}_ckpt_model_epoch_{}.safetensors",
                    self.get_model_arch(),
                    epoch - 1
                );
                // Check if the prior checkpoint exists, if it does delete it
                if PathBuf::from(&checkpoint_path).exists() {
                    std::fs::remove_file(&checkpoint_path)?;
                }
                // Save the current checkpoint
                let checkpoint_path = format!(
                    "redeem_{}_ckpt_model_epoch_{}.safetensors",
                    self.get_model_arch(),
                    epoch
                );
                self.get_mut_varmap().save(&checkpoint_path)?;
            }
        }

        Ok(step_metrics)
    }

    /// Fine-tune the model on a batch of training data.
    ///
    /// # Arguments
    /// * `training_data` - A vector of `PeptideData` instances representing the training data.
    /// * `modifications` - A map of modifications and their corresponding feature vectors.
    /// * `batch_size` - The batch size to use for training.
    /// * `learning_rate` - The learning rate to use for training.
    /// * `epochs` - The number of epochs to train for.
    fn fine_tune(
        &mut self,
        training_data: &Vec<PeptideData>,
        modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        batch_size: usize,
        learning_rate: f64,
        epochs: usize,
    ) -> Result<()> {
        // let num_batches = if training_data.len() < batch_size {
        //     1
        // } else {
        //     let full_batches = training_data.len() / batch_size;
        //     if training_data.len() % batch_size > 0 {
        //         full_batches + 1
        //     } else {
        //         full_batches
        //     }
        // };

        // info!(
        //     "Fine-tuning {} model on {} peptide features ({} batches) for {} epochs",
        //     self.get_model_arch(),
        //     training_data.len(),
        //     num_batches,
        //     epochs
        // );

        // let params = candle_nn::ParamsAdamW {
        //     lr: learning_rate,
        //     ..Default::default()
        // };
        // let mut opt = candle_nn::AdamW::new(self.get_mut_varmap().all_vars(), params)?;

        // for epoch in 0..epochs {
        //     let progress = Progress::new(num_batches, &format!("[fine-tuning] Epoch {}: ", epoch));
        //     let mut total_loss = 0.0;

        //     for batch_idx in 0..num_batches {
        //         let start = batch_idx * batch_size;
        //         let end = (start + batch_size).min(training_data.len());
        //         let batch_data = &training_data[start..end];

        //         let peptides: Vec<String> = batch_data
        //             .iter()
        //             .map(|p| remove_mass_shift(&p.sequence))
        //             .collect();
        //         let mods: Vec<String> = batch_data
        //             .iter()
        //             .map(|p| get_modification_string(&p.sequence, &modifications))
        //             .collect();
        //         let mod_sites: Vec<String> = batch_data
        //             .iter()
        //             .map(|p| get_modification_indices(&p.sequence))
        //             .collect();

        //         let charges = batch_data
        //             .iter()
        //             .filter_map(|p| p.charge)
        //             .collect::<Vec<_>>();
        //         let charges = if charges.len() == batch_data.len() {
        //             Some(charges)
        //         } else {
        //             None
        //         };

        //         let nces = batch_data.iter().filter_map(|p| p.nce).collect::<Vec<_>>();
        //         let nces = if nces.len() == batch_data.len() {
        //             Some(nces)
        //         } else {
        //             None
        //         };

        //         let instruments = batch_data
        //             .iter()
        //             .filter_map(|p| p.instrument.clone())
        //             .collect::<Vec<_>>();
        //         let instruments = if instruments.len() == batch_data.len() {
        //             Some(instruments)
        //         } else {
        //             None
        //         };

        //         let input_batch = self
        //             .encode_peptides(&peptides, &mods, &mod_sites, charges, nces, instruments)?
        //             .to_device(self.get_device())?;

        //         log::trace!(
        //             "[ModelInterface::fine_tune] input_batch shape: {:?}, device: {:?}",
        //             input_batch.shape(),
        //             input_batch.device()
        //         );

        //         let batch_targets = match self.property_type() {
        //             PropertyType::RT => PredictionResult::RTResult(
        //                 batch_data
        //                     .iter()
        //                     .map(|p| p.retention_time.unwrap_or_default())
        //                     .collect(),
        //             ),
        //             PropertyType::CCS => PredictionResult::CCSResult(
        //                 batch_data
        //                     .iter()
        //                     .map(|p| p.ion_mobility.unwrap_or_default())
        //                     .collect(),
        //             ),
        //             PropertyType::MS2 => PredictionResult::MS2Result(
        //                 batch_data
        //                     .iter()
        //                     .map(|p| p.ms2_intensities.clone().unwrap_or_default())
        //                     .collect(),
        //             ),
        //         };

        //         let target_batch = match batch_targets {
        //             PredictionResult::RTResult(ref values)
        //             | PredictionResult::CCSResult(ref values) => {
        //                 Tensor::new(values.clone(), &self.get_device())?
        //             }
        //             PredictionResult::MS2Result(ref spectra) => {
        //                 let max_len = spectra.iter().map(|s| s.len()).max().unwrap_or(1);
        //                 let feature_dim = spectra
        //                     .get(0)
        //                     .and_then(|s| s.get(0))
        //                     .map(|v| v.len())
        //                     .unwrap_or(1);
        //                 let mut padded_spectra = spectra.clone();
        //                 for s in &mut padded_spectra {
        //                     s.resize(max_len, vec![0.0; feature_dim]);
        //                 }
        //                 Tensor::new(padded_spectra.concat(), &self.get_device())?.reshape((
        //                     batch_data.len(),
        //                     max_len,
        //                     feature_dim,
        //                 ))?
        //             }
        //         }
        //         .to_device(self.get_device())?;

        //         let predicted = self.forward(&input_batch)?;
        //         let loss = candle_nn::loss::mse(&predicted, &target_batch)?;
        //         opt.backward_step(&loss)?;

        //         total_loss += loss.to_vec0::<f32>().unwrap_or(990.0);

        //         progress.update_description(&format!(
        //             "[fine-tuning] Epoch {}: Loss: {}",
        //             epoch,
        //             loss.to_vec0::<f32>()?
        //         ));
        //         progress.inc();
        //     }

        //     let avg_loss = total_loss / num_batches as f32;
        //     progress.update_description(&format!(
        //         "[fine-tuning] Epoch {}: Avg. Batch Loss: {}",
        //         epoch, avg_loss
        //     ));
        //     progress.finish();
        // }

        // Ok(())
        todo!()
    }

    /// Perform inference over a batch of peptides.
    fn inference(
        &self,
        inference_data: &Vec<PeptideData>,
        batch_size: usize,
        _modifications: HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
        rt_norm: TargetNormalization,
    ) -> Result<Vec<PeptideData>> {
        let num_batches = (inference_data.len() + batch_size - 1) / batch_size;
        info!(
            "Performing inference on {} peptide features ({} batches)",
            inference_data.len(),
            num_batches
        );

        let progress = Progress::new(inference_data.len(), "[inference] Batch:");
        let mut result: Vec<Option<PeptideData>> = vec![None; inference_data.len()];

        inference_data
            .par_chunks(batch_size)
            .enumerate()
            .map(|(batch_idx, batch_data)| {
                let start_idx = batch_idx * batch_size;
                let batch: PeptideBatchData = batch_data.into();

                let naked_sequences = &batch.naked_sequence;
                let mods = &batch.mods;
                let mod_sites = &batch.mod_sites;

                let charges = if batch.charges.iter().all(|c| c.is_some()) {
                    Some(batch.charges.iter().map(|c| c.unwrap()).collect::<Vec<_>>())
                } else {
                    None
                };

                let nces = if batch.nces.iter().all(|n| n.is_some()) {
                    Some(batch.nces.iter().map(|n| n.unwrap()).collect::<Vec<_>>())
                } else {
                    None
                };

                let instruments = if batch.instruments.iter().all(|i| i.is_some()) {
                    Some(batch.instruments.clone())
                } else {
                    None
                };

                let input_tensor = self
                    .encode_peptides(naked_sequences, mods, mod_sites, charges, nces, instruments)?
                    .to_device(self.get_device())?;
                let output = self.forward(&input_tensor)?;

                match self.property_type() {
                    PropertyType::RT | PropertyType::CCS => {
                        let predictions = output.to_vec1()?;
                        let updated: Vec<(usize, PeptideData)> = predictions
                            .into_iter()
                            .enumerate()
                            .map(|(i, pred)| {
                                let mut peptide = batch_data[i].clone();
                                match self.property_type() {
                                    PropertyType::RT => {
                                        peptide.retention_time = Some(match rt_norm {
                                            TargetNormalization::ZScore(mean, std) => pred * std + mean,
                                            TargetNormalization::MinMax(min, max) => {
                                                pred * (max - min) + min
                                            }
                                            TargetNormalization::None => pred,
                                        });
                                    }
                                    PropertyType::CCS => peptide.ion_mobility = Some(pred),
                                    _ => {}
                                };
                                (start_idx + i, peptide)
                            })
                            .collect();
                        Ok(updated)
                    }
                    PropertyType::MS2 => Err(anyhow::anyhow!(
                        "Inference not supported for MS2 models in batch mode"
                    )),
                }
            })
            .collect::<Result<Vec<Vec<(usize, PeptideData)>>>>()?
            .into_iter()
            .flatten()
            .for_each(|(idx, peptide)| {
                result[idx] = Some(peptide);
                progress.inc();
            });

        progress.finish();
        Ok(result.into_iter().flatten().collect())
    }

    /// Extract encoded input and target tensor for a batch of peptides.
    fn prepare_batch_inputs(
        &self,
        batch_data: &[PeptideData],
        _modifications: &HashMap<
            (String, Option<char>),
            crate::utils::peptdeep_utils::ModificationMap,
        >,
    ) -> Result<(Tensor, Tensor)> {
        use rayon::prelude::*;

        let batch: PeptideBatchData = batch_data.into();

        let naked_sequences = &batch.naked_sequence;
        let mods = &batch.mods;
        let mod_sites = &batch.mod_sites;

        let charges = if batch.charges.iter().all(|c| c.is_some()) {
            Some(batch.charges.iter().map(|c| c.unwrap()).collect::<Vec<_>>())
        } else {
            None
        };

        let nces = if batch.nces.iter().all(|n| n.is_some()) {
            Some(batch.nces.iter().map(|n| n.unwrap()).collect::<Vec<_>>())
        } else {
            None
        };

        let instruments = if batch.instruments.iter().all(|i| i.is_some()) {
            Some(batch.instruments.clone())
        } else {
            None
        };

        let input_batch = self
            .encode_peptides(naked_sequences, mods, mod_sites, charges, nces, instruments)?
            .to_device(self.get_device())?;

        let target_values: Vec<f32> = match self.property_type() {
            PropertyType::RT => batch
                .retention_times
                .iter()
                .map(|v| v.unwrap_or(0.0))
                .collect(),
            PropertyType::CCS => batch
                .ion_mobilities
                .iter()
                .map(|v| v.unwrap_or(0.0))
                .collect(),
            PropertyType::MS2 => {
                return Err(anyhow::anyhow!("MS2 training is not yet implemented"))
            }
        };

        let target_tensor = Tensor::new(target_values, &self.get_device())?;
        Ok((input_batch, target_tensor))
    }

    /// Set model to evaluation mode for inference
    /// This disables dropout and other training-specific layers.
    fn set_evaluation_mode(&mut self);

    /// Set model to training mode for training
    /// This enables dropout and other training-specific layers.
    fn set_training_mode(&mut self);

    fn get_property_type(&self) -> String;

    fn get_model_arch(&self) -> String;

    fn get_device(&self) -> &Device;

    fn get_mod_element_count(&self) -> usize;

    fn get_mod_to_feature(&self) -> &HashMap<Arc<[u8]>, Vec<f32>>;

    fn get_min_pred_intensity(&self) -> f32;

    fn get_mut_varmap(&mut self) -> &mut VarMap;

    fn print_summary(&self);
    fn print_weights(&self);

    /// Save model weights to a file in safetensors format.
    fn save(&mut self, path: &str) -> Result<()> {
        info!(
            "Saving {} model weights to: {:?}",
            self.get_model_arch(),
            path
        );
        self.get_mut_varmap().clone().save(&PathBuf::from(path))?;
        Ok(())
    }

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
#[derive(Clone)]
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
    pub rt_model: Option<RTModelWrapper>,

    /// Optional collision cross-section prediction model.
    pub ccs_model: Option<CCSModelWrapper>,

    /// Optional MS2 intensity prediction model.
    pub ms2_model: Option<MS2ModelWrapper>,
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
    /// models.rt_model = Some(RTModelWrapper::new());
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
    /// models.rt_model = Some(RTModelWrapper::new());
    /// assert!(models.is_not_empty());
    /// ```
    pub fn is_not_empty(&self) -> bool {
        self.rt_model.is_some() || self.ccs_model.is_some() || self.ms2_model.is_some()
    }
}
