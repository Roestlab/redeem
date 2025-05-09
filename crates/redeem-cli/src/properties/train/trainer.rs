use anyhow::{Context, Result};
use input::PropertyTrainConfig;
use load_data::load_peptide_data;
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::{rt_cnn_lstm_model::RTCNNLSTMModel, rt_cnn_transformer_model::RTCNNTFModel};
use redeem_properties::utils::data_handling::PeptideData;
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;
use std::path::PathBuf;
use candle_core::Device;

use crate::properties::load_data;

use super::input;

pub fn run_training(config: &PropertyTrainConfig) -> Result<()> {

    // Load training data
    let train_peptides: Vec<PeptideData> = load_peptide_data(&config.train_data)?;
    println!("Loaded {} training peptides", train_peptides.len());

    // Load validation data if specified
    let val_peptides = if let Some(ref val_path) = config.validation_data {
        Some(load_peptide_data(val_path).context("Failed to load validation data")?)
    } else {
        None
    };

    if let Some(ref val_data) = val_peptides {
        println!("Loaded {} validation peptides", val_data.len());
    } else {
        println!("No validation data provided.");
    }

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;

    let mut model: Box<dyn ModelInterface + Send + Sync> = match model_arch {
        "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new_untrained(device.clone())?),
        "rt_cnn_tf" => Box::new(RTCNNTFModel::new_untrained(device.clone())?),
        _ => return Err(anyhow::anyhow!("Unsupported model architecture: {}", model_arch)),
    };

    let modifications = load_modifications().context("Failed to load modifications")?;

    model.train(
        &train_peptides,
        val_peptides.as_ref(),
        modifications,
        config.batch_size,
        config.learning_rate as f64,
        config.epochs,
    )?;

    model.save(&config.output_file)?;
    println!("Model saved to: {}", config.output_file);

    Ok(())
}
