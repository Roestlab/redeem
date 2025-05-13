use anyhow::{Context, Result};
use redeem_properties::models::ccs_cnn_lstm_model::CCSCNNLSTMModel;
use redeem_properties::models::ccs_cnn_tf_model::CCSCNNTFModel;
use redeem_properties::models::ccs_model::load_collision_cross_section_model;
use redeem_properties::models::model_interface::ModelInterface;
use redeem_properties::models::rt_cnn_lstm_model::RTCNNLSTMModel;
use redeem_properties::models::rt_model::load_retention_time_model;
use redeem_properties::utils::data_handling::{PeptideData, RTNormalization};
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;

use crate::properties::inference::input::PropertyInferenceConfig;
use crate::properties::inference::output::write_peptide_data;
use crate::properties::load_data::load_peptide_data;
use crate::properties::util::write_bytes_to_file;

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {
    // Load inference data
    let (inference_data, norm_factor) = load_peptide_data(
        &config.inference_data,
        &config.model_arch,
        Some(config.nce),
        Some(config.instrument.clone()),
        Some("min_max".to_string()),
    )?;
    log::info!("Loaded {} peptides", inference_data.len());

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;

    let mut model: Box<dyn ModelInterface + Send + Sync> = match model_arch {
        "rt_cnn_lstm" => Box::new(RTCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "rt_cnn_tf" => Box::new(RTCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_lstm" => Box::new(CCSCNNLSTMModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        "ccs_cnn_tf" => Box::new(CCSCNNTFModel::new(
            &config.model_path,
            None,
            0,
            8,
            4,
            true,
            device.clone(),
        )?),
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported RT model architecture: {}",
                model_arch
            ));
        }
    };

    let modifications = load_modifications().context("Failed to load modifications")?;

    let start_time = std::time::Instant::now();
    model.set_evaluation_mode();
    let inference_results: Vec<PeptideData> = model.inference(
        &inference_data,
        config.batch_size,
        modifications,
        norm_factor,
    )?;
    log::info!("Inference completed in {:?}", start_time.elapsed());

    log::info!("Predictions saved to: {}", config.output_file);
    write_peptide_data(&inference_results, &config.output_file)?;

    let path = "redeem_inference_config.json";
    let json = serde_json::to_string_pretty(&config)?;
    println!("{}", json);
    let bytes = serde_json::to_vec_pretty(&config)?;
    write_bytes_to_file(path, &bytes)?;

    Ok(())
}
