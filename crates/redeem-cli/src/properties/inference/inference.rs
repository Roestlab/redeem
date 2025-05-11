use anyhow::{Context, Result};
use redeem_properties::utils::data_handling::PeptideData;
use redeem_properties::utils::peptdeep_utils::load_modifications;
use redeem_properties::utils::utils::get_device;
use redeem_properties::models::rt_model::load_retention_time_model;

use crate::properties::load_data::load_peptide_data;
use crate::properties::util::write_bytes_to_file;
use crate::properties::inference::input::PropertyInferenceConfig;
use crate::properties::inference::output::write_peptide_data;

pub fn run_inference(config: &PropertyInferenceConfig) -> Result<()> {

    // Load inference data
    let (inference_data, norm_factor) = load_peptide_data(&config.inference_data, Some(config.nce), Some(config.instrument.clone()), true)?;
    log::info!("Loaded {} peptides", inference_data.len());

    // Dispatch model training based on architecture
    let model_arch = config.model_arch.as_str();
    let device = get_device(&config.device)?;

    let mut model = load_retention_time_model(
        &config.model_path,
        None,
        &config.model_arch,
        device.clone(),
    )?;

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
