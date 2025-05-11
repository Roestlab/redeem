use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use clap::ArgMatches;
use anyhow::{Context, Result};

use crate::properties::util::validate_tsv_or_csv_file;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PropertyInferenceConfig {
    pub model_path: String,
    pub inference_data: String,
    pub output_file: String,
    pub model_arch: String,
    pub device: String,
    pub batch_size: usize,
    pub instrument: String,
    pub nce: i32,
}

impl Default for PropertyInferenceConfig {
    fn default() -> Self {
        PropertyInferenceConfig {
            model_path: String::new(),
            inference_data: String::new(),
            output_file: String::from("redeem_inference.csv"),
            model_arch: String::from("rt_cnn_tf"),
            device: String::from("cpu"),
            batch_size: 64,
            instrument: String::from("QE"),
            nce: 20,
        }
    }
}

impl PropertyInferenceConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let config_json = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let mut config: PropertyInferenceConfig = serde_json::from_str(&config_json)
            .unwrap_or_else(|_| PropertyInferenceConfig::default());

        // Apply CLI overrides
        if let Some(model_path) = matches.get_one::<String>("model_path") {
            config.model_path = model_path.clone();
        } else {
            config.model_path = config.model_path.clone();
        }

        if let Some(inference_data) = matches.get_one::<String>("inference_data") {
            validate_tsv_or_csv_file(inference_data)?;
            config.inference_data = inference_data.clone().to_string();
        } else {
            validate_tsv_or_csv_file(&config.inference_data)?;
        }

        if let Some(output_file) = matches.get_one::<String>("output_file") {
            config.output_file = output_file.clone();
        }

        if let Some(model_arch) = matches.get_one::<String>("model_arch") {
            config.model_arch = model_arch.clone();
        }

        Ok(config)
    }
}

