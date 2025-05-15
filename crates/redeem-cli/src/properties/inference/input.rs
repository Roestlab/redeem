use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use clap::ArgMatches;
use anyhow::{Context, Result};

use crate::properties::util::validate_tsv_or_csv_file;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PropertyInferenceConfig {
    pub version: String,
    pub model_path: String,
    pub inference_data: String,
    pub output_file: String,
    pub normalization: Option<String>,
    pub model_arch: String,
    pub device: String,
    pub batch_size: usize,
    pub instrument: String,
    pub nce: i32,
}

impl Default for PropertyInferenceConfig {
    fn default() -> Self {
        PropertyInferenceConfig {
            version: clap::crate_version!().to_string(),
            model_path: String::new(),
            inference_data: String::new(),
            output_file: String::from("redeem_inference.csv"),
            normalization: Some(String::from("min_max")),
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

        let partial: serde_json::Value = serde_json::from_str(&config_json)?;
        let mut config = PropertyInferenceConfig::default();

        macro_rules! load_or_default {
            ($field:ident) => {
                if let Some(val) = partial.get(stringify!($field)) {
                    if let Ok(parsed) = serde_json::from_value(val.clone()) {
                        config.$field = parsed;
                    } else {
                        log::warn!(
                            "Config Invalid value for '{}', using default: {:?}",
                            stringify!($field), config.$field
                        );
                    }
                } else {
                    log::warn!(
                        "Config Missing field '{}', using default: {:?}",
                        stringify!($field), config.$field
                    );
                }
            };
        }

        load_or_default!(model_path);
        load_or_default!(inference_data);
        load_or_default!(output_file);
        load_or_default!(normalization);
        load_or_default!(model_arch);
        load_or_default!(device);
        load_or_default!(batch_size);
        load_or_default!(instrument);
        load_or_default!(nce);

        // Apply CLI overrides
        if let Some(model_path) = matches.get_one::<String>("model_path") {
            config.model_path = model_path.clone();
        }
        if let Some(inference_data) = matches.get_one::<String>("inference_data") {
            validate_tsv_or_csv_file(inference_data)?;
            config.inference_data = inference_data.clone();
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

