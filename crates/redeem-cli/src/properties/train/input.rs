use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use clap::ArgMatches;
use anyhow::{Context, Result};

use crate::properties::util::validate_tsv_or_csv_file;


#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PropertyTrainConfig {
    pub version: String,
    pub train_data: String,
    pub validation_data: Option<String>,
    pub output_file: String,
    pub rt_normalization: Option<String>,
    pub model_arch: String,
    pub device: String,
    pub batch_size: usize,
    pub validation_batch_size: Option<usize>,
    pub learning_rate: f32,
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub checkpoint_file: Option<String>,
    pub instrument: String,
    pub nce: i32,
}

impl Default for PropertyTrainConfig {
    fn default() -> Self {
        PropertyTrainConfig {
            version: clap::crate_version!().to_string(),
            train_data: String::new(),
            validation_data: None,
            output_file: String::from("rt_cnn_tf.safetensors"),
            rt_normalization: Some(String::from("min_max")),
            model_arch: String::from("rt_cnn_tf"),
            device: String::from("cpu"),
            batch_size: 64,
            validation_batch_size: None,
            learning_rate: 1e-3,
            epochs: 10,
            early_stopping_patience: 5,
            checkpoint_file: None,
            instrument: String::from("QE"),
            nce: 20,
        }
    }
}

impl PropertyTrainConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> anyhow::Result<Self> {
        let config_json = fs::read_to_string(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

        let partial: serde_json::Value = serde_json::from_str(&config_json)?;
        let mut config = PropertyTrainConfig::default();

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

        load_or_default!(train_data);
        load_or_default!(validation_data);
        load_or_default!(output_file);
        load_or_default!(rt_normalization);
        load_or_default!(model_arch);
        load_or_default!(device);
        load_or_default!(batch_size);
        load_or_default!(validation_batch_size);
        load_or_default!(learning_rate);
        load_or_default!(epochs);
        load_or_default!(early_stopping_patience);
        load_or_default!(checkpoint_file);
        load_or_default!(instrument);
        load_or_default!(nce);

        // Apply CLI overrides
        if let Some(train_data) = matches.get_one::<String>("train_data") {
            config.train_data = train_data.clone();
        }
        if let Some(validation_data) = matches.get_one::<String>("validation_data") {
            config.validation_data = Some(validation_data.clone());
        }
        if let Some(output_file) = matches.get_one::<String>("output_file") {
            config.output_file = output_file.clone();
        }
        if let Some(model_arch) = matches.get_one::<String>("model_arch") {
            config.model_arch = model_arch.clone();
        }
        if let Some(checkpoint_file) = matches.get_one::<String>("checkpoint_file") {
            config.checkpoint_file = Some(checkpoint_file.clone());
        }

        Ok(config)
    }
}