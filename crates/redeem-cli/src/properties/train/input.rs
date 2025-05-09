use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use clap::ArgMatches;
use anyhow::{Context, Result};

#[derive(Debug, Deserialize, Clone)]
pub struct PropertyTrainConfig {
    pub train_data: String,
    pub validation_data: Option<String>,
    pub output_file: String,
    pub model_arch: String,
    pub device: String,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub instrument: String,
    pub nce: i32,
}

impl Default for PropertyTrainConfig {
    fn default() -> Self {
        PropertyTrainConfig {
            train_data: String::new(),
            validation_data: None,
            output_file: String::from("rt_cnn_tf.safetensors"),
            model_arch: String::from("rt_cnn_tf"),
            device: String::from("cpu"),
            batch_size: 64,
            learning_rate: 1e-3,
            epochs: 10,
            instrument: String::from("QE"),
            nce: 20,
        }
    }
}

impl PropertyTrainConfig {
    pub fn from_arguments(config_path: &PathBuf, matches: &ArgMatches) -> Result<Self> {
        let config_json = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let mut config: PropertyTrainConfig = serde_json::from_str(&config_json)
            .unwrap_or_else(|_| PropertyTrainConfig::default());

        // Apply CLI overrides
        if let Some(train_data) = matches.get_one::<String>("train_data") {
            validate_tsv_or_csv_file(train_data)?;
            config.train_data = train_data.clone().to_string();
        } else {
            validate_tsv_or_csv_file(&config.train_data)?;
        }

        if let Some(validation_data) = matches.get_one::<String>("validation_data") {
            validate_tsv_or_csv_file(validation_data)?;
            config.validation_data = Some(validation_data.clone().to_string());
        } else if let Some(val_data) = &config.validation_data {
            validate_tsv_or_csv_file(val_data)?;
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


pub fn validate_tsv_or_csv_file(path: &str) -> Result<()> {
    let pb = PathBuf::from(path);

    let ext = pb.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase());
    match ext.as_deref() {
        Some("tsv") | Some("csv") => {}
        _ => anyhow::bail!("File must have a .tsv or .csv extension: {}", path),
    }

    if !pb.exists() {
        anyhow::bail!("File does not exist: {}", path);
    }

    Ok(())
}
