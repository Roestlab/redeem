use anyhow::{Context, Result, Error};
use std::fs::File;
use std::path::PathBuf;
use std::io;
use std::fs;
// use std::error::Error;
use csv::ReaderBuilder;
use reqwest;
// use regex::Regex;
use candle_core::Error as CandleError;
use std::collections::HashMap;
use crate::models::rt_cnn_lstm_model::ModelConstants;

const MOD_TSV_URL: &str = "https://raw.githubusercontent.com/MannLabs/alphabase/main/alphabase/constants/const_files/modification.tsv";
const MOD_TSV_PATH: &str = "data/modification.tsv";

#[derive(Debug, serde::Deserialize)]
struct ModFeature {
    mod_name: String,
    composition: String,
    // Add other fields if needed
}

/// Parse the model constants from a YAML file.
pub fn parse_model_constants(path: &str) -> Result<ModelConstants> {
    let f = std::fs::File::open(path).map_err(|e| Error::msg(e.to_string()))?;
    // Holds the model constants.
    let constants: ModelConstants = serde_yaml::from_reader(f).map_err(|e| Error::msg(e.to_string()))?;
    Ok(constants)
}

fn ensure_mod_tsv_exists() -> Result<PathBuf, io::Error> {
    let path = PathBuf::from(MOD_TSV_PATH);
    
    // Ensure the parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    if !path.exists() {
        println!("Downloading modification.tsv...");
        let mut response = reqwest::blocking::get(MOD_TSV_URL)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut file = File::create(&path)?;
        response.copy_to(&mut file)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        println!("Download complete.");
    }
    Ok(path)
}

fn parse_mod_formula(formula: &str, mod_elem_to_idx: &HashMap<String, usize>, mod_feature_size: usize) -> Vec<f32> {
    let mut feature = vec![0.0; mod_feature_size];
    let elems: Vec<&str> = formula.trim_end_matches(')').split(')').collect();
    for elem in elems {
        let parts: Vec<&str> = elem.split('(').collect();
        if parts.len() == 2 {
            let chem = parts[0];
            let num: i32 = parts[1].parse().unwrap_or(0);
            if let Some(&idx) = mod_elem_to_idx.get(chem) {
                feature[idx] = num as f32;
            } else {
                feature[mod_feature_size - 1] += num as f32;
            }
        }
    }
    feature
}

pub fn load_mod_to_feature(constants: &ModelConstants) -> Result<HashMap<String, Vec<f32>>, Error> {
    let path = ensure_mod_tsv_exists()?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)?;

    // Create mod_elem_to_idx mapping
    let mod_elem_to_idx: HashMap<String, usize> = constants.mod_elements.iter()
        .enumerate()
        .map(|(i, elem)| (elem.clone(), i))
        .collect();

    let mod_feature_size = constants.mod_elements.len();

    let mut mod_to_feature = HashMap::new();

    for result in rdr.deserialize() {
        let record: ModFeature = result?;
        let feature_vector = parse_mod_formula(&record.composition, &mod_elem_to_idx, mod_feature_size);
        mod_to_feature.insert(record.mod_name, feature_vector);
    }

    Ok(mod_to_feature)
}
