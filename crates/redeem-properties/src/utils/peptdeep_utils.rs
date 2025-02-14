use anyhow::{Context, Result, Error};
use std::fs::File;
use std::ops::Index;
use std::path::PathBuf;
use std::io;
use std::fs;
use log::info;
use csv::ReaderBuilder;
use reqwest;
use regex::Regex;
use std::collections::HashMap;
use serde::Deserialize;
use zip::ZipArchive;

const MOD_TSV_URL: &str = "https://raw.githubusercontent.com/MannLabs/alphabase/main/alphabase/constants/const_files/modification.tsv";
const MOD_TSV_PATH: &str = "data/modification.tsv";

const PRETRAINED_MODELS_URL: &str = "https://github.com/singjc/redeem/releases/download/v0.1.0-alpha/peptdeep_generic_pretrained_models.zip";
const PRETRAINED_MODELS_ZIP: &str = "data/peptdeep_generic_pretrained_models.zip";
const PRETRAINED_MODELS_PATH: &str = "data/peptdeep_generic_pretrained_models";


// Constants and Utility Structs

const INSTRUMENT_DICT: &[(&str, usize)] = &[
    ("QE", 0),
    ("LUMOS", 1),
    ("TIMSTOF", 2),
    ("SCIEXTOF", 3),
    ("THERMOTOF", 4),
];

const MAX_INSTRUMENT_NUM: usize = 8;

const UNKNOWN_INSTRUMENT_NUM: usize = MAX_INSTRUMENT_NUM - 1;


pub fn download_pretrained_models_exist() -> Result<PathBuf, io::Error> {
    let zip_path = PathBuf::from(PRETRAINED_MODELS_ZIP);
    let extract_dir = PathBuf::from(PRETRAINED_MODELS_PATH);

    // Ensure the parent directory exists
    if let Some(parent) = zip_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download the zip file if it doesn't exist
    if !zip_path.exists() {
        info!("Downloading pretrained models...");
        let mut response = reqwest::blocking::get(PRETRAINED_MODELS_URL)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut file = File::create(&zip_path)?;
        io::copy(&mut response, &mut file)?;
    }

    // Unzip the file if the target directory doesn't exist
    if !extract_dir.exists() {
        info!("Unzipping pretrained models...");
        let file = File::open(&zip_path)?;
        let mut archive = ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = extract_dir.join(file.mangled_name());

            if file.name().ends_with('/') {
                // Create directory
                fs::create_dir_all(&outpath)?;
            } else {
                // Write file
                if let Some(parent) = outpath.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut outfile = File::create(&outpath)?;
                io::copy(&mut file, &mut outfile)?;
            }
        }
    }

    Ok(extract_dir)
}

pub fn parse_instrument_index(instrument: &str) -> usize {
    let upper_instrument = instrument.to_uppercase();
    
    INSTRUMENT_DICT.iter()
        .find(|&&(name, _)| name == upper_instrument)
        .map_or(UNKNOWN_INSTRUMENT_NUM, |&(_, index)| index)
}


#[derive(Clone, Debug, Deserialize)]
/// Represents the constants used in a model.
pub struct ModelConstants {
    /// The size of the amino acid embedding.
    pub aa_embedding_size: Option<usize>,
    /// The charge factor used in the model.
    pub charge_factor: Option<f32>,
    /// The list of instruments used in the model.
    pub instruments: Vec<String>,
    /// The maximum number of instruments allowed in the model.
    pub max_instrument_num: usize,
    /// The list of modification elements used in the model.
    pub mod_elements: Vec<String>,
    /// The NCE (Normalized Collision Energy) factor used in the model.
    pub nce_factor: Option<f32>,
}

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
        info!("Downloading modification.tsv...");
        let mut response = reqwest::blocking::get(MOD_TSV_URL)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let mut file = File::create(&path)?;
        response.copy_to(&mut file)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
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


// #[derive(Debug, Clone)]
pub struct ModificationMap {
    pub name: String,
    pub amino_acid: Option<char>, // Optional if not applicable
}


pub fn load_modifications() -> Result<HashMap<(String, Option<char>), ModificationMap>> {
    let path: PathBuf = ensure_mod_tsv_exists().context("Failed to ensure TSV exists")?;

    let mut rdr = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path).context("Failed to read TSV file")?;

    let mut modifications = HashMap::new();
    
    for result in rdr.records() {
        let record = result.context("Failed to read record")?;
        let mod_name = record.get(0).unwrap_or("").to_string();
        let unimod_mass: f64 = record.get(1).unwrap_or("0").parse().unwrap_or(0.0);
        
        // Convert mass to string with 4 decimal places
        let mass_key = format!("{:.4}", unimod_mass);
        
        // Extract amino acid from mod_name
        let amino_acid = mod_name.split('@').nth(1).and_then(|aa| aa.chars().next());

        // Create Modification struct
        let modification = ModificationMap {
            name: mod_name,
            amino_acid,
        };

        // Insert into HashMap
        modifications.insert((mass_key, amino_acid), modification);
    }

    Ok(modifications)
}

pub fn remove_mass_shift(peptide: &str) -> String {
    let re = Regex::new(r"\[.*?\]").unwrap();
    re.replace_all(peptide, "").to_string()
}


pub fn extract_masses(peptide: &str) -> Vec<f64> {
    let mut masses = Vec::new();
    let mut start = 0;

    while let Some(open_bracket) = peptide[start..].find('[') {
        let open_index = start + open_bracket;
        if let Some(close_index) = peptide[open_index..].find(']') {
            let mass_str = &peptide[open_index + 1..open_index + close_index];
            if let Ok(mass) = mass_str.trim_start_matches('+').parse::<f64>() {
                masses.push(mass);
            }
            start = open_index + close_index + 1; // Move past the current bracket
        } else {
            break; // No closing bracket found
        }
    }

    masses
}

pub fn extract_masses_and_indices(peptide: &str) -> Vec<(f64, usize)> {
    let re = Regex::new(r"\[([+-]?\d*\.?\d+)\]").unwrap();
    let mut results = Vec::new();
    let mut offset = 0;

    for cap in re.captures_iter(peptide) {
        if let (Some(whole_match), Some(mass_str)) = (cap.get(0), cap.get(1)) {
            if let Ok(mass) = mass_str.as_str().parse::<f64>() {
                let index = whole_match.start() - offset;
                results.push((mass, index));
                offset += whole_match.len();
            }
        }
    }

    results
}


pub fn get_modification_indices(peptide: &str) -> String {
    let re = Regex::new(r"\[.*?\]").unwrap();
    let mut indices = Vec::new();
    let mut offset = 1; // Offset by 1 for 0-based index

    for mat in re.find_iter(peptide) {
        let index = mat.start().saturating_sub(offset);
        indices.push(index.to_string());
        offset += mat.end() - mat.start();
    }

    indices.join(";")
}

pub fn get_modification_string(
    peptide: &str,
    modification_map: &HashMap<(String, Option<char>), ModificationMap>,
) -> String {
    let naked_peptide = remove_mass_shift(peptide);

    let extracted_masses_and_indices = extract_masses_and_indices(&peptide.to_string());

    let mut found_modifications = Vec::new();

    // Map modifications based on extracted masses and indices
    for (mass, index) in extracted_masses_and_indices {
        // Subtract 1 from index to get 0-based index, ensure it's within bounds
        let index = index.saturating_sub(1);
        let amino_acid = naked_peptide.chars().nth(index).unwrap_or('\0');

        if let Some(modification) = modification_map
            .get(&(format!("{:.4}", mass), Some(amino_acid)))
        {
            found_modifications.push(modification.name.clone());
        } else if let Some(modification) =
            modification_map.get(&(format!("{:.4}", mass), None))
        {
            found_modifications.push(modification.name.clone());
        }
    }

    found_modifications.join(";")
}



// TODO: Derive from PeptDep constants yaml
const IM_GAS_MASS: f64 = 28.0; 
const CCS_IM_COEF: f64 = 1059.62245; 

/// Calculates the reduced mass for CCS and mobility calculation.
///
/// This function computes the reduced mass using the formula:
/// reduced_mass = (precursor_mz * charge * IM_GAS_MASS) / (precursor_mz * charge + IM_GAS_MASS)
///
/// # Arguments
///
/// * `precursor_mz` - The precursor mass-to-charge ratio (m/z)
/// * `charge` - The charge of the ion
///
/// # Returns
///
/// The calculated reduced mass as a f64 value
pub fn get_reduced_mass(precursor_mz: f64, charge: f64) -> f64 {
    let reduced_mass = precursor_mz * charge;
    reduced_mass * IM_GAS_MASS / (reduced_mass + IM_GAS_MASS)
}

/// Converts CCS (Collision Cross Section) to mobility for Bruker (timsTOF) instruments.
///
/// This function calculates the mobility using the formula:
/// mobility = (ccs_value * sqrt(reduced_mass)) / (charge * CCS_IM_COEF)
///
/// # Arguments
///
/// * `ccs_value` - The Collision Cross Section value
/// * `charge` - The charge of the ion
/// * `precursor_mz` - The precursor mass-to-charge ratio (m/z)
///
/// # Returns
///
/// The calculated mobility as a f64 value
pub fn ccs_to_mobility_bruker(ccs_value: f64, charge: f64, precursor_mz: f64) -> f64 {
    let reduced_mass = get_reduced_mass(precursor_mz, charge);
    ccs_value * f64::sqrt(reduced_mass) / (charge * CCS_IM_COEF)
}


#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn test_ensure_pretrained_models_exist() {
        let result = download_pretrained_models_exist();
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_modification_indices() {
        // Compile the regex once for all tests
        // let re = Regex::new(r"\[.*?\]").unwrap();

        // Test cases
        let test_cases = vec![
            ("PEPTIDE", ""),
            ("PEPT[+15.9949]IDE", "3"),
            ("P[+15.9949]EPT[+79.99]IDE", "0;3"),
            ("TVQSLEIDLDSM[+15.9949]R", "11"),
            ("TVQS[+79.99]LEIDLDSM[+15.9949]R", "3;11"),
            ("[+42.0106]PEPTIDE", "0"),
            ("PEPTIDE[+42.0106]", "6"),
            ("P[+15.9949]EP[+79.99]T[+15.9949]IDE", "0;2;3"),
        ];

        for (peptide, expected) in test_cases {
            let result = get_modification_indices(peptide);
            println!("Peptide: {}, Expected: {}, Result: {}", peptide, expected, result);
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }
    }

    #[test]
    fn test_extract_masses_and_indices() {
        let test_cases = vec![
            ("PEPTIDE", vec![]),
            ("PEPT[+15.9949]IDE", vec![(15.9949, 4)]),
            ("P[+15.9949]EPT[+79.99]IDE", vec![(15.9949, 1), (79.99, 4)]),
            ("TVQSLEIDLDSM[+15.9949]R", vec![(15.9949, 12)]),
            ("TVQS[+79.99]LEIDLDSM[+15.9949]R", vec![(79.99, 4), (15.9949, 12)]),
            ("[+42.0106]PEPTIDE", vec![(42.0106, 0)]),
            ("PEPTIDE[+42.0106]", vec![(42.0106, 7)]),
            ("P[+15.9949]EP[+79.99]T[+15.9949]IDE", vec![(15.9949, 1), (79.99, 3), (15.9949, 4)]),
            ("PEPTIDE[]", vec![]),  // Empty modification
            ("P[+15.9949]EP[-79.99]TIDE", vec![(15.9949, 1), (-79.99, 3)]),  // Negative mass
            ("P[+15.9949]EP[79.99]TIDE", vec![(15.9949, 1), (79.99, 3)]),  // No plus sign
        ];

        for (peptide, expected) in test_cases {
            let result = extract_masses_and_indices(peptide);
            println!("Peptide: {}, Expected: {:?}, Result: {:?}", peptide, expected, result);
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }
    }

    #[test]
    fn test_get_modification_string() {
        let modification_map = load_modifications().unwrap();

        let test_cases = vec![
            ("PEPTIDE", ""),
            ("PEPT[+15.9949]IDE", "Oxidation@T"),
            ("P[+15.9949]EPT[+79.9663]IDE", "Oxidation@P;Phospho@T"),
            ("TVQSLEIDLDSM[+15.9949]R", "Oxidation@M"),
            ("TVQS[+79.9663]LEIDLDSM[+15.9949]R", "Phospho@S;Oxidation@M"),
            ("[+42.0106]PEPTIDE", "Acetyl@Protein_N-term"),
            ("PEPTIDE[+42.0106]", ""),
            ("P[+15.9949]EP[+79.9663]T[+15.9949]IDE", "Oxidation@P;Oxidation@T"),
        ];


        for (peptide, expected) in test_cases {
            let result = get_modification_string(peptide, &modification_map);
            println!("Peptide: {}, Expected: {}, Result: {}", peptide, expected, result);
            assert_eq!(result, expected, "Failed for peptide: {}", peptide);
        }

    }

    #[test]
    fn test_ccs_to_mobility_bruker() {
        let ccs_value = 0.032652;
        let charge = 2.0;
        let precursor_mz = 762.329553;

        let result = ccs_to_mobility_bruker(ccs_value, charge, precursor_mz);
        
        let expected = 8.078969627454307e-05;
        println!("Rust result: {}", result);
        println!("Python result: {}", expected);
        println!("Difference: {}", (result - expected).abs());

        assert_eq!(result, expected, "Failed for ccs_to_mobility_bruker");

    }

}