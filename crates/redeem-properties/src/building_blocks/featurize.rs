use anyhow::{Result, anyhow};
use std::{collections::HashMap, ops::Deref};
use ndarray::Array2;
use candle_core::{DType, Device, Tensor};

use crate::building_blocks::building_blocks::AA_EMBEDDING_SIZE;

/// Convert peptide sequences into AA ID array.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
/// 
/// Example:
/// ```rust
/// use redeem_properties::building_blocks::featurize::get_aa_indices;
/// use anyhow::Result;
/// use ndarray::Array2;
/// 
/// let seq = "AGHCEWQMKYR";
/// let result = get_aa_indices(seq).unwrap();
/// println!("aa_indices: {:?}", result);
/// let expect_out = Array2::from_shape_vec((1, 13), vec![0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]).unwrap();
/// assert_eq!(result.shape(), &[1, 13]);
/// assert_eq!(result, expect_out);
/// ```
pub fn get_aa_indices(seq: &str) -> Result<Array2<i64>> {
    let valid_aa = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // amino acids as defined in alphabase: https://github.com/MannLabs/alphabase/blob/main/alphabase/constants/const_files/amino_acid.tsv
    let filtered_seq: String = seq.chars().filter(|c| valid_aa.contains(*c)).collect();

    // TODO: Maybe this should be done higher up in the pipeline, and this should panic here instead.
    // But for now this is done to deal with cases like: -MQPLSKL
    if seq.len() != filtered_seq.len() {
        log::trace!("Invalid amino acid characters found in sequence: {:?}, stripping them out to {:?}", seq, filtered_seq);
    }

    let seq_len = filtered_seq.len();
    let mut result = Array2::<i64>::zeros((1, seq_len + 2));

    for (j, c) in filtered_seq.chars().enumerate() {
        let aa_index = (c as i64) - ('A' as i64) + 1;
        result[[0, j + 1]] = aa_index;
    }

    Ok(result)
}

/// Convert peptide sequences into ASCII code array.
///
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L115
pub fn get_ascii_indices(peptide_sequences: &[String], device: Device) -> Result<Tensor> {
    // println!("Peptide sequences to encode: {:?}", peptide_sequences);
    let max_len = peptide_sequences.iter().map(|s| s.len()).max().unwrap_or(0) + 2; // +2 for padding
    let batch_size = peptide_sequences.len();

    let mut aa_indices = vec![0u32; batch_size * max_len];

    for (i, peptide) in peptide_sequences.iter().enumerate() {
        for (j, c) in peptide.chars().enumerate() {
            aa_indices[i * max_len + j + 1] = c as u32; // +1 to skip the first padding
        }
    }
    let aa_indices_tensor =
        Tensor::from_slice(&aa_indices, (batch_size, max_len), &device)?;
    Ok(aa_indices_tensor)
}

/// One-hot encode amino acid indices and concatenate additional tensors.
pub fn aa_one_hot(aa_indices: &Tensor, cat_others: &[&Tensor]) -> Result<Tensor> {
    let (batch_size, seq_len) = aa_indices.shape().dims2()?;
    let num_classes = AA_EMBEDDING_SIZE;

    let mut one_hot_data = vec![0.0f32; batch_size * seq_len * num_classes];

    // Iterate over the 2D tensor directly
    for batch_idx in 0..batch_size {
        for seq_idx in 0..seq_len {
            let index = aa_indices.get(batch_idx)?.get(seq_idx)?.to_scalar::<f32>()?;
            let class_idx = index.round() as usize; // Round to nearest integer and convert to usize
            if class_idx < num_classes {
                one_hot_data[batch_idx * seq_len * num_classes + seq_idx * num_classes + class_idx] = 1.0;
            }
        }
    }

    // Convert the one_hot_data vector directly to a tensor
    let one_hot_tensor = Tensor::from_slice(&one_hot_data, (batch_size, seq_len, num_classes), aa_indices.device())
        .map_err(|e| anyhow!("{}", e))?;

    // Concatenate additional tensors if provided
    let mut output_tensor = one_hot_tensor;

    for other in cat_others {
        output_tensor = Tensor::cat(&[output_tensor, other.deref().clone()], 2)?;
    }

    Ok(output_tensor)
}


/// Get the modification features for a given set of modifications and modification sites.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L47
pub fn get_mod_features(mods: &str, mod_sites: &str, seq_len: usize, mod_feature_size: usize, mod_to_feature: HashMap<String, Vec<f32>>, device: Device) -> Result<Tensor> {
    let mod_names: Vec<&str> = mods.split(';').filter(|&s| !s.is_empty()).collect();
    let mod_sites: Vec<usize> = mod_sites
        .split(';')
        .filter(|&s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .collect();

    // let mod_feature_size = self.constants.mod_elements.len();

    let mut mod_x = vec![0.0f32; seq_len * mod_feature_size];

    for (mod_name, &site) in mod_names.iter().zip(mod_sites.iter()) {
        if let Some(feat) = mod_to_feature.get(*mod_name) {
            for (i, &value) in feat.iter().enumerate() {
                if site < seq_len {
                    mod_x[site * mod_feature_size + i] += value;
                }
            }
            // println!("Site: {}, feat: {:?}", site, feat);
        }
    }

    Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), &device)
        .map_err(|e| anyhow!("Failed to create tensor: {}", e))
}


const VALID_AA: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Precomputes amino acid index map from characters A-Z
fn aa_index_map() -> HashMap<char, i64> {
    VALID_AA
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i64 + 1))
        .collect()
}

/// Efficiently converts an amino acid sequence to a padded tensor of indices
pub fn aa_indices_tensor(seq: &str, device: &Device) -> Result<Tensor> {
    let map = aa_index_map();
    let filtered: Vec<i64> = seq
        .chars()
        .filter_map(|c| map.get(&c).copied())
        .collect();
    let mut indices = vec![0i64]; // padding start
    indices.extend(filtered);
    indices.push(0); // padding end

    Ok(Tensor::from_slice(&indices, (1, indices.len()), device)?.to_dtype(DType::F32)?.unsqueeze(2)?)
}


/// Optimized version of get_mod_features that avoids repeated parsing
pub fn get_mod_features_from_parsed(
    mod_names: &[&str],
    mod_sites: &[usize],
    seq_len: usize,
    mod_feature_size: usize,
    mod_to_feature: &HashMap<String, Vec<f32>>,
    device: &Device,
) -> Result<Tensor> {
    let mut mod_x = vec![0.0f32; seq_len * mod_feature_size];

    for (mod_name, &site) in mod_names.iter().zip(mod_sites.iter()) {
        if site >= seq_len {
            log::warn!("Skipping mod {} at invalid site {} (seq_len {})", mod_name, site, seq_len);
            continue;
        }
        if let Some(feat) = mod_to_feature.get(*mod_name) {
            for (i, &val) in feat.iter().enumerate() {
                mod_x[site * mod_feature_size + i] += val;
            }
        } else {
            log::warn!("Unknown modification feature: {}", mod_name);
        }
    }

    Ok(Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), device)
        .map_err(|e| anyhow!("Failed to create tensor: {}", e))?)
}


#[cfg(test)]
mod tests {
 
    use crate::utils::peptdeep_utils::load_mod_to_feature;
    use crate::utils::peptdeep_utils::parse_model_constants;
    use crate::utils::peptdeep_utils::ModelConstants;

    use super::*;
    use candle_core::Device;
    use candle_core::Tensor;
    use ndarray::Array2;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn test_get_aa_indices() {
        let seq = "AGHCEWQMKYR";
        let result = get_aa_indices(seq).unwrap();
        // expected result is [[0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]]
        let expect_out = Array2::from_shape_vec((1, 13), vec![0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]).unwrap();
        println!("{:?} - aa_indices: {:?}", seq, result);
        assert_eq!(result.shape(), &[1, 13]);
        assert_eq!(result, expect_out);
    }

    #[test]
    fn test_aa_indices_tensor(){
        let device = Device::Cpu;
        let seq = "AGHCEWQMKYR";
        let result = aa_indices_tensor(seq, &device).unwrap();
        // expected result is [[0, 1, 7, 8, 3, 5, 23, 17, 13, 11, 25, 18, 0]]
        let expect_out = Tensor::from_vec(vec!{0.0f32, 1.0f32, 7.0f32, 8.0f32, 3.0f32, 5.0f32, 23.0f32, 17.0f32, 13.0f32, 11.0f32, 25.0f32, 18.0f32, 0.0f32}, (1, 13), &device).unwrap();
        println!("{:?} - aa_indices_tensor: {:?}", seq, result.to_vec3::<f32>().unwrap());
        println!("result shape: {:?}", result.shape());
        assert_eq!(result.shape().dims(), &[1, 13, 1]);
        // assert_eq!(result.to_vec3::<f32>().unwrap(), expect_out.to_vec3::<f32>().unwrap());
    }

    #[test]
    fn test_get_mod_features() {
        let mods = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let mod_sites = "0;4;8";
        let seq_len = 11 + 2;
        let mod_feature_size = 109;

        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let constants: ModelConstants =
            parse_model_constants(constants_path.to_str().unwrap()).unwrap();
        let mod_to_feature: HashMap<String, Vec<f32>> = load_mod_to_feature(&constants).unwrap();

        let device = Device::Cpu;
        let tensor = get_mod_features(
            mods,
            mod_sites,
            seq_len,
            mod_feature_size,
            mod_to_feature,
            device,
        ).unwrap();
        println!("tensor shape: {:?}", tensor.shape());
        assert_eq!(tensor.shape().dims(), &[1, seq_len, mod_feature_size]);
    }

    #[test]
    fn test_get_mod_features_from_parsed() {
        let mods_str = "Acetyl@Protein N-term;Carbamidomethyl@C;Oxidation@M";
        let sites_str = "0;4;8";

        // Manually parse and split
        let mod_names: Vec<&str> = mods_str.split(';').filter(|s| !s.is_empty()).collect();
        let mod_sites: Vec<usize> = sites_str
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap())
            .collect();
        let seq_len = 11 + 2;
        let mod_feature_size = 109;

        let constants_path =
            PathBuf::from("data/models/alphapeptdeep/generic/rt.pth.model_const.yaml");
        let constants: ModelConstants =
            parse_model_constants(constants_path.to_str().unwrap()).unwrap();
        let mod_to_feature: HashMap<String, Vec<f32>> = load_mod_to_feature(&constants).unwrap();

        let device = Device::Cpu;
        let tensor = get_mod_features_from_parsed(
            &mod_names,
            &mod_sites,
            seq_len,
            mod_feature_size,
            &mod_to_feature,
            &device,
        ).unwrap();

        println!("tensor shape: {:?}", tensor.shape());

        assert_eq!(tensor.shape().dims(), &[1, seq_len, mod_feature_size]);

    }
}