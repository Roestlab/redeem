use anyhow::{Result, anyhow};
use std::{collections::HashMap, ops::Deref};
use ndarray::Array2;
use candle_core::{Device, Tensor};

use crate::building_blocks::building_blocks::AA_EMBEDDING_SIZE;

/// Convert peptide sequences into AA ID array.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
pub fn get_aa_indices(seq: &str) -> Result<Array2<i64>> {
    let seq_len = seq.len();
    let mut result = Array2::<i64>::zeros((1, seq_len + 2));

    for (j, c) in seq.chars().enumerate() {
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