use anyhow::{Result, anyhow};
use std::collections::HashMap;
use ndarray::Array2;
use candle_core::{Tensor, Device};

/// Convert peptide sequences into AA ID array.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
pub fn get_aa_indices(seq_array: &[String]) -> Result<Array2<i64>> {
    let seq_len = seq_array[0].len();
    let mut result = Array2::<i64>::zeros((seq_array.len(), seq_len + 2));

    for (i, seq) in seq_array.iter().enumerate() {
        for (j, c) in seq.chars().enumerate() {
            let aa_index = (c as i64) - ('A' as i64) + 1;
            result[[i, j + 1]] = aa_index;
        }
    }
    
    Ok(result)
}


/// One-hot encode amino acid indices.
pub fn aa_one_hot(aa_indices: &Tensor, aa_embedding_size: usize) -> Result<Tensor> {
    let (batch_size, seq_len) = aa_indices.shape().dims2()?;
    let num_classes = aa_embedding_size;

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
    Tensor::from_slice(&one_hot_data, (batch_size, seq_len, num_classes), aa_indices.device())
        .map_err(|e| anyhow!("{}", e))
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