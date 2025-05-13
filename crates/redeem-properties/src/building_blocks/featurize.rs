use anyhow::{Result, anyhow};
use std::collections::HashMap;
use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::building_blocks::building_blocks::AA_EMBEDDING_SIZE;


const VALID_AA: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Precomputes amino acid index map from characters A-Z
fn aa_index_map() -> HashMap<char, i64> {
    VALID_AA
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i64 + 1))
        .collect()
}


/// Convert peptide sequences into AA ID array.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L88
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


/// One-hot encode amino acid indices and concatenate additional tensors.
pub fn aa_one_hot(aa_indices: &Tensor, cat_others: &[&Tensor]) -> Result<Tensor> {
    let (batch_size, seq_len) = aa_indices.shape().dims2()?;
    log::trace!("[aa_one_hot] batch_size: {}, seq_len: {}", batch_size, seq_len);
    let num_classes = AA_EMBEDDING_SIZE;

    let indices = aa_indices.to_vec2::<f32>()?;
    let mut one_hot_data = vec![0.0f32; batch_size * seq_len * num_classes];

    one_hot_data
        .par_chunks_mut(seq_len * num_classes)
        .zip(indices.par_iter())
        .enumerate()
        .try_for_each(|(batch_idx, (chunk, row))| -> Result<()> {
            for (seq_idx, &fidx) in row.iter().enumerate() {
                if !fidx.is_finite() {
                    return Err(anyhow!(
                        "Invalid AA index: found NaN or Inf at batch {}, position {}: {}",
                        batch_idx, seq_idx, fidx
                    ));
                }

                if fidx < 0.0 {
                    return Err(anyhow!(
                        "Invalid AA index: negative value at batch {}, position {}: {}",
                        batch_idx, seq_idx, fidx
                    ));
                }

                let class_idx = fidx.round() as usize;
                if class_idx >= num_classes {
                    return Err(anyhow!(
                        "AA index out of bounds: got {}, but num_classes = {} (batch {}, position {})",
                        class_idx, num_classes, batch_idx, seq_idx
                    ));
                }

                let index = seq_idx * num_classes + class_idx;
                chunk[index] = 1.0;
            }
            Ok(())
        })?;

    let one_hot_tensor = Tensor::from_slice(
        &one_hot_data,
        (batch_size, seq_len, num_classes),
        aa_indices.device(),
    )
    .map_err(|e| anyhow!("Failed to create one-hot tensor: {}", e))?;

    if cat_others.is_empty() {
        Ok(one_hot_tensor)
    } else {
        let mut features = vec![one_hot_tensor];
        features.extend(cat_others.iter().cloned().cloned());
        Ok(Tensor::cat(&features, 2)?)
    }
}





/// Get the modification features for a given set of modifications and modification sites.
/// 
/// Based on https://github.com/MannLabs/alphapeptdeep/blob/450518a39a4cd7d03db391108ec8700b365dd436/peptdeep/model/featurize.py#L47
pub fn get_mod_features_from_parsed(
    mod_names: &[&str],
    mod_sites: &[usize],
    seq_len: usize,
    mod_feature_size: usize,
    mod_to_feature: &HashMap<String, Vec<f32>>,
    device: &Device,
) -> Result<Tensor> {
    // Initialize buffer with atomic wrappers
    let atomic_buffer: Vec<AtomicU32> = (0..seq_len * mod_feature_size)
        .map(|_| AtomicU32::new(0))
        .collect();

    mod_names
        .par_iter()
        .zip(mod_sites.par_iter())
        .for_each(|(&mod_name, &site)| {
            if site >= seq_len {
                log::warn!(
                    "Skipping mod {} at invalid site {} (seq_len {})",
                    mod_name, site, seq_len
                );
                return;
            }
            if let Some(feat) = mod_to_feature.get(mod_name) {
                for (i, &val) in feat.iter().enumerate() {
                    let idx = site * mod_feature_size + i;
                    let val_bits = val.to_bits();
                    atomic_buffer[idx].fetch_add(val_bits, Ordering::Relaxed);
                }
            } else {
                log::warn!("Unknown modification feature: {}", mod_name);
            }
        });

    // Convert atomic buffer back to f32
    let mod_x: Vec<f32> = atomic_buffer
        .into_iter()
        .map(|a| f32::from_bits(a.load(Ordering::Relaxed)))
        .collect();

    Tensor::from_slice(&mod_x, (1, seq_len, mod_feature_size), device)
        .map_err(|e| anyhow!("Failed to create tensor: {}", e))
}


#[cfg(test)]
mod tests {
 
    use crate::utils::peptdeep_utils::load_mod_to_feature;
    use crate::utils::peptdeep_utils::parse_model_constants;
    use crate::utils::peptdeep_utils::ModelConstants;

    use super::*;
    use candle_core::Device;
    use candle_core::Tensor;
    use std::collections::HashMap;
    use std::path::PathBuf;

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