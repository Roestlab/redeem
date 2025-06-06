use ndarray::{s, Array1, Axis};
// use ndarray_stats::QuantileExt;

use crate::error::TdcError;

/// Estimate q-values using target-decoy competition.
///
/// This function implements the simple target-decoy competition method to estimate q-values.
/// For a set of target and decoy PSMs meeting a specified score threshold, the false discovery
/// rate (FDR) is estimated as:
///
/// FDR = (Decoys + 1) / Targets
/// 
/// Adpated from: https://github.com/wfondrie/mokapot/blob/main/mokapot/qvalues.py#L28
///
/// # Arguments
///
/// * `scores` - A 1D array containing the scores to rank by.
/// * `target` - A 1D boolean array indicating if the entry is from a target (true) or decoy (false) hit.
/// * `desc` - A boolean indicating if higher scores are better (true) or if lower scores are better (false).
///
/// # Returns
///
/// A 1D array with the estimated q-value for each entry. The array is the same length as the `scores` and `target` arrays.
pub fn tdc(scores: &Array1<f32>, target: &Array1<bool>, desc: bool) -> Result<Array1<f32>, TdcError> {
    // Validate inputs
    if scores.len() != target.len() {
        return Err(TdcError::LengthMismatch);
    }

    // Check for NaN values and log information
    let nan_count = scores.iter().filter(|x| x.is_nan()).count();
    if nan_count > 0 {
        // Log first 10 scores for debugging
        log::error!("First 10 scores: {:?}", scores.iter().take(10).collect::<Vec<_>>());
        log::error!("Found {} NaN values in scores array (total scores: {})", nan_count, scores.len());
        return Err(TdcError::NaNFound(nan_count));
    }

    // Sort scores and target
    let mut sorted_indices = (0..scores.len()).collect::<Vec<usize>>();
    if desc {
        sorted_indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a])
                .expect("NaN check should have caught this - unexpected comparison failure")
        });
    } else {
        sorted_indices.sort_unstable_by(|&a, &b| {
            scores[a].partial_cmp(&scores[b])
                .expect("NaN check should have caught this - unexpected comparison failure")
        });
    }

    let sorted_scores = sorted_indices.iter().map(|&i| scores[i]).collect::<Array1<f32>>();
    let sorted_target = sorted_indices.iter().map(|&i| target[i]).collect::<Array1<bool>>();

    // Calculate cumulative sums
    let cum_targets = sorted_target.iter().scan(0, |acc, &x| {
        *acc += x as usize;
        Some(*acc)
    }).collect::<Array1<usize>>();

    let cum_decoys = sorted_target.iter().scan(0, |acc, &x| {
        *acc += (!x) as usize;
        Some(*acc)
    }).collect::<Array1<usize>>();

    let num_total = &cum_targets + &cum_decoys;

    // Calculate FDR
    let fdr = cum_decoys.mapv(|x| (x + 1) as f32) / cum_targets.mapv(|x| x.max(1) as f32);

    // Find unique scores and their indices
    let mut unique_scores = Vec::new();
    let mut indices = Vec::new();
    let mut current_score = sorted_scores[0];
    let mut count = 0;
    for &score in sorted_scores.iter() {
        if score != current_score {
            unique_scores.push(current_score);
            indices.push(count);
            current_score = score;
            count = 1;
        } else {
            count += 1;
        }
    }
    unique_scores.push(current_score);
    indices.push(count);

    // Flip arrays if necessary
    let fdr = if desc { fdr.slice(s![..;-1]).to_owned() } else { fdr };
    let num_total = if desc { num_total.slice(s![..;-1]).to_owned() } else { num_total };
    let unique_scores = if !desc { unique_scores.into_iter().rev().collect() } else { unique_scores };
    let indices = if !desc { indices.into_iter().rev().collect() } else { indices };

    // Calculate q-values
    let qvals = fdr2qvalue(&fdr, &num_total, &unique_scores, &indices);

    // Reorder q-values to match original order
    let mut final_qvals = Array1::<f32>::zeros(scores.len());
    for (i, &idx) in sorted_indices.iter().enumerate() {
        final_qvals[idx] = qvals[i];
    }

    Ok(final_qvals)
}

// /// Convert a list of FDRs to q-values.
// ///
// /// This turns a list of FDRs into q-values. All of the inputs are assumed to be sorted.
// /// 
// /// Adpated from: https://github.com/wfondrie/mokapot/blob/main/mokapot/qvalues.py#L148
// ///
// /// # Arguments
// ///
// /// * `fdr` - A vector of all unique FDR values.
// /// * `num_total` - A vector of the cumulative number of PSMs at each score.
// /// * `met` - A vector of the unique scores for each PSM.
// /// * `indices` - A vector where the value at index i indicates the number of PSMs that shared the unique FDR value in `fdr`.
// ///
// /// # Returns
// ///
// /// A vector of q-values.
// fn fdr2qvalue(fdr: &Array1<f32>, num_total: &Array1<usize>, met: &Vec<f32>, indices: &Vec<usize>) -> Array1<f32> {
//     let mut min_q: f32 = 1.0;
//     let mut qvals = Array1::<f32>::ones(fdr.len());
//     let mut prev_idx = 0;

//     for (idx, &count) in indices.iter().enumerate() {
//         let next_idx = prev_idx + count;
//         let group = prev_idx..next_idx;

//         let fdr_group = fdr.slice(s![group.clone()]);
//         let n_group = num_total.slice(s![group.clone()]);
        
//         let curr_fdr = fdr_group[n_group.argmax().unwrap()]; // This line is not working in ndarray v0.15.0, as argmax is only available in v0.16.0, but linfa uses v0.15.0. They are working on updating to v0.16.0. https://github.com/rust-ml/linfa/pull/371
//         if curr_fdr < min_q {
//             min_q = curr_fdr;
//         }

//         qvals.slice_mut(s![group]).fill(min_q);
//         prev_idx = next_idx;
//     }

//     qvals
// }

/// Convert a list of FDRs to q-values.
///
/// This turns a list of FDRs into q-values. All of the inputs are assumed to be sorted.
/// 
/// Adpated from: https://github.com/wfondrie/mokapot/blob/main/mokapot/qvalues.py#L148
///
/// # Arguments
///
/// * `fdr` - A vector of all unique FDR values.
/// * `num_total` - A vector of the cumulative number of PSMs at each score.
/// * `met` - A vector of the unique scores for each PSM.
/// * `indices` - A vector where the value at index i indicates the number of PSMs that shared the unique FDR value in `fdr`.
///
/// # Returns
///
/// A vector of q-values.
fn fdr2qvalue(fdr: &Array1<f32>, num_total: &Array1<usize>, met: &Vec<f32>, indices: &Vec<usize>) -> Array1<f32> {
    let mut min_q: f32 = 1.0;
    let mut qvals = Array1::<f32>::ones(fdr.len());
    let mut prev_idx = 0;

    for (idx, &count) in indices.iter().enumerate() {
        let next_idx = prev_idx + count;
        let group = prev_idx..next_idx;

        let fdr_group = fdr.slice(s![group.clone()]);
        let n_group = num_total.slice(s![group.clone()]);

        // Manual implementation of argmax for ndarray v0.15.0
        let mut max_index = 0;
        let mut max_value = n_group[0];
        for (i, &value) in n_group.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        let curr_fdr = fdr_group[max_index];
        if curr_fdr < min_q {
            min_q = curr_fdr;
        }

        qvals.slice_mut(s![group]).fill(min_q);
        prev_idx = next_idx;
    }

    qvals
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq; // For floating-point comparisons

    #[test]
    fn test_tdc_basic_working_case_descending() {
        // Test case where higher scores are better (desc = true)
        let scores = array![3.2, 1.5, 4.0, 2.1, 5.5];
        let target = array![true, false, true, false, true];
        
        let result = tdc(&scores, &target, true).unwrap();
        
        // Check basic properties
        assert_eq!(result.len(), 5);
        // Q-values should be monotonically increasing
        for i in 0..result.len()-1 {
            assert!(result[i] <= result[i+1], "Q-values should be monotonically increasing");
        }
        // First entry (highest score) should have lowest q-value
        assert!(result[4] < result[0]); // 5.5 is highest score at index 4
    }

    #[test]
    fn test_tdc_basic_working_case_ascending() {
        // Test case where lower scores are better (desc = false)
        let scores = array![3.2, 1.5, 4.0, 2.1, 5.5];
        let target = array![true, false, true, false, true];
        
        let result = tdc(&scores, &target, false).unwrap();
        
        assert_eq!(result.len(), 5);
        // Q-values should be monotonically increasing
        for i in 0..result.len()-1 {
            assert!(result[i] <= result[i+1], "Q-values should be monotonically increasing");
        }
        // First entry (lowest score) should have lowest q-value
        assert!(result[1] < result[4]); // 1.5 is lowest score at index 1
    }

    #[test]
    fn test_tdc_with_nan_values() {
        let _ = env_logger::builder().is_test(true).try_init();
        let scores = array![3.2, f32::NAN, 4.0, 2.1, 5.5];
        let target = array![true, false, true, false, true];
        
        let result = tdc(&scores, &target, true);
        println!("Error result: {:?}", result);  
        
        assert!(result.is_err());
        match result.unwrap_err() {
            TdcError::NaNFound(count) => {
                println!("Found {} NaN values", count);  
                assert_eq!(count, 1);
            },
            _ => panic!("Expected NaNFound error"),
        }
    }

    #[test]
    fn test_tdc_length_mismatch() {
        // Test case with mismatched array lengths
        let scores = array![3.2, 1.5, 4.0];
        let target = array![true, false];
        
        let result = tdc(&scores, &target, true);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            TdcError::LengthMismatch => (),
            _ => panic!("Expected LengthMismatch error"),
        }
    }

    // #[test]
    // fn test_tdc_edge_cases() {
    //     // Test edge cases - empty arrays
    //     let scores = array![];
    //     let target = array![];
        
    //     let result = tdc(&scores, &target, true).unwrap();
    //     assert_eq!(result.len(), 0);

    //     // All targets
    //     let scores = array![1.0, 2.0, 3.0];
    //     let target = array![true, true, true];
    //     let result = tdc(&scores, &target, true).unwrap();
    //     assert_abs_diff_eq!(result, array![0.0, 0.0, 0.0], epsilon = 1e-6);

    //     // All decoys
    //     let scores = array![1.0, 2.0, 3.0];
    //     let target = array![false, false, false];
    //     let result = tdc(&scores, &target, true).unwrap();
    //     // All q-values should be infinity (or very large)
    //     assert!(result.iter().all(|&x| x > 1e6));
    // }
}