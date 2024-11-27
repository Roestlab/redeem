use ndarray::{s, Array1, Axis};
use ndarray_stats::QuantileExt;

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
pub fn tdc(scores: &Array1<f32>, target: &Array1<bool>, desc: bool) -> Array1<f32> {
    // Sort scores and target
    let mut sorted_indices = (0..scores.len()).collect::<Vec<usize>>();
    if desc {
        sorted_indices.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    } else {
        sorted_indices.sort_unstable_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
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

    final_qvals
}

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
        
        let curr_fdr = fdr_group[n_group.argmax().unwrap()];
        if curr_fdr < min_q {
            min_q = curr_fdr;
        }

        qvals.slice_mut(s![group]).fill(min_q);
        prev_idx = next_idx;
    }

    qvals
}
