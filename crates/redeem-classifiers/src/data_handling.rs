use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use crate::error::TdcError;
use crate::stats::tdc;

#[derive(Debug, Clone)]
pub struct PsmMetadata {
    /// Spectrum id
    pub spec_id: Vec<String>,
    /// File identifier
    pub file_id: Vec<usize>,
    /// Feature names
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub x: Array2<f32>,
    pub y: Array1<i32>,
    pub is_train: Array1<bool>,
    pub is_top_peak: Array1<bool>,
    pub tg_num_id: Array1<i32>,
    pub classifier_score: Array1<f32>,
    pub psm_metadata: PsmMetadata,
}

impl Experiment {
    pub fn new(x: Array2<f32>, y: Array1<i32>, psm_metadata: PsmMetadata) -> Self {
        let n_samples = x.nrows();
        Experiment {
            x,
            y,
            is_train: Array1::from_elem(n_samples, false),
            is_top_peak: Array1::from_elem(n_samples, false),
            tg_num_id: Array1::from_elem(n_samples, 0),
            classifier_score: Array1::from_elem(n_samples, 0.0),
            psm_metadata,
        }
    }

    pub fn log_input_data_summary(&self) {
        println!("----- Input Data Summary -----");
        println!("Info: {} Target PSMs and {} Decoy PSMs", self.y.iter().filter(|&&v| v == 1).count(), self.y.iter().filter(|&&v| v == -1).count());
        println!("Info: {} feature scores (columns)", self.x.ncols());
        println!("-------------------------------");
    }

    /// Update labels based on scores and a specified FDR threshold.
    ///
    /// This method is used during model training to define positive examples,
    /// which are traditionally the target PSMs that fall within a specified
    /// FDR threshold.
    /// This method is adapted from MS2Rescore
    ///
    /// # Arguments
    ///
    /// * `scores` - The scores used to rank the PSMs.
    /// * `eval_fdr` - The false discovery rate threshold to use.
    /// * `desc` - Are higher scores better?
    ///
    /// # Returns
    ///
    /// An Array1<i32> where 1 indicates a positive example, -1 indicates a negative example,
    /// and 0 removes the PSM from training. Typically, 0 is reserved for targets below
    /// the specified FDR threshold.
    pub fn update_labels(&self, scores: &Array1<f32>, eval_fdr: f32, desc: bool) -> Array1<i32> {
        let targets = &self.y.mapv(|v| v == 1);
        let qvals = tdc(scores, targets, desc).unwrap_or_else(|e| {
            panic!(
                "Fatal error in update_labels: {}. \
                 Scores likely contained {} NaN values when computing qvalues with target-decoy competition. \
                 This suggests a problem in the scoring function.",
                e,
                if let TdcError::NaNFound(n) = e { n } else { 0 }
            );
        });
        
        let unlabeled = (&qvals.mapv(|v| v > eval_fdr)) & targets;
        
        let mut new_labels = Array1::ones(qvals.len());
        for (i, &target) in targets.iter().enumerate() {
            if !target {
                new_labels[i] = -1;
            } else if unlabeled[i] {
                new_labels[i] = 0;
            }
        }
        
        new_labels
    }

    /// Update the "rank" feature column based on new classifier scores.
    ///
    /// This re-ranks all PSMs per spectrum (grouped by file_id and spec_id),
    /// and sets the rank column in `self.x` accordingly (1 = best).
    ///
    /// Also logs the percentage of PSMs whose rank changed.
    ///
    /// # Arguments
    /// * `scores` - The current classifier scores (same length as rows in `x`)
    /// * `metadata` - PSM metadata with file_id and spec_id for grouping
    pub fn update_rank_feature(&mut self, scores: &Array1<f32>, metadata: &PsmMetadata) {
        // 1. Locate the "rank" feature index
        let Some(rank_feature_idx) = metadata
            .feature_names
            .iter()
            .position(|name| name == "rank")
        else {
            log::warn!("No 'rank' feature found in feature_names — skipping rank update.");
            return;
        };

        // 2. Group PSMs by (file_id, spec_id)
        let mut spectrum_groups: HashMap<(usize, &str), Vec<(usize, f32)>> = HashMap::new();
        for i in 0..self.x.nrows() {
            spectrum_groups
                .entry((metadata.file_id[i], metadata.spec_id[i].as_str()))
                .or_default()
                .push((i, scores[i]));
        }

        let mut changed_ranks = 0;

        // 3. For each group, sort by score descending and assign new rank
        for group in spectrum_groups.values_mut() {
            group.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, (row_idx, _)) in group.iter().enumerate() {
                let old_rank = self.x[[*row_idx, rank_feature_idx]] as usize;
                let new_rank = rank + 1;
                if old_rank != new_rank {
                    changed_ranks += 1;
                }
                self.x[[*row_idx, rank_feature_idx]] = new_rank as f32;
            }
        }

        let total = self.x.nrows();
        let pct_changed = (changed_ranks as f64 / total as f64) * 100.0;

        log::debug!(
            "Updated rank feature for {} spectrum groups. Rank changed for {:.2}% of PSMs ({} of {}).",
            spectrum_groups.len(),
            pct_changed,
            changed_ranks,
            total
        );
    }

    /// Extracts the "rank" feature column as a 1D array of `u32`s.
    ///
    /// # Returns
    /// * `Ok(Array1<u32>)` containing the rank values (one per row in `x`)
    /// * `Err` if "rank" is not found in the feature names
    pub fn get_rank_column(&self) -> anyhow::Result<Array1<u32>> {
        let Some(rank_idx) = self
            .psm_metadata
            .feature_names
            .iter()
            .position(|name| name == "rank")
        else {
            anyhow::bail!("'rank' feature not found in feature_names");
        };

        let rank_f32 = self.x.column(rank_idx);

        let rank_u32 = rank_f32
            .iter()
            .map(|&val| {
                if val.is_finite() && val >= 0.0 {
                    val.round() as u32
                } else {
                    0 // fallback: treat NaNs or negatives as rank 0 (could also bail or panic if preferred)
                }
            })
            .collect::<Array1<u32>>();

        Ok(rank_u32)
    }


    pub fn get_top_test_peaks(&self) -> Experiment {
        let mask = &self.is_train.mapv(|x| !x) & &self.is_top_peak;
        self.filter(&mask)
    }
    

    pub fn get_decoy_peaks(&self) -> Experiment {
        // Sage represents decoy peaks as -1
        let mask = &self.y.mapv(|v| v == -1);  
        self.filter(mask)
    }

    pub fn get_target_peaks(&self) -> Experiment {
        // Sage represents target peaks as 1
        let mask = &self.y.mapv(|v| v != 1);  
        self.filter(mask)
    }

    pub fn get_top_decoy_peaks(&self) -> Experiment {
        let mask = &self.y.mapv(|v| v == 0) & &self.is_top_peak;
        self.filter(&mask)
    }
    
    pub fn get_top_target_peaks(&self) -> Experiment {
        let mask = &self.y.mapv(|v| v != 0) & &self.is_top_peak;
        self.filter(&mask)
    }

    pub fn get_feature_matrix(&self) -> Array2<f32> {
        self.x.clone()
    }

    /// Filter the experiment by applying a boolean mask to all row-aligned fields.
    ///
    /// This includes:
    /// - Feature matrix `x`
    /// - Labels `y`
    /// - Training/test flags `is_train`
    /// - Top peak flags `is_top_peak`
    /// - Target group identifiers `tg_num_id`
    /// - Classifier scores `classifier_score`
    /// - PSM metadata: `spec_id`, `file_id` (feature names are retained as-is)
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean mask (`Array1<bool>`) of the same length as the number of PSMs (rows in `x`)
    ///
    /// # Returns
    ///
    /// A new `Experiment` instance with only rows where `mask[i] == true`
    pub fn filter(&self, mask: &Array1<bool>) -> Experiment {
        let selected_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        fn filter_vec<T: Clone>(v: &Vec<T>, indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&i| v[i].clone()).collect()
        }

        Experiment {
            x: self.x.select(Axis(0), &selected_indices),
            y: self.y.select(Axis(0), &selected_indices),
            is_train: self.is_train.select(Axis(0), &selected_indices),
            is_top_peak: self.is_top_peak.select(Axis(0), &selected_indices),
            tg_num_id: self.tg_num_id.select(Axis(0), &selected_indices),
            classifier_score: self.classifier_score.select(Axis(0), &selected_indices),
            psm_metadata: PsmMetadata {
                spec_id: filter_vec(&self.psm_metadata.spec_id, &selected_indices),
                file_id: filter_vec(&self.psm_metadata.file_id, &selected_indices),
                feature_names: self.psm_metadata.feature_names.clone(), // not row-aligned
            },
        }
    }
    

    pub fn split_for_xval(&mut self, fraction: f32, is_test: bool) {
        let mut rng = thread_rng();
        let n_samples = self.x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if !is_test {
            indices.shuffle(&mut rng);
        } else {
            indices.sort_unstable();
        }

        let n_train = (n_samples as f32 * fraction) as usize;
        for (i, &idx) in indices.iter().enumerate() {
            self.is_train[idx] = i < n_train;
        }
    }

    pub fn get_train_psms(&self) -> Experiment {
        let mask = &self.is_train;  
        self.filter(mask)
    }

    pub fn get_test_psms(&self) -> Experiment {
        let mask = &self.is_train.mapv(|x| !x);  
        self.filter(mask)
    }
    
    pub fn remove_psms(&mut self, indices_to_remove: &[usize]) {
        let keep = (0..self.x.nrows())
            .filter(|&i| !indices_to_remove.contains(&i))
            .collect::<Vec<_>>();

        self.x = self.x.select(Axis(0), &keep);
        self.y = self.y.select(Axis(0), &keep);
        self.is_train = self.is_train.select(Axis(0), &keep);
        self.is_top_peak = self.is_top_peak.select(Axis(0), &keep);
        self.tg_num_id = self.tg_num_id.select(Axis(0), &keep);
        self.classifier_score = self.classifier_score.select(Axis(0), &keep);
    }
}

