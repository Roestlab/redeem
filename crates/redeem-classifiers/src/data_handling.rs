use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};

use crate::stats::tdc;

#[derive(Debug, Clone)]
pub struct Experiment {
    pub x: Array2<f32>,
    pub y: Array1<i32>,
    pub is_train: Array1<bool>,
    pub is_top_peak: Array1<bool>,
    pub tg_num_id: Array1<i32>,
    pub classifier_score: Array1<f32>,
}

impl Experiment {
    pub fn new(x: Array2<f32>, y: Array1<i32>) -> Self {
        let n_samples = x.nrows();
        Experiment {
            x,
            y,
            is_train: Array1::from_elem(n_samples, false),
            is_top_peak: Array1::from_elem(n_samples, false),
            tg_num_id: Array1::from_elem(n_samples, 0),
            classifier_score: Array1::from_elem(n_samples, 0.0),
        }
    }

    pub fn log_input_data_summary(&self) {
        println!("----- Input Data Summary -----");
        println!("Info: {} Target PSMs and {} Decoy PSMs", self.y.iter().filter(|&&v| v == 1).count(), self.y.iter().filter(|&&v| v == -1).count());
        println!("Info: {} scores (columns)", self.x.ncols());
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
        let qvals = tdc(scores, targets, desc);
        
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

    pub fn filter(&self, mask: &Array1<bool>) -> Experiment {
        Experiment {
            x: self.x.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            y: self.y.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            is_train: self.is_train.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            is_top_peak: self.is_top_peak.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            tg_num_id: self.tg_num_id.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
            classifier_score: self.classifier_score.select(Axis(0), &mask.iter().enumerate().filter_map(|(i, &m)| if m { Some(i) } else { None }).collect::<Vec<_>>()),
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

