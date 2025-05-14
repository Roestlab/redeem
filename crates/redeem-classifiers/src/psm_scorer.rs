use std::f64;

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

use crate::data_handling::{Experiment, PsmMetadata};

use crate::models::utils::{ModelParams, ModelType};
#[cfg(feature = "xgboost")]
use crate::models::xgboost::XGBoostClassifier;
#[cfg(feature = "linfa")]
use crate::models::svm::SVMClassifier;
use crate::models::gbdt::GBDTClassifier;


pub trait SemiSupervisedModel {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    );
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;
    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32>;
}

pub struct SemiSupervisedLearner {
    model: Box<dyn SemiSupervisedModel>,
    train_fdr: f32,
    xeval_num_iter: usize,
    class_pct: Option<(f64, f64)>,
}

impl SemiSupervisedLearner {
    /// Create a new SemiSupervisedLearner
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of model to use
    /// * `train_fdr` - The FDR threshold to use for training
    /// * `xeval_num_iter` - The number of iterations to use for cross-validation
    /// * `class_pct` - (f64, f64) The percentage of targets and decoys to use for training
    ///
    /// # Returns
    ///
    /// A new SemiSupervisedLearner
    pub fn new(
        model_type: ModelType,
        learning_rate: f32, 
        train_fdr: f32,
        xeval_num_iter: usize,
        class_pct: Option<(f64, f64)>,
    ) -> Self {
        let model: Box<dyn SemiSupervisedModel> = match model_type {
            ModelType::GBDT { max_depth, num_boost_round, debug, training_optimization_level, loss_type } => {
                let params = ModelParams {
                    learning_rate,
                    model_type: ModelType::GBDT {
                        max_depth,
                        num_boost_round,
                        debug,
                        training_optimization_level,
                        loss_type
                    },
                };
                Box::new(GBDTClassifier::new(params))
            }
            #[cfg(feature = "xgboost")]
            ModelType::XGBoost { max_depth, num_boost_round, early_stopping_rounds, verbose_eval } => {
                let params = ModelParams {
                    learning_rate,
                    model_type: ModelType::XGBoost { max_depth, num_boost_round, early_stopping_rounds, verbose_eval },
                };
                Box::new(XGBoostClassifier::new(params))
            }
            #[cfg(feature = "linfa")]
            ModelType::SVM { eps, c, kernel, gaussian_kernel_eps, polynomial_kernel_constant, polynomial_kernel_degree } => {
                let params = ModelParams {
                    learning_rate,
                    model_type: ModelType::SVM {
                        eps,
                        c,
                        kernel,
                        gaussian_kernel_eps,
                        polynomial_kernel_constant,
                        polynomial_kernel_degree,
                    },
                };
                Box::new(SVMClassifier::new(params))
            }
            #[cfg(not(any(feature = "xgboost", feature = "linfa")))]
            _ => panic!("No model selected. Please enable the 'xgboost' or 'linfa' feature."),
        };

        SemiSupervisedLearner {
            model,
            train_fdr,
            xeval_num_iter,
            class_pct,
        }
    }

    /// Initialize the best feature
    ///
    /// Adapted from MS2Rescore
    ///
    /// # Arguments
    ///
    /// * `experiment` - The experiment to use
    /// * `eval_fdr` - The FDR threshold to use for evaluation
    pub fn init_best_feature(
        &mut self,
        experiment: &Experiment,
        eval_fdr: f32,
    ) -> (usize, usize, Array1<i32>, bool, Array1<f32>) {
        /// Helper function to count targets by feature
        let targets_count_by_feature = |desc: bool| -> Vec<usize> {
            (0..experiment.x.ncols())
                .map(|col| {
                    let scores = experiment.x.column(col).to_owned();
                    let labels = experiment.update_labels(&scores, eval_fdr, desc);
                    labels.iter().filter(|&&x| x == 1).count()
                })
                .collect()
        };

        // Find the best feature
        let mut best_feat = 0;
        let mut best_positives = 0;
        let mut new_labels = Array1::zeros(experiment.x.nrows());
        let mut best_desc = false;

        for desc in &[true, false] {
            let num_passing = targets_count_by_feature(*desc);
            let feat_idx = num_passing
                .iter()
                .enumerate()
                .max_by_key(|&(_, count)| count)
                .map(|(idx, _)| idx)
                .unwrap();
            let num_passing = num_passing[feat_idx];

            if num_passing > best_positives {
                best_positives = num_passing;
                best_feat = feat_idx;
                let scores = experiment.x.column(feat_idx).to_owned();
                new_labels = experiment.update_labels(&scores, eval_fdr, *desc);
                best_desc = *desc;
            }
        }

        log::trace!(
            "Best feature: {} with {} positives",
            best_feat,
            best_positives
        );

        if best_positives == 0 {
            panic!("No PSMs found below the 'eval_fdr' {}", eval_fdr);
        }

        let best_feature_scores = experiment.x.column(best_feat).to_owned();

        (
            best_feat,
            best_positives,
            new_labels,
            best_desc,
            best_feature_scores,
        )
    }

    /// Remove unlabeled PSMs
    ///
    /// This function removes PSMs with a label of 0 from the experiment. These PSMs are not used for training.
    ///
    /// # Arguments
    ///
    /// * `experiment` - The experiment to use
    ///
    /// # Returns
    ///
    /// The experiment with the unlabeled PSMs removed
    fn remove_unlabeled_psms(&self, experiment: &mut Experiment) {
        let indices_to_remove: Vec<usize> = experiment
            .y
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == 0 { Some(i) } else { None })
            .collect();
        log::trace!("Removing {} unlabeled PSMs", indices_to_remove.len());
        experiment.remove_psms(&indices_to_remove);
    }

    /// Create folds for cross-validation
    ///
    /// # Arguments
    ///
    /// * `experiment` - The experiment to use
    /// * `n_folds` - The number of folds to create
    /// * `target_pct` - The percentage of targets to use for training
    /// * `decoy_pct` - The percentage of decoys to use for training
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the training and testing experiments for each fold
    fn create_folds(
        &self,
        experiment: &Experiment,
        n_folds: usize,
        target_pct: Option<f64>,
        decoy_pct: Option<f64>,
    ) -> Vec<(Experiment, Experiment)> {
        let mut rng = thread_rng();
        let n_samples = experiment.x.nrows();
    
        // Separate targets and decoys
        let targets: Vec<_> = (0..n_samples).filter(|&i| experiment.y[i] == 1).collect();
        let decoys: Vec<_> = (0..n_samples).filter(|&i| experiment.y[i] == -1).collect();
    
        // If neither target_pct nor decoy_pct is set, use the full data
        let use_full_data = target_pct.is_none() && decoy_pct.is_none();

        if !use_full_data {
            log::info!("Using {} % of targets and {} % of decoys for training", target_pct.unwrap_or(1.0) * 100.0, decoy_pct.unwrap_or(1.0) * 100.0);
        }
    
        // Calculate the number of targets and decoys to include in each fold
        let targets_per_fold = if use_full_data {
            (targets.len() as f64 / n_folds as f64).ceil() as usize
        } else {
            (targets.len() as f64 * target_pct.unwrap_or(1.0) / n_folds as f64).ceil() as usize
        };
        let decoys_per_fold = if use_full_data {
            (decoys.len() as f64 / n_folds as f64).ceil() as usize
        } else {
            (decoys.len() as f64 * decoy_pct.unwrap_or(1.0) / n_folds as f64).ceil() as usize
        };
    
        (0..n_folds)
            .map(|i| {
                // Randomly select targets and decoys for the test set
                let test_targets: Vec<_> = targets.choose_multiple(&mut rng, targets_per_fold).cloned().collect();
                let test_decoys: Vec<_> = decoys.choose_multiple(&mut rng, decoys_per_fold).cloned().collect();
    
                // Remaining targets and decoys after selecting test samples
                let remaining_targets: Vec<_> = targets.iter().filter(|t| !test_targets.contains(t)).cloned().collect();
                let remaining_decoys: Vec<_> = decoys.iter().filter(|d| !test_decoys.contains(d)).cloned().collect();
    
                // Randomly select targets and decoys for the training set
                let train_targets: Vec<_> = if use_full_data {
                    remaining_targets.clone()
                } else {
                    remaining_targets
                        .choose_multiple(
                            &mut rng,
                            (remaining_targets.len() as f64 * target_pct.unwrap_or(1.0)).round() as usize,
                        )
                        .cloned()
                        .collect()
                };
                let train_decoys: Vec<_> = if use_full_data {
                    remaining_decoys.clone()
                } else {
                    remaining_decoys
                        .choose_multiple(
                            &mut rng,
                            (remaining_decoys.len() as f64 * decoy_pct.unwrap_or(1.0)).round() as usize,
                        )
                        .cloned()
                        .collect()
                };
    
                // Combine training and testing indices
                let train_indices: Vec<_> = train_targets.into_iter().chain(train_decoys.into_iter()).collect();
                let test_indices: Vec<_> = test_targets.into_iter().chain(test_decoys.into_iter()).collect();
    
                // Create masks for training and testing sets
                let mut train_mask = Array1::from_elem(n_samples, false);
                for &idx in &train_indices {
                    train_mask[idx] = true;
                }
    
                let mut test_mask = Array1::from_elem(n_samples, false);
                for &idx in &test_indices {
                    test_mask[idx] = true;
                }
    
                // Filter the experiment to create training and testing sets
                let train_exp = experiment.filter(&train_mask);
                let test_exp = experiment.filter(&test_mask);
    
                log::trace!(
                    "Preparing fold {} with {} training samples ({} targets and {} decoys) and {} testing samples ({} targets and {} decoys)",
                    i,
                    train_exp.x.nrows(),
                    train_exp.y.iter().filter(|&&x| x == 1).count(),
                    train_exp.y.iter().filter(|&&x| x == -1).count(),
                    test_exp.x.nrows(),
                    test_exp.y.iter().filter(|&&x| x == 1).count(),
                    test_exp.y.iter().filter(|&&x| x == -1).count()
                );
    
                (train_exp, test_exp)
            })
            .collect()
    }
    
    

    /// Fit the SemiSupervisedLearner
    ///
    /// # Arguments
    ///
    /// * `x` - The features to use, shape (n_samples, n_features)
    /// * `y` - The labels to use, shape (n_samples,)
    ///
    /// # Returns
    ///
    /// The predictions for the input features
    pub fn fit(&mut self, x: Array2<f32>, y: Array1<i32>, psm_metadata: PsmMetadata) -> Array1<f32> {

        let mut experiment = Experiment::new(x.clone(), y.clone(), psm_metadata.clone());

        experiment.log_input_data_summary();

        // Get initial best feature
        let (_best_feat, _best_positives, mut new_labels, best_desc, _best_feature_scores) =
            self.init_best_feature(&experiment, self.train_fdr);

        // println!("Original labels: {:?}", experiment.y);
        experiment.y = new_labels.clone();
        // println!("New labels: {:?}", experiment.y);

        let folds = self.create_folds(&experiment, self.xeval_num_iter, self.class_pct.map(|(t, _d)| t), self.class_pct.map(|(_t, d)| d));

        for (fold, (mut train_exp, test_exp)) in folds.into_iter().enumerate() {
            
            let n_samples = experiment.x.nrows();
            log::info!("Learning on Cross-Validation Fold: {} with {} training samples", fold, train_exp.x.nrows());

            let mut all_predictions = Array1::zeros(n_samples);

            self.remove_unlabeled_psms(&mut train_exp);

            train_exp.split_for_xval(0.80, false);

            let train_indices: Vec<usize> = train_exp.is_train
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect();

            let test_indices: Vec<usize> = train_exp.is_train
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if !val { Some(i) } else { None })  
            .collect();
            

            self.model
                .fit(&train_exp.x.select(ndarray::Axis(0), &train_indices), &train_exp.y.select(ndarray::Axis(0), &train_indices).to_vec(), Some(&train_exp.x.select(ndarray::Axis(0), &test_indices)), Some(&train_exp.y.select(ndarray::Axis(0), &test_indices).to_vec()));
            
            let fold_predictions = Array1::from(self.model.predict_proba(&test_exp.x));

            // Update predictions
            for (i, pred) in fold_predictions.iter().enumerate() {
                all_predictions[test_exp.tg_num_id[i] as usize] = *pred;
            }

            new_labels = experiment.update_labels(&all_predictions, self.train_fdr, best_desc);
            experiment.y = new_labels;

            experiment.update_rank_feature(&all_predictions, &experiment.psm_metadata.clone());

        }

        // Final prediction on the entire dataset
        log::info!("Final prediction on the entire dataset");
        let experiment = Experiment::new(x, y, psm_metadata);

        // self.model
        //     .fit(&experiment.x, &experiment.y.to_vec(), None, None);
        Array1::from(self.model.predict_proba(&experiment.x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csv::ReaderBuilder;
    use ndarray::{Array1, Array2};
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;

    fn read_features_tsv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_path(path)?;

        let mut data = Vec::new();

        for result in reader.records() {
            let record = result?;
            let row: Vec<f32> = record
                .iter()
                .map(|field| field.parse::<f32>())
                .collect::<Result<_, _>>()?;
            data.push(row);
        }

        let n_samples = data.len();
        let n_features = data[0].len();

        Array2::from_shape_vec(
            (n_samples, n_features),
            data.into_iter().flatten().collect(),
        )
        .map_err(|e| e.into())
    }

    fn read_labels_tsv(path: &str) -> Result<Array1<i32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'\t')
            .from_path(path)?;

        let labels: Vec<i32> = reader
            .records()
            .map(|r| {
                let record = r?;
                let value = record.get(0).ok_or_else(|| "Empty row".to_string())?;
                value.parse::<i32>().map_err(|e| e.into())
            })
            .collect::<Result<_, Box<dyn Error>>>()?;

        Ok(Array1::from_vec(labels))
    }

    fn save_predictions_to_csv(
        predictions: &Array1<f32>,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_path)?;

        for &pred in predictions.iter() {
            writeln!(file, "{}", pred)?;
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "xgboost")]
    fn test_xgb_semi_supervised_learner() {
        // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let xgb_params = ModelType::XGBoost {
                max_depth: 8,
                num_boost_round: 100,
                early_stopping_rounds: 10,
                verbose_eval: false,
            };
        let mut learner = SemiSupervisedLearner::new(
            xgb_params,
            0.001,
            1.0,
            2,
            Some((0.2, 0.5))
        );
        let predictions = learner.fit(x, y.clone());

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }

    #[test]
    #[cfg(feature = "linfa")]
    fn test_svm_semi_supervised_learner() {
        // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let params = ModelType::SVM  {
                eps: 0.1,
                c: (1.0, 1.0),
                kernel: "linear".to_string(),
                gaussian_kernel_eps: 0.1,
                polynomial_kernel_constant: 1.0,
                polynomial_kernel_degree: 3.0
            };
        let mut learner = SemiSupervisedLearner::new(
            params,
            0.001,
            1.0,
            1000,
            Some((0.2, 0.5))
        );
        let predictions = learner.fit(x, y.clone());

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }
}
