use std::f64;

use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::data_handling::{Experiment, PsmMetadata};
use crate::math::{Array1, Array2};
use crate::preprocessing;

use crate::models::classifier_trait::ClassifierModel;
use crate::models::factory;
#[cfg(feature = "svm")]
use crate::models::svm::SVMClassifier;
use crate::config::{ModelConfig, ModelType};

// Legacy `SemiSupervisedModel` behavior is provided by implementations of
// `models::classifier_trait::ClassifierModel`. The learner stores a boxed
// `ClassifierModel` trait object.

pub struct SemiSupervisedLearner {
    model: Box<dyn ClassifierModel>,
    train_fdr: f32,
    xeval_num_iter: usize,
    class_pct: Option<(f64, f64)>,
    /// If true, fit a standard scaler on the input features and use the
    /// scaled features for training and evaluation.
    scale_features: bool,
    /// If true, normalize final prediction scores to zero-mean/unit-variance
    /// before producing the final output.
    normalize_scores: bool,
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
        scale_features: bool,
        normalize_scores: bool,
    ) -> Self {
        // Centralize construction to the models factory which returns a
        // `Box<dyn ClassifierModel>`.
        let cfg = ModelConfig {
            learning_rate,
            model_type,
        };
        let model: Box<dyn ClassifierModel> = factory::build_model(cfg);

        SemiSupervisedLearner {
            model,
            train_fdr,
            xeval_num_iter,
            class_pct,
            scale_features,
            normalize_scores,
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
        // Helper function to count targets by feature
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
            log::info!(
                "Using {} % of targets and {} % of decoys for training",
                target_pct.unwrap_or(1.0) * 100.0,
                decoy_pct.unwrap_or(1.0) * 100.0
            );
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
                let train_indices: Vec<_> = train_targets.into_iter().chain(train_decoys).collect();
                let test_indices: Vec<_> = test_targets.into_iter().chain(test_decoys).collect();

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
    pub fn fit(
        &mut self,
        x: Array2<f32>,
        y: Array1<i32>,
        psm_metadata: PsmMetadata,
    ) -> anyhow::Result<(Array1<f32>, Array1<u32>)> {
        // Optionally scale features before building the Experiment. We fit the
        // scaler on the supplied feature matrix and transform the full matrix
        // so training/evaluation use standardized features.
        let x = if self.scale_features {
            log::info!("Fitting scaler and transforming input features");
            preprocessing::fit_transform(&x)
        } else {
            x
        };

        let mut experiment = Experiment::new(x.clone(), y.clone(), psm_metadata.clone());

        experiment.log_input_data_summary();

        // Get initial best feature
        let (_best_feat, _best_positives, mut new_labels, best_desc, _best_feature_scores) =
            self.init_best_feature(&experiment, self.train_fdr);

        experiment.y = new_labels.clone();

        let folds = self.create_folds(
            &experiment,
            self.xeval_num_iter,
            self.class_pct.map(|(t, _d)| t),
            self.class_pct.map(|(_t, d)| d),
        );

        for (fold, (mut train_exp, test_exp)) in folds.into_iter().enumerate() {
            let n_samples = experiment.x.nrows();
            log::info!(
                "Learning on Cross-Validation Fold: {} with {} training samples",
                fold,
                train_exp.x.nrows()
            );

            let mut all_predictions = Array1::zeros(n_samples);

            self.remove_unlabeled_psms(&mut train_exp);

            train_exp.split_for_xval(0.80, false);

            let train_indices: Vec<usize> = train_exp
                .is_train
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if val { Some(i) } else { None })
                .collect();

            let test_indices: Vec<usize> = train_exp
                .is_train
                .iter()
                .enumerate()
                .filter_map(|(i, &val)| if !val { Some(i) } else { None })
                .collect();

            let train_x_subset = train_exp.x.select_rows(&train_indices);
            let train_y_subset = train_exp.y.select(&train_indices);
            let eval_x_subset = train_exp.x.select_rows(&test_indices);
            let eval_y_subset = train_exp.y.select(&test_indices);

            self.model.fit(
                &train_x_subset,
                train_y_subset.as_slice(),
                Some(&eval_x_subset),
                Some(eval_y_subset.as_slice()),
            );

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
        let mut experiment = Experiment::new(x, y, psm_metadata);

        // self.model
        //     .fit(&experiment.x, &experiment.y.to_vec(), None, None);
        let mut final_predictions = Array1::from(self.model.predict_proba(&experiment.x));

        // Log basic statistics of final predictions before normalization so
        // we can see whether they are already constant (which would lead to
        // zeroed normalized scores).
        {
            let s = final_predictions.as_slice();
            if !s.is_empty() {
                let len = s.len() as f32;
                let mean = s.iter().copied().sum::<f32>() / len;
                let mut var = 0f32;
                for &v in s.iter() {
                    let d = v - mean;
                    var += d * d;
                }
                let std = (var / len).sqrt();
                log::debug!(
                    "Final predictions before normalization: len={}, mean={:.6}, std={:.6}, min={}, max={}",
                    s.len(),
                    mean,
                    std,
                    s.iter().cloned().fold(f32::INFINITY, f32::min),
                    s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            } else {
                log::debug!("Final predictions before normalization: empty");
            }
        }

        if self.normalize_scores {
            log::info!("Normalizing final prediction scores (zero-mean, unit-variance)");
            preprocessing::normalize_scores(final_predictions.as_mut_slice());

            // Log stats after normalization
            let s = final_predictions.as_slice();
            if !s.is_empty() {
                let len = s.len() as f32;
                let mean = s.iter().copied().sum::<f32>() / len;
                let mut var = 0f32;
                for &v in s.iter() {
                    let d = v - mean;
                    var += d * d;
                }
                let std = (var / len).sqrt();
                log::debug!(
                    "Final predictions after normalization: len={}, mean={:.6}, std={:.6}, min={}, max={}",
                    s.len(),
                    mean,
                    std,
                    s.iter().cloned().fold(f32::INFINITY, f32::min),
                    s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            }
        }

        experiment.update_rank_feature(&final_predictions, &experiment.psm_metadata.clone());
        let updated_ranks = experiment.get_rank_column()?;

        Ok((final_predictions, updated_ranks))
    }
}

#[cfg(test)]
mod tests {
    use csv::ReaderBuilder;
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;

    use crate::math::{Array1, Array2};

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    #[ignore]
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
        let mut learner =
            SemiSupervisedLearner::new(xgb_params, 0.001, 1.0, 2, Some((0.2, 0.5)), false, false);
        let metadata = crate::data_handling::PsmMetadata {
            file_id: vec![0usize; x.nrows()],
            spec_id: vec!["spec".to_string(); x.nrows()],
            feature_names: (0..x.ncols()).map(|i| format!("f{}", i)).collect(),
        };

        let (predictions, _ranks) = learner.fit(x, y.clone(), metadata).unwrap();

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }

    #[test]
    #[cfg(feature = "svm")]
    #[ignore]
    fn test_svm_semi_supervised_learner() {
        // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let params = ModelType::SVM {
            eps: 0.1,
            c: (1.0, 1.0),
            kernel: "linear".to_string(),
            gaussian_kernel_eps: 0.1,
            polynomial_kernel_constant: 1.0,
            polynomial_kernel_degree: 3.0,
        };
        let mut learner =
            SemiSupervisedLearner::new(params, 0.001, 1.0, 1000, Some((0.2, 0.5)), false, false);
        let metadata = crate::data_handling::PsmMetadata {
            file_id: vec![0usize; x.nrows()],
            spec_id: vec!["spec".to_string(); x.nrows()],
            feature_names: (0..x.ncols()).map(|i| format!("f{}", i)).collect(),
        };

        let (predictions, _ranks) = learner.fit(x, y.clone(), metadata).unwrap();

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        // save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }
}
