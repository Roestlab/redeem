use ndarray::{Array1, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::data_handling::Experiment;
use crate::models::xgboost::XGBoostClassifier;


pub enum ModelType {
    XGBoost,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "xgboost" => Ok(ModelType::XGBoost),
            // Add other model types as needed
            _ => Err(format!("Unknown model type: {}", s)),
        }
    }
}


pub trait SemiSupervisedModel {
    fn fit(&mut self, x: &Array2<f32>, y: &[i32], x_eval: Option<&Array2<f32>>, y_eval: Option<&[i32]>);
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;
    fn predict_proba(&self, x: &Array2<f32>) -> Vec<f32>;
}


pub struct SemiSupervisedLearner {
    model: Box<dyn SemiSupervisedModel>,
    threshold: f32,
    ss_num_iter: usize,
    xeval_num_iter: usize,
}

impl SemiSupervisedLearner {
    pub fn new(model_type: ModelType, threshold: f32, ss_num_iter: usize, xeval_num_iter: usize) -> Self {
        let model: Box<dyn SemiSupervisedModel> = match model_type {
            ModelType::XGBoost => Box::new(XGBoostClassifier::new()),
        };

        SemiSupervisedLearner {
            model,
            threshold,
            ss_num_iter,
            xeval_num_iter,
        }
    }


    pub fn fit(&mut self, x: Array2<f32>, y: Array1<i32>) -> Array1<f32> {
        let mut experiment = Experiment::new(x.clone(), y.clone());
        
        for fold in 0..self.xeval_num_iter {
            println!("Fold: {}", fold);
            let n_samples = experiment.x.nrows();
            let mut all_predictions = Array1::zeros(n_samples);

            // Split data for cross-validation
            // Using 50% for training, 50% for testing. TODO: make this configurable?
            experiment.split_for_xval(0.5, false);  

            let train_exp = experiment.get_train_peaks();
            let test_exp = experiment.filter(&experiment.is_train.mapv(|x| !x));

            self.model.fit(&train_exp.x, &train_exp.y.to_vec(), Some(&test_exp.x), Some(&test_exp.y.to_vec()));
            let fold_predictions = Array1::from(self.model.predict_proba(&test_exp.x));

            // Update predictions
            for (i, pred) in fold_predictions.iter().enumerate() {
                all_predictions[test_exp.tg_num_id[i] as usize] = *pred;
            }

            let new_labels = all_predictions.mapv(|p| if p > self.threshold { 1 } else { -1 });

            if new_labels == experiment.y {
                break;
            }

            experiment.y = new_labels;
        }

        // Final prediction on the entire dataset
        println!("Final prediction on the entire dataset");
        let experiment = Experiment::new(x, y);
        self.model.fit(&experiment.x, &experiment.y.to_vec(), None, None);
        Array1::from(self.model.predict_proba(&experiment.x))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;
    use csv::ReaderBuilder;
    use ndarray::{Array2, Array1};


    fn read_features_tsv(path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_path(path)?;
    
        let mut data = Vec::new();
    
        for result in reader.records() {
            let record = result?;
            let row: Vec<f32> = record.iter()
                .map(|field| field.parse::<f32>())
                .collect::<Result<_, _>>()?;
            data.push(row);
        }
    
        let n_samples = data.len();
        let n_features = data[0].len();
    
        Array2::from_shape_vec((n_samples, n_features), data.into_iter().flatten().collect())
            .map_err(|e| e.into())
    }
    
    fn read_labels_tsv(path: &str) -> Result<Array1<i32>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b'\t')
            .from_path(path)?;
    
        let labels: Vec<i32> = reader.records()
            .map(|r| {
                let record = r?;
                let value = record.get(0).ok_or_else(|| "Empty row".to_string())?;
                value.parse::<i32>().map_err(|e| e.into())
            })
            .collect::<Result<_, Box<dyn Error>>>()?;
    
        Ok(Array1::from_vec(labels))
    }

    fn save_predictions_to_csv(predictions: &Array1<f32>, file_path: &str) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(file_path)?;
        
        for &pred in predictions.iter() {
            writeln!(file, "{}", pred)?;
        }
    
        Ok(())
    }

    #[test]
    fn test_semi_supervised_learner() {
       // Load the test data from the TSV files
        let x = read_features_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_scores_for_testing.csv").unwrap();
        let y = read_labels_tsv("/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/sage_labels_for_testing.csv").unwrap();
        // Convert y to [0, 1]
        let y = y.mapv(|x| if x == 1 { 0 } else { 1 });

        println!("Loaded features shape: {:?}", x.shape());
        println!("Loaded labels shape: {:?}", y.shape());

        // Create and train your SemiSupervisedLearner
        let mut learner = SemiSupervisedLearner::new(ModelType::XGBoost, 0.5, 1, 1);
        let predictions = learner.fit(x, y.clone());

        println!("Labels: {:?}", y);

        // Evaluate the predictions
        println!("Predictions: {:?}", predictions);
        save_predictions_to_csv(&predictions, "/home/singjc/Documents/github/sage_bruker/20241115_single_file_redeem/predictions.csv").unwrap();
    }
}