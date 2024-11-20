use std::collections::HashMap;

use ndarray::Array2;
use sage_core::ml::matrix::Matrix;
use rayon::prelude::*;
use xgboost::{ parameters::{tree::{TreeBoosterParametersBuilder, TreeMethod}, BoosterParametersBuilder, BoosterType, TrainingParametersBuilder}, Booster, DMatrix};

use sage_core::scoring::Feature;

use crate::psm_scorer::SemiSupervisedModel;


pub struct XGBoostClassifier {
    booster: Option<Booster>,
    params: HashMap<String, String>,
}

impl XGBoostClassifier {
    pub fn new() -> Self {
        let mut params = HashMap::new();
        params.insert("objective".to_string(), "binary:logistic".to_string());
        params.insert("eval_metric".to_string(), "logloss".to_string());
        
        XGBoostClassifier {
            booster: None,
            params,
        }
    }
}

impl SemiSupervisedModel for XGBoostClassifier {
    fn fit(&mut self, x: &Array2<f32>, y: &[i32]) {
        // Convert feature matrix into DMatrix
        let mut dmat = DMatrix::from_dense(x.as_slice().unwrap(), x.nrows()).unwrap();

        // Set ground truth labels for the training matrix
        dmat.set_labels(&y.iter().map(|&l| l as f32).collect::<Vec<f32>>()).unwrap();


        // configure the tree-based learning model's parameters
        let tree_params = TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.3)
        .build().unwrap();

        // overall configuration for Booster
        let booster_params = BoosterParametersBuilder::default()
        .booster_type(BoosterType::Tree(tree_params))
        .verbose(true)
        .build().unwrap();


        // Create Training Parameters with evaluation sets if needed
        let training_params = TrainingParametersBuilder::default()
            .dtrain(&dmat)
            .boost_rounds(100)
            .booster_params(booster_params)
            .build()
            .unwrap();

        // Train the model and store the booster
        self.booster = Some(Booster::train(&training_params).unwrap());
    }

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        let dmat = DMatrix::from_dense(x.as_slice().unwrap(), x.nrows()).unwrap();
        self.booster.as_ref().unwrap().predict(&dmat).unwrap()
    }

    fn predict_proba(&self, x: &Array2<f32>) -> Vec<f32> {
        self.predict(x)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array1};

    #[test]
    fn test_xgboost_classifier() {
        // Create a feature matrix with 5 features and 10 samples
        let x = Array2::from_shape_vec((10, 5), vec![
            0.1,  1.0, 5.0,  0.2, -0.3,
            0.4, -1.0, 5.0,  0.8,  0.1,
            0.6,  1.0, 5.0,  1.2,  0.2,
            0.9, -1.0, 5.0,  1.8, -0.1,
            1.2,  1.0, 5.0,  2.4,  0.3,
            1.5, -1.0, 5.0,  3.0,  0.0,
            1.8,  1.0, 5.0,  3.6, -0.2,
            2.1, -1.0, 5.0,  4.2,  0.4,
            2.4,  1.0, 5.0,  4.8, -0.1,
            2.7, -1.0, 5.0,  5.4,  0.2,
        ]).unwrap();

        // Create a target vector perfectly correlated with the second feature
        let y = Array1::from_vec(vec![1i32, -1i32,
                                       1i32, -1i32,
                                       1i32, -1i32,
                                       1i32, -1i32,
                                       1i32, -1i32]);

        // Initialize the XGBoost classifier
        let mut classifier = XGBoostClassifier::new();

        // Fit the classifier
        classifier.fit(&x, &y.to_vec());

        // Make predictions
        let predictions = classifier.predict(&x);

        println!("Predictions: {:?}", predictions);
        println!("y: {:?}", y.to_vec());

        // Check that predictions are reasonable
        assert_eq!(predictions.len(), y.len());
        
        // You can add more specific assertions depending on what you expect from your model
        for (_i, &pred) in predictions.iter().enumerate() {
            assert!(pred >= -1f32 && pred <= 1f32); // Example assertion for binary-like output
        }
    }
}
