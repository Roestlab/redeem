use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use sage_core::ml::matrix::Matrix;
use xgboost::{
    parameters::{
        learning::{LearningTaskParametersBuilder, Objective},
        tree::{TreeBoosterParametersBuilder, TreeMethod},
        BoosterParametersBuilder, BoosterType, TrainingParametersBuilder,
    },
    Booster, DMatrix,
};

use sage_core::scoring::Feature;

use crate::models::utils::{ModelType, ModelParams};
use crate::psm_scorer::SemiSupervisedModel;

pub struct XGBoostClassifier {
    booster: Option<Booster>,
    params: ModelParams,
}

impl XGBoostClassifier {
    pub fn new(params: ModelParams) -> Self {
        XGBoostClassifier {
            booster: None,
            params,
        }
    }
}

impl SemiSupervisedModel for XGBoostClassifier {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        // Convert y to [0, 1] for XGBoost binary regression
        // Note: we set targets (original 1) as 1 and decoys (original -1) as 0, so that the scores are positive for targets and negative for decoys
        // TODO: this maybe should be done outside of the model
        let y = y.iter().map(|&l| if l == 1 { 1 } else { 0 }).collect::<Vec<i32>>();
        let y_eval = y_eval.map(|y_e| y_e.iter().map(|&l| if l == 1 { 1 } else { 0 }).collect::<Vec<i32>>());
    
        // Convert feature matrix into DMatrix
        let mut dmat = DMatrix::from_dense(x.as_slice().unwrap(), x.nrows()).unwrap();
        dmat.set_labels(&y.iter().map(|&l| l as f32).collect::<Vec<f32>>()).unwrap();
    
        let mut eval_matrix = None;
        let dmat_eval = if let (Some(x_e), Some(y_e)) = (x_eval, y_eval) {
            let mut matrix = DMatrix::from_dense(x_e.as_slice().unwrap(), x_e.nrows()).unwrap();
            matrix.set_labels(&y_e.iter().map(|&l| l as f32).collect::<Vec<f32>>()).unwrap();
            eval_matrix = Some(matrix);
            Some(vec![
                (&dmat, "train"),
                (eval_matrix.as_ref().unwrap(), "test"),
            ])
        } else {
            None
        };
    
        if let ModelType::XGBoost {
            max_depth,
            num_boost_round
        } = &self.params.model_type
        {
            // Configure learning objective
            let learning_params = LearningTaskParametersBuilder::default()
                .objective(Objective::BinaryLogisticRaw)
                .build()
                .unwrap();
    
            // Configure the tree-based learning model's parameters
            let tree_params = TreeBoosterParametersBuilder::default()
                .max_depth(*max_depth)
                .eta(self.params.learning_rate)
                .build()
                .unwrap();
    
            // Overall configuration for Booster
            let booster_params = BoosterParametersBuilder::default()
                .booster_type(BoosterType::Tree(tree_params))
                .learning_params(learning_params)
                .verbose(false)
                .build()
                .unwrap();
    
            // Create Training Parameters with evaluation sets if needed
            let training_params = TrainingParametersBuilder::default()
                .dtrain(&dmat)
                .boost_rounds(*num_boost_round)
                .booster_params(booster_params)
                .evaluation_sets(dmat_eval.as_deref())
                .build()
                .unwrap();
    
            // Train the model and store the booster
            self.booster = Some(Booster::train(&training_params).unwrap());
        } else {
            eprintln!("Error: Expected ModelType::XGBoost but got another type.");
        }
    }
    

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        let dmat = DMatrix::from_dense(x.as_slice().unwrap(), x.nrows()).unwrap();
        self.booster.as_ref().unwrap().predict(&dmat).unwrap()
    }

    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        self.predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_xgboost_classifier() {
        // Create a feature matrix with 5 features and 10 samples
        let x = Array2::from_shape_vec(
            (10, 5),
            vec![
                0.1, 1.0, 5.0, 0.2, -0.3, 0.4, -1.0, 5.0, 0.8, 0.1, 0.6, 1.0, 5.0, 1.2, 0.2, 0.9,
                -1.0, 5.0, 1.8, -0.1, 1.2, 1.0, 5.0, 2.4, 0.3, 1.5, -1.0, 5.0, 3.0, 0.0, 1.8, 1.0,
                5.0, 3.6, -0.2, 2.1, -1.0, 5.0, 4.2, 0.4, 2.4, 1.0, 5.0, 4.8, -0.1, 2.7, -1.0, 5.0,
                5.4, 0.2,
            ],
        )
        .unwrap();

        // Create a target vector perfectly correlated with the second feature
        let y = Array1::from_vec(vec![
            1i32, -1i32, 1i32, -1i32, 1i32, -1i32, 1i32, -1i32, 1i32, -1i32,
        ]);

        // Convert y to [0, 1]
        let y = y.mapv(|x| if x == 1 { 0 } else { 1 });

        println!("y.to_vec(): {:?}", y.to_vec());

        // Initialize the XGBoost classifier
        let mut classifier = XGBoostClassifier::new(ModelParams::default());

        // Fit the classifier
        classifier.fit(&x, &y.to_vec(), None, None);

        // Make predictions
        let predictions = classifier.predict(&x);

        println!("Predictions: {:?}", predictions);
        println!("y: {:?}", y.to_vec());

        // Check that predictions are reasonable
        // assert_eq!(predictions.len(), y.len());

        // // You can add more specific assertions depending on what you expect from your model
        // for (_i, &pred) in predictions.iter().enumerate() {
        //     assert!(pred >= -1f32 && pred <= 1f32); // Example assertion for binary-like output
        // }

        // // Get attribute names
        // let attr_names = classifier.booster.unwrap().get_attribute("weight").unwrap().unwrap();
        // println!("Attribute names: {:?}", attr_names);

        // Predict Contributions
        let mut dmat = DMatrix::from_dense(&x.as_slice().unwrap(), x.nrows()).unwrap();
        dmat.set_labels(&y.iter().map(|&l| l as f32).collect::<Vec<f32>>())
            .unwrap();

        let contributions = classifier
            .booster
            .as_ref()
            .unwrap()
            .predict_contributions(&dmat)
            .unwrap();
        println!("Contributions: {:?}", contributions);

        let interactions = classifier
            .booster
            .as_ref()
            .unwrap()
            .predict_interactions(&dmat)
            .unwrap();
        println!("Interactions: {:?}", interactions);
    }
}
