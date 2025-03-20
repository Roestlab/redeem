use ndarray::Array2;
use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec};
use gbdt::gradient_boost::GBDT;


use crate::models::utils::{ModelType, ModelParams};
use crate::psm_scorer::SemiSupervisedModel;

/// Gradient Boosting Decision Tree (GBDT) classifier
pub struct GBDTClassifier {
    model: Option<GBDT>,
    params: ModelParams,
}

impl GBDTClassifier {
    pub fn new(params: ModelParams) -> Self {
        GBDTClassifier {
            model: None,
            params,
        }
    }
}

impl SemiSupervisedModel for GBDTClassifier {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        _x_eval: Option<&Array2<f32>>,
        _y_eval: Option<&[i32]>,
    ) {
        let feature_size = x.ncols();
        
        if let ModelType::GBDT {
            max_depth,
            num_boost_round,
            debug,
            training_optimization_level,
            loss_type,
        } = &self.params.model_type {
            let mut config = Config::new();
            
            config.set_feature_size(feature_size);
            config.set_shrinkage(self.params.learning_rate);
            config.set_max_depth(*max_depth);
            config.set_iterations(*num_boost_round as usize);
            config.set_debug(*debug);
            config.set_training_optimization_level(*training_optimization_level);
            config.set_loss(loss_type); 


            let mut gbdt = GBDT::new(&config);

            let mut train_x = DataVec::new();

            for (i, row) in x.outer_iter().enumerate() {
                let mut train_row = Vec::new();
                for (j, &val) in row.iter().enumerate() {
                    train_row.push(val);
                }
                train_x.push(Data::new_training_data(train_row, 1.0, y[i] as f32, None));
            }

            gbdt.fit(&mut train_x);

            self.model = Some(gbdt);
        } else {
            panic!("Error: Expected ModelType::GBDT params, got {:?}", self.params.model_type);
        }
        
    }

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        let mut test_x = DataVec::new();
        for row in x.outer_iter() {
            let mut test_row = Vec::new();
            for &val in row.iter() {
                test_row.push(val);
            }
            test_x.push(Data::new_training_data(test_row, 1.0, 0.0, None));
        }
        let predictions = self.model.as_ref().unwrap().decision_function(&test_x);
        predictions
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
    fn test_gbdt_classifier() {
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
        let params = ModelParams {
            learning_rate: 0.1,
            model_type: ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 3,
                debug: true,
                training_optimization_level: 2,
                loss_type: "LogLikelyhood".to_string(),
            },
        };

        let mut classifier = GBDTClassifier::new(params);

        // Fit the classifier
        classifier.fit(&x, &y.to_vec(), None, None);

        // Make predictions
        let predictions = classifier.predict(&x);

        println!("Predictions: {:?}", predictions);
        println!("y: {:?}", y.to_vec());

        // Check that predictions are reasonable
        // assert_eq!(predictions.len(), y.len());

    }
}
