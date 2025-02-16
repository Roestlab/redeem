use linfa::dataset::Pr;
use linfa::traits::Predict;
use linfa::Dataset;
use linfa_svm::SvmParams;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};

use crate::models::utils::{ModelParams, ModelType};
use crate::psm_scorer::SemiSupervisedModel;

pub struct SVMClassifier {
    model: Option<Svm<f64, Pr>>,
    params: ModelParams,
    predictions: Option<linfa::DatasetBase<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>, ndarray::ArrayBase<ndarray::OwnedRepr<Pr>, ndarray::Dim<[usize; 1]>>>>,
}

impl SVMClassifier {
    pub fn new(params: ModelParams) -> Self {
        SVMClassifier {
            model: None,
            params,
            predictions: None,
        }
    }
}

impl SemiSupervisedModel for SVMClassifier {
    fn fit(
        &mut self,
        x: &Array2<f32>,
        y: &[i32],
        x_eval: Option<&Array2<f32>>,
        y_eval: Option<&[i32]>,
    ) {
        // Convert y to [0, 1] for regular binary labels
        // Note: we set targets (original 1) as 1 and decoys (original -1) as 0, so that the scores are positive for targets and negative for decoys
        // TODO: this maybe should be done outside of the model
        // let y = y.iter().map(|&l| if l == 1 { 1 } else { 0 }).collect::<Vec<i32>>();
        // Convert y to [true, false] for binary classification
        let y = y.iter().map(|&l| l == 1).collect::<Vec<bool>>();

        // let y_eval = y_eval.map(|y_e| y_e.iter().map(|&l| if l == 1 { 1 } else { 0 }).collect::<Vec<i32>>());
        let y_eval = y_eval.map(|y_e| y_e.iter().map(|&l| l == 1).collect::<Vec<bool>>());

        // Convert y and y_eval to ArrayBase<OwnedRepr<i32>, Dim<[usize; 1]>>
        let y = Array1::from_vec(y);
        // let y = CountedTargets::new(y);

        let y_eval = y_eval.map(|y_e| Array1::from_vec(y_e));

        // Convert feature matrix from f32 to f64
        let x_f64 = x.mapv(|v| v as f64);

        // Convert feature matrix into Dataset
        let dataset = Dataset::new(x_f64.to_owned(), y);

        // Extract SVM parameters from ModelParams
        if let ModelType::SVM {
            eps,
            c,
            kernel,
            gaussian_kernel_eps,
            polynomial_kernel_constant,
            polynomial_kernel_degree,
        } = &self.params.model_type
        {
            let (c1, c2) = *c;
            let eps = *eps;
            let kernel = kernel.clone();

            let mut model: SvmParams<f64, Pr> =
                Svm::<f64, Pr>::params().eps(eps).pos_neg_weights(c1, c2);

            // Chain the kernel configuration based on the kernel type
            model = match kernel.as_str() {
                "linear" => model.linear_kernel(),
                "gauss" => model.gaussian_kernel(*gaussian_kernel_eps),
                "poly" => model.polynomial_kernel(*polynomial_kernel_constant, *polynomial_kernel_degree),
                _ => {
                    eprintln!("Error: Unsupported kernel type: {}. Valid options are: linear, gauss, poly", kernel);
                    return; // Exit early if the kernel type is unsupported
                }
            };

            // Fit the model
            self.model = Some(<SvmParams<f64, Pr> as linfa::traits::Fit<_, _, _>>::fit(&model, &dataset).unwrap());
        } else {
            eprintln!("Error: Expected ModelType::SVM but got another type.");
        }
    }

    fn predict(&self, x: &Array2<f32>) -> Vec<f32> {
        todo!()
    }

    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32> {
        // Convert feature matrix from f32 to f64
        let x_f64 = x.mapv(|v| v as f64);
        let predictions = self.model.as_ref().unwrap().predict(x_f64);
        self.predictions = Some(predictions.clone());
        // let tmp = predictions.records();
        let tmp: Vec<Pr> = predictions.targets().to_vec();
        // Convert predictions from ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> to Vec<f32>
        tmp.iter().map(|&v| *v).collect::<Vec<f32>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::DatasetBase;
    use linfa::metrics::ToConfusionMatrix;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_svm_classifier() {
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

        // Create SVM parameters
        let params = ModelParams {
            learning_rate: 0.001,
            model_type: ModelType::SVM {
                eps: 0.0000001,
                c: (1.0, 1.0),
                kernel: "linear".to_string(),
                gaussian_kernel_eps: 0.1,
                polynomial_kernel_constant: 1.0,
                polynomial_kernel_degree: 1.0,
            },
        };

        // Initialize the SVM classifier
        let mut classifier = SVMClassifier::new(params);

        // Fit the classifier
        classifier.fit(&x, &y.to_vec(), None, None);

        // Make predictions
        let predictions = classifier.predict_proba(&x);

        println!("Predictions: {:?}", predictions);

        // Convert predictions to Array1<bool> (binary classification)
        // Convert predictions (Vec<f32>) to Array1<f32>
        let preds = Array1::from_vec(predictions).mapv(|x| x > 0.5);

        // Convert ground truth to Array1<bool>
        let y = y.mapv(|x| x != 0);

        // Create a DatasetBase for ground truth
        let ground_truth = DatasetBase::new(x.clone(), y);

        // Compute the confusion matrix using fully qualified syntax
        let cm = <Array1<bool> as ToConfusionMatrix<bool, _>>::confusion_matrix(&preds, &ground_truth).unwrap();

        println!("Confusion Matrix: {:?}", cm);
    }
}
