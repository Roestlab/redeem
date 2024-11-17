//! Univariate feature selection methods following scikit-learn's API.
//! 
//! See: https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use statrs::{distribution::{Continuous, ContinuousCDF, FisherSnedecor}, statistics::Statistics};


/// Compute row-wise (squared) Euclidean norms of a 2D array.
///
/// This function calculates the Euclidean norm for each row in the input matrix.
/// If `squared` is true, it returns the squared norms instead of the regular norms.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing the input data.
/// * `squared` - A boolean indicating whether to return squared norms.
///
/// # Returns
///
/// An array of shape (n_samples,) containing the row-wise (squared) Euclidean norms.
///
/// # Examples
///
/// ```rust
/// let x = Array2::from_shape_vec((3, 3), vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
/// ]).unwrap();
/// let norms = row_norms(&x, false);
/// println!("Row norms: {:?}", norms);
/// ```
pub fn row_norms<S>(x: &ArrayBase<S, Ix2>, squared: bool) -> Array1<f64>
where
    S: Data<Elem = f64>,
{
    let n_samples = x.nrows();
    let mut norms = Array1::zeros(n_samples);

    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let sum_of_squares: f64 = row.iter().map(|&val| val.powi(2)).sum();
        norms[i] = if squared {
            sum_of_squares
        } else {
            sum_of_squares.sqrt()
        };
    }

    norms
}


/// Compute Pearson's r for each feature and the target.
///
/// Pearson's r is also known as the Pearson correlation coefficient.
/// This function tests the individual effect of each regressor
/// on the target variable. It is a scoring function used in feature
/// selection procedures.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing
///   the data matrix (features).
/// * `y` - A 1D array of shape (n_samples,) representing the target vector.
/// * `center` - A boolean indicating whether to center the data.
///     If true, both `X` and `y` will be centered by subtracting their means.
/// * `force_finite` - A boolean indicating whether to force the correlation
///     coefficients to be finite. If true, non-finite values will be replaced
///     with 0.0.
///
/// # Returns
///
/// An array of shape (n_features,) containing the Pearson's r correlation
/// coefficients for each feature.
///
/// # Examples
///
/// ```rust
/// let x = Array2::from_shape_vec((10, 5), vec![/* your data */]).unwrap();
/// let y = Array1::from_vec(vec![/* your target */]);
/// let correlation_coefficients = r_regression(&x, &y, true, true);
/// println!("Correlation coefficients: {:?}", correlation_coefficients);
/// ```
pub fn r_regression(x: &Array2<f64>, y: &Array1<f64>, center: bool, force_finite: bool) -> Array1<f64> {
    let n_samples = x.nrows() as f64;
    let n_features = x.ncols();

    let mut y_centered = y.to_owned();
    let mut x_means = Array1::zeros(n_features);
    let mut x_norms = Array1::zeros(n_features);

    if center {
        let y_mean = y.mean().unwrap();
        y_centered -= y_mean;

        for (i, col) in x.columns().into_iter().enumerate() {
            let col_mean = col.mean();
            x_means[i] = col_mean;
        }

        // Compute the scaled standard deviations via moments
        let x_squared_norms = row_norms(&x.t(), true);
        x_norms = (&x_squared_norms - n_samples * &x_means.mapv(|m| m.powi(2))).mapv(f64::sqrt);
    } else {
        x_norms = row_norms(&x.t(), false);
    }

    let mut correlation_coefficient = Array1::zeros(n_features);
    for (i, col) in x.columns().into_iter().enumerate() {
        let centered_col = if center {
            col.mapv(|v| v - x_means[i])
        } else {
            col.to_owned()
        };
        correlation_coefficient[i] = centered_col.dot(&y_centered);
    }

    let y_norm = y_centered.dot(&y_centered).sqrt();

    correlation_coefficient /= &x_norms;
    correlation_coefficient /= y_norm;

    if force_finite {
        for val in correlation_coefficient.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }
    }

    correlation_coefficient
}


/// Univariate linear regression tests returning F-statistic and p-values.
///
/// This function performs a quick linear model test for assessing
/// the effect of a single regressor on the target variable,
/// sequentially for many regressors.
///
/// # Parameters
///
/// * `x` - A 2D array of shape (n_samples, n_features) representing
///   the data matrix (features).
/// * `y` - A 1D array of shape (n_samples,) representing the target vector.
/// * `center` - A boolean indicating whether to center the data.
/// * `force_finite` - A boolean indicating whether to force F-statistics
///   and associated p-values to be finite.
///
/// # Returns
///
/// A tuple containing:
/// - An array of shape (n_features,) with F-statistics for each feature.
/// - An array of shape (n_features,) with p-values associated with each F-statistic.
///
/// # Examples
///
/// ```rust
/// let x = Array2::from_shape_vec((10, 5), vec![/* your data */]).unwrap();
/// let y = Array1::from_vec(vec![/* your target */]);
/// let (f_statistic, p_values) = f_regression(&x, &y, true, true);
/// println!("F-statistic: {:?}", f_statistic);
/// println!("p-values: {:?}", p_values);
/// ```
pub fn f_regression(x: &Array2<f64>, y: &Array1<f64>, center: bool, force_finite: bool) -> (Array1<f64>, Array1<f64>) {
    let correlation_coefficient = r_regression(x, y, center, force_finite);
    let deg_of_freedom = y.len() as f64 - if center { 2.0 } else { 1.0 };

    // Calculate squared correlation coefficients
    let corr_coef_squared = correlation_coefficient.mapv(|x| x.powi(2));

    // Calculate F-statistic
    let mut f_statistic = &corr_coef_squared / (1.0 - &corr_coef_squared) * deg_of_freedom;
    let mut p_values = Array1::zeros(f_statistic.len());

    // Create an F-distribution object for calculating p-values
    let f_dist = FisherSnedecor::new(1.0, deg_of_freedom).unwrap();
    for (i, &f) in f_statistic.iter().enumerate() {
        p_values[i] = 1.0 - f_dist.cdf(f);
    }

    if force_finite {
        for i in 0..f_statistic.len() {
            if !f_statistic[i].is_finite() {
                if f_statistic[i].is_infinite() {
                    f_statistic[i] = f64::MAX;
                    p_values[i] = 0.0;
                } else if f_statistic[i].is_nan() {
                    f_statistic[i] = 0.0;
                    p_values[i] = 1.0;
                }
            }
        }
    }

    (f_statistic, p_values)
}


/// A struct for selecting the k best features based on F-scores.
///
/// This struct implements a feature selection method similar to scikit-learn's SelectKBest
/// with f_regression as the scoring function.
pub struct SelectKBest {
    /// The number of top features to select.
    k: usize,
}

impl SelectKBest {
    /// Creates a new SelectKBest instance.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of top features to select.
    ///
    /// # Returns
    ///
    /// A new SelectKBest instance.
    pub fn new(k: usize) -> Self {
        SelectKBest { k }
    }

    /// Fits the SelectKBest model and returns the indices of the k best features.
    ///
    /// # Arguments
    ///
    /// * `x` - The feature matrix (n_samples x n_features).
    /// * `y` - The target vector.
    /// * `center` - A boolean indicating whether to center the data. Default is true.
    /// * `force_finite` - A boolean indicating whether to force F-statistics and associated p-values to be finite. Default is true.
    ///
    /// # Returns
    ///
    /// A vector of indices corresponding to the k best features.
    pub fn fit(&self, x: &Array2<f64>, y: &Array1<f64>, center: Option<bool>, force_finite: Option<bool>) -> Vec<usize> {
        let center = center.unwrap_or(true);
        let force_finite = force_finite.unwrap_or(true);
    
        let (f_scores, _) = f_regression(x, y, center, force_finite);

        // Create a vector of indices
        let mut indices: Vec<usize> = (0..f_scores.len()).collect();
        
        // Sort indices based on scores in ascending order using a stable sort
        indices.sort_by(|&i, &j| f_scores[i].partial_cmp(&f_scores[j]).unwrap_or(std::cmp::Ordering::Equal));
    
        // Select top k features by taking the last k elements
        indices.iter().rev().take(self.k).cloned().collect()
    }
    
}



#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_select_k_best() {
        // Create a feature matrix with 5 features and 10 samples
        // Features: [random, collinear with target, constant, collinear with feature 1, noise]
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
        let y = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        // Create a SelectKBest instance to select the top 3 features
        let selector = SelectKBest::new(3);

        // Fit the selector and get the indices of the best features
        let selected_indices = selector.fit(&x, &y, Some(true), Some(true));

        // Print f-scores for debugging
        println!("Selected feature indices: {:?}", selected_indices);
        let (f_scores, p_values) = f_regression(&x, &y, true, true);
        println!("F-scores: {:?}", f_scores);
        println!("P-values: {:?}", p_values);
        // Print selected features for debugging
        println!("Selected features:");
        for i in selected_indices.iter() {
            println!("Feature {:?}: {:?}",i, x.column(*i));
        }
        

        // Check that we got 3 indices
        assert_eq!(selected_indices.len(), 3);

        // Check that the indices are within the valid range (0 to 4)
        assert!(selected_indices.iter().all(|&idx| idx < 5));

        // Check that the indices are unique
        assert!(selected_indices.iter().collect::<std::collections::HashSet<_>>().len() == 3);

        // The second feature (index 1) should definitely be selected as it's perfectly correlated with the target
        assert!(selected_indices.contains(&1));

        // The third feature (index 2) should not be selected as it's constant
        assert!(!selected_indices.contains(&2));

    }
}
