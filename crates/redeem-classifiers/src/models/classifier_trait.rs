use crate::math::Array2;

/// A small trait abstraction for classifier models used by the semi-supervised
/// learner. This mirrors the existing internal `SemiSupervisedModel` API but
/// centralizes the contract in the `models` module so implementations can live
/// next to model code.
pub trait ClassifierModel {
    /// Fit the model. `y` uses the crate convention (1 for target, -1 for decoy)
    fn fit(&mut self, x: &Array2<f32>, y: &[i32], x_eval: Option<&Array2<f32>>, y_eval: Option<&[i32]>);

    /// Predict raw scores (may be margins or probabilistic depending on impl)
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;

    /// Predict probabilities (0..1) when available. Implementations that only
    /// produce margins should convert appropriately.
    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32>;

    /// Optional human readable name for the model
    fn name(&self) -> &str { "classifier" }
}
