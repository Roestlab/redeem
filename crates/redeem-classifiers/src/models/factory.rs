use crate::models::utils::ModelConfig;
use crate::models::classifier_trait::ClassifierModel;

/// Build a boxed classifier model from a `ModelConfig`.
/// Currently this is a thin factory implemented as a single function.
pub fn build_model(params: ModelConfig) -> Box<dyn ClassifierModel> {
    match params.model_type {
        crate::config::ModelType::XGBoost { .. } => {
            #[cfg(feature = "xgboost")]
            {
                return Box::new(crate::models::xgboost::XGBoostClassifier::new(params));
            }

            #[cfg(not(feature = "xgboost"))]
            {
                panic!("xgboost feature is not enabled");
            }
        }
    crate::config::ModelType::GBDT { .. } => Box::new(crate::models::gbdt::GBDTClassifier::new(params)),
    #[cfg(feature = "svm")]
    crate::config::ModelType::SVM { .. } => Box::new(crate::models::svm::SVMClassifier::new(params)),
    _ => panic!("unsupported or disabled model feature requested in ModelConfig"),
    }
}
