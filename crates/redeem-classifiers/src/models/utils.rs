use serde::{Deserialize, Serialize};



#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelParams {
    pub learning_rate: f32, 

    #[serde(flatten)]
    pub model_type: ModelType, 
}

#[derive(Deserialize, Serialize, Debug, Clone)]
// #[serde(tag = "model")]
pub enum ModelType {
    #[cfg(feature = "xgboost")]
    XGBoost {
        max_depth: u32,
        num_boost_round: u32,
    },
    #[cfg(feature = "linfa")]
    SVM {
        eps: f64,
        c: (f64, f64),
        kernel: String,
        gaussian_kernel_eps: f64,
        polynomial_kernel_constant: f64,
        polynomial_kernel_degree: f64,
    },
    GBDT {
        max_depth: u32,
        num_boost_round: u32,
        debug: bool,
        training_optimization_level: u8,
        loss_type: String,
    },
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::GBDT {
            max_depth: 6,
            num_boost_round: 3,
            debug: false,
            training_optimization_level: 2,
            loss_type:"LogLikelyhood".to_string(),
        }
    }
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "gbdt" => Ok(ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 3,
                debug: false,
                training_optimization_level: 2,
                loss_type:"LogLikelyhood".to_string(),
            }),
            #[cfg(feature = "xgboost")]
            "xgboost" => Ok(ModelType::XGBoost {
                max_depth: 6,
                num_boost_round: 3,
            }),
            #[cfg(feature = "linfa")]
            "svm" => Ok(ModelType::SVM {
                eps: 0.1,
                c: (1.0, 1.0),
                kernel: "linear".to_string(),
                gaussian_kernel_eps: 0.1,
                polynomial_kernel_constant: 1.0,
                polynomial_kernel_degree: 3.0,
            }),
            _ => Err(format!("Unknown model type: {}. To use xgboost or svm, please compile with `--features xgboost` or `--features linfa`", s)),
        }
    }
}

impl ModelParams {
    pub fn new(learning_rate: f32, model_type: ModelType) -> Self {
        Self {
            learning_rate,
            model_type,
        }
    }
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            model_type: ModelType::GBDT {
                max_depth: 6,
                num_boost_round: 3,
                debug: false,
                training_optimization_level: 2,
                loss_type:"LogLikelyhood".to_string(),
            },
        }
    }
}
