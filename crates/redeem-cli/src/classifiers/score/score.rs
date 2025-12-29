//! CLI scoring helpers for redeem-classifiers.
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use redeem_classifiers::config::ModelConfig;
use redeem_classifiers::data_handling::PsmMetadata;
use redeem_classifiers::io::read_pin_tsv;
use redeem_classifiers::math::Array1;
use redeem_classifiers::psm_scorer::SemiSupervisedLearner;

/// Parameters for running the semi-supervised scorer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoreConfig {
    pub model: ModelConfig,
    pub train_fdr: f32,
    pub xeval_num_iter: usize,
    pub max_iterations: usize,
    pub class_pct: Option<(f64, f64)>,
    pub scale_features: bool,
    pub normalize_scores: bool,
}

impl Default for ScoreConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            train_fdr: 0.01,
            xeval_num_iter: 5,
            max_iterations: 10,
            class_pct: None,
            scale_features: false,
            normalize_scores: false,
        }
    }
}

/// Scoring outputs from the semi-supervised learner.
#[derive(Debug)]
pub struct ScoreResult {
    pub predictions: Array1<f32>,
    pub ranks: Array1<u32>,
    pub metadata: PsmMetadata,
}

/// Load a scorer configuration from a JSON file.
pub fn load_score_config<P: AsRef<Path>>(path: P) -> Result<ScoreConfig> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config: {}", path.as_ref().display()))?;
    let config: ScoreConfig = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse config: {}", path.as_ref().display()))?;
    Ok(config)
}

/// Run the semi-supervised scorer using a Percolator .pin TSV input.
pub fn score_pin_with_config<P: AsRef<Path>>(pin_path: P, config_path: P) -> Result<ScoreResult> {
    let config = load_score_config(config_path)?;
    score_pin(pin_path, &config)
}

/// Run the semi-supervised scorer using a Percolator .pin TSV input.
pub fn score_pin<P: AsRef<Path>>(pin_path: P, config: &ScoreConfig) -> Result<ScoreResult> {
    let pin_data = read_pin_tsv(pin_path)?;
    let mut learner = SemiSupervisedLearner::new(
        config.model.model_type.clone(),
        config.model.learning_rate,
        config.train_fdr,
        config.xeval_num_iter,
        config.max_iterations,
        config.class_pct,
        config.scale_features,
        config.normalize_scores,
    );

    let (predictions, ranks) = learner.fit(pin_data.x, pin_data.y, pin_data.metadata.clone())?;

    Ok(ScoreResult {
        predictions,
        ranks,
        metadata: pin_data.metadata,
    })
}
