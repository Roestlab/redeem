# redeem-classifiers

A Rust library for semi-supervised rescoring of peptide-spectrum matches (PSMs) using machine learning classifiers, inspired by tools like [Percolator](http://percolator.ms/) and [mokapot](https://github.com/wfondrie/mokapot).

`redeem-classifiers` implements an iterative semi-supervised learning loop that trains a classifier to distinguish target from decoy PSMs, refining labels at each iteration using target-decoy competition (TDC) q-value estimation.

## Features

- **Semi-supervised PSM rescoring** via `SemiSupervisedLearner` with configurable FDR thresholds, cross-validation folds, and iteration limits
- **Multiple classifier backends**:
  - **GBDT** (Gradient Boosting Decision Trees) — always available, no native dependencies
  - **XGBoost** — optional, requires `libxgboost` (`--features xgboost`)
  - **SVM** (Support Vector Machine) — optional (`--features svm`)
- **Percolator PIN file reader** for loading standard `.pin` TSV files
- **Univariate feature selection** (ANOVA F-test, Pearson's r) following scikit-learn conventions
- **Preprocessing** — standard scaling and score normalization
- **Custom lightweight math primitives** (`Array1`, `Array2`) with no external ndarray dependency
- **HTML reporting** with interactive Plotly charts (score histograms, P-P plots)
- **Target-decoy competition** q-value estimation

## Installation

Add `redeem-classifiers` to your `Cargo.toml`:

```toml
[dependencies]
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "master" }
```

### Feature flags

| Feature | Description | Extra requirements |
|---------|-------------|-------------------|
| *(default)* | GBDT classifier only | None |
| `xgboost` | Enable XGBoost classifier | `clang`, `libstdc++-dev` |
| `svm` | Enable SVM classifier | None |

```toml
# Enable all classifiers
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "master", features = ["xgboost", "svm"] }
```

> **Note:** The XGBoost feature wraps the native XGBoost C++ library, which requires a C++ toolchain:
> ```bash
> sudo apt update
> sudo apt install build-essential clang libstdc++-12-dev
> ```

## Quick Start

### Semi-supervised rescoring with GBDT

```rust
use redeem_classifiers::config::ModelType;
use redeem_classifiers::data_handling::{Experiment, PsmMetadata, RankGrouping};
use redeem_classifiers::math::{Array1, Array2};
use redeem_classifiers::psm_scorer::SemiSupervisedLearner;

// Configure the model
let model_type = ModelType::GBDT {
    max_depth: 6,
    num_boost_round: 3,
    debug: false,
    training_optimization_level: 2,
    loss_type: "LogLikelyhood".to_string(),
};

// Create the semi-supervised learner
let mut learner = SemiSupervisedLearner::new(
    model_type,
    0.1,                    // learning_rate
    0.01,                   // train_fdr
    3,                      // cross-validation folds
    10,                     // max iterations
    None,                   // class_pct (target/decoy sampling)
    true,                   // scale_features
    true,                   // normalize_scores
    RankGrouping::SpecId,   // rank grouping strategy
);

// Build an Experiment from your data
let experiment = Experiment::new(
    x,              // Array2<f32> feature matrix
    y,              // Array1<i32> labels (1 = target, -1 = decoy)
    psm_metadata,   // PsmMetadata (spec_id, file_id, feature_names, ...)
    RankGrouping::SpecId,
);

// Run semi-supervised learning
let results = learner.fit_predict(&mut experiment)?;
```

### Loading Percolator PIN files

```rust
use redeem_classifiers::io::percolator_pin::{PinReaderConfig, read_pin};

let config = PinReaderConfig::default();
let pin_data = read_pin("path/to/results.pin", &config)?;
// pin_data.x    -> Array2<f32> feature matrix
// pin_data.y    -> Array1<i32> labels
// pin_data.metadata -> PsmMetadata
```

### Feature selection

```rust
use redeem_classifiers::feature_selection::univariate_selection::{f_classif, SelectKBest};

// ANOVA F-test for classification
let (f_scores, p_values) = f_classif(&x_f64, &y);

// Select top-k features
let selector = SelectKBest::new(k);
let x_selected = selector.fit_transform(&x_f64, &y);
```

## Crate Structure

```
redeem-classifiers/
├── src/
│   ├── lib.rs                    # Public module exports
│   ├── config.rs                 # ModelConfig, ModelType enum
│   ├── data_handling.rs          # Experiment, PsmMetadata, label updates
│   ├── error.rs                  # Error types
│   ├── psm_scorer.rs             # SemiSupervisedLearner (core training loop)
│   ├── preprocessing.rs          # Scaler, score normalization
│   ├── stats.rs                  # TDC q-value estimation
│   ├── models/                   # Classifier implementations
│   │   ├── classifier_trait.rs   #   ClassifierModel trait
│   │   ├── factory.rs            #   Runtime model construction
│   │   ├── gbdt.rs               #   GBDT (always available)
│   │   ├── xgboost.rs            #   XGBoost (feature-gated)
│   │   └── svm.rs                #   SVM (feature-gated)
│   ├── feature_selection/        # Feature selection methods
│   │   └── univariate_selection.rs  # ANOVA F-test, Pearson's r, SelectKBest
│   ├── io/                       # File format readers
│   │   └── percolator_pin.rs     #   Percolator .pin TSV reader
│   ├── math/                     # Lightweight array primitives
│   │   ├── matrix.rs             #   Array2<T>
│   │   └── vector.rs             #   Array1<T>
│   └── report/                   # HTML report generation
│       ├── report.rs             #   Report builder re-exports
│       └── plots.rs              #   Plotly-based charts
├── examples/                     # Runnable examples
│   ├── gbdt_semi_supervised_learning.rs
│   ├── xgb_semi_supervised_learning.rs
│   ├── svm_semi_supervised_learning.rs
│   └── xgb_synthetic.rs
└── tests/                        # Integration tests
```

## Supported Classifiers

| Classifier | Feature flag | Backend crate | Description |
|-----------|-------------|--------------|-------------|
| GBDT | *(default)* | [`gbdt`](https://github.com/singjc/gbdt-rs) | Gradient boosting decision trees |
| XGBoost | `xgboost` | [`xgb`](https://crates.io/crates/xgb) | XGBoost via C++ bindings |
| SVM | `svm` | [`light-svm`](https://crates.io/crates/light-svm) | Support vector machine |

All classifiers implement the `ClassifierModel` trait:

```rust
pub trait ClassifierModel: Send {
    fn fit(&mut self, x: &Array2<f32>, y: &[i32],
           x_eval: Option<&Array2<f32>>, y_eval: Option<&[i32]>);
    fn predict(&self, x: &Array2<f32>) -> Vec<f32>;
    fn predict_proba(&mut self, x: &Array2<f32>) -> Vec<f32>;
    fn clone_box(&self) -> Box<dyn ClassifierModel>;
    fn name(&self) -> &str;
    fn feature_weights(&self) -> Option<Vec<f32>>;
}
```

## License

See the repository [LICENSE](../../LICENSE) file.
