<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png" alt="ReDeem_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png" alt="ReDeem_Logo" width="500">
    <img alt="ReDeem Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/singjc/redeem/raw/master/img/redeem_logo.png">
  </picture>
</p>

---

# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

ReDeeM is a Rust crate designed for implementing deep learning models specifically tailored for mass spectrometry data. The primary goal of this project is to facilitate the prediction of peptide properties and to develop classifier scoring models (TDA). 

### Usage

The ReDeeM crates are designed to be used as a library in other projects, i.e. in Sage. To use the ReDeeM crates, add the following to your `Cargo.toml` file:

```toml
[dependencies]
redeem-properties = { git = "https://github.com/singjc/redeem.git", branch = "master" }
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "master" }
```

**Note**: The ReDeeM crates are still under development and are not yet available on crates.io.

#### Python

Install the Python bindings using [maturin](https://www.maturin.rs):

```bash
pip install maturin
git clone https://github.com/singjc/redeem.git
cd redeem/crates/redeem-properties-py
maturin develop
```

Quick example:

```python
import redeem_properties_py

# Load from shipped pretrained weights (downloaded automatically on first use)
rt_model = redeem_properties_py.RTModel.from_pretrained("rt")
rt_values = rt_model.predict(["PEPTIDE", "SEQUENCE"], ["", ""], ["", ""])
print(rt_values)  # numpy array of RT predictions

# Or load from a custom model file
rt_model = redeem_properties_py.RTModel(
    model_path="path/to/rt.pth",
    arch="rt_cnn_lstm",
    constants_path="path/to/rt.pth.model_const.yaml",
)
```

### Current Crates

The ReDeeM project consists of three primary crates:

1. **redeem-properties**: 
   - This crate focuses on deep learning models for peptide property prediction. It implements models for predicting retention time (RT), ion mobility (IM), and MS2 fragment intensities using the Candle library.
   - The models can be trained, fine-tuned on new data and can be saved in the safetensor format for later use.
   
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    AlphaPept RT Model | `rt_cnn_lstm` | CNN-LSTM | :heavy_check_mark:
    AlphaPept MS2 Model | `ms2_bert` | Bert | :heavy_check_mark:
    AlphaPept CCS Model | `ccs_cnn_lstm` | CNN-LSTM | :heavy_check_mark:
    RT Model | `rt_tf_lstm` | CNN-Transformer | :heavy_check_mark:
    CCS Model | `ccs_tf_lstm` | CNN-Transformer | :heavy_check_mark:

2. **redeem-classifiers**:
   - This crate is aimed at developing semi-supervised scoring classifier models. The goal is to create models for separating target peptides from decoys.
  
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    XGBoost Classifier | `redeem_classifiers::XGBoostClassifier` | XGBoost | :heavy_check_mark:
    GBDT Classifier | `redeem_classifiers::GBDTClassifier` | GBDT | :heavy_check_mark:
    SVM Classifier | `redeem_classifiers::SVMClassifier` | SVM | :heavy_check_mark:

3. **redeem-properties-py**:
   - Python bindings for `redeem-properties` via [PyO3](https://pyo3.rs), exposing RT, CCS, and MS2 prediction models to Python.
   - Build with [maturin](https://www.maturin.rs) for use in Python workflows.

> [!NOTE]
> To use the XGBoost classifier, or the SVM classifier, you need to compile with the `--features xgboost` or `--features linfa` flag respectively.

> [!IMPORTANT]
> The XGBoost crate is a wrapper around the original XGBoost library, which requires clang/c++ to be installed on the system. On Ubuntu, you can do the following:
    
    ```bash
    sudo apt update
    sudo apt install build-essential
    sudo apt install clang
    sudo apt install libstdc++-12-dev
    ```
