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

### Current Crates

The ReDeeM project consists of two primary crates:

1. **redeem-properties**: 
   - This crate focuses on deep learning models for peptide property prediction. It implements models for predicting retention time (RT), ion mobility (IM), and MS2 fragment intensities using the Candle library.
   - The models are implemented using the Candle library, which is a Rust library for building deep learning models.
   - The models can be fine-tuned on new data and can be saved in the safetensor format for later use.
   
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    AlphaPept RT Model | `redeem_properties::RTCNNLSTMModel` | CNN-LSTM | :heavy_check_mark:
    AlphaPept MS2 Model | `redeem_properties::MS2BertModel` | Bert | :heavy_check_mark:
    AlphaPept IM Model | `redeem_properties::CCSCNNLSTMModel` | CNN-LSTM | :heavy_check_mark:

2. **redeem-classifiers**:
   - This crate is aimed at developing semi-supervised scoring classifier models. The goal is to create models for separating target peptides from decoys.
  
   - Current Models
  
    Model | Name | Architecture | Implemented
    --- | --- | --- | ---
    XGBoost Classifier | `redeem_classifiers::XGBoostClassifier` | XGBoost | :heavy_check_mark:
    GBDT Classifier | `redeem_classifiers::GBDTClassifier` | GBDT | :heavy_check_mark:
    SVM Classifier | `redeem_classifiers::SVMClassifier` | SVM | :heavy_check_mark: