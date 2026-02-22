# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

ReDeeM is a Rust crate designed for implementing deep learning models specifically tailored for mass spectrometry data. The primary goal of this project is to facilitate the prediction of peptide properties and to develop classifier scoring models (TDA).

<em>[TODO.md spec & Kanban Board](https://bit.ly/3fCwKfM)</em>

### Todo

- [ ] Save target normalization min/max values in safetensor.  
- [ ] implement early stopping during fine-tuning  
- [ ] freeze certain layers for fine-tuning?  
- [ ] xgboost/semi-supervised learning param cleanup  
- [ ] Clean up code / remove comments and unneeded debug macros  

### In Progress


### Done ✓

- [x] implement learning rate scheduler  
- [x] Implement XGBoost classifier PSM scoring  
- [x] Implement redeem property prediction into Sage  
- [x] Refactor peptdeep RT CNN-LSTM model to use refactored codebase  
- [x] add MS2 class model interface  
- [x] add CCS class model interface  
- [x] Implement peptdeep CCS CNN-LSTM model  
- [x] Implement peptdeep MS2 Bert model  
- [x] Implement fine tuning for peptdeep RT CNN-LSTM model  
- [x] implement peptideep RT CNN-LSTM model  
- [x] implement fine tuning for peptdeep MS2 Bert model  
- [x] Implement fine tuning for peptdeep CCS CNN-LSTM model  
- [x] Batch prediction (peptide properties)  
- [x] Open PR for pub bert encoder struct  

