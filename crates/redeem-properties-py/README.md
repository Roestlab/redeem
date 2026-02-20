# redeem-properties-py

Python bindings for the [redeem-properties](../redeem-properties) Rust crate, exposing peptide property prediction models (RT, CCS, MS2) via [PyO3](https://pyo3.rs).

## Installation

### From source (requires Rust + maturin)

```bash
pip install maturin
cd crates/redeem-properties-py
maturin develop
```

### Build a wheel

```bash
# CPU-only
maturin build --release

# With CUDA support
maturin build --release --features cuda
```

## Usage

### Retention Time (RT) Prediction

```python
import redeem_properties_py

model = redeem_properties_py.RTModel(
    model_path="path/to/rt.pth",
    arch="rt_cnn_lstm",
    constants_path="path/to/rt.pth.model_const.yaml",
)

sequences  = ["AGHCEWQMKYR", "PEPTIDE"]
mods       = ["Acetyl@Protein N-term;Oxidation@M", ""]
mod_sites  = ["0;8", ""]

rt_values = model.predict(sequences, mods, mod_sites)
print(rt_values)  # numpy.ndarray of shape (2,)
```

### Collision Cross Section (CCS) Prediction

```python
import redeem_properties_py

model = redeem_properties_py.CCSModel(
    model_path="path/to/ccs.pth",
    arch="ccs_cnn_lstm",
    constants_path="path/to/ccs.pth.model_const.yaml",
)

sequences  = ["AGHCEWQMKYR", "PEPTIDE"]
mods       = ["Oxidation@M", ""]
mod_sites  = ["8", ""]
charges    = [2, 3]

ccs_values = model.predict(sequences, mods, mod_sites, charges)
print(ccs_values)  # numpy.ndarray of shape (2,)
```

### MS2 Fragment Intensity Prediction

```python
import redeem_properties_py

model = redeem_properties_py.MS2Model(
    model_path="path/to/ms2.pth",
    arch="ms2_bert",
    constants_path="path/to/ms2.pth.model_const.yaml",
)

sequences   = ["AGHCEWQMKYR"]
mods        = ["Oxidation@M"]
mod_sites   = ["8"]
charges     = [2]
nces        = [20]
instruments = ["QE"]

intensities = model.predict(sequences, mods, mod_sites, charges, nces, instruments)
# List of 2-D numpy arrays, one per peptide
print(intensities[0].shape)
```

## Supported Architectures

| Model | `arch` value |
|-------|-------------|
| AlphaPeptDeep RT CNN-LSTM | `rt_cnn_lstm` |
| AlphaPeptDeep RT CNN-Transformer | `rt_cnn_tf` |
| AlphaPeptDeep CCS CNN-LSTM | `ccs_cnn_lstm` |
| AlphaPeptDeep CCS CNN-Transformer | `ccs_cnn_tf` |
| AlphaPeptDeep MS2 BERT | `ms2_bert` |
