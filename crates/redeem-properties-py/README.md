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

Peptides can be passed with inline modification annotations — the binding handles
parsing automatically. Both mass-shift (`[+X.X]`) and UniMod (`(UniMod:N)`) notation
are supported.

### Retention Time (RT) Prediction

```python
import redeem_properties_py

# From the shipped pretrained weights using the pretrained model registry
# Accepted names: "rt", "alphapeptdeep-rt", "alphapeptdeep-rt-cnn-lstm",
#                 "redeem-rt", "redeem-rt-cnn-tf"
model = redeem_properties_py.RTModel.from_pretrained("rt")

# Or load from a custom model file
model = redeem_properties_py.RTModel(
    model_path="path/to/rt.pth",
    arch="rt_cnn_lstm",
    constants_path="path/to/rt.pth.model_const.yaml",
)

# Peptides with inline modifications — no separate mod/site strings needed
rt_values = model.predict([
    "AGHCEWQMKYR",
    "SEQU[+42.0106]ENCE",        # mass-shift notation
    "SEQUEN(UniMod:4)CE",        # UniMod notation
])
print(rt_values)  # numpy.ndarray of shape (3,)
```

### Collision Cross Section (CCS) Prediction

```python
import redeem_properties_py

# From the shipped pretrained weights
# Accepted names: "ccs", "alphapeptdeep-ccs", "alphapeptdeep-ccs-cnn-lstm",
#                 "redeem-ccs", "redeem-ccs-cnn-tf"
model = redeem_properties_py.CCSModel.from_pretrained("ccs")

# Or load from a custom model file
model = redeem_properties_py.CCSModel(
    model_path="path/to/ccs.pth",
    arch="ccs_cnn_lstm",
    constants_path="path/to/ccs.pth.model_const.yaml",
)

ccs_values = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
)
print(ccs_values)  # numpy.ndarray of shape (2,)
```

### MS2 Fragment Intensity Prediction

```python
import redeem_properties_py

# From the shipped pretrained weights
# Accepted names: "ms2", "alphapeptdeep-ms2", "alphapeptdeep-ms2-bert"
model = redeem_properties_py.MS2Model.from_pretrained("ms2")

# Or load from a custom model file
model = redeem_properties_py.MS2Model(
    model_path="path/to/ms2.pth",
    arch="ms2_bert",
    constants_path="path/to/ms2.pth.model_const.yaml",
)

intensities = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 2],
    nces=[20, 20],
    instruments=["QE", "QE"],
)
# List of 2-D numpy arrays, one per peptide
print(intensities[0].shape)
```

## Pretrained Model Names

The `from_pretrained` method accepts the following names (case-insensitive):

| Short name | Full name | Model class |
|------------|-----------|-------------|
| `"rt"` | `"alphapeptdeep-rt-cnn-lstm"` | `RTModel` |
| `"redeem-rt"` | `"redeem-rt-cnn-tf"` | `RTModel` |
| `"ccs"` | `"alphapeptdeep-ccs-cnn-lstm"` | `CCSModel` |
| `"redeem-ccs"` | `"redeem-ccs-cnn-tf"` | `CCSModel` |
| `"ms2"` | `"alphapeptdeep-ms2-bert"` | `MS2Model` |

Model files are looked up in this order:
1. `$REDEEM_PRETRAINED_MODELS_DIR/<path>`
2. `data/pretrained_models/` relative to the working directory
3. `$HOME/.local/share/redeem/models/<path>`

## Supported Architectures (manual loading)

| Model | `arch` value |
|-------|-------------|
| AlphaPeptDeep RT CNN-LSTM | `rt_cnn_lstm` |
| AlphaPeptDeep RT CNN-Transformer | `rt_cnn_tf` |
| AlphaPeptDeep CCS CNN-LSTM | `ccs_cnn_lstm` |
| AlphaPeptDeep CCS CNN-Transformer | `ccs_cnn_tf` |
| AlphaPeptDeep MS2 BERT | `ms2_bert` |
