# redeem-properties-py

Python bindings for the [redeem-properties](../redeem-properties) Rust crate, exposing peptide property prediction models (RT, CCS, MS2) via [PyO3](https://pyo3.rs).

## Installation

### From PyPI

By default, installing from PyPI provides a CPU-only build with embedded pretrained weights:

```bash
pip install redeem_properties
```

To install with CUDA support, you can compile the package from the source distribution (sdist) provided on PyPI. This requires the Rust toolchain to be installed on your system:

```bash
# Install with CUDA support and embedded pretrained weights
pip install redeem_properties --no-binary redeem_properties --config-settings=cargo-args="--features=cuda,pretrained"
```

### From source (requires Rust + maturin)

```bash
pip install maturin
cd redeem-properties-py
maturin develop
```

For `predict_df` support install pandas or polars as extras:

```bash
pip install "redeem_properties[pandas]"   # or [polars]
# from source:
pip install ".[pandas]"
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
import redeem_properties

# From the shipped pretrained weights using the pretrained model registry
# Accepted names: "rt", "alphapeptdeep-rt", "alphapeptdeep-rt-cnn-lstm",
#                 "redeem-rt", "redeem-rt-cnn-tf"
model = redeem_properties.RTModel.from_pretrained("rt")

# Or load from a custom model file
model = redeem_properties.RTModel(
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

# DataFrame output (pandas or polars)
rt_df = model.predict_df(["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"])
# columns: peptide, rt
rt_df_polars = model.predict_df(["AGHCEWQMKYR"], framework="polars")
```

### Collision Cross Section (CCS) Prediction

```python
import redeem_properties

# From the shipped pretrained weights
# Accepted names: "ccs", "alphapeptdeep-ccs", "alphapeptdeep-ccs-cnn-lstm",
#                 "redeem-ccs", "redeem-ccs-cnn-tf"
model = redeem_properties.CCSModel.from_pretrained("ccs")

# Or load from a custom model file
model = redeem_properties.CCSModel(
    model_path="path/to/ccs.pth",
    arch="ccs_cnn_lstm",
    constants_path="path/to/ccs.pth.model_const.yaml",
)

ccs_values = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
)
# List of dicts, one per peptide-charge combination
for res in ccs_values:
    print(res["ccs"])     # predicted CCS value (Å²)
    print(res["charge"])  # charge state used for this prediction

# DataFrame output — columns: peptide, ccs, charge
ccs_df = model.predict_df(["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"], charges=[2, 3])
ccs_df_polars = model.predict_df(["AGHCEWQMKYR"], charges=2, framework="polars")
```

### MS2 Fragment Intensity Prediction

```python
import redeem_properties

# From the shipped pretrained weights
# Accepted names: "ms2", "alphapeptdeep-ms2", "alphapeptdeep-ms2-bert"
model = redeem_properties.MS2Model.from_pretrained("ms2")

# Or load from a custom model file
model = redeem_properties.MS2Model(
    model_path="path/to/ms2.pth",
    arch="ms2_bert",
    constants_path="path/to/ms2.pth.model_const.yaml",
)

results = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
    nces=20,
    instruments="QE",
)
# Each element is a dict with intensities + fragment annotations
for res in results:
    print(res["intensities"].shape)  # (n_positions, 8)
    print(res["ion_types"])          # ["b", "b", "y", "y", "b_nl", "b_nl", "y_nl", "y_nl"]
    print(res["ion_charges"])        # [1, 2, 1, 2, 1, 2, 1, 2]
    print(res["b_ordinals"])         # [1, 2, ..., n_positions]
    print(res["y_ordinals"])         # [n_positions, ..., 1]

# Long-format DataFrame — one row per (peptide, ion_type, fragment_charge, ordinal):
ms2_df = model.predict_df(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
    nces=20,
    instruments="QE",
)
# columns: peptide, ion_type, fragment_charge, ordinal, intensity
print(ms2_df.head())

# polars variant
ms2_df_polars = model.predict_df(
    ["AGHCEWQMKYR"],
    charges=2, nces=20,
    framework="polars",
)
```

### Unified prediction with PropertyPrediction

The `PropertyPrediction` helper loads RT, CCS and MS2 models (all optional)
and returns a single long-format DataFrame combining scalar predictions
(RT/CCS) with per-fragment MS2 rows. By default `annotate_mz=True` so
precursor and fragment m/z values are included when a charge is supplied.

```python
import redeem_properties as rp

# Create the unified predictor (loads pretrained models by default)
prop = rp.PropertyPrediction()

peptides = [
    "SKEEET[+79.9663]SIDVAGKP",
    "LPILVPSAKKAIYM",
    "RTPKIQVYSRHPAE",
]
charges = [2, 3]
nces = 20
instruments = "QE"

# Long-format DataFrame: one row per fragment. Columns include
# peptide, charge, nce, instrument, rt, ccs, precursor_mz, ion_type,
# fragment_charge, ordinal, intensity, and mz (when annotate_mz=True).
# Because we provided 3 peptides and 2 charges, this will predict
# 6 combinations (Cartesian product).
df = prop.predict_df(
    peptides,
    charges=charges,
    nces=nces,
    instruments=instruments,
    annotate_mz=True,
)

print(df.columns.tolist())
print(df.head())
```

If you only need scalar predictions (RT/CCS) and not MS2, construct
`PropertyPrediction(predict_ms2=False)` and call `predict_df` — it will
return one row per peptide and will still include `precursor_mz` when
`annotate_mz=True` and `charges` are provided.

```python
prop_scalar = rp.PropertyPrediction(predict_ms2=False)
df_scalar = prop_scalar.predict_df(peptides, charges=charges, annotate_mz=True)
print(df_scalar.head())
```

## Pretrained Model Names

The `from_pretrained` method accepts the following names (case-insensitive):

| Short name | Full name | Model class |
|------------|-----------|-------------|
| `"alphapeptdeep-rt"` | `"alphapeptdeep-rt-cnn-lstm"` | `RTModel` |
| `"redeem-rt"` | `"redeem-rt-cnn-tf"` | `RTModel` |
| `"alphapeptdeep-ccs"` | `"alphapeptdeep-ccs-cnn-lstm"` | `CCSModel` |
| `"redeem-ccs"` | `"redeem-ccs-cnn-tf"` | `CCSModel` |
| `"alphapeptdeep-ms2"` | `"alphapeptdeep-ms2-bert"` | `MS2Model` |

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
