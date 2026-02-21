# Getting Started

This guide will walk you through installing `redeem_properties` and using it to predict Retention Time (RT), Collision Cross Section (CCS), and MS2 fragment intensities for peptides.

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

For `predict_df` support, install pandas or polars as extras:

```bash
pip install "redeem_properties[pandas]"   # or [polars]
```

## Usage Examples

Peptides can be passed with inline modification annotations — the binding handles parsing automatically. Both mass-shift (`[+X.X]`) and UniMod (`(UniMod:N)`) notation are supported.

### Retention Time (RT) Prediction

```python
import redeem_properties

# Load the pretrained RT model
model = redeem_properties.RTModel.from_pretrained("rt")

# Predict RT for a list of peptides
rt_values = model.predict([
    "AGHCEWQMKYR",
    "SEQU[+42.0106]ENCE",        # mass-shift notation
    "SEQUEN(UniMod:4)CE",        # UniMod notation
])
print(rt_values)  # numpy.ndarray of shape (3,)

# Get results as a pandas DataFrame
rt_df = model.predict_df(["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"])
print(rt_df)
```

### Collision Cross Section (CCS) Prediction

```python
import redeem_properties

# Load the pretrained CCS model
model = redeem_properties.CCSModel.from_pretrained("ccs")

# Predict CCS for peptides with specific charge states
ccs_values = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
)

# List of dicts, one per peptide-charge combination
for res in ccs_values:
    print(f"CCS: {res['ccs']:.2f} Å², Charge: {res['charge']}")

# Get results as a pandas DataFrame
ccs_df = model.predict_df(["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"], charges=[2, 3])
print(ccs_df)
```

### MS2 Fragment Intensity Prediction

```python
import redeem_properties

# Load the pretrained MS2 model
model = redeem_properties.MS2Model.from_pretrained("ms2")

# Predict MS2 fragment intensities
results = model.predict(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
    nces=20,
    instruments="QE",
)

# Each element is a dict with intensities + fragment annotations
for res in results:
    print(f"Intensities shape: {res['intensities'].shape}")
    print(f"Ion types: {res['ion_types']}")

# Get results as a long-format DataFrame
ms2_df = model.predict_df(
    ["AGHCEWQMKYR", "SEQU[+42.0106]ENCE"],
    charges=[2, 3],
    nces=20,
    instruments="QE",
)
print(ms2_df.head())
```

### Unified Prediction with PropertyPrediction

The `PropertyPrediction` helper loads RT, CCS, and MS2 models and returns a single long-format DataFrame combining scalar predictions (RT/CCS) with per-fragment MS2 rows.

```python
import redeem_properties as rp

# Create the unified predictor (loads pretrained models by default)
prop = rp.PropertyPrediction()

peptides = [
    "SKEEET[+79.9663]SIDVAGKP",
    "LPILVPSAKKAIYM",
    "RTPKIQVYSRHPAE",
]

# Predict all properties and return a unified DataFrame
df = prop.predict_df(
    peptides,
    charges=[2, 3],
    nces=20,
    instruments="QE",
    annotate_mz=True,
)

print(df.head())
```
