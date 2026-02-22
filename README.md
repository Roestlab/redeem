<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new.png" alt="ReDeem_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new_dark.png" alt="ReDeem_Logo" width="500">
    <img alt="ReDeem Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/singjc/redeem/raw/develop/img/redeem_logo_new_dark.png">
  </picture>
</p>

---

# ReDeeM: Repository for Deep Learning Models for Mass Spectrometry

ReDeeM is a Rust workspace for mass spectrometry proteomics, providing deep learning models for peptide property prediction and machine learning classifiers for PSM rescoring. It is designed to be used as a library in other tools (e.g. [Sage](https://github.com/lazear/sage)).

## Crates

| Crate | Description | Docs |
|-------|-------------|------|
| [`redeem-cli`](redeem-cli/) | Command-line interface for ReDeeM | [README](redeem-cli/README.md) |
| [`redeem-classifiers`](redeem-classifiers/) | Semi-supervised PSM rescoring (GBDT, XGBoost, SVM) | [README](redeem-classifiers/README.md) |
| [`redeem-properties`](redeem-properties/) | Peptide property prediction (RT, CCS, MS2) using candle | [README](redeem-properties/README.md) |
| [`redeem-properties-py`](redeem-properties-py/) | Python bindings for `redeem-properties` via PyO3 | [README](redeem-properties-py/README.md) · [Docs](https://redeem-properties.readthedocs.io/) |

## Installation

### Rust

> [!NOTE]
> The ReDeeM crates are still under development and are not yet available on crates.io.

```toml
[dependencies]
redeem-properties = { git = "https://github.com/singjc/redeem.git", branch = "master" }
redeem-classifiers = { git = "https://github.com/singjc/redeem.git", branch = "master" }
```

### Python

```bash
pip install redeem_properties
```

For DataFrame output support:

```bash
pip install "redeem_properties[pandas]"   # or [polars]
```

## Quick Example

### Python

```python
import redeem_properties as rp

# Create a unified predictor (loads pretrained models by default)
model = rp.PropertyPrediction()

peptides = [
    "SKEEET[+79.9663]SIDVAGKP",
    "LPILVPSAKKAIYM",
    "RTPKIQVYSRHPAE",
]

df = model.predict_df(
    peptides,
    charges=[2, 3],
    nces=20,
    instruments="timsTOF",
    annotate_mz=True,
    annotate_mobility=True
)
```

```python
>>> df.head()
                    peptide  charge  nce instrument         rt         ccs  ion_mobility  precursor_mz ion_type  fragment_charge  ordinal    intensity          mz
0  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        2   790.534363  216.134268
1  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        3   822.035767  345.176861
2  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        4  1272.754517  474.219454
3  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        b                1        5  1806.533691  603.262047
4  SKEEET[+79.9663]SIDVAGKP       2   20    timsTOF  26.884516  535.355408      1.324961     785.35581        y                1        9   218.158798  967.449573
```

See the [Python bindings README](redeem-properties-py/README.md) for full usage, including MS2 prediction, DataFrame output, and the unified `PropertyPrediction` helper.

### Rust

```rust
use redeem_properties::pretrained::{locate_pretrained_model, PretrainedModel};
use redeem_properties::models::rt_model::RTModelWrapper;
use candle_core::Device;
use std::sync::Arc;

let model_path = locate_pretrained_model(PretrainedModel::RedeemRtCnnTf)?;
let model = RTModelWrapper::new(&model_path, None::<&str>, "rt_cnn_tf", Device::Cpu)?;

let sequences = vec![Arc::from(b"PEPTIDEK".as_slice())];
let mods = vec![Arc::from(b"".as_slice())];
let mod_sites = vec![Arc::from(b"".as_slice())];
let result = model.predict(&sequences, &mods, &mod_sites)?;
```

See each crate's README for detailed API documentation and examples.

