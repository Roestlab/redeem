use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use redeem_properties::models::ccs_model::CCSModelWrapper;
use redeem_properties::models::model_interface::PredictionResult;
use redeem_properties::models::ms2_model::MS2ModelWrapper;
use redeem_properties::models::rt_model::RTModelWrapper;
use redeem_properties::pretrained::{locate_pretrained_model, PretrainedModel};

fn strings_to_arcs(v: Vec<String>) -> Vec<Arc<[u8]>> {
    v.into_iter()
        .map(|s| Arc::from(s.into_bytes().into_boxed_slice()))
        .collect()
}

fn opt_strings_to_arcs(v: Vec<Option<String>>) -> Vec<Option<Arc<[u8]>>> {
    v.into_iter()
        .map(|opt| opt.map(|s| Arc::from(s.into_bytes().into_boxed_slice())))
        .collect()
}

/// Parse a pretrained model name and validate it belongs to the expected family.
///
/// Returns `(arch_str, model_path)` on success.  The `family` argument is one of
/// `"rt"`, `"ccs"`, or `"ms2"` and is used only for the error message.
fn resolve_pretrained(name: &str, family: &str) -> PyResult<(String, PathBuf)> {
    let pm = PretrainedModel::from_str(name)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let arch = pm.arch();
    if !arch.starts_with(family) {
        return Err(PyRuntimeError::new_err(format!(
            "Pretrained model '{}' (arch '{}') is not a {} model. \
             Pass the correct name to the matching class.",
            name, arch, family
        )));
    }

    let model_path = locate_pretrained_model(pm)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok((arch.to_string(), model_path))
}

/// Python wrapper for the RT (retention time) prediction model.
#[pyclass]
struct RTModel {
    inner: RTModelWrapper,
}

#[pymethods]
impl RTModel {
    /// Create a new RTModel.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "rt_cnn_lstm" or "rt_cnn_tf").
    ///     constants_path (str, optional): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path=None, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: Option<String>,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let wrapper = RTModelWrapper::new(
            std::path::Path::new(&model_path),
            constants_path.as_deref().map(std::path::Path::new),
            &arch,
            device,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load an RTModel from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    /// On first use the search order is:
    ///   1. ``$REDEEM_PRETRAINED_MODELS_DIR/<name>``
    ///   2. ``data/pretrained_models/`` relative to the current directory
    ///   3. ``$HOME/.local/share/redeem/models/<name>``
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"rt"`` / ``"alphapeptdeep-rt"`` / ``"alphapeptdeep-rt-cnn-lstm"``
    ///   - ``"redeem-rt"`` / ``"redeem-rt-cnn-tf"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     RTModel: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "rt")?;
        let device = get_device(use_cuda)?;
        let wrapper = RTModelWrapper::new(&model_path, None::<std::path::PathBuf>.as_ref(), &arch, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Predict retention times for a list of peptide sequences.
    ///
    /// Args:
    ///     sequences (list[str]): Peptide sequences.
    ///     mods (list[str]): Modification strings per peptide.
    ///     mod_sites (list[str]): Modification site strings per peptide.
    ///
    /// Returns:
    ///     numpy.ndarray: 1-D array of predicted RT values (f32).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let seqs = strings_to_arcs(sequences);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(mod_sites);

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match result {
            PredictionResult::RTResult(values) => {
                Ok(PyArray1::from_slice_bound(py, &values))
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }
}

/// Python wrapper for the CCS (collision cross section) prediction model.
#[pyclass]
struct CCSModel {
    inner: CCSModelWrapper,
}

#[pymethods]
impl CCSModel {
    /// Create a new CCSModel.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "ccs_cnn_lstm" or "ccs_cnn_tf").
    ///     constants_path (str): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: String,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let wrapper = CCSModelWrapper::new(&model_path, &constants_path, &arch, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load a CCSModel from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"ccs"`` / ``"alphapeptdeep-ccs"`` / ``"alphapeptdeep-ccs-cnn-lstm"``
    ///   - ``"redeem-ccs"`` / ``"redeem-ccs-cnn-tf"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     CCSModel: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "ccs")?;
        let device = get_device(use_cuda)?;
        // CCSModelWrapper requires a non-optional constants_path. For pretrained .pth files
        // the constants are embedded in the file itself, so model_path is passed for both.
        let wrapper = CCSModelWrapper::new(&model_path, &model_path, &arch, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Predict CCS values for a list of peptide sequences.
    ///
    /// Args:
    ///     sequences (list[str]): Peptide sequences.
    ///     mods (list[str]): Modification strings per peptide.
    ///     mod_sites (list[str]): Modification site strings per peptide.
    ///     charges (list[int]): Charge states per peptide.
    ///
    /// Returns:
    ///     numpy.ndarray: 1-D array of predicted CCS values (f32).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
        charges: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let seqs = strings_to_arcs(sequences);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(mod_sites);

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc, charges)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match result {
            PredictionResult::CCSResult(values) => {
                Ok(PyArray1::from_slice_bound(py, &values))
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }
}

/// Python wrapper for the MS2 fragment intensity prediction model.
#[pyclass]
struct MS2Model {
    inner: MS2ModelWrapper,
}

#[pymethods]
impl MS2Model {
    /// Create a new MS2Model.
    ///
    /// Args:
    ///     model_path (str): Path to the model file.
    ///     arch (str): Model architecture (e.g. "ms2_bert").
    ///     constants_path (str): Path to the constants/config file.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    #[new]
    #[pyo3(signature = (model_path, arch, constants_path, use_cuda=false))]
    fn new(
        model_path: String,
        arch: String,
        constants_path: String,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let device = get_device(use_cuda)?;
        let wrapper = MS2ModelWrapper::new(&model_path, &constants_path, &arch, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Load an MS2Model from the shipped pretrained weights.
    ///
    /// Models are located using the ``redeem_properties`` pretrained-model registry.
    ///
    /// Accepted ``name`` values (case-insensitive):
    ///   - ``"ms2"`` / ``"alphapeptdeep-ms2"`` / ``"alphapeptdeep-ms2-bert"``
    ///
    /// Args:
    ///     name (str): Pretrained model identifier.
    ///     use_cuda (bool): Whether to use CUDA for inference. Defaults to False.
    ///
    /// Returns:
    ///     MS2Model: A ready-to-use model loaded with pretrained weights.
    #[staticmethod]
    #[pyo3(signature = (name, use_cuda=false))]
    fn from_pretrained(name: String, use_cuda: bool) -> PyResult<Self> {
        let (arch, model_path) = resolve_pretrained(&name, "ms2")?;
        let device = get_device(use_cuda)?;
        // MS2ModelWrapper requires a non-optional constants_path. For pretrained .pth files
        // the constants are embedded in the file itself, so model_path is passed for both.
        let wrapper = MS2ModelWrapper::new(&model_path, &model_path, &arch, device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: wrapper })
    }

    /// Predict MS2 fragment intensities for a list of peptide sequences.
    ///
    /// Args:
    ///     sequences (list[str]): Peptide sequences.
    ///     mods (list[str]): Modification strings per peptide.
    ///     mod_sites (list[str]): Modification site strings per peptide.
    ///     charges (list[int]): Charge states per peptide.
    ///     nces (list[int]): Normalized collision energies per peptide.
    ///     instruments (list[str | None], optional): Instrument names per peptide.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of 2-D arrays of fragment intensities (f32),
    ///         one per peptide.
    #[pyo3(signature = (sequences, mods, mod_sites, charges, nces, instruments=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<String>,
        mods: Vec<String>,
        mod_sites: Vec<String>,
        charges: Vec<i32>,
        nces: Vec<i32>,
        instruments: Option<Vec<Option<String>>>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
        let seqs = strings_to_arcs(sequences);
        let mods_arc = strings_to_arcs(mods);
        let sites_arc = strings_to_arcs(mod_sites);
        let n = seqs.len();
        let instr_vec = match instruments {
            Some(v) => opt_strings_to_arcs(v),
            None => vec![None; n],
        };

        let result = self
            .inner
            .predict(&seqs, &mods_arc, &sites_arc, charges, nces, instr_vec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        match result {
            PredictionResult::MS2Result(matrices) => {
                let mut out = Vec::with_capacity(matrices.len());
                for matrix in matrices {
                    let rows = matrix.len();
                    let cols = if rows > 0 { matrix[0].len() } else { 0 };
                    let arr =
                        ndarray::Array2::from_shape_fn((rows, cols), |(r, c)| matrix[r][c]);
                    out.push(PyArray2::from_array_bound(py, &arr));
                }
                Ok(out)
            }
            _ => Err(PyRuntimeError::new_err("Unexpected prediction result type")),
        }
    }
}

#[cfg(feature = "cuda")]
fn get_device(use_cuda: bool) -> PyResult<candle_core::Device> {
    if use_cuda {
        candle_core::Device::new_cuda(0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

#[cfg(not(feature = "cuda"))]
fn get_device(use_cuda: bool) -> PyResult<candle_core::Device> {
    if use_cuda {
        Err(PyRuntimeError::new_err(
            "CUDA support is not compiled in. Rebuild with --features cuda.",
        ))
    } else {
        Ok(candle_core::Device::Cpu)
    }
}

/// Python bindings for the redeem-properties peptide property prediction models.
#[pymodule]
fn redeem_properties_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RTModel>()?;
    m.add_class::<CCSModel>()?;
    m.add_class::<MS2Model>()?;
    Ok(())
}
