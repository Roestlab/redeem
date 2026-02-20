"""
redeem_properties_py
====================
Python bindings for the redeem-properties Rust crate, exposing peptide property
prediction models for retention time (RT), collisional cross-section (CCS), and
MS2 fragment intensities.

All three model classes delegate inference to the compiled Rust extension
(``_lib``).  The ``predict_df`` convenience methods additionally return results
as a ``pandas`` or ``polars`` DataFrame.

Quick start
-----------
>>> import redeem_properties_py as rp
>>>
>>> rt_model  = rp.RTModel.from_pretrained("rt")
>>> ccs_model = rp.CCSModel.from_pretrained("ccs")
>>> ms2_model = rp.MS2Model.from_pretrained("ms2")
>>>
>>> # numpy arrays / list[dict]
>>> rt_values   = rt_model.predict(["PEPTIDE", "SEQU[+42.0106]ENCE"])
>>> ccs_results = ccs_model.predict(["PEPTIDE"], charges=[2])
>>> ms2_results = ms2_model.predict(["PEPTIDE"], charges=[2], nces=[20])
>>>
>>> # pandas DataFrames
>>> rt_df  = rt_model.predict_df(["PEPTIDE", "SEQU[+42.0106]ENCE"])
>>> ccs_df = ccs_model.predict_df(["PEPTIDE"], charges=[2])
>>> ms2_df = ms2_model.predict_df(["PEPTIDE"], charges=[2], nces=[20])
"""

from __future__ import annotations

from typing import Optional

from redeem_properties_py._lib import (  # noqa: F401  (re-exported)
    CCSModel as _CCSLib,
    MS2Model as _MS2Lib,
    RTModel as _RTLib,
    locate_pretrained as locate_pretrained,
    validate_pretrained as validate_pretrained,
)

__all__ = ["RTModel", "CCSModel", "MS2Model", "locate_pretrained"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(data: dict, framework: str):
    """Build a DataFrame from a column dict using *framework* ('pandas'/'polars')."""
    if framework == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for predict_df(framework='pandas'). "
                "Install it with: pip install redeem-properties-py[pandas]"
            ) from exc
        return pd.DataFrame(data)
    elif framework == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "polars is required for predict_df(framework='polars'). "
                "Install it with: pip install redeem-properties-py[polars]"
            ) from exc
        return pl.DataFrame(data)
    else:
        raise ValueError(f"Unknown framework '{framework}'. Use 'pandas' or 'polars'.")


# ---------------------------------------------------------------------------
# RTModel
# ---------------------------------------------------------------------------

class RTModel:
    """Retention time prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"rt_cnn_lstm"``).
    constants_path:
        Optional path to the ``.yaml`` constants file.
    use_cuda:
        Whether to run inference on GPU (requires CUDA build). Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _RTLib(model_path, arch, constants_path=constants_path, use_cuda=use_cuda)

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "RTModel":
        """Load an RTModel from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"rt"``, ``"alphapeptdeep-rt-cnn-lstm"``, ``"redeem-rt-cnn-tf"``.

        Parameters
        ----------
        name:
            Pretrained model identifier.
        use_cuda:
            Whether to run inference on GPU. Default ``False``.
        """
        obj = cls.__new__(cls)
        obj._inner = _RTLib.from_pretrained(name, use_cuda=use_cuda)
        return obj

    def predict(self, peptides: list[str]):
        """Predict retention times for a list of peptides.

        Peptides may contain inline modification annotations
        (``[+X.X]`` mass-shift or ``(UniMod:N)`` notation).

        Parameters
        ----------
        peptides:
            List of peptide sequences.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of predicted RT values, one per peptide.
        """
        return self._inner.predict(peptides)

    def predict_df(self, peptides: list[str], framework: str = "pandas"):
        """Predict retention times and return the result as a DataFrame.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide`` (str), ``rt`` (float32).
        """
        rt_values = self.predict(peptides)
        return _make_df({"peptide": peptides, "rt": rt_values}, framework)


# ---------------------------------------------------------------------------
# CCSModel
# ---------------------------------------------------------------------------

class CCSModel:
    """Collisional cross-section prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"ccs_cnn_lstm"``).
    constants_path:
        Path to the ``.yaml`` constants file (required).
    use_cuda:
        Whether to run inference on GPU. Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _CCSLib(model_path, arch, constants_path, use_cuda=use_cuda)

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "CCSModel":
        """Load a CCSModel from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"ccs"``, ``"alphapeptdeep-ccs-cnn-lstm"``, ``"redeem-ccs-cnn-tf"``.
        """
        obj = cls.__new__(cls)
        obj._inner = _CCSLib.from_pretrained(name, use_cuda=use_cuda)
        return obj

    def predict(self, peptides: list[str], charges: list[int]):
        """Predict CCS values for a list of peptides.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide.

        Returns
        -------
        list[dict]
            One dict per peptide with keys:

            * ``"ccs"`` – predicted CCS value (Å²).
            * ``"charge"`` – charge state used for the prediction.
        """
        return self._inner.predict(peptides, charges)

    def predict_df(
        self,
        peptides: list[str],
        charges: list[int],
        framework: str = "pandas",
    ):
        """Predict CCS values and return the result as a DataFrame.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide.
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide`` (str), ``ccs`` (float32), ``charge`` (int).
        """
        results = self.predict(peptides, charges)
        return _make_df(
            {
                "peptide": peptides,
                "ccs": [r["ccs"] for r in results],
                "charge": [r["charge"] for r in results],
            },
            framework,
        )


# ---------------------------------------------------------------------------
# MS2Model
# ---------------------------------------------------------------------------

class MS2Model:
    """MS2 fragment intensity prediction model.

    Parameters
    ----------
    model_path:
        Path to the ``.pth`` model weights file.
    arch:
        Model architecture string (e.g. ``"ms2_bert"``).
    constants_path:
        Path to the ``.yaml`` constants file (required).
    use_cuda:
        Whether to run inference on GPU. Default ``False``.
    """

    def __init__(
        self,
        model_path: str,
        arch: str,
        constants_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        self._inner = _MS2Lib(model_path, arch, constants_path, use_cuda=use_cuda)

    @classmethod
    def from_pretrained(cls, name: str, use_cuda: bool = False) -> "MS2Model":
        """Load an MS2Model from the shipped pretrained weights.

        Accepted *name* values (case-insensitive):
        ``"ms2"``, ``"alphapeptdeep-ms2-bert"``.
        """
        obj = cls.__new__(cls)
        obj._inner = _MS2Lib.from_pretrained(name, use_cuda=use_cuda)
        return obj

    def predict(
        self,
        peptides: list[str],
        charges: list[int],
        nces: list[int],
        instruments: Optional[list[Optional[str]]] = None,
    ):
        """Predict MS2 fragment intensities for a list of peptides.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Charge state per peptide.
        nces:
            Normalized collision energy per peptide.
        instruments:
            Instrument name per peptide (optional).

        Returns
        -------
        list[dict]
            One dict per peptide with keys:

            * ``"intensities"`` – 2-D float32 array ``(n_positions, 8)``.
            * ``"ion_types"`` – list of 8 ion-type strings.
            * ``"ion_charges"`` – list of 8 fragment charge integers.
            * ``"b_ordinals"`` – 1-D int array ``[1, …, n_positions]``.
            * ``"y_ordinals"`` – 1-D int array ``[n_positions, …, 1]``.
        """
        return self._inner.predict(peptides, charges, nces, instruments=instruments)

    def predict_df(
        self,
        peptides: list[str],
        charges: list[int],
        nces: list[int],
        instruments: Optional[list[Optional[str]]] = None,
        framework: str = "pandas",
    ):
        """Predict MS2 fragment intensities and return a long-format DataFrame.

        Each row represents one (peptide, ion_type, fragment_charge, ordinal) combination.

        Parameters
        ----------
        peptides:
            List of peptide sequences (inline modifications supported).
        charges:
            Precursor charge state per peptide.
        nces:
            Normalized collision energy per peptide.
        instruments:
            Instrument name per peptide (optional).
        framework:
            ``'pandas'`` (default) or ``'polars'``.

        Returns
        -------
        pandas.DataFrame or polars.DataFrame
            Columns: ``peptide``, ``ion_type``, ``fragment_charge``,
            ``ordinal``, ``intensity``.

        Example
        -------
        >>> df = ms2_model.predict_df(
        ...     ["AGHCEWQMKYR"],
        ...     charges=[2], nces=[20], instruments=["QE"],
        ... )
        >>> df.head()
           peptide ion_type  fragment_charge  ordinal  intensity
        0  AGHCEWQMKYR       b                1        1      0.123
        1  AGHCEWQMKYR       b                2        1      0.045
        ...
        """
        results = self.predict(peptides, charges, nces, instruments=instruments)

        b_ion_types = {"b", "b_nl"}
        pep_col: list[str] = []
        ion_type_col: list[str] = []
        frag_charge_col: list[int] = []
        ordinal_col: list[int] = []
        intensity_col: list[float] = []

        for pep, res in zip(peptides, results):
            intensities = res["intensities"]
            ion_types = res["ion_types"]
            frag_charges = res["ion_charges"]
            b_ords = res["b_ordinals"]
            y_ords = res["y_ordinals"]
            n_pos, n_types = intensities.shape
            for r in range(n_pos):
                for c in range(n_types):
                    t = ion_types[c]
                    ordinal = int(b_ords[r]) if t in b_ion_types else int(y_ords[r])
                    pep_col.append(pep)
                    ion_type_col.append(t)
                    frag_charge_col.append(int(frag_charges[c]))
                    ordinal_col.append(ordinal)
                    intensity_col.append(float(intensities[r, c]))

        return _make_df(
            {
                "peptide": pep_col,
                "ion_type": ion_type_col,
                "fragment_charge": frag_charge_col,
                "ordinal": ordinal_col,
                "intensity": intensity_col,
            },
            framework,
        )
