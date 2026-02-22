"""Basic smoke tests for the redeem_properties Python bindings.

These tests exercise import paths, class/function existence, docstrings,
and basic error paths — none require pretrained model files.
"""

import pytest


# ---------------------------------------------------------------------------
# Import & existence
# ---------------------------------------------------------------------------


def test_module_importable():
    """The top-level wrapper package should be importable."""
    import redeem_properties  # noqa: F401


def test_raw_lib_importable():
    """The low-level Rust extension should also be importable."""
    from redeem_properties import _lib  # noqa: F401


def test_classes_exist():
    import redeem_properties

    assert hasattr(redeem_properties, "RTModel"), "RTModel not found"
    assert hasattr(redeem_properties, "CCSModel"), "CCSModel not found"
    assert hasattr(redeem_properties, "MS2Model"), "MS2Model not found"


def test_utility_functions_exist():
    import redeem_properties

    for name in (
        "compute_precursor_mz",
        "compute_fragment_mzs",
        "compute_peptide_mz_info",
        "match_fragment_mzs",
        "ccs_to_mobility",
        "locate_pretrained",
        "validate_pretrained",
        "download_pretrained_models",
    ):
        assert hasattr(redeem_properties, name), f"{name} not found"


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


def test_rt_model_has_docstring():
    from redeem_properties._lib import RTModel

    assert RTModel.__doc__, "RTModel missing docstring"


def test_ccs_model_has_docstring():
    from redeem_properties._lib import CCSModel

    assert CCSModel.__doc__, "CCSModel missing docstring"


def test_ms2_model_has_docstring():
    from redeem_properties._lib import MS2Model

    assert MS2Model.__doc__, "MS2Model missing docstring"


# ---------------------------------------------------------------------------
# Class method existence
# ---------------------------------------------------------------------------


def test_rt_model_has_from_pretrained():
    from redeem_properties import RTModel

    assert hasattr(RTModel, "from_pretrained"), "RTModel missing from_pretrained"


def test_ccs_model_has_from_pretrained():
    from redeem_properties import CCSModel

    assert hasattr(CCSModel, "from_pretrained"), "CCSModel missing from_pretrained"


def test_ms2_model_has_from_pretrained():
    from redeem_properties import MS2Model

    assert hasattr(MS2Model, "from_pretrained"), "MS2Model missing from_pretrained"


def test_models_have_predict_df():
    from redeem_properties import RTModel, CCSModel, MS2Model

    for cls in (RTModel, CCSModel, MS2Model):
        assert hasattr(cls, "predict_df"), f"{cls.__name__} missing predict_df"


# ---------------------------------------------------------------------------
# Error paths (no model files needed)
# ---------------------------------------------------------------------------


def test_from_pretrained_unknown_name_raises():
    """Unknown pretrained model names should raise RuntimeError."""
    from redeem_properties._lib import RTModel

    with pytest.raises(RuntimeError):
        RTModel.from_pretrained("completely_unknown_model_xyz")


def test_from_pretrained_wrong_family_raises():
    """Passing a CCS model name to RTModel.from_pretrained should raise."""
    from redeem_properties._lib import RTModel

    with pytest.raises(RuntimeError):
        RTModel.from_pretrained("ccs")


def test_rt_model_nonexistent_path_raises():
    from redeem_properties._lib import RTModel

    with pytest.raises(RuntimeError):
        RTModel(
            model_path="/nonexistent/path/rt.pth",
            arch="rt_cnn_lstm",
            constants_path="/nonexistent/path/rt.pth.model_const.yaml",
        )


def test_ccs_model_nonexistent_path_raises():
    from redeem_properties._lib import CCSModel

    with pytest.raises(RuntimeError):
        CCSModel(
            model_path="/nonexistent/path/ccs.pth",
            arch="ccs_cnn_lstm",
            constants_path="/nonexistent/path/ccs.pth.model_const.yaml",
        )


def test_ms2_model_nonexistent_path_raises():
    from redeem_properties._lib import MS2Model

    with pytest.raises(RuntimeError):
        MS2Model(
            model_path="/nonexistent/path/ms2.pth",
            arch="ms2_bert",
            constants_path="/nonexistent/path/ms2.pth.model_const.yaml",
        )
