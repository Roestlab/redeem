"""Basic smoke tests for the redeem_properties_py module."""

import pytest


def test_module_importable():
    import redeem_properties_py  # noqa: F401


def test_classes_exist():
    import redeem_properties_py

    assert hasattr(redeem_properties_py, "RTModel"), "RTModel not found in module"
    assert hasattr(redeem_properties_py, "CCSModel"), "CCSModel not found in module"
    assert hasattr(redeem_properties_py, "MS2Model"), "MS2Model not found in module"


def test_rt_model_has_docstring():
    import redeem_properties_py

    assert redeem_properties_py.RTModel.__doc__, "RTModel missing docstring"


def test_ccs_model_has_docstring():
    import redeem_properties_py

    assert redeem_properties_py.CCSModel.__doc__, "CCSModel missing docstring"


def test_ms2_model_has_docstring():
    import redeem_properties_py

    assert redeem_properties_py.MS2Model.__doc__, "MS2Model missing docstring"


def test_rt_model_has_from_pretrained():
    import redeem_properties_py

    assert hasattr(redeem_properties_py.RTModel, "from_pretrained"), \
        "RTModel missing from_pretrained classmethod"


def test_ccs_model_has_from_pretrained():
    import redeem_properties_py

    assert hasattr(redeem_properties_py.CCSModel, "from_pretrained"), \
        "CCSModel missing from_pretrained classmethod"


def test_ms2_model_has_from_pretrained():
    import redeem_properties_py

    assert hasattr(redeem_properties_py.MS2Model, "from_pretrained"), \
        "MS2Model missing from_pretrained classmethod"


def test_from_pretrained_unknown_name_raises():
    """Unknown pretrained model names should raise RuntimeError."""
    import redeem_properties_py

    with pytest.raises(RuntimeError):
        redeem_properties_py.RTModel.from_pretrained("completely_unknown_model_xyz")


def test_from_pretrained_wrong_family_raises():
    """Passing a CCS model name to RTModel.from_pretrained should raise RuntimeError."""
    import redeem_properties_py

    with pytest.raises(RuntimeError):
        # "ccs" is a valid pretrained name but belongs to the CCS family, not RT
        redeem_properties_py.RTModel.from_pretrained("ccs")


def test_rt_model_nonexistent_path_raises():
    import redeem_properties_py

    with pytest.raises(RuntimeError):
        redeem_properties_py.RTModel(
            model_path="/nonexistent/path/rt.pth",
            arch="rt_cnn_lstm",
            constants_path="/nonexistent/path/rt.pth.model_const.yaml",
        )


def test_ccs_model_nonexistent_path_raises():
    import redeem_properties_py

    with pytest.raises(RuntimeError):
        redeem_properties_py.CCSModel(
            model_path="/nonexistent/path/ccs.pth",
            arch="ccs_cnn_lstm",
            constants_path="/nonexistent/path/ccs.pth.model_const.yaml",
        )


def test_ms2_model_nonexistent_path_raises():
    import redeem_properties_py

    with pytest.raises(RuntimeError):
        redeem_properties_py.MS2Model(
            model_path="/nonexistent/path/ms2.pth",
            arch="ms2_bert",
            constants_path="/nonexistent/path/ms2.pth.model_const.yaml",
        )


if __name__ == "__main__":
    test_module_importable()
    test_classes_exist()
    test_rt_model_has_docstring()
    test_ccs_model_has_docstring()
    test_ms2_model_has_docstring()
    test_rt_model_has_from_pretrained()
    test_ccs_model_has_from_pretrained()
    test_ms2_model_has_from_pretrained()
    test_from_pretrained_unknown_name_raises()
    test_from_pretrained_wrong_family_raises()
    test_rt_model_nonexistent_path_raises()
    test_ccs_model_nonexistent_path_raises()
    test_ms2_model_nonexistent_path_raises()
    print("All tests passed.")
