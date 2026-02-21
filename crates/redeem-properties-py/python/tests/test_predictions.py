import pytest
import numpy as np
import pandas as pd
import polars as pl
import redeem_properties as rp

TEST_PEPTIDES = [
    "AKPLMELIER",
    "TEM[+15.9949]VTISDASQR",
    "AGKFPSLLTHNENMVAK",
    "LSELDDRADALQAGASQFETSAAK",
    "FLLQDTVELR",
    "SVTEQGAELSNEER",
    "EHALLAYTLGVK",
    "TVQSLEIDLDSM[+15.9949]R",
    "VVSQYSSLLSPMSVNAVM[+15.9949]K",
    "TFLALINQVFPAEEDSKK",
]

@pytest.fixture(scope="module")
def rt_model():
    return rp.RTModel.from_pretrained("rt")

@pytest.fixture(scope="module")
def ccs_model():
    return rp.CCSModel.from_pretrained("ccs")

@pytest.fixture(scope="module")
def ms2_model():
    return rp.MS2Model.from_pretrained("ms2")

@pytest.fixture(scope="module")
def property_prediction(rt_model, ccs_model, ms2_model):
    return rp.PropertyPrediction(rt_model=rt_model, ccs_model=ccs_model, ms2_model=ms2_model)

def test_rt_model_predict(rt_model):
    results = rt_model.predict(TEST_PEPTIDES)
    assert len(results) == len(TEST_PEPTIDES)
    for res in results:
        assert isinstance(res, (float, np.floating))

def test_rt_model_predict_df(rt_model):
    df = rt_model.predict_df(TEST_PEPTIDES, framework="pandas")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(TEST_PEPTIDES)
    assert "peptide" in df.columns
    assert "rt" in df.columns

def test_ccs_model_predict(ccs_model):
    charges = [2] * len(TEST_PEPTIDES)
    results = ccs_model.predict(TEST_PEPTIDES, charges=charges)
    assert len(results) == len(TEST_PEPTIDES)
    for res in results:
        assert isinstance(res, dict)
        assert "ccs" in res

def test_ccs_model_predict_df(ccs_model):
    charges = [2] * len(TEST_PEPTIDES)
    df = ccs_model.predict_df(TEST_PEPTIDES, charges=charges, framework="polars", annotate_mobility=True)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == len(TEST_PEPTIDES)
    assert "peptide" in df.columns
    assert "ccs" in df.columns
    assert "ion_mobility" in df.columns

def test_ms2_model_predict(ms2_model):
    charges = [2] * len(TEST_PEPTIDES)
    nces = [25.0] * len(TEST_PEPTIDES)
    instruments = ["Lumos"] * len(TEST_PEPTIDES)
    results = ms2_model.predict(TEST_PEPTIDES, charges=charges, nces=nces, instruments=instruments)
    assert len(results) == len(TEST_PEPTIDES)
    for res in results:
        assert isinstance(res, dict)
        assert "intensities" in res
        assert "ion_types" in res
        assert "ion_charges" in res

def test_ms2_model_predict_df(ms2_model):
    charges = [2] * len(TEST_PEPTIDES)
    nces = [25.0] * len(TEST_PEPTIDES)
    instruments = ["Lumos"] * len(TEST_PEPTIDES)
    df = ms2_model.predict_df(TEST_PEPTIDES, charges=charges, nces=nces, instruments=instruments, framework="pandas", annotate_mz=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > len(TEST_PEPTIDES) # Should be expanded
    assert "peptide" in df.columns
    assert "intensity" in df.columns
    assert "mz" in df.columns
    assert "precursor_mz" in df.columns

def test_property_prediction_predict_df(property_prediction):
    charges = [2] * len(TEST_PEPTIDES)
    nces = [25.0] * len(TEST_PEPTIDES)
    instruments = ["Lumos"] * len(TEST_PEPTIDES)
    df = property_prediction.predict_df(
        TEST_PEPTIDES, 
        charges=charges, 
        nces=nces, 
        instruments=instruments, 
        framework="pandas", 
        annotate_mz=True, 
        annotate_mobility=True
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > len(TEST_PEPTIDES)
    assert "peptide" in df.columns
    assert "rt" in df.columns
    assert "ccs" in df.columns
    assert "ion_mobility" in df.columns
    assert "intensity" in df.columns
    assert "mz" in df.columns
    assert "precursor_mz" in df.columns

def test_property_prediction_predict_df_no_ms2(rt_model, ccs_model):
    property_prediction_no_ms2 = rp.PropertyPrediction(rt_model=rt_model, ccs_model=ccs_model, ms2_model=None, predict_ms2=False)
    charges = [2] * len(TEST_PEPTIDES)
    df = property_prediction_no_ms2.predict_df(
        TEST_PEPTIDES, 
        charges=charges, 
        framework="pandas", 
        annotate_mz=True, 
        annotate_mobility=True
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(TEST_PEPTIDES)
    assert "peptide" in df.columns
    assert "rt" in df.columns
    assert "ccs" in df.columns
    assert "ion_mobility" in df.columns
    assert "precursor_mz" in df.columns
    assert "intensity" not in df.columns
    assert "mz" not in df.columns

def test_utils():
    mz = rp.compute_precursor_mz("AKPLMELIER", 2)
    assert isinstance(mz, float)
    assert mz > 0

    frag_mzs = rp.compute_fragment_mzs("AKPLMELIER", 2)
    assert isinstance(frag_mzs, dict)
    assert len(frag_mzs) > 0

    mob = rp.ccs_to_mobility(400.0, 2.0, 500.0)
    assert isinstance(mob, float)
    assert mob > 0
