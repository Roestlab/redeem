"""Tests for the m/z utility functions exposed via redeem_properties.

These exercise the pure-computation functions that do NOT require any pretrained
model files – they just need the compiled Rust extension.
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROTON_MASS = 1.007276


def _approx(val, *, rel=1e-4, abs=1e-6):
    return pytest.approx(val, rel=rel, abs=abs)


# ---------------------------------------------------------------------------
# compute_precursor_mz
# ---------------------------------------------------------------------------


class TestComputePrecursorMz:
    """Tests for redeem_properties.compute_precursor_mz."""

    def test_simple_peptide_charge2(self):
        from redeem_properties import compute_precursor_mz

        mz = compute_precursor_mz("PEPTIDE", 2)
        # "PEPTIDE" neutral mass ≈ 799.3600, so (mass + 2*proton) / 2 ≈ 400.6877
        assert isinstance(mz, float)
        assert mz > 0.0
        assert 399.0 < mz < 402.0

    def test_simple_peptide_charge1(self):
        from redeem_properties import compute_precursor_mz

        mz1 = compute_precursor_mz("PEPTIDE", 1)
        mz2 = compute_precursor_mz("PEPTIDE", 2)
        # charge-1 should be roughly 2*mz2 - proton
        assert mz1 > mz2

    def test_different_charges(self):
        from redeem_properties import compute_precursor_mz

        mz_vals = [compute_precursor_mz("PEPTIDE", z) for z in range(1, 5)]
        # Higher charge ⇒ lower m/z
        for a, b in zip(mz_vals, mz_vals[1:]):
            assert a > b

    def test_modified_peptide(self):
        """A modification should change the m/z relative to the unmodified form."""
        from redeem_properties import compute_precursor_mz

        mz_plain = compute_precursor_mz("PEPTIDE", 2)
        mz_mod = compute_precursor_mz("PEPT[+79.9663]IDE", 2)
        assert mz_mod > mz_plain

    def test_invalid_sequence_raises(self):
        from redeem_properties import compute_precursor_mz

        with pytest.raises(RuntimeError):
            compute_precursor_mz("", 2)


# ---------------------------------------------------------------------------
# compute_fragment_mzs
# ---------------------------------------------------------------------------


class TestComputeFragmentMzs:
    """Tests for redeem_properties.compute_fragment_mzs."""

    def test_returns_dict_with_expected_keys(self):
        from redeem_properties import compute_fragment_mzs

        result = compute_fragment_mzs("PEPTIDE", 1)
        assert isinstance(result, dict)
        for key in ("ion_types", "charges", "ordinals", "mzs"):
            assert key in result, f"missing key '{key}'"

    def test_lengths_consistent(self):
        from redeem_properties import compute_fragment_mzs

        result = compute_fragment_mzs("PEPTIDE", 1)
        n = len(result["mzs"])
        assert n > 0
        assert len(result["ion_types"]) == n
        assert len(result["charges"]) == n
        assert len(result["ordinals"]) == n

    def test_ion_types_are_b_or_y(self):
        from redeem_properties import compute_fragment_mzs

        result = compute_fragment_mzs("PEPTIDE", 2)
        assert set(result["ion_types"]).issubset({"b", "y"})

    def test_charge_range(self):
        from redeem_properties import compute_fragment_mzs

        max_charge = 3
        result = compute_fragment_mzs("PEPTIDE", max_charge)
        assert all(1 <= c <= max_charge for c in result["charges"])

    def test_mz_positive(self):
        from redeem_properties import compute_fragment_mzs

        result = compute_fragment_mzs("PEPTIDE", 1)
        assert all(m > 0 for m in result["mzs"])

    def test_higher_max_charge_gives_more_fragments(self):
        from redeem_properties import compute_fragment_mzs

        r1 = compute_fragment_mzs("PEPTIDE", 1)
        r2 = compute_fragment_mzs("PEPTIDE", 2)
        assert len(r2["mzs"]) > len(r1["mzs"])


# ---------------------------------------------------------------------------
# compute_peptide_mz_info  (precursor + fragments in one call)
# ---------------------------------------------------------------------------


class TestComputePeptideMzInfo:
    """Tests for redeem_properties.compute_peptide_mz_info."""

    def test_returns_dict_with_precursor_mz(self):
        from redeem_properties import compute_peptide_mz_info

        result = compute_peptide_mz_info("PEPTIDE", 2, 1)
        assert "precursor_mz" in result
        assert isinstance(result["precursor_mz"], float)
        assert result["precursor_mz"] > 0

    def test_precursor_matches_standalone(self):
        from redeem_properties import compute_precursor_mz, compute_peptide_mz_info

        standalone = compute_precursor_mz("PEPTIDE", 2)
        combined = compute_peptide_mz_info("PEPTIDE", 2, 1)
        assert combined["precursor_mz"] == _approx(standalone)

    def test_fragment_keys_present(self):
        from redeem_properties import compute_peptide_mz_info

        result = compute_peptide_mz_info("PEPTIDE", 2, 2)
        for key in ("ion_types", "charges", "ordinals", "mzs"):
            assert key in result


# ---------------------------------------------------------------------------
# match_fragment_mzs
# ---------------------------------------------------------------------------


class TestMatchFragmentMzs:
    """Tests for redeem_properties.match_fragment_mzs."""

    def test_round_trip(self):
        """Matching theoretical fragments back to themselves should give exact mzs."""
        from redeem_properties import compute_fragment_mzs, match_fragment_mzs

        frags = compute_fragment_mzs("PEPTIDE", 1)
        matched = match_fragment_mzs(
            "PEPTIDE",
            1,
            frags["ion_types"],
            frags["charges"],
            frags["ordinals"],
        )
        for orig, recon in zip(frags["mzs"], matched):
            assert orig == _approx(recon)

    def test_unmatched_returns_nan(self):
        """A nonsense ordinal should give NaN."""
        from redeem_properties import match_fragment_mzs

        matched = match_fragment_mzs(
            "PEPTIDE",
            1,
            ["b"],
            [1],
            [9999],  # ordinal beyond peptide length
        )
        assert len(matched) == 1
        assert math.isnan(matched[0])


# ---------------------------------------------------------------------------
# ccs_to_mobility
# ---------------------------------------------------------------------------


class TestCcsToMobility:
    """Tests for redeem_properties.ccs_to_mobility (Bruker conversion)."""

    def test_positive_result(self):
        from redeem_properties import ccs_to_mobility

        inv_k0 = ccs_to_mobility(300.0, 2.0, 500.0)
        assert isinstance(inv_k0, float)
        assert inv_k0 > 0

    def test_higher_ccs_gives_higher_mobility(self):
        from redeem_properties import ccs_to_mobility

        low = ccs_to_mobility(200.0, 2.0, 500.0)
        high = ccs_to_mobility(400.0, 2.0, 500.0)
        # Higher CCS ⇒ larger 1/K₀
        assert high > low

    def test_higher_charge_effect(self):
        from redeem_properties import ccs_to_mobility

        z1 = ccs_to_mobility(300.0, 1.0, 500.0)
        z2 = ccs_to_mobility(300.0, 2.0, 500.0)
        # Different charges should give different 1/K₀
        assert z1 != z2
