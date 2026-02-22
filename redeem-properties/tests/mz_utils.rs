//! Integration tests for mz_utils: precursor/product m/z computation.

use redeem_properties::utils::mz_utils::{
    compute_peptide_mz_info, compute_precursor_mz, compute_product_mzs, match_product_mzs,
};

// ---------------------------------------------------------------------------
// Precursor m/z
// ---------------------------------------------------------------------------

#[test]
fn precursor_mz_simple_peptide() {
    // PEPTIDE at charge 2: monoisotopic mass ≈ 799.36 Da → m/z ≈ 400.69
    let mz = compute_precursor_mz("PEPTIDE", 2).unwrap();
    assert!(mz > 399.0 && mz < 402.0, "PEPTIDE+2 m/z = {}", mz);
}

#[test]
fn precursor_mz_increases_with_mass() {
    let mz_short = compute_precursor_mz("PEP", 2).unwrap();
    let mz_long = compute_precursor_mz("PEPTIDEK", 2).unwrap();
    assert!(mz_long > mz_short, "longer peptide should have higher m/z");
}

#[test]
fn precursor_mz_decreases_with_charge() {
    let mz_z2 = compute_precursor_mz("PEPTIDE", 2).unwrap();
    let mz_z3 = compute_precursor_mz("PEPTIDE", 3).unwrap();
    assert!(
        mz_z2 > mz_z3,
        "higher charge should give lower m/z: z2={} z3={}",
        mz_z2,
        mz_z3
    );
}

#[test]
fn precursor_mz_modified_peptide() {
    let mz_unmod = compute_precursor_mz("PEPTMIDE", 2).unwrap();
    let mz_ox = compute_precursor_mz("PEPTM[+15.9949]IDE", 2).unwrap();
    assert!(
        (mz_ox - mz_unmod - 15.9949 / 2.0).abs() < 0.05,
        "oxidation should add ~8 Da to z2: unmod={} ox={}",
        mz_unmod,
        mz_ox
    );
}

#[test]
fn precursor_mz_zero_charge_errors() {
    assert!(compute_precursor_mz("PEPTIDE", 0).is_err());
}

#[test]
fn precursor_mz_negative_charge_errors() {
    assert!(compute_precursor_mz("PEPTIDE", -1).is_err());
}

// ---------------------------------------------------------------------------
// Product m/z
// ---------------------------------------------------------------------------

#[test]
fn product_mzs_contain_b_and_y_ions() {
    let ions = compute_product_mzs("PEPTIDE", 1).unwrap();
    let has_b = ions.iter().any(|i| i.ion_type == "b");
    let has_y = ions.iter().any(|i| i.ion_type == "y");
    assert!(has_b, "should contain b ions");
    assert!(has_y, "should contain y ions");
}

#[test]
fn product_mzs_all_positive() {
    let ions = compute_product_mzs("AGHCEWQMKYR", 2).unwrap();
    for ion in &ions {
        assert!(ion.mz > 0.0, "negative m/z: {:?}", ion);
        assert!(ion.ordinal >= 1, "ordinal < 1: {:?}", ion);
        assert!(ion.charge >= 1, "charge < 1: {:?}", ion);
    }
}

#[test]
fn product_mzs_higher_max_charge_gives_more_ions() {
    let ions_z1 = compute_product_mzs("PEPTIDE", 1).unwrap();
    let ions_z2 = compute_product_mzs("PEPTIDE", 2).unwrap();
    assert!(
        ions_z2.len() >= ions_z1.len(),
        "z2 ({}) should have >= ions than z1 ({})",
        ions_z2.len(),
        ions_z1.len()
    );
}

#[test]
fn product_mzs_invalid_charge_errors() {
    assert!(compute_product_mzs("PEPTIDE", 0).is_err());
}

// ---------------------------------------------------------------------------
// Combined info
// ---------------------------------------------------------------------------

#[test]
fn peptide_mz_info_has_both_precursor_and_products() {
    let info = compute_peptide_mz_info("PEPTIDE", 2, 1).unwrap();
    assert!(info.precursor_mz > 0.0);
    assert!(!info.product_ions.is_empty());
}

// ---------------------------------------------------------------------------
// match_product_mzs
// ---------------------------------------------------------------------------

#[test]
fn match_product_mzs_finds_known_ions() {
    let ions = compute_product_mzs("PEPTIDE", 2).unwrap();
    let types = vec!["b".to_string(), "y".to_string()];
    let charges = vec![1, 1];
    let ordinals = vec![2, 3];

    let mzs = match_product_mzs(&ions, &types, &charges, &ordinals);
    assert_eq!(mzs.len(), 2);
    assert!(!mzs[0].is_nan(), "b2+1 should match");
    assert!(!mzs[1].is_nan(), "y3+1 should match");
}

#[test]
fn match_product_mzs_returns_nan_for_missing() {
    let ions = compute_product_mzs("PEP", 1).unwrap();
    // Ordinal 99 should not exist for a 3-residue peptide
    let types = vec!["b".to_string()];
    let charges = vec![1];
    let ordinals = vec![99];

    let mzs = match_product_mzs(&ions, &types, &charges, &ordinals);
    assert!(mzs[0].is_nan(), "should be NaN for missing ordinal");
}
