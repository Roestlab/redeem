//! Integration tests for Experiment construction and data_handling.

use redeem_classifiers::data_handling::{Experiment, PsmMetadata};
use redeem_classifiers::math::{Array1, Array2};

fn make_metadata(n: usize) -> PsmMetadata {
    PsmMetadata {
        spec_id: (0..n).map(|i| format!("spec_{}", i)).collect(),
        file_id: vec![0; n],
        feature_names: vec!["f1".to_string(), "f2".to_string()],
        scan_nr: None,
        exp_mass: None,
    }
}

// ---------------------------------------------------------------------------
// Experiment construction
// ---------------------------------------------------------------------------

#[test]
fn experiment_new_valid() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    let y = Array1::from_vec(vec![1, -1, 1, -1]);
    let meta = make_metadata(4);
    let exp = Experiment::new(x, y, meta);
    assert!(exp.is_ok());
}

#[test]
fn experiment_new_dimension_mismatch() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    let y = Array1::from_vec(vec![1, -1]); // wrong length
    let meta = make_metadata(4);
    let exp = Experiment::new(x, y, meta);
    assert!(exp.is_err(), "should error on dimension mismatch");
}

#[test]
fn experiment_new_single_class_target_only() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0; 6]).unwrap();
    let y = Array1::from_vec(vec![1, 1, 1]); // no decoys
    let meta = make_metadata(3);
    let exp = Experiment::new(x, y, meta);
    assert!(exp.is_err(), "should error with only targets");
}

#[test]
fn experiment_new_single_class_decoy_only() {
    let x = Array2::from_shape_vec((3, 2), vec![1.0; 6]).unwrap();
    let y = Array1::from_vec(vec![-1, -1, -1]); // no targets
    let meta = make_metadata(3);
    let exp = Experiment::new(x, y, meta);
    assert!(exp.is_err(), "should error with only decoys");
}

#[test]
fn experiment_fields_initialized() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
    let y = Array1::from_vec(vec![1, -1, 1, -1]);
    let meta = make_metadata(4);
    let exp = Experiment::new(x, y, meta).unwrap();

    assert_eq!(exp.is_train.len(), 4);
    assert_eq!(exp.is_top_peak.len(), 4);
    assert_eq!(exp.tg_num_id.len(), 4);
    assert_eq!(exp.classifier_score.len(), 4);

    // All is_train should start as false
    for v in exp.is_train.iter() {
        assert!(!v);
    }
}
