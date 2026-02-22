//! Integration tests for model_interface types and the CosineWithWarmup scheduler.

use redeem_properties::models::model_interface::PredictionResult;
use redeem_properties::utils::utils::CosineWithWarmup;
use redeem_properties::utils::utils::LRScheduler;

// ---------------------------------------------------------------------------
// PredictionResult
// ---------------------------------------------------------------------------

#[test]
fn rt_result_len() {
    let r = PredictionResult::RTResult(vec![1.0, 2.0, 3.0]);
    assert_eq!(r.len(), 3);
}

#[test]
fn ccs_result_len() {
    let r = PredictionResult::CCSResult(vec![400.0, 500.0]);
    assert_eq!(r.len(), 2);
}

#[test]
fn ms2_result_len() {
    let inner = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let r = PredictionResult::MS2Result(vec![inner.clone(), inner]);
    assert_eq!(r.len(), 2);
}

#[test]
fn prediction_result_len_zero() {
    let empty = PredictionResult::RTResult(vec![]);
    assert_eq!(empty.len(), 0);
    let non_empty = PredictionResult::CCSResult(vec![1.0]);
    assert_eq!(non_empty.len(), 1);
}

// ---------------------------------------------------------------------------
// CosineWithWarmup learning rate scheduler
// ---------------------------------------------------------------------------

#[test]
fn cosine_warmup_starts_at_zero() {
    let sched = CosineWithWarmup::new(1e-3, 100, 1000, 0.5);
    // At step 0 (before any step()) the LR should be 0 (warmup fraction = 0/100)
    let lr = sched.get_last_lr();
    assert!(lr.abs() < 1e-10, "LR at step 0 should be ~0, got {}", lr);
}

#[test]
fn cosine_warmup_reaches_peak() {
    let mut sched = CosineWithWarmup::new(1e-3, 10, 1000, 0.5);
    for _ in 0..10 {
        sched.step();
    }
    let lr = sched.get_last_lr();
    assert!(
        (lr - 1e-3).abs() < 1e-6,
        "LR after warmup should be ~1e-3, got {}",
        lr
    );
}

#[test]
fn cosine_warmup_decays_after_peak() {
    let mut sched = CosineWithWarmup::new(1e-3, 10, 100, 0.5);
    // Warmup
    for _ in 0..10 {
        sched.step();
    }
    let peak = sched.get_last_lr();

    // Step further into decay
    for _ in 0..50 {
        sched.step();
    }
    let later = sched.get_last_lr();
    assert!(
        later < peak,
        "LR should decay after warmup: peak={} later={}",
        peak,
        later
    );
}

#[test]
fn cosine_warmup_lr_never_negative() {
    let mut sched = CosineWithWarmup::new(1e-3, 10, 100, 0.5);
    for _ in 0..200 {
        sched.step();
        assert!(
            sched.get_last_lr() >= 0.0,
            "LR should never be negative at step"
        );
    }
}

#[test]
fn cosine_warmup_monotonic_during_warmup() {
    let mut sched = CosineWithWarmup::new(1e-3, 50, 1000, 0.5);
    let mut prev = 0.0;
    for _ in 0..50 {
        sched.step();
        let lr = sched.get_last_lr();
        assert!(lr >= prev, "LR should be non-decreasing during warmup");
        prev = lr;
    }
}
