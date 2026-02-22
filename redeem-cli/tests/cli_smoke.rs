//! CLI binary smoke tests using assert_cmd.
//!
//! These tests exercise the compiled `redeem` binary to verify that
//! argument parsing, help text, and error handling work end-to-end.

use assert_cmd::Command;
use predicates::prelude::*;

fn cmd() -> Command {
    Command::cargo_bin("redeem").unwrap()
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

#[test]
fn no_args_shows_help() {
    cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn help_flag() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("properties"))
        .stdout(predicate::str::contains("classifiers"));
}

#[test]
fn version_flag() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("redeem"));
}

// ---------------------------------------------------------------------------
// Properties subcommand
// ---------------------------------------------------------------------------

#[test]
fn properties_no_subcommand_errors() {
    // The CLI hits unreachable!() when no subcommand is given to `properties`
    cmd()
        .arg("properties")
        .assert()
        .failure();
}

#[test]
fn properties_train_no_config_errors() {
    cmd()
        .args(["properties", "train"])
        .assert()
        .failure();
}

#[test]
fn properties_train_nonexistent_config_errors() {
    cmd()
        .args(["properties", "train", "/nonexistent/config.json"])
        .assert()
        .failure();
}

#[test]
fn properties_inference_no_config_errors() {
    cmd()
        .args(["properties", "inference"])
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// Classifiers subcommand
// ---------------------------------------------------------------------------

#[test]
fn classifiers_no_subcommand_errors() {
    // The CLI hits unreachable!() when no subcommand is given to `classifiers`
    cmd()
        .arg("classifiers")
        .assert()
        .failure();
}

#[test]
fn classifiers_score_no_pin_errors() {
    cmd()
        .args(["classifiers", "score"])
        .assert()
        .failure();
}

#[test]
fn classifiers_score_nonexistent_pin_errors() {
    cmd()
        .args(["classifiers", "score", "/nonexistent/results.pin"])
        .assert()
        .failure();
}
