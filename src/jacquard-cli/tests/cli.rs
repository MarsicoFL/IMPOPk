//! Integration tests for the jacquard CLI binary.

use assert_cmd::Command;

/// Test that `jacquard --help` runs successfully
#[test]
fn test_jacquard_help() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("jacquard").unwrap();
    cmd.arg("--help");
    cmd.assert().success();
}

/// Test that `jacquard` fails without required arguments
#[test]
fn test_jacquard_missing_args() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("jacquard").unwrap();
    cmd.assert().failure();
}
