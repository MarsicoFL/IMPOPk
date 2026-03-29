use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use serde::Deserialize;

#[derive(Deserialize)]
struct Spec {
    name: String,
    workdir: String,
    script: String,
    rust_bin: String,
    tests: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    name: String,
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[test]
#[ignore] // Legacy shell script bin/jacquard/jacquard_coeffs.sh no longer exists
fn parity_specs() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let spec_dir = manifest.join("tests/parity");
    for entry in fs::read_dir(&spec_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let spec: Spec = toml::from_str(&fs::read_to_string(entry.path())?)?;
            run_spec(&manifest, &spec)?;
        }
    }
    Ok(())
}

fn run_spec(root: &Path, spec: &Spec) -> Result<(), Box<dyn Error>> {
    for case in &spec.tests {
        let script_out = run_script(root, spec, case)?;
        let rust_out = run_rust(root, spec, case)?;
        let script_stdout = normalize(&script_out.stdout);
        let rust_stdout = normalize(&rust_out.stdout);
        assert_eq!(
            script_stdout, rust_stdout,
            "stdout mismatch for spec '{}' case '{}':\nlegacy:\n{}\nrust:\n{}",
            spec.name, case.name, script_stdout, rust_stdout
        );

        let script_stderr = normalize(&script_out.stderr);
        let rust_stderr = normalize(&rust_out.stderr);
        assert_eq!(
            script_stderr, rust_stderr,
            "stderr mismatch for spec '{}' case '{}':\nlegacy:\n{}\nrust:\n{}",
            spec.name, case.name, script_stderr, rust_stderr
        );

        assert!(
            script_out.status.success() && rust_out.status.success(),
            "non-zero exit status for spec '{}' case '{}'",
            spec.name,
            case.name
        );
    }
    Ok(())
}

fn run_script(root: &Path, spec: &Spec, case: &TestCase) -> Result<std::process::Output, Box<dyn Error>> {
    let workdir = root.join(&spec.workdir);
    let script_path = workdir.join(&spec.script);
    let mut cmd = Command::new(script_path);
    cmd.current_dir(&workdir);
    cmd.args(&case.args);
    for (k, v) in &case.env {
        cmd.env(k, v);
    }
    Ok(cmd.output()?)
}

fn run_rust(root: &Path, spec: &Spec, case: &TestCase) -> Result<std::process::Output, Box<dyn Error>> {
    let workdir = root.join(&spec.workdir);
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin(&spec.rust_bin)?;
    cmd.current_dir(&workdir);
    cmd.args(&case.args);
    for (k, v) in &case.env {
        cmd.env(k, v);
    }
    Ok(cmd.output()?)
}

fn normalize(buf: &[u8]) -> String {
    let mut s = String::from_utf8_lossy(buf).replace("\r\n", "\n");
    while s.ends_with('\n') {
        s.pop();
    }
    s
}
