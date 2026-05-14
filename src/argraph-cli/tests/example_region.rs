//! Integration test against the bundled example GFA (chr12:60-60.01 Mb,
//! the panarg gold standard). Counts must match: 18 snp + 1 indel + 1 microsat.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../src/argraph-cli
    manifest.parent().unwrap().parent().unwrap().to_path_buf() // .../IMPOPk
}

fn example_gfa() -> PathBuf {
    workspace_root().join("data/examples/argraph/input/pangenome.gfa")
}

fn count_lines(output: &str) -> usize {
    output.lines().filter(|l| !l.is_empty()).count()
}

#[test]
fn stats_subcommand_finds_expected_counts() {
    let bin = assert_cmd::cargo::cargo_bin("argraph");
    let out = Command::new(bin)
        .args(["stats", "--gfa"])
        .arg(example_gfa())
        .output()
        .expect("argraph stats failed to spawn");
    assert!(out.status.success(), "stats failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Expected: 18 SNP, 1 indel, 1 microsat, 20 bubbles total
    let lines: Vec<&str> = stdout.lines().collect();
    let mut snp = 0u64;
    let mut indel = 0u64;
    let mut microsat = 0u64;
    let mut bubbles = 0u64;
    for l in &lines {
        let mut f = l.split('\t');
        let (k, v) = (f.next().unwrap_or(""), f.next().unwrap_or("0"));
        let v: u64 = v.parse().unwrap_or(0);
        match k {
            "snp" => snp = v,
            "indel" => indel = v,
            "microsat" => microsat = v,
            "bubbles" => bubbles = v,
            _ => {}
        }
    }
    assert_eq!(snp, 18, "SNPs: {}", stdout);
    assert_eq!(indel, 1, "indel: {}", stdout);
    assert_eq!(microsat, 1, "microsat: {}", stdout);
    assert_eq!(bubbles, 20, "bubbles: {}", stdout);
}

#[test]
fn classify_subcommand_emits_one_row_per_bubble() {
    let bin = assert_cmd::cargo::cargo_bin("argraph");
    let out = Command::new(bin)
        .args(["classify", "--gfa"])
        .arg(example_gfa())
        .args(["--output", "-"])
        .output()
        .expect("argraph classify failed to spawn");
    assert!(out.status.success(), "classify failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    // header + 20 rows
    assert_eq!(count_lines(&stdout), 21, "row count off:\n{}", stdout);
    assert!(stdout.lines().next().unwrap().starts_with("bubble_id\t"), "header missing");
}
