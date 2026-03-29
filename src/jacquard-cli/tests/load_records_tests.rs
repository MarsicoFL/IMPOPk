//! Tests for load_records and run() validation in jacquard-cli main.rs
//!
//! Tests the file I/O parsing logic, header detection, sorting, and
//! edge cases. Uses the binary for haplotype validation error messages.

use std::fs;
use std::process::Command;
use tempfile::TempDir;

// ── load_records reimplementation for unit testing ─────────────────────

#[derive(Clone, Debug)]
struct Record {
    chrom: String,
    start: i64,
    end: i64,
    hap_a: String,
    hap_b: String,
}

fn load_records(path: &std::path::Path) -> anyhow::Result<Vec<Record>> {
    use anyhow::{bail, Context};
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut line_index = 0_usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if line_index == 0 {
            let first = line.split('\t').next().unwrap_or("");
            if first.eq_ignore_ascii_case("chrom") {
                line_index += 1;
                continue;
            }
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            bail!("incomplete IBS row at line {}", line_index + 1);
        }
        let start: i64 = fields[1]
            .parse()
            .with_context(|| format!("invalid start position on line {}", line_index + 1))?;
        let end: i64 = fields[2]
            .parse()
            .with_context(|| format!("invalid end position on line {}", line_index + 1))?;
        records.push(Record {
            chrom: fields[0].to_string(),
            start,
            end,
            hap_a: fields[3].to_string(),
            hap_b: fields[4].to_string(),
        });
        line_index += 1;
    }

    records.sort_by(|a, b| {
        a.chrom
            .cmp(&b.chrom)
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.end.cmp(&b.end))
    });
    Ok(records)
}

#[test]
fn load_records_basic_with_header() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(
        &path,
        "chrom\tstart\tend\tgroup.a\tgroup.b\tidentity\n\
         chr1\t100\t200\tA#1\tB#1\t0.99\n\
         chr1\t200\t300\tA#2\tB#2\t0.98\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].chrom, "chr1");
    assert_eq!(records[0].start, 100);
    assert_eq!(records[0].end, 200);
    assert_eq!(records[0].hap_a, "A#1");
    assert_eq!(records[0].hap_b, "B#1");
}

#[test]
fn load_records_without_header() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    // No "chrom" header — first line is data
    fs::write(
        &path,
        "chr1\t100\t200\tA#1\tB#1\t0.99\n\
         chr1\t200\t300\tA#2\tB#2\t0.98\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 2);
}

#[test]
fn load_records_case_insensitive_header() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(
        &path,
        "CHROM\tSTART\tEND\tGROUP.A\tGROUP.B\tIDENTITY\n\
         chr1\t100\t200\tA#1\tB#1\t0.99\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 1);
}

#[test]
fn load_records_empty_lines_skipped() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(
        &path,
        "chrom\tstart\tend\tgroup.a\tgroup.b\n\
         \n\
         chr1\t100\t200\tA#1\tB#1\n\
         \n\
         chr1\t300\t400\tA#2\tB#2\n\
         \n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 2);
}

#[test]
fn load_records_empty_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(&path, "").unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 0);
}

#[test]
fn load_records_too_few_fields_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(&path, "chr1\t100\t200\tA#1\n").unwrap(); // only 4 fields

    assert!(load_records(&path).is_err());
}

#[test]
fn load_records_invalid_start_position_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(&path, "chr1\tabc\t200\tA#1\tB#1\n").unwrap();

    assert!(load_records(&path).is_err());
}

#[test]
fn load_records_invalid_end_position_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(&path, "chr1\t100\txyz\tA#1\tB#1\n").unwrap();

    assert!(load_records(&path).is_err());
}

#[test]
fn load_records_nonexistent_file_errors() {
    assert!(load_records(std::path::Path::new("/no/such/file.tsv")).is_err());
}

#[test]
fn load_records_sorted_by_chrom_then_start() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    // Unsorted input
    fs::write(
        &path,
        "chr2\t500\t600\tA#1\tB#1\n\
         chr1\t300\t400\tA#1\tB#1\n\
         chr1\t100\t200\tA#1\tB#1\n\
         chr2\t100\t200\tA#1\tB#1\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 4);
    // Should be sorted: chr1:100, chr1:300, chr2:100, chr2:500
    assert_eq!(records[0].chrom, "chr1");
    assert_eq!(records[0].start, 100);
    assert_eq!(records[1].chrom, "chr1");
    assert_eq!(records[1].start, 300);
    assert_eq!(records[2].chrom, "chr2");
    assert_eq!(records[2].start, 100);
    assert_eq!(records[3].chrom, "chr2");
    assert_eq!(records[3].start, 500);
}

#[test]
fn load_records_sorted_by_end_tiebreaker() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(
        &path,
        "chr1\t100\t300\tA#1\tB#1\n\
         chr1\t100\t200\tA#1\tB#1\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records[0].end, 200); // shorter end first
    assert_eq!(records[1].end, 300);
}

#[test]
fn load_records_extra_columns_accepted() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(
        &path,
        "chr1\t100\t200\tA#1\tB#1\t0.999\textra1\textra2\n",
    )
    .unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records.len(), 1);
}

#[test]
fn load_records_negative_coordinates() {
    // Negative coordinates are valid i64 values but semantically wrong.
    // load_records should parse them without error (the run() function validates later).
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ibs.tsv");
    fs::write(&path, "chr1\t-100\t200\tA#1\tB#1\n").unwrap();

    let records = load_records(&path).unwrap();
    assert_eq!(records[0].start, -100);
}

// ── hap_key reimplementation ───────────────────────────────────────────

fn hap_key(raw: &str) -> String {
    let mut parts = raw.split('#');
    match (parts.next(), parts.next()) {
        (Some(sample), Some(hap)) => format!("{}#{}", sample, hap),
        _ => raw.to_string(),
    }
}

#[test]
fn hap_key_strips_scaffold_suffix() {
    assert_eq!(hap_key("HG00096#1#scaffold:0-5000"), "HG00096#1");
}

#[test]
fn hap_key_preserves_two_part() {
    assert_eq!(hap_key("HG00096#1"), "HG00096#1");
}

#[test]
fn hap_key_no_hash_returns_raw() {
    assert_eq!(hap_key("nohash"), "nohash");
}

// ── CLI validation tests (via binary) ──────────────────────────────────

fn jacquard_binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_jacquard"))
}

#[test]
fn cli_duplicate_hap_a_rejected() {
    let dir = TempDir::new().unwrap();
    let ibs = dir.path().join("ibs.tsv");
    fs::write(&ibs, "chrom\tstart\tend\tgroup.a\tgroup.b\n").unwrap();

    let output = jacquard_binary()
        .arg("--ibs").arg(&ibs)
        .arg("--hap-a1").arg("X#1")
        .arg("--hap-a2").arg("X#1")  // duplicate
        .arg("--hap-b1").arg("Y#1")
        .arg("--hap-b2").arg("Y#2")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("distinct"), "Expected distinct error, got: {}", stderr);
}

#[test]
fn cli_overlapping_groups_rejected() {
    let dir = TempDir::new().unwrap();
    let ibs = dir.path().join("ibs.tsv");
    fs::write(&ibs, "chrom\tstart\tend\tgroup.a\tgroup.b\n").unwrap();

    let output = jacquard_binary()
        .arg("--ibs").arg(&ibs)
        .arg("--hap-a1").arg("X#1")
        .arg("--hap-a2").arg("X#2")
        .arg("--hap-b1").arg("X#1")  // overlaps with group A
        .arg("--hap-b2").arg("Y#2")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("distinct") || stderr.contains("overlap"),
        "Expected overlap error, got: {}", stderr);
}

#[test]
fn cli_empty_ibs_file_rejected() {
    let dir = TempDir::new().unwrap();
    let ibs = dir.path().join("ibs.tsv");
    fs::write(&ibs, "chrom\tstart\tend\tgroup.a\tgroup.b\n").unwrap();

    let output = jacquard_binary()
        .arg("--ibs").arg(&ibs)
        .arg("--hap-a1").arg("A#1")
        .arg("--hap-a2").arg("A#2")
        .arg("--hap-b1").arg("B#1")
        .arg("--hap-b2").arg("B#2")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no data rows"), "Expected 'no data rows', got: {}", stderr);
}

#[test]
fn cli_nonexistent_ibs_file_rejected() {
    let output = jacquard_binary()
        .arg("--ibs").arg("/nonexistent/file.tsv")
        .arg("--hap-a1").arg("A#1")
        .arg("--hap-a2").arg("A#2")
        .arg("--hap-b1").arg("B#1")
        .arg("--hap-b2").arg("B#2")
        .output()
        .unwrap();

    assert!(!output.status.success());
}
