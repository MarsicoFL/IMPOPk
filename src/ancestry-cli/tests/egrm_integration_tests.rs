//! Integration tests for eGRM output (PairwiseAccumulator, GCTA binary format).
//!
//! Targets: edge cases in similarity parsing, GCTA file format correctness,
//! centering arithmetic, and extract_sample_id parsing.
//! All tests use only the public API: parse_similarity_for_egrm + write_gcta_grm.
//!
//! Cycle 62 — testing agent.

use impopk_ancestry_cli::{parse_similarity_for_egrm, write_gcta_grm};
use std::io::Write;
use tempfile::NamedTempFile;

fn create_sim_file(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{}", content).unwrap();
    f.flush().unwrap();
    f
}

/// Helper: parse sim file and write GRM, return (bin_data, n_bin_data, id_content)
fn parse_and_write(content: &str, center: bool) -> (Vec<u8>, Vec<u8>, String) {
    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("grm");
    let prefix_str = prefix.to_str().unwrap();

    write_gcta_grm(&acc, prefix_str, center).unwrap();

    let bin = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    let n_bin = std::fs::read(format!("{}.grm.N.bin", prefix_str)).unwrap();
    let ids = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
    (bin, n_bin, ids)
}

fn read_f32(data: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(data[index * 4..(index + 1) * 4].try_into().unwrap())
}

// ── Parsing edge cases ──

#[test]
fn missing_group_a_column_returns_error() {
    let content = "query.name\tstart\tend\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tB#1#chr1:0-10000\t0.99\n";
    let f = create_sim_file(content);
    let result = parse_similarity_for_egrm(f.path(), "estimated.identity");
    assert!(result.is_err());
}

#[test]
fn missing_group_b_column_returns_error() {
    let content = "query.name\tstart\tend\tgroup.a\testimated.identity\n\
                   chr1\t0\t10000\tA#1#chr1:0-10000\t0.99\n";
    let f = create_sim_file(content);
    let result = parse_similarity_for_egrm(f.path(), "estimated.identity");
    assert!(result.is_err());
}

#[test]
fn missing_identity_column_returns_error() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\n\
                   chr1\t0\t10000\tA#1#chr1:0-10000\tB#1#chr1:0-10000\n";
    let f = create_sim_file(content);
    let result = parse_similarity_for_egrm(f.path(), "estimated.identity");
    assert!(result.is_err());
}

#[test]
fn custom_identity_column_name_works() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\tmy_custom_id\n\
                   chr1\t0\t10000\tA#1#chr1:0-10000\tB#1#chr1:0-10000\t0.995\n";
    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "my_custom_id").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("custom");
    let prefix_str = prefix.to_str().unwrap();

    write_gcta_grm(&acc, prefix_str, false).unwrap();

    let ids = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
    assert_eq!(ids.trim().lines().count(), 2);
}

#[test]
fn parse_nonexistent_file_returns_error() {
    let result = parse_similarity_for_egrm(
        std::path::Path::new("/nonexistent/similarity.tsv"),
        "estimated.identity",
    );
    assert!(result.is_err());
}

#[test]
fn header_only_file_write_fails() {
    // Empty accumulator → write_gcta_grm should fail with "No samples"
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n";
    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("empty");
    let result = write_gcta_grm(&acc, prefix.to_str().unwrap(), false);
    assert!(result.is_err());
}

#[test]
fn self_comparisons_same_haplotype_skipped() {
    // Same sample#hap → skipped. Only cross-haplotype pair kept.
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tHG00733#1#chr1:0-10000\tHG00733#1#chr1:5000-15000\t1.000\n\
                   chr1\t0\t10000\tHG00733#1#chr1:0-10000\tHG00733#2#chr1:0-10000\t0.998\n";
    let (bin, _, ids) = parse_and_write(content, false);

    // 2 samples → 3 entries
    assert_eq!(bin.len(), 12);
    assert_eq!(ids.trim().lines().count(), 2);
}

#[test]
fn different_haplotypes_same_individual_kept() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tHG00733#1#chr1:0-10000\tHG00733#2#chr1:0-10000\t0.998\n";
    let (bin, _, ids) = parse_and_write(content, false);
    assert_eq!(ids.trim().lines().count(), 2);
    assert_eq!(bin.len(), 12); // 2 samples → 3 lower-triangle entries
}

#[test]
fn multiple_windows_averaged_in_output() {
    // 3 windows for same pair → average should appear in GRM binary
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#chr1:0-10000\tB#1#chr1:0-10000\t0.990\n\
                   chr1\t10000\t20000\tA#1#chr1:10000-20000\tB#1#chr1:10000-20000\t0.994\n\
                   chr1\t20000\t30000\tA#1#chr1:20000-30000\tB#1#chr1:20000-30000\t0.998\n";
    let (bin, n_bin, _) = parse_and_write(content, false);

    // [1,0] position = index 1 in lower triangle
    let avg = read_f32(&bin, 1);
    assert!((avg - 0.994).abs() < 1e-5, "expected ~0.994, got {}", avg);

    // Count at [1,0] should be 3
    let count = read_f32(&n_bin, 1);
    assert!((count - 3.0).abs() < 1e-5, "expected 3 windows, got {}", count);
}

// ── extract_sample_id behavior (tested indirectly) ──

#[test]
fn sample_id_extraction_standard_format() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tHG00733#1#chr1:0-10000\tNA19240#2#chr1:0-10000\t0.993\n";
    let (_, _, ids) = parse_and_write(content, false);
    assert_eq!(ids.trim().lines().count(), 2);
    // FID should be the sample name without haplotype
    assert!(ids.contains("HG00733"));
    assert!(ids.contains("NA19240"));
}

#[test]
fn sample_id_extraction_no_hash() {
    // If group name has no #, it's used as-is
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tsampleA\tsampleB\t0.99\n";
    let (_, _, ids) = parse_and_write(content, false);
    assert_eq!(ids.trim().lines().count(), 2);
    assert!(ids.contains("sampleA"));
    assert!(ids.contains("sampleB"));
}

// ── GCTA GRM output format ──

#[test]
fn gcta_grm_file_sizes_correct_3_samples() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.99\n\
                   chr1\t0\t10000\tA#1#c\tC#1#c\t0.98\n\
                   chr1\t0\t10000\tB#1#c\tC#1#c\t0.97\n";
    let (bin, n_bin, ids) = parse_and_write(content, false);

    let n = 3;
    let entries = n * (n + 1) / 2; // 6
    assert_eq!(bin.len(), entries * 4);
    assert_eq!(n_bin.len(), entries * 4);
    assert_eq!(ids.trim().lines().count(), n);
}

#[test]
fn gcta_id_file_format_fid_tab_iid() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tHG00733#1#c\tNA19240#2#c\t0.99\n";
    let (_, _, ids) = parse_and_write(content, false);

    for line in ids.trim().lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 2, "expected FID\\tIID: {}", line);
    }
}

#[test]
fn gcta_grm_two_samples_binary_values() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.995\n";
    let (bin, _, _) = parse_and_write(content, false);

    // Lower triangle for 2 samples: [0,0], [1,0], [1,1]
    assert_eq!(bin.len(), 12);

    let v00 = read_f32(&bin, 0); // diagonal, no data → 0.0
    let v10 = read_f32(&bin, 1); // A vs B → 0.995
    let v11 = read_f32(&bin, 2); // diagonal, no data → 0.0

    assert!((v00 - 0.0).abs() < 1e-5, "v00={}", v00);
    assert!((v10 - 0.995).abs() < 1e-5, "v10={}", v10);
    assert!((v11 - 0.0).abs() < 1e-5, "v11={}", v11);
}

#[test]
fn gcta_grm_centering_subtracts_grand_mean() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.995\n";
    let (bin, _, _) = parse_and_write(content, true);

    // Grand mean = 0.995 (only 1 entry with data)
    // [0,0] = 0.0 - 0.995 = -0.995
    // [1,0] = 0.995 - 0.995 = 0.0
    // [1,1] = 0.0 - 0.995 = -0.995
    let v00 = read_f32(&bin, 0);
    let v10 = read_f32(&bin, 1);
    let v11 = read_f32(&bin, 2);

    assert!((v00 - (-0.995)).abs() < 1e-4, "v00={}", v00);
    assert!((v10 - 0.0).abs() < 1e-4, "v10={}", v10);
    assert!((v11 - (-0.995)).abs() < 1e-4, "v11={}", v11);
}

#[test]
fn centering_vs_uncentered_differ() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.99\n\
                   chr1\t0\t10000\tA#1#c\tC#1#c\t0.98\n\
                   chr1\t0\t10000\tB#1#c\tC#1#c\t0.97\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let p_uc = dir.path().join("uc");
    let p_c = dir.path().join("c");

    write_gcta_grm(&acc, p_uc.to_str().unwrap(), false).unwrap();
    write_gcta_grm(&acc, p_c.to_str().unwrap(), true).unwrap();

    let uc = std::fs::read(format!("{}.grm.bin", p_uc.to_str().unwrap())).unwrap();
    let c = std::fs::read(format!("{}.grm.bin", p_c.to_str().unwrap())).unwrap();

    assert_ne!(uc, c, "centered and uncentered should differ");
}

#[test]
fn count_file_correct_values() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.99\n\
                   chr1\t10000\t20000\tA#1#c\tB#1#c\t0.98\n\
                   chr1\t20000\t30000\tA#1#c\tB#1#c\t0.97\n";
    let (_, n_bin, _) = parse_and_write(content, false);

    // 2 samples → 3 entries: [0,0]=0, [1,0]=3, [1,1]=0
    let c00 = read_f32(&n_bin, 0);
    let c10 = read_f32(&n_bin, 1);
    let c11 = read_f32(&n_bin, 2);

    assert!((c00 - 0.0).abs() < 1e-5);
    assert!((c10 - 3.0).abs() < 1e-5);
    assert!((c11 - 0.0).abs() < 1e-5);
}

// ── Scaling ──

#[test]
fn ten_samples_file_sizes_scale() {
    let mut content = String::from("query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n");
    for i in 0..10u32 {
        for j in (i + 1)..10 {
            content.push_str(&format!(
                "chr1\t0\t10000\tS{}#1#c\tS{}#1#c\t0.99{}\n",
                i, j, i % 10
            ));
        }
    }

    let f = create_sim_file(&content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("scale");
    let prefix_str = prefix.to_str().unwrap();

    write_gcta_grm(&acc, prefix_str, false).unwrap();

    let n = 10usize;
    let entries = n * (n + 1) / 2; // 55
    let bin_size = std::fs::metadata(format!("{}.grm.bin", prefix_str)).unwrap().len();
    let id_lines = std::fs::read_to_string(format!("{}.grm.id", prefix_str))
        .unwrap()
        .trim()
        .lines()
        .count();

    assert_eq!(bin_size, (entries * 4) as u64);
    assert_eq!(id_lines, n);
}

#[test]
fn centering_full_matrix_entries_sum_near_zero() {
    // With 4 samples, all pairs have data. Centering should make mean ≈ 0.
    let mut content = String::from("query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n");
    let identities = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94];
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    for (k, &(i, j)) in pairs.iter().enumerate() {
        content.push_str(&format!(
            "chr1\t0\t10000\tS{}#1#c\tS{}#1#c\t{}\n",
            i, j, identities[k]
        ));
    }

    let (bin_c, _, _) = parse_and_write(&content, true);

    // Read all 10 entries (4 samples → 4*5/2 = 10) and compute sum of data-bearing ones
    // Off-diagonal positions that have data: (1,0), (2,0), (3,0), (2,1), (3,1), (3,2)
    let data_positions: Vec<usize> = vec![
        1 * (1 + 1) / 2 + 0, // (1,0) = 1
        2 * (2 + 1) / 2 + 0, // (2,0) = 3
        3 * (3 + 1) / 2 + 0, // (3,0) = 6
        2 * (2 + 1) / 2 + 1, // (2,1) = 4
        3 * (3 + 1) / 2 + 1, // (3,1) = 7
        3 * (3 + 1) / 2 + 2, // (3,2) = 8
    ];

    let mut sum = 0.0f64;
    for &pos in &data_positions {
        sum += read_f32(&bin_c, pos) as f64;
    }
    let mean = sum / data_positions.len() as f64;
    assert!(
        mean.abs() < 1e-4,
        "mean of centered data entries = {}", mean
    );
}

// ── Invalid data ──

#[test]
fn non_parseable_identity_returns_error() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\tnot_a_number\n";
    let f = create_sim_file(content);
    let result = parse_similarity_for_egrm(f.path(), "estimated.identity");
    assert!(result.is_err());
}

#[test]
fn nan_identity_does_not_panic() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\tNaN\n";
    let f = create_sim_file(content);
    // NaN is parseable. The code should not panic.
    let _result = parse_similarity_for_egrm(f.path(), "estimated.identity");
}

// ── Single sample edge ──

#[test]
fn only_self_comparisons_produces_zero_samples() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#chr1:0-10000\tA#1#chr1:5000-15000\t1.000\n";
    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("self_only");
    let result = write_gcta_grm(&acc, prefix.to_str().unwrap(), false);
    assert!(result.is_err()); // 0 samples → error
}

// ── Bidirectional pairs coalesce ──

#[test]
fn bidirectional_pairs_averaged_together() {
    // A→B and B→A in the input should both contribute to the same cell
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.990\n\
                   chr1\t0\t10000\tB#1#c\tA#1#c\t0.998\n";
    let (bin, n_bin, _) = parse_and_write(content, false);

    // Average: (0.990 + 0.998) / 2 = 0.994
    let v10 = read_f32(&bin, 1);
    assert!((v10 - 0.994).abs() < 1e-5, "v10={}", v10);

    // Count should be 2
    let c10 = read_f32(&n_bin, 1);
    assert!((c10 - 2.0).abs() < 1e-5);
}

// ── Multi-chromosome accumulation ──

#[test]
fn many_pairs_all_unique_samples() {
    // Each pair involves a unique pair of samples
    let mut content = String::from("query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n");
    for i in 0..5u32 {
        for j in (i + 1)..5 {
            content.push_str(&format!(
                "chr1\t0\t10000\tS{}#1#c\tS{}#1#c\t{}\n",
                i, j, 0.99 - 0.001 * (i * 5 + j) as f64
            ));
        }
    }

    let f = create_sim_file(&content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("multi");
    write_gcta_grm(&acc, prefix.to_str().unwrap(), false).unwrap();

    // 5 samples → 15 lower-triangle entries
    let bin = std::fs::read(format!("{}.grm.bin", prefix.to_str().unwrap())).unwrap();
    assert_eq!(bin.len(), 15 * 4);
}
