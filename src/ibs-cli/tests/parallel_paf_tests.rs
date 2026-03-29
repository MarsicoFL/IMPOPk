//! Tests for parallel window processing in ibs-from-paf.
//!
//! Verifies that parallel processing with rayon produces identical results
//! to sequential processing, and that the threading infrastructure works
//! correctly.

use hprc_ibs::paf::{self, PafAlignment};
use rayon::prelude::*;
use std::collections::HashSet;

fn make_alignment(hap: &str, start: u64, end: u64, mismatches: Vec<u64>) -> PafAlignment {
    PafAlignment {
        hap_id: hap.to_string(),
        target_start: start,
        target_end: end,
        gap_identity: 1.0 - (mismatches.len() as f64 / (end - start).max(1) as f64),
        mismatch_positions: mismatches,
        aligned_bases: end - start,
    }
}

fn generate_test_alignments(n_haps: usize, region_len: u64) -> Vec<PafAlignment> {
    let mut alns = Vec::new();
    for i in 0..n_haps {
        let hap_id = format!("HAP{:03}#{}", i / 2, (i % 2) + 1);
        // Each haplotype has alignment covering the full region
        // with deterministic mismatches based on hap index
        let mut mismatches = Vec::new();
        let step = (region_len / 100).max(1);
        for pos in (0..region_len).step_by(step as usize) {
            // Different haplotypes have different mismatch patterns
            if (pos / step + i as u64) % 7 == 0 {
                mismatches.push(pos);
            }
        }
        alns.push(make_alignment(&hap_id, 0, region_len, mismatches));
    }
    alns.sort_by_key(|a| a.target_start);
    alns
}

// -----------------------------------------------------------------------
// Parallel vs sequential consistency
// -----------------------------------------------------------------------

#[test]
fn test_parallel_matches_sequential_small() {
    let alns = vec![
        make_alignment("A#1", 0, 1000, vec![100, 200, 300]),
        make_alignment("B#1", 0, 1000, vec![100, 400, 500]),
        make_alignment("C#1", 0, 1000, vec![200, 300, 600]),
    ];

    let windows: Vec<(u64, u64)> = (0..10).map(|i| (i * 100, (i + 1) * 100)).collect();

    // Sequential
    let sequential: Vec<Vec<paf::PairwiseIdentity>> = windows
        .iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    // Parallel
    let parallel: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    assert_eq!(sequential.len(), parallel.len());
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(
            seq.len(),
            par.len(),
            "Window {} has different pair counts: seq={}, par={}",
            i, seq.len(), par.len()
        );
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.group_a, p.group_a, "Window {}: group_a mismatch", i);
            assert_eq!(s.group_b, p.group_b, "Window {}: group_b mismatch", i);
            assert!(
                (s.identity - p.identity).abs() < 1e-10,
                "Window {}: identity mismatch: {} vs {}",
                i, s.identity, p.identity
            );
        }
    }
}

#[test]
fn test_parallel_matches_sequential_large() {
    // Simulate a realistic scenario: 20 haplotypes, 100kb region, 10kb windows
    let alns = generate_test_alignments(20, 100_000);
    let window_size = 10_000u64;
    let region_end = 100_000u64;

    let windows: Vec<(u64, u64)> = {
        let mut ws = Vec::new();
        let mut start = 0u64;
        while start < region_end {
            let end = (start + window_size).min(region_end);
            ws.push((start, end));
            start = end;
        }
        ws
    };

    let sequential: Vec<Vec<paf::PairwiseIdentity>> = windows
        .iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    let parallel: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    assert_eq!(sequential.len(), parallel.len());
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(seq.len(), par.len(), "Window {} pair count mismatch", i);
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.group_a, p.group_a);
            assert_eq!(s.group_b, p.group_b);
            assert!((s.identity - p.identity).abs() < 1e-10);
        }
    }
}

#[test]
fn test_parallel_with_query_ref_filter() {
    let alns = vec![
        make_alignment("Q1#1", 0, 1000, vec![50, 150]),
        make_alignment("Q2#1", 0, 1000, vec![50, 250]),
        make_alignment("R1#1", 0, 1000, vec![150, 350]),
        make_alignment("R2#1", 0, 1000, vec![250, 450]),
    ];

    let mut qf = HashSet::new();
    qf.insert("Q1".to_string());
    qf.insert("Q2".to_string());
    let mut rf = HashSet::new();
    rf.insert("R1".to_string());
    rf.insert("R2".to_string());

    let windows: Vec<(u64, u64)> = (0..10).map(|i| (i * 100, (i + 1) * 100)).collect();

    let sequential: Vec<Vec<paf::PairwiseIdentity>> = windows
        .iter()
        .map(|&(s, e)| {
            paf::compute_window_pairwise(&alns, s, e, "CHM13", Some(&qf), Some(&rf), 0.0)
        })
        .collect();

    let parallel: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| {
            paf::compute_window_pairwise(&alns, s, e, "CHM13", Some(&qf), Some(&rf), 0.0)
        })
        .collect();

    assert_eq!(sequential.len(), parallel.len());
    for (i, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
        assert_eq!(seq.len(), par.len(), "Window {} pair count mismatch", i);
        for (s, p) in seq.iter().zip(par.iter()) {
            assert_eq!(s.group_a, p.group_a);
            assert_eq!(s.group_b, p.group_b);
            assert!((s.identity - p.identity).abs() < 1e-10);
        }
    }

    // Verify only cross-set pairs are emitted
    for pairs in &parallel {
        for pair in pairs {
            let a_is_query =
                qf.contains(paf::extract_sample_from_hap(&pair.group_a));
            let b_is_ref =
                rf.contains(paf::extract_sample_from_hap(&pair.group_b));
            let a_is_ref =
                rf.contains(paf::extract_sample_from_hap(&pair.group_a));
            let b_is_query =
                qf.contains(paf::extract_sample_from_hap(&pair.group_b));
            assert!(
                (a_is_query && b_is_ref) || (a_is_ref && b_is_query),
                "Pair {}-{} is not a cross-set pair",
                pair.group_a, pair.group_b
            );
        }
    }
}

#[test]
fn test_parallel_preserves_order() {
    // Verify that par_iter().collect() preserves window order
    let alns = generate_test_alignments(10, 50_000);
    let windows: Vec<(u64, u64)> = (0..50).map(|i| (i * 1000, (i + 1) * 1000)).collect();

    let results: Vec<(u64, u64, usize)> = windows
        .par_iter()
        .map(|&(s, e)| {
            let pairs = paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0);
            (s, e, pairs.len())
        })
        .collect();

    // Verify order matches input window order
    for (i, &(s, e, _)) in results.iter().enumerate() {
        assert_eq!(s, windows[i].0, "Window {} start mismatch", i);
        assert_eq!(e, windows[i].1, "Window {} end mismatch", i);
    }
}

#[test]
fn test_parallel_empty_windows() {
    let alns = vec![
        make_alignment("A#1", 500, 600, vec![510, 520]),
    ];

    // Windows that don't overlap any alignment
    let windows: Vec<(u64, u64)> = vec![(0, 100), (100, 200), (200, 300), (700, 800)];

    let results: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    // All windows should produce empty results (only 1 haplotype = no pairs)
    for (i, pairs) in results.iter().enumerate() {
        assert!(pairs.is_empty(), "Window {} should have no pairs", i);
    }
}

#[test]
fn test_parallel_cutoff_filter() {
    let alns = vec![
        make_alignment("A#1", 0, 1000, (0..50).map(|i| i * 20).collect()),
        make_alignment("B#1", 0, 1000, vec![]),
    ];

    let windows: Vec<(u64, u64)> = (0..10).map(|i| (i * 100, (i + 1) * 100)).collect();

    // High cutoff should filter most pairs
    let high_cutoff: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.99))
        .collect();

    // Low cutoff should keep all pairs
    let low_cutoff: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    let high_total: usize = high_cutoff.iter().map(|p| p.len()).sum();
    let low_total: usize = low_cutoff.iter().map(|p| p.len()).sum();
    assert!(high_total <= low_total, "High cutoff should produce fewer or equal pairs");
    assert!(low_total > 0, "Low cutoff should produce some pairs");
}

#[test]
fn test_parallel_deterministic() {
    // Run the same parallel computation multiple times and verify identical results
    let alns = generate_test_alignments(30, 50_000);
    let windows: Vec<(u64, u64)> = (0..50).map(|i| (i * 1000, (i + 1) * 1000)).collect();

    let run1: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    let run2: Vec<Vec<paf::PairwiseIdentity>> = windows
        .par_iter()
        .map(|&(s, e)| paf::compute_window_pairwise(&alns, s, e, "CHM13", None, None, 0.0))
        .collect();

    for (i, (r1, r2)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(r1.len(), r2.len(), "Window {} pair count differs between runs", i);
        for (p1, p2) in r1.iter().zip(r2.iter()) {
            assert_eq!(p1.group_a, p2.group_a);
            assert_eq!(p1.group_b, p2.group_b);
            assert!((p1.identity - p2.identity).abs() < 1e-15);
        }
    }
}

// -----------------------------------------------------------------------
// CLI --threads flag
// -----------------------------------------------------------------------

#[test]
fn test_from_paf_threads_flag_accepted() {
    use assert_cmd::Command;
    let dir = tempfile::tempdir().unwrap();
    let paf = dir.path().join("test.paf");
    std::fs::write(&paf, "").unwrap();
    let output = dir.path().join("out.tsv");

    // --threads should be accepted (even if computation fails due to empty PAF)
    let cmd = Command::cargo_bin("ibs-from-paf")
        .unwrap()
        .args([
            "-a", paf.to_str().unwrap(),
            "--region", "chr1:1-1000",
            "--size", "100",
            "--output", output.to_str().unwrap(),
            "--threads", "2",
        ])
        .output()
        .unwrap();

    // Should succeed (empty PAF = no alignments, writes header only)
    assert!(cmd.status.success(), "Exit status: {:?}, stderr: {}", cmd.status, String::from_utf8_lossy(&cmd.stderr));
}

#[test]
fn test_from_paf_threads_zero_uses_default() {
    use assert_cmd::Command;
    let dir = tempfile::tempdir().unwrap();
    let paf = dir.path().join("test.paf");
    std::fs::write(&paf, "").unwrap();
    let output = dir.path().join("out.tsv");

    let cmd = Command::cargo_bin("ibs-from-paf")
        .unwrap()
        .args([
            "-a", paf.to_str().unwrap(),
            "--region", "chr1:1-1000",
            "--size", "100",
            "--output", output.to_str().unwrap(),
            "--threads", "0",
        ])
        .output()
        .unwrap();

    assert!(cmd.status.success());
    let stderr = String::from_utf8_lossy(&cmd.stderr);
    assert!(stderr.contains("threads"), "Should report thread count");
}

// -----------------------------------------------------------------------
// Batch window generation
// -----------------------------------------------------------------------

#[test]
fn test_window_generation_covers_region() {
    let region_start = 1u64;
    let region_end = 133324548u64;
    let window_size = 10000u64;

    let windows: Vec<(u64, u64)> = {
        let mut ws = Vec::new();
        let mut start = region_start;
        while start <= region_end {
            let end = (start + window_size - 1).min(region_end);
            ws.push((start, end));
            start = end + 1;
        }
        ws
    };

    // First window starts at region_start
    assert_eq!(windows[0].0, region_start);
    // Last window ends at region_end
    assert_eq!(windows.last().unwrap().1, region_end);
    // Windows are contiguous (no gaps)
    for i in 1..windows.len() {
        assert_eq!(
            windows[i].0,
            windows[i - 1].1 + 1,
            "Gap between windows {} and {}",
            i - 1, i
        );
    }
    // Expected count
    let expected = (region_end - region_start + window_size) / window_size;
    assert_eq!(windows.len() as u64, expected);
}

#[test]
fn test_window_generation_single_window() {
    let windows: Vec<(u64, u64)> = {
        let mut ws = Vec::new();
        let mut start = 1u64;
        let region_end = 500u64;
        let window_size = 10000u64;
        while start <= region_end {
            let end = (start + window_size - 1).min(region_end);
            ws.push((start, end));
            start = end + 1;
        }
        ws
    };

    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0], (1, 500));
}

#[test]
fn test_window_generation_exact_fit() {
    let windows: Vec<(u64, u64)> = {
        let mut ws = Vec::new();
        let mut start = 1u64;
        let region_end = 100u64;
        let window_size = 10u64;
        while start <= region_end {
            let end = (start + window_size - 1).min(region_end);
            ws.push((start, end));
            start = end + 1;
        }
        ws
    };

    assert_eq!(windows.len(), 10);
    assert_eq!(windows[0], (1, 10));
    assert_eq!(windows[9], (91, 100));
}
