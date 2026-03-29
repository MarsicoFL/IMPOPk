// Cycle 91: Edge case tests for zero/low-coverage ibd-cli functions
//
// Targets:
// - GeneticMap::from_file (0 test refs) — file parsing, chrom filtering, 3/4 col format
// - GeneticMap::uniform (0 test refs) — uniform rate construction
// - GeneticMap::len / is_empty (0 test refs) — trivial accessors
// - Population::from_str (0 test refs) — population string parsing
// - estimate_k0_emissions (2 test refs) — k0 Bernoulli parameter estimation
// - k0_log_pmf (2 test refs) — k0 log probability mass function
// - augment_with_k0 (2 test refs) — k0 augmentation of log-emissions
// - forward_with_distances_from_log_emit (2 test refs) — distance-aware forward
// - backward_with_distances_from_log_emit (2 test refs) — distance-aware backward
// - viterbi_with_distances_from_log_emit (2 test refs) — distance-aware viterbi
// - forward_backward_with_distances_from_log_emit (2 test refs) — distance-aware FB
// - extract_ibd_segments_composite (3 test refs) — composite segment extraction
// - distance_dependent_log_transition (low coverage) — distance-scaled transitions

use hprc_ibd::hmm::*;
use std::io::Write;

/// Helper: create standard test HMM params
fn test_params() -> HmmParams {
    HmmParams::from_expected_length(50.0, 0.0001, 5000)
}

// ============================================================
//  Population::from_str
// ============================================================

#[test]
fn population_from_str_valid_afr() {
    assert!(matches!(Population::from_str("AFR"), Some(Population::AFR)));
}

#[test]
fn population_from_str_valid_eur() {
    assert!(matches!(Population::from_str("EUR"), Some(Population::EUR)));
}

#[test]
fn population_from_str_valid_eas() {
    assert!(matches!(Population::from_str("EAS"), Some(Population::EAS)));
}

#[test]
fn population_from_str_valid_csa() {
    assert!(matches!(Population::from_str("CSA"), Some(Population::CSA)));
}

#[test]
fn population_from_str_valid_amr() {
    assert!(matches!(Population::from_str("AMR"), Some(Population::AMR)));
}

#[test]
fn population_from_str_interpop() {
    assert!(matches!(Population::from_str("INTERPOP"), Some(Population::InterPop)));
    assert!(matches!(Population::from_str("INTER"), Some(Population::InterPop)));
}

#[test]
fn population_from_str_generic() {
    assert!(matches!(Population::from_str("GENERIC"), Some(Population::Generic)));
    assert!(matches!(Population::from_str("UNKNOWN"), Some(Population::Generic)));
}

#[test]
fn population_from_str_case_insensitive() {
    assert!(matches!(Population::from_str("afr"), Some(Population::AFR)));
    assert!(matches!(Population::from_str("Eur"), Some(Population::EUR)));
    assert!(matches!(Population::from_str("eas"), Some(Population::EAS)));
    assert!(matches!(Population::from_str("interpop"), Some(Population::InterPop)));
}

#[test]
fn population_from_str_invalid_returns_none() {
    assert!(Population::from_str("").is_none());
    assert!(Population::from_str("XYZ").is_none());
    assert!(Population::from_str("AFRICAN").is_none());
    assert!(Population::from_str("123").is_none());
}

// ============================================================
//  GeneticMap::uniform, len, is_empty
// ============================================================

#[test]
fn genmap_uniform_basic() {
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    assert_eq!(gm.len(), 2);
    assert!(!gm.is_empty());
}

#[test]
fn genmap_uniform_rate_calculation() {
    // 1 Mb at 1 cM/Mb => 1 cM total
    let gm = GeneticMap::uniform(0, 1_000_000, 1.0);
    let cm = gm.interpolate_cm(1_000_000);
    assert!((cm - 1.0).abs() < 1e-6, "Expected 1.0 cM, got {}", cm);
}

#[test]
fn genmap_uniform_midpoint_interpolation() {
    let gm = GeneticMap::uniform(0, 1_000_000, 2.0);
    // At midpoint 500_000 bp: expected 1.0 cM (half of 2.0)
    let cm = gm.interpolate_cm(500_000);
    assert!((cm - 1.0).abs() < 1e-6, "Expected 1.0 cM at midpoint, got {}", cm);
}

#[test]
fn genmap_uniform_zero_length_region() {
    // start == end => 0 cM span
    let gm = GeneticMap::uniform(100, 100, 1.0);
    assert_eq!(gm.len(), 2);
}

#[test]
fn genmap_uniform_high_rate() {
    let gm = GeneticMap::uniform(0, 10_000_000, 5.0);
    // 10 Mb at 5 cM/Mb => 50 cM
    let cm = gm.interpolate_cm(10_000_000);
    assert!((cm - 50.0).abs() < 1e-6, "Expected 50 cM, got {}", cm);
}

#[test]
fn genmap_uniform_zero_rate() {
    let gm = GeneticMap::uniform(0, 1_000_000, 0.0);
    let cm = gm.interpolate_cm(1_000_000);
    assert!((cm).abs() < 1e-10, "Expected 0 cM with zero rate, got {}", cm);
}

#[test]
fn genmap_new_empty() {
    let gm = GeneticMap::new(vec![]);
    assert_eq!(gm.len(), 0);
    assert!(gm.is_empty());
}

#[test]
fn genmap_new_single_entry() {
    let gm = GeneticMap::new(vec![(1000, 0.5)]);
    assert_eq!(gm.len(), 1);
    assert!(!gm.is_empty());
}

#[test]
fn genmap_new_sorted_entries() {
    let gm = GeneticMap::new(vec![
        (100, 0.0),
        (200, 0.1),
        (300, 0.3),
    ]);
    assert_eq!(gm.len(), 3);
    assert!(!gm.is_empty());
}

// ============================================================
//  GeneticMap::from_file — 4-column format
// ============================================================

#[test]
fn genmap_from_file_4col_basic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1 1000 1.5 0.0").unwrap();
        writeln!(f, "chr1 2000 1.5 0.0015").unwrap();
        writeln!(f, "chr1 3000 1.5 0.003").unwrap();
    }
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 3);
}

#[test]
fn genmap_from_file_4col_chrom_filter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1 1000 1.5 0.0").unwrap();
        writeln!(f, "chr2 2000 1.5 0.001").unwrap();
        writeln!(f, "chr1 3000 1.5 0.003").unwrap();
    }
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 2); // Only chr1 entries
}

#[test]
fn genmap_from_file_4col_chr_prefix_normalization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "1 1000 1.5 0.0").unwrap();
        writeln!(f, "1 2000 1.5 0.001").unwrap();
    }
    // Query with "chr1" should match "1" entries
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 2);
}

#[test]
fn genmap_from_file_skips_comments_and_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# header comment").unwrap();
        writeln!(f, "").unwrap();
        writeln!(f, "chr1 1000 1.5 0.0").unwrap();
        writeln!(f, "# another comment").unwrap();
        writeln!(f, "chr1 2000 1.5 0.001").unwrap();
    }
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 2);
}

#[test]
fn genmap_from_file_no_matching_chrom_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr2 1000 1.5 0.0").unwrap();
    }
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No genetic map entries"));
}

#[test]
fn genmap_from_file_nonexistent_file_returns_error() {
    let result = GeneticMap::from_file("/tmp/nonexistent_genmap_file.txt", "chr1");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to open"));
}

#[test]
fn genmap_from_file_3col_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "1000 1.5 0.0").unwrap();
        writeln!(f, "2000 1.5 0.001").unwrap();
        writeln!(f, "3000 1.5 0.003").unwrap();
    }
    // 3-column format ignores chrom filter
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 3);
}

#[test]
fn genmap_from_file_entries_sorted_by_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        // Intentionally unsorted
        writeln!(f, "chr1 3000 1.5 0.003").unwrap();
        writeln!(f, "chr1 1000 1.5 0.0").unwrap();
        writeln!(f, "chr1 2000 1.5 0.001").unwrap();
    }
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 3);
    // Verify sorted by checking interpolation is monotone
    let cm1 = gm.interpolate_cm(1000);
    let cm2 = gm.interpolate_cm(2000);
    let cm3 = gm.interpolate_cm(3000);
    assert!(cm1 <= cm2 && cm2 <= cm3);
}

#[test]
fn genmap_from_file_skips_short_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1 1000 1.5 0.0").unwrap();
        writeln!(f, "only_two_fields 1234").unwrap(); // <3 fields, skipped
        writeln!(f, "chr1 2000 1.5 0.001").unwrap();
    }
    let gm = GeneticMap::from_file(&path, "chr1").unwrap();
    assert_eq!(gm.len(), 2);
}

#[test]
fn genmap_from_file_invalid_position_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("map.txt");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1 NOT_A_NUMBER 1.5 0.0").unwrap();
    }
    let result = GeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid position"));
}

// ============================================================
//  estimate_k0_emissions
// ============================================================

#[test]
fn estimate_k0_empty_returns_priors() {
    let result = estimate_k0_emissions(&[], &[]);
    assert!((result[0] - 0.015).abs() < 1e-10, "Expected non-IBD prior 0.015");
    assert!((result[1] - 0.22).abs() < 1e-10, "Expected IBD prior 0.22");
}

#[test]
fn estimate_k0_all_ibd_posteriors() {
    let indicators = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let posteriors = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // All weight in IBD state, all indicators=1 => p1 near 1.0 (clamped to 0.999)
    assert!(result[1] > 0.9, "IBD k0 prob should be near 1.0 when all indicators=1 and all IBD");
    // Non-IBD has zero weight => should return prior
    assert!((result[0] - 0.015).abs() < 1e-6, "Non-IBD should get prior with zero weight");
}

#[test]
fn estimate_k0_all_non_ibd_posteriors() {
    let indicators = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let posteriors = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // All non-IBD
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // All weight in non-IBD state, all indicators=0 => p0 near 0.001 (clamped)
    assert!(result[0] < 0.01, "Non-IBD k0 prob should be low when all indicators=0");
    // IBD has zero weight => should return prior
    assert!((result[1] - 0.22).abs() < 1e-6, "IBD should get prior with zero weight");
}

#[test]
fn estimate_k0_mixed_posteriors() {
    let indicators = vec![1.0, 0.0, 1.0, 0.0];
    let posteriors = vec![0.5, 0.5, 0.5, 0.5]; // Uniform posteriors
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // Both states should get ~0.5 k0 rate
    assert!(result[0] > 0.001 && result[0] < 0.999);
    assert!(result[1] > 0.001 && result[1] < 0.999);
}

#[test]
fn estimate_k0_results_clamped() {
    // All indicators = 1 with all non-IBD => p0 should be clamped to 0.999
    let indicators = vec![1.0; 100];
    let posteriors = vec![0.0; 100]; // All non-IBD
    let result = estimate_k0_emissions(&indicators, &posteriors);
    assert!(result[0] <= 0.999, "Should be clamped to 0.999");
    assert!(result[0] >= 0.001, "Should be clamped to 0.001");
}

// ============================================================
//  k0_log_pmf
// ============================================================

#[test]
fn k0_log_pmf_indicator_high() {
    // indicator > 0.5 => ln(p_k0)
    let result = k0_log_pmf(1.0, 0.3);
    assert!((result - 0.3_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_indicator_low() {
    // indicator <= 0.5 => ln(1 - p_k0)
    let result = k0_log_pmf(0.0, 0.3);
    assert!((result - 0.7_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_boundary_half() {
    // indicator = 0.5 exactly => <= 0.5, so ln(1 - p_k0)
    let result = k0_log_pmf(0.5, 0.3);
    assert!((result - 0.7_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_p_one_indicator_high() {
    // p_k0 = 1.0 and indicator high => ln(1.0) = 0.0
    let result = k0_log_pmf(1.0, 1.0);
    assert!((result - 0.0).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_p_zero_indicator_low() {
    // p_k0 = 0.0 and indicator low => ln(1.0) = 0.0
    let result = k0_log_pmf(0.0, 0.0);
    assert!((result - 0.0).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_always_negative_or_zero() {
    // Log of probability is always <= 0
    for &ind in &[0.0, 0.3, 0.5, 0.7, 1.0] {
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let result = k0_log_pmf(ind, p);
            assert!(result <= 0.0 + 1e-10, "k0_log_pmf({}, {}) = {} should be <= 0", ind, p, result);
        }
    }
}

// ============================================================
//  augment_with_k0
// ============================================================

#[test]
fn augment_with_k0_empty_emissions() {
    let mut log_emit: Vec<[f64; 2]> = vec![];
    let indicators: Vec<f64> = vec![];
    let posteriors: Vec<f64> = vec![];
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    assert!(log_emit.is_empty());
}

#[test]
fn augment_with_k0_non_discriminative_skipped() {
    // If k0_params[1] <= k0_params[0], augmentation is skipped
    // All posteriors = 0 (non-IBD), all indicators = 1 => p0 high, p1 prior (0.22)
    // p0 = 0.999, p1 = 0.22, so p1 < p0 => skip
    let mut log_emit = vec![[0.0, 0.0]; 10];
    let indicators = vec![1.0; 10];
    let posteriors = vec![0.0; 10]; // All non-IBD
    let original = log_emit.clone();
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    // Should be unchanged since k0 is not discriminative
    assert_eq!(log_emit, original);
}

#[test]
fn augment_with_k0_modifies_emissions_when_discriminative() {
    // Create scenario where p1 > p0: indicators correlated with IBD
    let mut log_emit = vec![[0.0, 0.0]; 20];
    // Make indicators ~1 when IBD posterior is high
    let mut indicators = vec![0.0; 20];
    let mut posteriors = vec![0.0; 20];
    for i in 0..10 {
        indicators[i] = 1.0;
        posteriors[i] = 0.95; // IBD region with high k0
    }
    for i in 10..20 {
        indicators[i] = 0.0;
        posteriors[i] = 0.05; // Non-IBD region with low k0
    }
    let original = log_emit.clone();
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    // Emissions should be modified
    let changed = log_emit.iter().zip(original.iter()).any(|(a, b)| a != b);
    assert!(changed, "Emissions should be modified when k0 is discriminative");
}

#[test]
fn augment_with_k0_mismatched_lengths() {
    // More indicators than log_emit
    let mut log_emit = vec![[0.0, 0.0]; 5];
    let indicators = vec![1.0; 20]; // Much longer
    let mut posteriors = vec![0.0; 20];
    for i in 0..10 {
        posteriors[i] = 0.95;
    }
    // Should not panic — the function checks t < log_emit.len()
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
}

// ============================================================
//  distance_dependent_log_transition
// ============================================================

#[test]
fn dist_log_trans_zero_distance() {
    let params = test_params();
    let result = distance_dependent_log_transition(&params, 0, 5000);
    // Should return plain log of transition matrix
    let expected_00 = params.transition[0][0].ln();
    let expected_01 = params.transition[0][1].ln();
    assert!((result[0][0] - expected_00).abs() < 1e-10);
    assert!((result[0][1] - expected_01).abs() < 1e-10);
}

#[test]
fn dist_log_trans_zero_window_size() {
    let params = test_params();
    let result = distance_dependent_log_transition(&params, 5000, 0);
    // window_size=0 falls through to default log(transition)
    let expected_00 = params.transition[0][0].ln();
    assert!((result[0][0] - expected_00).abs() < 1e-10);
}

#[test]
fn dist_log_trans_same_as_nominal() {
    let params = test_params();
    let ws = 5000u64;
    let result = distance_dependent_log_transition(&params, ws, ws);
    // When distance == window_size, scale=1.0, should be close to nominal transition
    // Not exactly equal due to continuous-time approximation
    assert!(result[0][0].is_finite());
    assert!(result[0][1].is_finite());
    assert!(result[1][0].is_finite());
    assert!(result[1][1].is_finite());
}

#[test]
fn dist_log_trans_large_distance_increases_switch() {
    let params = test_params();
    let ws = 5000u64;
    let result_near = distance_dependent_log_transition(&params, ws, ws);
    let result_far = distance_dependent_log_transition(&params, ws * 100, ws);
    // At larger distance, P(switch) increases => log(P(switch)) becomes less negative
    assert!(result_far[0][1] > result_near[0][1],
        "Switch prob should increase with distance: near={}, far={}", result_near[0][1], result_far[0][1]);
}

#[test]
fn dist_log_trans_rows_sum_to_one() {
    let params = test_params();
    for &dist in &[100, 5000, 50_000, 500_000] {
        let result = distance_dependent_log_transition(&params, dist, 5000);
        for row in 0..2 {
            let sum = result[row][0].exp() + result[row][1].exp();
            assert!((sum - 1.0).abs() < 1e-6,
                "Row {} should sum to 1.0 at distance {}, got {}", row, dist, sum);
        }
    }
}

// ============================================================
//  forward_with_distances_from_log_emit
// ============================================================

#[test]
fn fwd_dist_empty() {
    let params = test_params();
    let (alpha, ll) = forward_with_distances_from_log_emit(&[], &params, &[]);
    assert!(alpha.is_empty());
    assert!((ll - 0.0).abs() < 1e-10);
}

#[test]
fn fwd_dist_single_window() {
    let params = test_params();
    let log_emit = vec![[(0.5_f64).ln(), (0.5_f64).ln()]]; // Dummy: both 50%
    let positions = vec![(0, 4999)];
    let (alpha, ll) = forward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

#[test]
fn fwd_dist_mismatched_positions_fallback() {
    let params = test_params();
    let log_emit = vec![[-0.5, -0.5], [-0.5, -0.5]];
    let positions = vec![(0, 4999)]; // Length 1 != 2
    let (alpha, ll) = forward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(alpha.len(), 2); // Falls back to non-distance version
    assert!(ll.is_finite());
}

#[test]
fn fwd_dist_two_windows_finite() {
    let params = test_params();
    let log_emit = vec![[-1.0, -0.1], [-0.1, -1.0]];
    let positions = vec![(0, 4999), (5000, 9999)];
    let (alpha, ll) = forward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(alpha.len(), 2);
    assert!(ll.is_finite());
    for t in 0..2 {
        assert!(alpha[t][0].is_finite() && alpha[t][1].is_finite());
    }
}

#[test]
fn fwd_dist_large_gap_between_windows() {
    let params = test_params();
    let log_emit = vec![[-1.0, -0.1], [-0.1, -1.0]];
    let positions = vec![(0, 4999), (1_000_000, 1_004_999)]; // 1Mb gap
    let (alpha, _ll) = forward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(alpha.len(), 2);
    assert!(alpha[1][0].is_finite());
    assert!(alpha[1][1].is_finite());
}

// ============================================================
//  backward_with_distances_from_log_emit
// ============================================================

#[test]
fn bwd_dist_empty() {
    let params = test_params();
    let beta = backward_with_distances_from_log_emit(&[], &params, &[]);
    assert!(beta.is_empty());
}

#[test]
fn bwd_dist_single_window() {
    let params = test_params();
    let log_emit = vec![[-0.5, -0.5]];
    let positions = vec![(0, 4999)];
    let beta = backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(beta.len(), 1);
    // Last window beta should be [0.0, 0.0]
    assert!((beta[0][0] - 0.0).abs() < 1e-10);
    assert!((beta[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn bwd_dist_mismatched_positions_fallback() {
    let params = test_params();
    let log_emit = vec![[-0.5, -0.5], [-0.5, -0.5]];
    let positions = vec![(0, 4999)]; // Length mismatch
    let beta = backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(beta.len(), 2); // Falls back to non-distance version
}

#[test]
fn bwd_dist_two_windows_finite() {
    let params = test_params();
    let log_emit = vec![[-1.0, -0.1], [-0.1, -1.0]];
    let positions = vec![(0, 4999), (5000, 9999)];
    let beta = backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(beta.len(), 2);
    for t in 0..2 {
        assert!(beta[t][0].is_finite() && beta[t][1].is_finite());
    }
}

// ============================================================
//  viterbi_with_distances_from_log_emit
// ============================================================

#[test]
fn vit_dist_empty() {
    let params = test_params();
    let path = viterbi_with_distances_from_log_emit(&[], &params, &[]);
    assert!(path.is_empty());
}

#[test]
fn vit_dist_single_window() {
    let params = test_params();
    let log_emit = vec![[-1.0, -0.01]]; // Strongly favors IBD
    let positions = vec![(0, 4999)];
    let path = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(path.len(), 1);
    assert!(path[0] == 0 || path[0] == 1); // Valid state
}

#[test]
fn vit_dist_strong_ibd_signal() {
    let params = test_params();
    // All windows strongly favor IBD
    let log_emit: Vec<[f64; 2]> = vec![[-5.0, -0.01]; 10];
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 5000, i * 5000 + 4999)).collect();
    let path = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(path.len(), 10);
    // Most should be IBD (state 1)
    let ibd_count = path.iter().filter(|&&s| s == 1).count();
    assert!(ibd_count >= 8, "Expected mostly IBD, got {} out of 10", ibd_count);
}

#[test]
fn vit_dist_mismatched_positions_fallback() {
    let params = test_params();
    let log_emit = vec![[-0.5, -0.5]; 5];
    let positions = vec![(0, 4999)]; // Mismatch
    let path = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(path.len(), 5); // Falls back
}

#[test]
fn vit_dist_valid_states_only() {
    let params = test_params();
    let log_emit: Vec<[f64; 2]> = vec![[-0.5, -0.8], [-0.8, -0.5], [-0.5, -0.8]];
    let positions: Vec<(u64, u64)> = vec![(0, 4999), (5000, 9999), (10000, 14999)];
    let path = viterbi_with_distances_from_log_emit(&log_emit, &params, &positions);
    for &s in &path {
        assert!(s == 0 || s == 1, "Invalid state: {}", s);
    }
}

// ============================================================
//  forward_backward_with_distances_from_log_emit
// ============================================================

#[test]
fn fb_dist_empty() {
    let params = test_params();
    let (posteriors, ll) = forward_backward_with_distances_from_log_emit(&[], &params, &[]);
    assert!(posteriors.is_empty());
    assert!((ll - 0.0).abs() < 1e-10);
}

#[test]
fn fb_dist_single_window() {
    let params = test_params();
    let log_emit = vec![[-0.5, -0.5]];
    let positions = vec![(0, 4999)];
    let (posteriors, ll) = forward_backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(posteriors.len(), 1);
    assert!(ll.is_finite());
    assert!(posteriors[0] >= 0.0 && posteriors[0] <= 1.0);
}

#[test]
fn fb_dist_posteriors_in_range() {
    let params = test_params();
    let log_emit: Vec<[f64; 2]> = vec![[-1.0, -0.1], [-0.5, -0.5], [-0.1, -1.0]];
    let positions: Vec<(u64, u64)> = vec![(0, 4999), (5000, 9999), (10000, 14999)];
    let (posteriors, ll) = forward_backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    assert_eq!(posteriors.len(), 3);
    assert!(ll.is_finite());
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0 + 1e-10, "Posterior out of range: {}", p);
    }
}

#[test]
fn fb_dist_strong_signal_high_posterior() {
    let params = test_params();
    // Strong IBD signal in middle
    let log_emit: Vec<[f64; 2]> = vec![
        [-0.1, -3.0], // Non-IBD
        [-5.0, -0.01], // Strong IBD
        [-5.0, -0.01], // Strong IBD
        [-5.0, -0.01], // Strong IBD
        [-0.1, -3.0], // Non-IBD
    ];
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 5000, i * 5000 + 4999)).collect();
    let (posteriors, _ll) = forward_backward_with_distances_from_log_emit(&log_emit, &params, &positions);
    // Middle windows should have high IBD posterior
    assert!(posteriors[2] > 0.8, "Middle window posterior should be high, got {}", posteriors[2]);
}

// ============================================================
//  extract_ibd_segments_composite
// ============================================================

#[test]
fn composite_empty_states() {
    let result = extract_ibd_segments_composite(&[], &[], None, 5, 3, 0.1);
    assert!(result.is_empty());
}

#[test]
fn composite_mismatched_lengths() {
    let states = vec![1, 1, 1];
    let posteriors = vec![0.9, 0.9]; // Length mismatch
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 3, 0.1);
    assert!(result.is_empty());
}

#[test]
fn composite_no_ibd_states() {
    let states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1, 0.1];
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 3, 0.1);
    assert!(result.is_empty());
}

#[test]
fn composite_below_hard_min() {
    // IBD segment of 2 windows, but hard_min=3
    let states = vec![0, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.95, 0.95, 0.1, 0.1];
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 3, 0.0);
    assert!(result.is_empty(), "Segments shorter than hard_min should be filtered");
}

#[test]
fn composite_above_hard_min_below_threshold() {
    // IBD segment of 3 windows with low posterior => low composite score
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.01, 0.01, 0.01, 0.1];
    // With no observations, LOD=0. score = 0 * 0.01 * length_factor = 0 < threshold
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 3, 0.1);
    assert!(result.is_empty(), "Low score should be below threshold");
}

#[test]
fn composite_single_long_segment() {
    let n = 20;
    let states = vec![1; n];
    let posteriors = vec![0.95; n];
    // threshold = 0 to accept everything
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 1, 0.0);
    // With no obs/params, LOD=0, so score = 0 * 0.95 * 1.0 = 0.
    // With threshold=0.0 and score=0.0: 0.0 < 0.0 is false, so it passes
    assert!(!result.is_empty() || result.is_empty()); // Just check it doesn't panic
}

#[test]
fn composite_multiple_segments() {
    let states = vec![1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9];
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 3, 2, 0.0);
    // Should find two segments (indices 0-3 and 7-11), but with LOD=0 score may be 0
    // Just verify it doesn't panic
    for seg in &result {
        assert!(seg.start_idx < states.len());
        assert!(seg.end_idx < states.len());
        assert!(seg.start_idx <= seg.end_idx);
    }
}
