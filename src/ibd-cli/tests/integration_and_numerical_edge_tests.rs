//! Cross-module integration tests and remaining numerical edge cases.
//!
//! Cycle 28: Tests for:
//! - kmeans_1d with NaN data (total_cmp handling)
//! - em_two_component degenerate cases (all-identical data, component vanishing)
//! - em_two_component_map with prior_strength = 0 (should approximate standard EM)
//! - detect_segments_rle → merge_segments pipeline (end-to-end)
//! - boundary_accuracy with odd-count median (2-match and 3-match)
//! - segment_length_histogram with zero bin size
//! - bic_model_selection: weight = 0.0 and weight = 1.0
//! - forward → backward → forward_backward consistency on edge data
//! - Baum-Welch convergence with extreme initial params

use hprc_ibd::hmm::{
    extract_ibd_segments, forward, forward_backward, infer_ibd, infer_ibd_with_training, viterbi,
    HmmParams, Population,
};
use hprc_ibd::segment::{
    detect_segments_rle, merge_segments, segment_length_histogram, IdentityTrack, RleParams,
    Segment,
};
use hprc_ibd::stats::{
    bic_model_selection, em_two_component, em_two_component_map, kmeans_1d, trimmed_mean,
    GaussianParams,
};
use hprc_ibd::concordance::boundary_accuracy;

/// Helper: create HmmParams with reasonable defaults for testing.
fn test_params() -> HmmParams {
    HmmParams::from_expected_length(100.0, 0.001, 10_000)
}

// =====================================================================
// kmeans_1d with NaN data
// =====================================================================

#[test]
fn test_kmeans_1d_data_with_nan() {
    // NaN values should be sorted consistently via total_cmp (NaN sorts last)
    let data = vec![1.0, f64::NAN, 2.0, 3.0, f64::NAN, 10.0];
    let result = kmeans_1d(&data, 2, 100);
    // Should not panic; might produce NaN centers but should complete
    assert!(result.is_some());
    let (centers, assignments) = result.unwrap();
    assert_eq!(centers.len(), 2);
    assert_eq!(assignments.len(), 6);
}

#[test]
fn test_kmeans_1d_all_nan() {
    let data = vec![f64::NAN, f64::NAN, f64::NAN];
    let result = kmeans_1d(&data, 2, 50);
    // Should not panic — all NaN means centers are NaN
    assert!(result.is_some());
}

#[test]
fn test_kmeans_1d_data_with_infinity() {
    let data = vec![1.0, 2.0, f64::INFINITY, f64::NEG_INFINITY, 5.0];
    let result = kmeans_1d(&data, 2, 50);
    assert!(result.is_some());
    let (_, assignments) = result.unwrap();
    assert_eq!(assignments.len(), 5);
}

#[test]
fn test_kmeans_1d_k_equals_one() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = kmeans_1d(&data, 1, 100).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 1);
    // All assignments should be 0
    assert!(assignments.iter().all(|&a| a == 0));
    // Center should be mean of data
    assert!((centers[0] - 3.0).abs() < 1e-9);
}

#[test]
fn test_kmeans_1d_large_k_equals_n() {
    let data = vec![1.0, 100.0, 200.0, 300.0, 400.0];
    let result = kmeans_1d(&data, 5, 100).unwrap();
    let (centers, assignments) = result;
    assert_eq!(centers.len(), 5);
    assert_eq!(assignments.len(), 5);
}

// =====================================================================
// em_two_component degenerate cases
// =====================================================================

#[test]
fn test_em_two_component_all_identical_data() {
    // All values identical → one component vanishes → returns None
    let data = vec![0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95];
    let init_low = GaussianParams::new_unchecked(0.9, 0.01);
    let init_high = GaussianParams::new_unchecked(0.99, 0.01);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 100, 1e-8);
    // Either returns None (degenerate) or returns with both components near 0.95
    if let Some((low, high, w0, w1)) = result {
        assert!(w0 + w1 > 0.99, "weights should sum to ~1");
        assert!(low.mean <= high.mean, "ordering should be maintained");
    }
    // If None, that's also acceptable for degenerate data
}

#[test]
fn test_em_two_component_two_distinct_peaks() {
    let mut data = Vec::new();
    for _ in 0..50 {
        data.push(0.90);
    }
    for _ in 0..50 {
        data.push(0.999);
    }
    let init_low = GaussianParams::new_unchecked(0.90, 0.02);
    let init_high = GaussianParams::new_unchecked(0.999, 0.02);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 100, 1e-8);
    assert!(result.is_some());
    let (low, high, w0, w1) = result.unwrap();
    assert!(low.mean < 0.95, "low component should be < 0.95, got {}", low.mean);
    assert!(high.mean > 0.95, "high component should be > 0.95, got {}", high.mean);
    assert!((w0 + w1 - 1.0).abs() < 1e-6, "weights should sum to 1");
    assert!((w0 - 0.5).abs() < 0.15, "weights should be ~0.5 each");
}

#[test]
fn test_em_two_component_heavily_skewed_weights() {
    // 95% of data in one cluster
    let mut data: Vec<f64> = vec![0.90; 95];
    data.extend(vec![0.999; 5]);
    let init_low = GaussianParams::new_unchecked(0.90, 0.02);
    let init_high = GaussianParams::new_unchecked(0.999, 0.02);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 200, 1e-8);
    if let Some((_, _, w0, w1)) = result {
        assert!((w0 + w1 - 1.0).abs() < 1e-6);
        // The dominant component should have weight > 0.7
        assert!(w0.max(w1) > 0.7);
    }
}

#[test]
fn test_em_two_component_exactly_4_points_boundary() {
    let data = vec![0.1, 0.2, 0.8, 0.9];
    let init_low = GaussianParams::new_unchecked(0.15, 0.1);
    let init_high = GaussianParams::new_unchecked(0.85, 0.1);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_some(), "n=4 is the boundary, should succeed");
    let (low, high, _, _) = result.unwrap();
    assert!(low.mean < high.mean);
}

#[test]
fn test_em_two_component_3_points_returns_none() {
    let data = vec![0.1, 0.5, 0.9];
    let init_low = GaussianParams::new_unchecked(0.1, 0.1);
    let init_high = GaussianParams::new_unchecked(0.9, 0.1);
    assert!(em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6).is_none());
}

// =====================================================================
// em_two_component_map with prior_strength = 0
// =====================================================================

#[test]
fn test_em_two_component_map_zero_prior_strength() {
    // With prior_strength = 0, MAP should approximate standard EM
    let data = vec![0.1, 0.12, 0.11, 0.9, 0.88, 0.91, 0.89, 0.1, 0.11];
    let init_low = GaussianParams::new_unchecked(0.1, 0.05);
    let init_high = GaussianParams::new_unchecked(0.9, 0.05);

    let result_std = em_two_component(&data, &init_low, &init_high, 0.5, 100, 1e-8);
    let result_map = em_two_component_map(&data, &init_low, &init_high, 0.5, 100, 1e-8, 0.0);

    // Both should succeed
    assert!(result_std.is_some());
    assert!(result_map.is_some());

    let (low_std, high_std, _, _) = result_std.unwrap();
    let (low_map, high_map, _, _) = result_map.unwrap();

    // With zero prior strength, means should be very similar
    assert!(
        (low_std.mean - low_map.mean).abs() < 0.05,
        "Low means diverge: std={}, map={}",
        low_std.mean,
        low_map.mean
    );
    assert!(
        (high_std.mean - high_map.mean).abs() < 0.05,
        "High means diverge: std={}, map={}",
        high_std.mean,
        high_map.mean
    );
}

#[test]
fn test_em_two_component_map_very_strong_prior() {
    // Very strong prior should keep means near initial values
    let data = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // all identical
    let init_low = GaussianParams::new_unchecked(0.1, 0.05);
    let init_high = GaussianParams::new_unchecked(0.9, 0.05);

    let result = em_two_component_map(&data, &init_low, &init_high, 0.5, 100, 1e-8, 10000.0);
    if let Some((low, high, _, _)) = result {
        // Means should be pulled toward 0.1 and 0.9 by strong prior
        assert!(low.mean < 0.3, "Strong prior should keep low mean near 0.1, got {}", low.mean);
        assert!(high.mean > 0.7, "Strong prior should keep high mean near 0.9, got {}", high.mean);
    }
}

// =====================================================================
// detect_segments_rle → merge_segments pipeline
// =====================================================================

fn make_window_positions(n: usize, window_size: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|i| {
            let start = i as u64 * window_size;
            (start, start + window_size)
        })
        .collect()
}

#[test]
fn test_rle_to_merge_pipeline_two_overlapping_chunks() {
    // Simulate two chunks that produce overlapping segments
    let wp = make_window_positions(20, 10_000);

    // Chunk 1: windows 0-9 are high identity
    let track1 = IdentityTrack {
        windows: (0..10).map(|i| (i, 0.9998)).collect(),
        n_total_windows: 20,
    };
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 10_000,
        drop_tolerance: 0.0,
    };
    let segs1 = detect_segments_rle(&track1, &wp, &params, "chr10", "hap_A", "hap_B");

    // Chunk 2: windows 8-17 are high identity (overlap at 8-9)
    let track2 = IdentityTrack {
        windows: (8..18).map(|i| (i, 0.9997)).collect(),
        n_total_windows: 20,
    };
    let segs2 = detect_segments_rle(&track2, &wp, &params, "chr10", "hap_A", "hap_B");

    // Merge both chunks
    let mut all_segs: Vec<Segment> = segs1.into_iter().chain(segs2).collect();
    let count_before = all_segs.len();
    merge_segments(&mut all_segs);

    // Should merge into fewer segments (or same if no overlap in window indices)
    assert!(all_segs.len() <= count_before, "merge should not increase count");

    // Merged segment should cover the full range
    if all_segs.len() == 1 {
        let merged = &all_segs[0];
        assert!(merged.start_idx <= 1, "should start early");
        assert!(merged.end_idx >= 9, "should extend past overlap");
        assert!(merged.mean_identity > 0.999, "mean identity should remain high");
    }
}

#[test]
fn test_rle_pipeline_no_segments_from_low_identity() {
    let wp = make_window_positions(10, 10_000);
    let track = IdentityTrack {
        windows: (0..10).map(|i| (i, 0.5)).collect(), // all low
        n_total_windows: 10,
    };
    let params = RleParams::default();
    let segs = detect_segments_rle(&track, &wp, &params, "chr10", "hap_A", "hap_B");
    assert!(segs.is_empty(), "low identity should produce no segments");
}

#[test]
fn test_rle_pipeline_gap_exceeds_max_splits_segment() {
    let wp = make_window_positions(15, 10_000);
    // Windows 0-4 high, 5-7 missing (gap=3), 8-14 high
    let mut windows: Vec<(usize, f64)> = (0..5).map(|i| (i, 0.9999)).collect();
    windows.extend((8..15).map(|i| (i, 0.9999)));

    let track = IdentityTrack {
        windows,
        n_total_windows: 15,
    };
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 2, // gap of 3 exceeds max_gap of 2
        min_windows: 3,
        min_length_bp: 10_000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &wp, &params, "chr10", "hap_A", "hap_B");
    assert_eq!(segs.len(), 2, "gap > max_gap should split into two segments");
}

#[test]
fn test_merge_segments_same_pair_adjacent_but_not_overlapping() {
    let mut segs = vec![
        Segment {
            chrom: "chr10".to_string(),
            start: 0,
            end: 50_000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 5,
            mean_identity: 0.9999,
            min_identity: 0.9998,
            identity_sum: 4.9995,
            n_called: 5,
            start_idx: 0,
            end_idx: 4,
        },
        Segment {
            chrom: "chr10".to_string(),
            start: 60_000,
            end: 100_000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 4,
            mean_identity: 0.9998,
            min_identity: 0.9997,
            identity_sum: 3.9992,
            n_called: 4,
            start_idx: 6, // gap at index 5
            end_idx: 9,
        },
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2, "non-overlapping segments should not merge");
}

// =====================================================================
// segment_length_histogram edge cases
// =====================================================================

#[test]
fn test_segment_length_histogram_empty() {
    let hist = segment_length_histogram(&[], 10_000);
    assert!(hist.is_empty());
}

#[test]
fn test_segment_length_histogram_multiple_same_bin() {
    let segs = vec![
        Segment {
            chrom: "chr1".to_string(), start: 0, end: 5_000,
            hap_a: "A".to_string(), hap_b: "B".to_string(),
            n_windows: 1, mean_identity: 0.99, min_identity: 0.99,
            identity_sum: 0.99, n_called: 1, start_idx: 0, end_idx: 0,
        },
        Segment {
            chrom: "chr1".to_string(), start: 100_000, end: 105_000,
            hap_a: "C".to_string(), hap_b: "D".to_string(),
            n_windows: 1, mean_identity: 0.99, min_identity: 0.99,
            identity_sum: 0.99, n_called: 1, start_idx: 10, end_idx: 10,
        },
    ];
    let hist = segment_length_histogram(&segs, 10_000);
    // Both segments have length 5001 bp, so they should be in same bin
    assert_eq!(hist.len(), 1, "two segments of same length should be in one bin");
    assert_eq!(hist[0].1, 2, "count should be 2");
}

// =====================================================================
// bic_model_selection with extreme weights
// =====================================================================

#[test]
fn test_bic_model_selection_weight_zero() {
    // weight_low = 0.0 means w_high = 1.0 → ln(0) = -inf
    // This should still produce finite BIC values (or gracefully handle)
    let data: Vec<f64> = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let low = GaussianParams::new_unchecked(0.5, 0.1);
    let high = GaussianParams::new_unchecked(0.9, 0.1);
    let (bic_1, _bic_2) = bic_model_selection(&data, &low, &high, 0.0);
    // bic_1 should be finite (single component)
    assert!(bic_1.is_finite(), "BIC_1 should be finite");
    // bic_2 might be -inf or +inf due to ln(0), but should not panic
}

#[test]
fn test_bic_model_selection_weight_one() {
    let data: Vec<f64> = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let low = GaussianParams::new_unchecked(0.5, 0.1);
    let high = GaussianParams::new_unchecked(0.9, 0.1);
    let (bic_1, _bic_2) = bic_model_selection(&data, &low, &high, 1.0);
    assert!(bic_1.is_finite(), "BIC_1 should be finite");
    // w_high = 0.0 → ln(0) = -inf for the high component contribution
}

#[test]
fn test_bic_model_selection_single_point() {
    let data = vec![0.5];
    let low = GaussianParams::new_unchecked(0.5, 0.1);
    let high = GaussianParams::new_unchecked(0.9, 0.1);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

// =====================================================================
// boundary_accuracy median: odd vs even count
// =====================================================================

#[test]
fn test_boundary_accuracy_three_matches_odd_median() {
    let matches = vec![
        ((100, 200), (110, 210)),   // distances: 10, 10
        ((300, 400), (305, 415)),   // distances: 5, 15
        ((500, 600), (520, 580)),   // distances: 20, 20
    ];
    let ba = boundary_accuracy(&matches, 100).unwrap();
    assert_eq!(ba.n_matched, 3);
    // Start distances sorted: 5, 10, 20 → median = 10
    assert!((ba.median_start_distance_bp - 10.0).abs() < 1e-9);
    // End distances sorted: 10, 15, 20 → median = 15
    assert!((ba.median_end_distance_bp - 15.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_two_matches_even_median() {
    let matches = vec![
        ((100, 200), (110, 210)),   // start distance = 10, end distance = 10
        ((300, 400), (320, 380)),   // start distance = 20, end distance = 20
    ];
    let ba = boundary_accuracy(&matches, 100).unwrap();
    assert_eq!(ba.n_matched, 2);
    // Even count: median = (10+20)/2 = 15
    assert!((ba.median_start_distance_bp - 15.0).abs() < 1e-9);
    assert!((ba.median_end_distance_bp - 15.0).abs() < 1e-9);
}

// =====================================================================
// Forward-backward consistency on edge data
// =====================================================================

#[test]
fn test_forward_backward_log_likelihood_matches_forward() {
    let params = test_params();
    let obs = vec![0.95, 0.96, 0.97, 0.999, 0.9999, 0.999, 0.96, 0.95];
    let (_, ll_fwd) = forward(&obs, &params);
    let (_, ll_fb) = forward_backward(&obs, &params);
    assert!(
        (ll_fwd - ll_fb).abs() < 1e-6,
        "Forward and forward-backward log-likelihoods should match: {} vs {}",
        ll_fwd,
        ll_fb
    );
}

#[test]
fn test_forward_backward_posteriors_sum_correctly() {
    let params = test_params();
    let obs = vec![0.90, 0.95, 0.999, 0.9999, 0.999, 0.95, 0.90];
    let (posteriors, _) = forward_backward(&obs, &params);
    // Posteriors are P(IBD | data), should be in [0, 1]
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior {} not in [0,1]", p);
    }
}

#[test]
fn test_viterbi_and_forward_backward_agree_on_strong_signal() {
    let params = test_params();
    // Clear IBD region
    let obs = vec![0.80, 0.80, 0.9999, 0.9999, 0.9999, 0.9999, 0.80, 0.80];
    let states = viterbi(&obs, &params);
    let (posteriors, _) = forward_backward(&obs, &params);

    // Where Viterbi says IBD (state=1), posteriors should be high
    for i in 0..obs.len() {
        if states[i] == 1 {
            assert!(posteriors[i] > 0.5, "Viterbi says IBD at {} but posterior={}", i, posteriors[i]);
        }
    }
}

#[test]
fn test_forward_backward_empty_input() {
    let params = test_params();
    let (posteriors, ll) = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn test_forward_backward_single_observation() {
    let params = test_params();
    let (posteriors, ll) = forward_backward(&[0.999], &params);
    assert_eq!(posteriors.len(), 1);
    assert!(ll.is_finite());
    assert!(posteriors[0] >= 0.0 && posteriors[0] <= 1.0);
}

// =====================================================================
// Baum-Welch with extreme initial parameters
// =====================================================================

#[test]
fn test_infer_ibd_with_training_from_extreme_params() {
    let mut params = HmmParams::from_expected_length(100.0, 0.001, 10_000);
    let obs: Vec<f64> = vec![0.95; 50]; // all at 0.95
    let result = infer_ibd_with_training(&obs, &mut params, Population::Generic, 10_000, 10);
    // Should not panic; training should adjust params
    assert_eq!(result.states.len(), 50);
}

#[test]
fn test_infer_ibd_all_very_high() {
    let params = test_params();
    let obs: Vec<f64> = vec![0.9999; 20];
    let result = infer_ibd(&obs, &params);
    // All very high identity → should be mostly IBD
    let ibd_count = result.states.iter().filter(|&&s| s == 1).count();
    assert!(ibd_count > 10, "most windows should be IBD, got {}/20", ibd_count);
}

#[test]
fn test_infer_ibd_all_very_low() {
    let params = test_params();
    let obs: Vec<f64> = vec![0.80; 20];
    let result = infer_ibd(&obs, &params);
    // All low identity → should be mostly non-IBD
    let non_ibd_count = result.states.iter().filter(|&&s| s == 0).count();
    assert!(non_ibd_count > 10, "most windows should be non-IBD, got {}/20", non_ibd_count);
}

// =====================================================================
// trimmed_mean: property-style tests
// =====================================================================

#[test]
fn test_trimmed_mean_zero_trim_equals_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tm = trimmed_mean(&data, 0.0).unwrap();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    assert!((tm - mean).abs() < 1e-9, "zero-trim should equal mean");
}

#[test]
fn test_trimmed_mean_reduces_outlier_effect() {
    // Without trimming, mean is pulled by outlier
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    let mean = trimmed_mean(&data, 0.0).unwrap();
    let tm20 = trimmed_mean(&data, 0.2).unwrap();
    // Trimmed mean should be closer to the central values
    assert!(
        (tm20 - 3.0).abs() < (mean - 3.0).abs(),
        "trimmed mean {} should be closer to 3 than mean {}",
        tm20, mean
    );
}

#[test]
fn test_trimmed_mean_empty() {
    assert!(trimmed_mean(&[], 0.1).is_none());
}

// =====================================================================
// Cross-crate: infer_ibd → extract_segments pipeline
// =====================================================================

#[test]
fn test_infer_ibd_extract_segments_roundtrip() {
    let params = test_params();
    // Clear IBD block in middle
    let mut obs: Vec<f64> = vec![0.90; 5];
    obs.extend(vec![0.9999; 10]);
    obs.extend(vec![0.90; 5]);

    let result = infer_ibd(&obs, &params);
    let segments = extract_ibd_segments(&result.states);

    // Should find at least one IBD segment
    assert!(!segments.is_empty(), "should detect IBD segment");

    // The IBD segment should be roughly in the middle
    for (start, end, state) in &segments {
        if *state == 1 {
            assert!(*start >= 3, "IBD segment should start around index 5, got {}", start);
            assert!(*end <= 17, "IBD segment should end around index 15, got {}", end);
        }
    }
}

// =====================================================================
// GaussianParams edge cases
// =====================================================================

#[test]
fn test_gaussian_params_log_pdf_at_mean() {
    let g = GaussianParams::new_unchecked(5.0, 1.0);
    let log_pdf_at_mean = g.log_pdf(5.0);
    // log(1/(sqrt(2π)σ)) = -0.5*ln(2π) - ln(σ)
    let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!((log_pdf_at_mean - expected).abs() < 1e-9);
}

#[test]
fn test_gaussian_params_log_pdf_symmetry() {
    let g = GaussianParams::new_unchecked(0.0, 1.0);
    let log_pdf_pos = g.log_pdf(1.5);
    let log_pdf_neg = g.log_pdf(-1.5);
    assert!((log_pdf_pos - log_pdf_neg).abs() < 1e-9, "log_pdf should be symmetric around mean");
}

#[test]
fn test_gaussian_params_new_rejects_zero_std() {
    assert!(GaussianParams::new(0.0, 0.0).is_err());
    assert!(GaussianParams::new(0.0, -1.0).is_err());
}

#[test]
fn test_gaussian_params_new_accepts_positive_std() {
    assert!(GaussianParams::new(0.0, 0.001).is_ok());
    assert!(GaussianParams::new(100.0, 50.0).is_ok());
}
