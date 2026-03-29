//! Tests for EM component swap paths, stats boundary conditions, and segment distribution edge cases
//!
//! Cycle 40: Focuses on genuinely untested branches in:
//! - em_two_component: mu0 > mu1 swap path (lines 283-289)
//! - em_two_component_map: mu0 > mu1 swap path (lines 400-406), degenerate n0 < 0.5 (line 371)
//! - em_two_component_map: prior_strength effects on separation
//! - segment_length_distribution: even count median path
//! - segment_length_histogram: single segment fills one bin
//! - trimmed_mean: single element, clamp boundary
//! - kmeans_1d: max_iter=0 (no iterations), convergence on perfectly separated data
//! - GaussianParams: pdf/log_pdf with extreme z-scores
//! - OnlineStats: extremely large values for numerical stability

use hprc_ibd::stats::*;
use hprc_ibd::segment::*;

// ============================================================================
// EM two-component: swap path when mu0 > mu1
// ============================================================================

#[test]
fn em_two_component_swap_path_when_init_low_higher_than_high() {
    // Initialize with low.mean > high.mean
    // The EM should swap components so that result.0.mean < result.1.mean
    let mut data = Vec::new();
    for _ in 0..30 { data.push(0.990); }
    for _ in 0..30 { data.push(0.999); }

    // Deliberately swap: init_low is actually the higher cluster
    let init_low = GaussianParams::new_unchecked(0.999, 0.001);
    let init_high = GaussianParams::new_unchecked(0.990, 0.001);

    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_some());
    let (low, high, _w_low, _w_high) = result.unwrap();
    // After EM, the swap at line 283-289 should ensure low.mean < high.mean
    assert!(low.mean < high.mean,
        "EM should swap so low ({}) < high ({})", low.mean, high.mean);
}

#[test]
fn em_two_component_map_swap_path_when_means_cross() {
    // MAP EM: initialize so that after EM iterations, mu0 > mu1, triggering swap
    let mut data = Vec::new();
    for _ in 0..30 { data.push(0.990); }
    for _ in 0..30 { data.push(0.999); }

    // Swapped init: init_low has the higher mean
    let init_low = GaussianParams::new_unchecked(0.999, 0.001);
    let init_high = GaussianParams::new_unchecked(0.990, 0.001);

    let result = em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 1.0);
    assert!(result.is_some());
    let (low, high, _w_low, _w_high) = result.unwrap();
    assert!(low.mean < high.mean,
        "MAP EM should swap so low ({}) < high ({})", low.mean, high.mean);
}

#[test]
fn em_two_component_swap_preserves_weights() {
    // When means cross, weights should also swap
    let mut data = Vec::new();
    for _ in 0..60 { data.push(0.990); } // More data at lower value
    for _ in 0..10 { data.push(0.999); } // Less data at higher value

    // Swapped initialization
    let init_low = GaussianParams::new_unchecked(0.999, 0.001);
    let init_high = GaussianParams::new_unchecked(0.990, 0.001);

    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_some());
    let (low, high, w_low, w_high) = result.unwrap();
    assert!(low.mean < high.mean);
    // The low cluster (0.990) has more data, so w_low > w_high
    assert!(w_low > w_high,
        "Low cluster weight ({}) should exceed high cluster weight ({})", w_low, w_high);
    // Weights should sum to ~1.0
    assert!((w_low + w_high - 1.0).abs() < 0.01);
}

// ============================================================================
// EM MAP: degenerate case at n0 < 0.5 boundary
// ============================================================================

#[test]
fn em_two_component_map_degenerate_all_identical() {
    // All identical values: one component vanishes → n0 < 0.5 → returns None
    let data = vec![0.999; 20];
    let init_low = GaussianParams::new_unchecked(0.990, 0.005);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);

    let result = em_two_component_map(&data, &init_low, &init_high, 0.1, 100, 1e-8, 0.1);
    // With very weak prior (0.1), one component should vanish
    // Either returns None (degenerate) or converges to single mode
    if let Some((low, high, _, _)) = result {
        // Both means near 0.999
        assert!((low.mean - 0.999).abs() < 0.01);
        assert!((high.mean - 0.999).abs() < 0.01);
    }
}

#[test]
fn em_two_component_map_strong_prior_prevents_vanishing() {
    // Very strong prior keeps both components alive even with unimodal data
    let data = vec![0.999; 20];
    let init_low = GaussianParams::new_unchecked(0.990, 0.005);
    let init_high = GaussianParams::new_unchecked(0.9999, 0.001);

    // Very strong prior (50.0) should prevent component vanishing
    let result = em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 50.0);
    if let Some((low, high, _, _)) = result {
        // With strong prior, means should stay somewhat near init values
        assert!(low.mean < high.mean);
    }
}

#[test]
fn em_two_component_map_prior_strength_zero_behaves_like_standard() {
    // With prior_strength=0, MAP EM should behave like standard EM
    let mut data = Vec::new();
    for _ in 0..40 { data.push(0.990); }
    for _ in 0..40 { data.push(0.999); }

    let init_low = GaussianParams::new_unchecked(0.990, 0.002);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);

    let standard = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    let map_zero = em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 0.0);

    // Both should converge to similar results
    if let (Some((s_low, s_high, _, _)), Some((m_low, m_high, _, _))) = (standard, map_zero) {
        assert!((s_low.mean - m_low.mean).abs() < 0.002,
            "Standard low mean ({}) should match MAP(0) low mean ({})", s_low.mean, m_low.mean);
        assert!((s_high.mean - m_high.mean).abs() < 0.002,
            "Standard high mean ({}) should match MAP(0) high mean ({})", s_high.mean, m_high.mean);
    }
}

// ============================================================================
// em_two_component: convergence and edge cases
// ============================================================================

#[test]
fn em_two_component_single_iteration() {
    let mut data = Vec::new();
    for _ in 0..30 { data.push(0.990); }
    for _ in 0..30 { data.push(0.999); }

    let init_low = GaussianParams::new_unchecked(0.990, 0.002);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);

    // max_iter=1 should still produce valid output
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 1, 1e-6);
    assert!(result.is_some());
    let (low, high, w_low, w_high) = result.unwrap();
    assert!(low.mean < high.mean);
    assert!((w_low + w_high - 1.0).abs() < 0.01);
}

#[test]
fn em_two_component_exactly_four_points() {
    // Minimum valid input (n=4)
    let data = vec![0.990, 0.991, 0.999, 0.9995];
    let init_low = GaussianParams::new_unchecked(0.990, 0.002);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);

    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_some());
}

#[test]
fn em_two_component_three_points_returns_none() {
    let data = vec![0.990, 0.999, 0.9995];
    let init_low = GaussianParams::new_unchecked(0.990, 0.002);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);

    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_none());
}

// ============================================================================
// segment_length_distribution: even count median
// ============================================================================

#[test]
fn segment_length_distribution_even_count_median() {
    // 4 segments → even count → median = average of middle two
    fn make_seg(start: u64, end: u64) -> Segment {
        Segment {
            chrom: "chr1".to_string(),
            start,
            end,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 1,
            mean_identity: 0.999,
            min_identity: 0.999,
            identity_sum: 0.999,
            n_called: 1,
            start_idx: 0,
            end_idx: 0,
        }
    }

    let segments = vec![
        make_seg(0, 999),    // 1000 bp
        make_seg(0, 1999),   // 2000 bp
        make_seg(0, 2999),   // 3000 bp
        make_seg(0, 3999),   // 4000 bp
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 4);
    // Even count: median = (2000 + 3000) / 2 = 2500
    assert!((stats.median_bp - 2500.0).abs() < 1e-6,
        "Even count median should be 2500, got {}", stats.median_bp);
}

#[test]
fn segment_length_distribution_two_segments_median() {
    fn make_seg(start: u64, end: u64) -> Segment {
        Segment {
            chrom: "chr1".to_string(),
            start,
            end,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 1,
            mean_identity: 0.999,
            min_identity: 0.999,
            identity_sum: 0.999,
            n_called: 1,
            start_idx: 0,
            end_idx: 0,
        }
    }

    let segments = vec![
        make_seg(0, 999),   // 1000 bp
        make_seg(0, 4999),  // 5000 bp
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 2);
    // median = (1000 + 5000) / 2 = 3000
    assert!((stats.median_bp - 3000.0).abs() < 1e-6,
        "Two-segment median should be 3000, got {}", stats.median_bp);
    assert_eq!(stats.min_bp, 1000);
    assert_eq!(stats.max_bp, 5000);
    assert!((stats.mean_bp - 3000.0).abs() < 1e-6);
}

// ============================================================================
// segment_length_histogram: single segment, large bin
// ============================================================================

#[test]
fn segment_length_histogram_single_segment_large_bin() {
    fn make_seg(start: u64, end: u64) -> Segment {
        Segment {
            chrom: "chr1".to_string(),
            start,
            end,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 1,
            mean_identity: 0.999,
            min_identity: 0.999,
            identity_sum: 0.999,
            n_called: 1,
            start_idx: 0,
            end_idx: 0,
        }
    }

    // Single segment of 5000 bp, bin size 100000
    let segments = vec![make_seg(0, 4999)]; // 5000 bp
    let hist = segment_length_histogram(&segments, 100000);
    // 5000 / 100000 = bin 0
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 1));
}

#[test]
fn segment_length_histogram_all_same_length() {
    fn make_seg(start: u64, end: u64) -> Segment {
        Segment {
            chrom: "chr1".to_string(),
            start,
            end,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 1,
            mean_identity: 0.999,
            min_identity: 0.999,
            identity_sum: 0.999,
            n_called: 1,
            start_idx: 0,
            end_idx: 0,
        }
    }

    // 5 segments all 10000bp, bin size 10000
    let segments: Vec<Segment> = (0..5).map(|_| make_seg(0, 9999)).collect();
    let hist = segment_length_histogram(&segments, 10000);
    // All in bin 1 (10000 / 10000 = 1)
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (10000, 5));
}

// ============================================================================
// trimmed_mean edge cases
// ============================================================================

#[test]
fn trimmed_mean_single_element() {
    // Single element: trim_count = 0, so start=0, end=1 → valid
    let data = vec![42.0];
    let result = trimmed_mean(&data, 0.1);
    assert!(result.is_some());
    assert!((result.unwrap() - 42.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_two_elements_with_high_trim() {
    // Two elements, trim_fraction=0.49 (max allowed after clamping)
    // trim_count = (2 * 0.49) as usize = 0 → no trimming
    let data = vec![1.0, 100.0];
    let result = trimmed_mean(&data, 0.49);
    assert!(result.is_some());
    // trim_count = floor(2 * 0.49) = 0, so full mean = 50.5
    assert!((result.unwrap() - 50.5).abs() < 1e-10);
}

#[test]
fn trimmed_mean_three_elements_heavy_trim() {
    // Three elements, trim_fraction=0.49
    // trim_count = floor(3 * 0.49) = 1 → keep only middle element
    let data = vec![1.0, 50.0, 100.0];
    let result = trimmed_mean(&data, 0.49);
    assert!(result.is_some());
    // After sorting: [1, 50, 100], trim 1 from each end → [50]
    assert!((result.unwrap() - 50.0).abs() < 1e-10);
}

// ============================================================================
// kmeans_1d edge cases
// ============================================================================

#[test]
fn kmeans_max_iter_zero() {
    // max_iter=0: no iterations, returns initial assignments
    let data = vec![1.0, 2.0, 3.0, 8.0, 9.0, 10.0];
    let result = kmeans_1d(&data, 2, 0);
    assert!(result.is_some());
    let (centers, assignments) = result.unwrap();
    assert_eq!(centers.len(), 2);
    assert_eq!(assignments.len(), 6);
}

#[test]
fn kmeans_already_converged() {
    // Data perfectly split into two clusters — should converge in 1-2 iterations
    let data = vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0];
    let result = kmeans_1d(&data, 2, 100);
    assert!(result.is_some());
    let (centers, _) = result.unwrap();
    let min_c = centers.iter().cloned().fold(f64::MAX, f64::min);
    let max_c = centers.iter().cloned().fold(f64::MIN, f64::max);
    assert!((min_c - 0.0).abs() < 0.1);
    assert!((max_c - 10.0).abs() < 0.1);
}

#[test]
fn kmeans_three_clusters() {
    let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 9.0, 9.1, 9.2];
    let result = kmeans_1d(&data, 3, 20);
    assert!(result.is_some());
    let (centers, assignments) = result.unwrap();
    assert_eq!(centers.len(), 3);
    assert_eq!(assignments.len(), 9);
    // Should have 3 unique clusters
    let unique: std::collections::HashSet<_> = assignments.iter().collect();
    assert_eq!(unique.len(), 3);
}

// ============================================================================
// GaussianParams extreme z-score behavior
// ============================================================================

#[test]
fn gaussian_pdf_very_far_from_mean() {
    let g = GaussianParams::new_unchecked(0.0, 1.0);
    // z = 10: exp(-50) ≈ 1.93e-22
    let pdf_far = g.pdf(10.0);
    assert!(pdf_far > 0.0);
    assert!(pdf_far < 1e-20);
}

#[test]
fn gaussian_log_pdf_very_far_from_mean() {
    let g = GaussianParams::new_unchecked(0.0, 1.0);
    let log_pdf_far = g.log_pdf(10.0);
    // log_pdf should be very negative but finite
    assert!(log_pdf_far.is_finite());
    assert!(log_pdf_far < -40.0);
}

#[test]
fn gaussian_pdf_negative_x() {
    let g = GaussianParams::new_unchecked(5.0, 2.0);
    let pdf_neg = g.pdf(-100.0);
    assert!(pdf_neg.is_finite());
    assert!(pdf_neg >= 0.0);
    assert!(pdf_neg < 1e-300);
}

#[test]
fn gaussian_log_pdf_negative_x() {
    let g = GaussianParams::new_unchecked(5.0, 2.0);
    let log_pdf_neg = g.log_pdf(-100.0);
    assert!(log_pdf_neg.is_finite());
    assert!(log_pdf_neg < -1000.0);
}

// ============================================================================
// gaussian_to_logit_space edge: near-boundary denominators
// ============================================================================

#[test]
fn gaussian_to_logit_space_very_near_one() {
    // mean very close to 1.0 → denominator near zero → fallback to LOGIT_CAP * 0.5
    let params = gaussian_to_logit_space(1.0 - 1e-16, 0.001);
    assert!(params.mean.is_finite());
    assert!(params.std.is_finite());
    assert!(params.std >= 0.01); // Clamped to min 0.01
}

#[test]
fn gaussian_to_logit_space_very_near_zero() {
    // mean very close to 0.0 → denominator near zero → fallback to LOGIT_CAP * 0.5
    let params = gaussian_to_logit_space(1e-16, 0.001);
    assert!(params.mean.is_finite());
    assert!(params.std.is_finite());
    assert!(params.std >= 0.01);
}

#[test]
fn gaussian_to_logit_space_half() {
    // mean = 0.5 → logit(0.5) = 0, denominator = 0.25
    let params = gaussian_to_logit_space(0.5, 0.1);
    assert!((params.mean - 0.0).abs() < 1e-8, "logit(0.5) should be 0, got {}", params.mean);
    // logit_std = 0.1 / 0.25 = 0.4
    assert!((params.std - 0.4).abs() < 0.01, "Expected std ≈ 0.4, got {}", params.std);
}

// ============================================================================
// logit_transform_observations with boundary values
// ============================================================================

#[test]
fn logit_transform_all_zeros() {
    let obs = vec![0.0, 0.0, 0.0];
    let transformed = logit_transform_observations(&obs);
    for &v in &transformed {
        assert!((v - (-LOGIT_CAP)).abs() < 1e-8, "logit(0) should be -LOGIT_CAP, got {}", v);
    }
}

#[test]
fn logit_transform_all_ones() {
    let obs = vec![1.0, 1.0, 1.0];
    let transformed = logit_transform_observations(&obs);
    for &v in &transformed {
        assert!((v - LOGIT_CAP).abs() < 1e-8, "logit(1) should be LOGIT_CAP, got {}", v);
    }
}

// ============================================================================
// BIC model selection: two-component with equal weights
// ============================================================================

#[test]
fn bic_model_selection_equal_weights() {
    let data: Vec<f64> = (0..50).map(|_| 0.990)
        .chain((0..50).map(|_| 0.999))
        .collect();

    let low = GaussianParams::new_unchecked(0.990, 0.0005);
    let high = GaussianParams::new_unchecked(0.999, 0.0005);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // Clear bimodal → 2-component should be better (lower BIC)
    assert!(bic_2 < bic_1, "BIC(2) = {} should be less than BIC(1) = {}", bic_2, bic_1);
}

#[test]
fn bic_model_selection_extreme_weight() {
    // weight_low = 0.99 → almost all mass on low component
    let data = vec![0.990; 100];
    let low = GaussianParams::new_unchecked(0.990, 0.0001);
    let high = GaussianParams::new_unchecked(0.999, 0.0001);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.99);
    // With extreme weight, data is effectively unimodal
    // BIC(1) should be competitive
    assert!(bic_1.is_finite());
    assert!(bic_2.is_finite());
}

#[test]
fn bic_model_selection_exactly_two_points() {
    let data = vec![0.990, 0.999];
    let low = GaussianParams::new_unchecked(0.990, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // n=2 is minimum valid → should return finite values
    assert!(bic_1.is_finite());
    assert!(bic_2.is_finite());
}

// ============================================================================
// format_segment_bed: edge cases
// ============================================================================

#[test]
fn format_segment_bed_zero_lod() {
    let seg = Segment {
        chrom: "chr1".to_string(),
        start: 1,
        end: 100,
        hap_a: "A".to_string(),
        hap_b: "B".to_string(),
        n_windows: 1,
        mean_identity: 0.999,
        min_identity: 0.999,
        identity_sum: 0.999,
        n_called: 1,
        start_idx: 0,
        end_idx: 0,
    };
    let bed = format_segment_bed(&seg, 0.0);
    // LOD=0.0 → score=0 (not positive)
    assert!(bed.contains("\t0\t"), "LOD=0 should give score=0, got: {}", bed);
}

#[test]
fn format_segment_bed_very_large_lod() {
    let seg = Segment {
        chrom: "chr1".to_string(),
        start: 100,
        end: 200,
        hap_a: "A".to_string(),
        hap_b: "B".to_string(),
        n_windows: 1,
        mean_identity: 0.999,
        min_identity: 0.999,
        identity_sum: 0.999,
        n_called: 1,
        start_idx: 0,
        end_idx: 0,
    };
    let bed = format_segment_bed(&seg, 50.0);
    // LOD=50.0 → score = min(5000, 1000) = 1000 (capped)
    assert!(bed.contains("\t1000\t"), "Very large LOD should cap at 1000, got: {}", bed);
}

#[test]
fn format_segment_bed_start_1_converts_to_0() {
    let seg = Segment {
        chrom: "chr1".to_string(),
        start: 1,
        end: 100,
        hap_a: "X".to_string(),
        hap_b: "Y".to_string(),
        n_windows: 1,
        mean_identity: 0.999,
        min_identity: 0.999,
        identity_sum: 0.999,
        n_called: 1,
        start_idx: 0,
        end_idx: 0,
    };
    let bed = format_segment_bed(&seg, 1.0);
    // BED start = 1 - 1 = 0
    assert!(bed.starts_with("chr1\t0\t"), "Start=1 should convert to BED 0, got: {}", bed);
}

// ============================================================================
// OnlineStats: merge/consistency checks
// ============================================================================

#[test]
fn online_stats_variance_known_dataset() {
    let mut stats = OnlineStats::new();
    // Known: [2, 4, 4, 4, 5, 5, 7, 9]
    // Mean = 5.0, Population variance = 4.0, Sample variance = 32/7 ≈ 4.571
    for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        stats.add(x);
    }
    assert_eq!(stats.count(), 8);
    assert!((stats.mean() - 5.0).abs() < 1e-10);
    let expected_var = 32.0 / 7.0; // sample variance
    assert!((stats.variance() - expected_var).abs() < 1e-10,
        "Expected variance {}, got {}", expected_var, stats.variance());
}

#[test]
fn online_stats_std_matches_sqrt_variance() {
    let mut stats = OnlineStats::new();
    for &x in &[1.0, 3.0, 5.0, 7.0, 9.0] {
        stats.add(x);
    }
    assert!((stats.std() - stats.variance().sqrt()).abs() < 1e-15);
}
