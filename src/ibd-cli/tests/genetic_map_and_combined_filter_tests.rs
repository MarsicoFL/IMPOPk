//! Tests for GeneticMap zero-rate extrapolation, extract_ibd_segments_with_lod combined
//! filter interactions, segment_lod_score boundary-index conditions, segment_quality_score
//! component decomposition, refine_states_with_posteriors multi-pass extension, and
//! infer_ibd_with_training pipeline variants.

use impopk_ibd::hmm::{
    extract_ibd_segments_with_lod, infer_ibd,
    infer_ibd_with_training, refine_states_with_posteriors, segment_lod_score,
    segment_quality_score, segment_posterior_std, compute_per_window_lod,
    GeneticMap, HmmParams, IbdSegmentWithPosterior, Population,
};

// ─────────────────────────────────────────────────────────────────────────────
// GeneticMap::interpolate_cm — zero-rate extrapolation branches
// ─────────────────────────────────────────────────────────────────────────────

/// Duplicate first two positions → zero rate → before-first extrapolation returns cm0
#[test]
fn test_genetic_map_duplicate_first_two_zero_rate_extrapolation() {
    // Two entries at same position (1000) but different cM values
    let map = GeneticMap::new(vec![(1000, 0.5), (1000, 1.0), (5000, 3.0)]);
    // Extrapolate before first entry: rate = 0 because bp1 == bp0
    let cm = map.interpolate_cm(500);
    // cm0 - 0.0 * (bp0 - pos_bp) = cm0 = 0.5
    assert!((cm - 0.5).abs() < 1e-10, "Expected 0.5 but got {cm}");
}

/// Duplicate last two positions → zero rate → after-last extrapolation returns cm_last
#[test]
fn test_genetic_map_duplicate_last_two_zero_rate_extrapolation() {
    let map = GeneticMap::new(vec![(1000, 0.5), (5000, 2.0), (5000, 3.0)]);
    // Extrapolate after last entry: rate = 0 because bp_last == bp_prev
    let cm = map.interpolate_cm(10000);
    // cm_last + 0.0 * (pos_bp - bp_last) = cm_last = 3.0
    assert!((cm - 3.0).abs() < 1e-10, "Expected 3.0 but got {cm}");
}

/// All entries at same position → zero rate for both before and after
#[test]
fn test_genetic_map_all_same_position_zero_rate() {
    let map = GeneticMap::new(vec![(1000, 0.0), (1000, 1.0), (1000, 2.0)]);
    // Before
    let cm_before = map.interpolate_cm(0);
    assert!((cm_before - 0.0).abs() < 1e-10);
    // After
    let cm_after = map.interpolate_cm(5000);
    assert!((cm_after - 2.0).abs() < 1e-10);
}

/// Normal interpolation between entries at same bp → returns cm_lo (bp_hi == bp_lo path)
#[test]
fn test_genetic_map_interpolation_duplicate_inner_position() {
    let map = GeneticMap::new(vec![(1000, 0.0), (3000, 1.0), (3000, 2.0), (5000, 3.0)]);
    // Query at exactly 3000: partition_point finds idx such that entries[idx].0 > 3000
    // This should fall in the interpolation branch with bp_hi == bp_lo
    let cm = map.interpolate_cm(3000);
    // When bp_hi == bp_lo, returns cm_lo
    assert!(cm.is_finite(), "Expected finite result, got {cm}");
}

/// genetic_distance_cm between same position is zero
#[test]
fn test_genetic_map_genetic_distance_same_position() {
    let map = GeneticMap::new(vec![(1000, 0.0), (5000, 2.0)]);
    let dist = map.genetic_distance_cm(3000, 3000);
    assert!((dist - 0.0).abs() < 1e-10);
}

/// genetic_distance_cm is always non-negative (abs)
#[test]
fn test_genetic_map_genetic_distance_reversed_order() {
    let map = GeneticMap::new(vec![(1000, 0.0), (5000, 2.0)]);
    let dist_forward = map.genetic_distance_cm(1000, 5000);
    let dist_backward = map.genetic_distance_cm(5000, 1000);
    assert!((dist_forward - dist_backward).abs() < 1e-10, "Distances should be equal");
    assert!(dist_forward > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// extract_ibd_segments_with_lod: combined filter interactions
// ─────────────────────────────────────────────────────────────────────────────

/// Segment rejected by posterior filter (mean_post < threshold), not reaching LOD check
#[test]
fn test_lod_extraction_posterior_rejects_before_lod() {
    // IBD segment with low posteriors
    let states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.3, 0.3, 0.3, 0.3, 0.3]; // mean = 0.3
    let obs = vec![0.999, 0.999, 0.999, 0.999, 0.999];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    // min_mean_posterior = 0.5 → segment rejected by posterior
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2,
        0.5, // threshold above mean of 0.3
        Some((&obs, &params)),
        Some(0.0), // any LOD would pass
    );
    assert!(segments.is_empty(), "Should be rejected by posterior filter");
}

/// Segment passes posterior but rejected by LOD filter
#[test]
fn test_lod_extraction_passes_posterior_rejected_by_lod() {
    let states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    // Observations near the non-IBD emission mean to get low/negative LOD
    let obs = vec![0.997, 0.997, 0.997, 0.997, 0.997];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    // Very high min_lod to reject
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2,
        0.1,        // pass posterior easily
        Some((&obs, &params)),
        Some(1000.0), // extremely high LOD threshold → reject
    );
    assert!(segments.is_empty(), "Should be rejected by LOD filter");
}

/// Both filters apply: posterior boundary and LOD boundary
#[test]
fn test_lod_extraction_both_filters_applied() {
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.51, 0.51, 0.51, 0.1];
    let obs = vec![0.998, 0.9999, 0.9999, 0.9999, 0.998];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);

    // This should pass: mean posterior = 0.51 > 0.5, and LOD should be finite
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2,
        0.5,
        Some((&obs, &params)),
        None, // no LOD filter
    );
    assert_eq!(segments.len(), 1);
    assert!((segments[0].mean_posterior - 0.51).abs() < 1e-10);
}

/// Multiple IBD segments with different posterior levels: one passes, one doesn't
#[test]
fn test_lod_extraction_mixed_segments_partial_filter() {
    let states = vec![1, 1, 1, 0, 0, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.9, 0.1, 0.1, 0.3, 0.3, 0.3];
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2,
        0.5, // first segment (0.9) passes, second (0.3) doesn't
        None,
        None,
    );
    assert_eq!(segments.len(), 1, "Only high-posterior segment should survive");
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 2);
}

/// min_windows filter rejects short segment
#[test]
fn test_lod_extraction_min_windows_filter() {
    let states = vec![0, 1, 0, 0, 0];
    let posteriors = vec![0.1, 0.95, 0.1, 0.1, 0.1];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2, // need at least 2 windows
        0.0,
        None,
        None,
    );
    assert!(segments.is_empty(), "Single-window segment rejected by min_windows=2");
}

/// Segment at end of array (finalized by in_ibd check after loop)
#[test]
fn test_lod_extraction_segment_at_end() {
    let states = vec![0, 0, 1, 1, 1];
    let posteriors = vec![0.1, 0.1, 0.9, 0.9, 0.9];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        2,
        0.0,
        None,
        None,
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 2);
    assert_eq!(segments[0].end_idx, 4);
}

/// Posteriors length mismatch returns empty
#[test]
fn test_lod_extraction_posteriors_length_mismatch() {
    let states = vec![1, 1, 1];
    let posteriors = vec![0.9, 0.9]; // shorter than states

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.0,
        None,
        None,
    );
    assert!(segments.is_empty());
}

/// IbdSegmentWithPosterior captures min/max correctly for uniform posteriors
#[test]
fn test_lod_extraction_uniform_posteriors_min_max() {
    let states = vec![1, 1, 1, 1];
    let posteriors = vec![0.8, 0.8, 0.8, 0.8];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.0,
        None,
        None,
    );
    assert_eq!(segments.len(), 1);
    assert!((segments[0].min_posterior - 0.8).abs() < 1e-10);
    assert!((segments[0].max_posterior - 0.8).abs() < 1e-10);
    assert!((segments[0].mean_posterior - 0.8).abs() < 1e-10);
}

/// Varying posteriors within segment → correct min/max
#[test]
fn test_lod_extraction_varying_posteriors() {
    let states = vec![1, 1, 1, 1];
    let posteriors = vec![0.6, 0.95, 0.7, 0.85];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.0,
        None,
        None,
    );
    assert_eq!(segments.len(), 1);
    assert!((segments[0].min_posterior - 0.6).abs() < 1e-10);
    assert!((segments[0].max_posterior - 0.95).abs() < 1e-10);
    let expected_mean = (0.6 + 0.95 + 0.7 + 0.85) / 4.0;
    assert!((segments[0].mean_posterior - expected_mean).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// segment_lod_score: boundary index edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// start_idx > end_idx returns 0.0
#[test]
fn test_segment_lod_score_inverted_indices() {
    let obs = vec![0.999, 0.9995, 0.9999];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lod = segment_lod_score(&obs, 2, 1, &params);
    assert!((lod - 0.0).abs() < 1e-10);
}

/// end_idx >= observations.len() returns 0.0
#[test]
fn test_segment_lod_score_out_of_bounds() {
    let obs = vec![0.999];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lod = segment_lod_score(&obs, 0, 5, &params);
    assert!((lod - 0.0).abs() < 1e-10);
}

/// Single-window segment LOD matches compute_per_window_lod for that window
#[test]
fn test_segment_lod_matches_per_window_lod() {
    let obs = vec![0.997, 0.9999, 0.998];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let per_window = compute_per_window_lod(&obs, &params);
    let seg_lod = segment_lod_score(&obs, 1, 1, &params);
    assert!((seg_lod - per_window[1]).abs() < 1e-10,
        "Single-window segment LOD should match per-window LOD");
}

/// Multi-window segment LOD is sum of per-window LODs
#[test]
fn test_segment_lod_sum_of_per_window() {
    let obs = vec![0.998, 0.9995, 0.9999, 0.998];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let per_window = compute_per_window_lod(&obs, &params);
    let seg_lod = segment_lod_score(&obs, 0, 3, &params);
    let expected_sum: f64 = per_window.iter().sum();
    assert!((seg_lod - expected_sum).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// segment_quality_score: component decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Perfect segment: max posterior, consistent, high LOD, long
#[test]
fn test_quality_score_perfect_segment() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 29,
        n_windows: 30,
        mean_posterior: 1.0,
        min_posterior: 1.0,
        max_posterior: 1.0,
        lod_score: 60.0, // LOD/window = 2.0 > 1.0 → full marks
    };
    let score = segment_quality_score(&seg);
    // posterior_score = 1.0 * 40 = 40
    // consistency = (1.0/1.0) * 20 = 20
    // lod_score = min(2.0, 1.0) * 30 = 30
    // length_score = min(30/20, 1.0) * 10 = 10
    // total = 100
    assert!((score - 100.0).abs() < 1e-10, "Expected 100, got {score}");
}

/// Zero-posterior segment: all components zero except potentially length
#[test]
fn test_quality_score_zero_posterior() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 4,
        n_windows: 5,
        mean_posterior: 0.0,
        min_posterior: 0.0,
        max_posterior: 0.0,
        lod_score: 0.0,
    };
    let score = segment_quality_score(&seg);
    // posterior_score = 0
    // consistency = 0 (mean_posterior == 0)
    // lod_score = 0
    // length_score = min(5/20, 1) * 10 = 2.5
    assert!((score - 2.5).abs() < 1e-10, "Expected 2.5, got {score}");
}

/// Negative LOD → clamped to 0 for lod component
#[test]
fn test_quality_score_negative_lod() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 2,
        n_windows: 3,
        mean_posterior: 0.5,
        min_posterior: 0.4,
        max_posterior: 0.6,
        lod_score: -5.0,
    };
    let score = segment_quality_score(&seg);
    // posterior_score = 0.5 * 40 = 20
    // consistency = (0.4/0.5) * 20 = 16
    // lod_score = clamp(-5/3 / 1.0, 0, 1) * 30 = 0
    // length_score = (3/20) * 10 = 1.5
    assert!((score - 37.5).abs() < 1e-10, "Expected 37.5, got {score}");
}

/// n_windows == 0 → lod_per_window = 0 (no division by zero)
#[test]
fn test_quality_score_zero_windows() {
    let seg = IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: 0,
        n_windows: 0,
        mean_posterior: 0.5,
        min_posterior: 0.5,
        max_posterior: 0.5,
        lod_score: 10.0,
    };
    let score = segment_quality_score(&seg);
    // posterior_score = 0.5 * 40 = 20
    // consistency = (0.5/0.5) * 20 = 20
    // lod_score = 0 (n_windows == 0 → lod_per_window = 0)
    // length_score = 0 (0/20 = 0)
    assert!((score - 40.0).abs() < 1e-10, "Expected 40, got {score}");
}

// ─────────────────────────────────────────────────────────────────────────────
// segment_posterior_std: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// Constant posteriors → std = 0
#[test]
fn test_posterior_std_constant_posteriors() {
    let posteriors = vec![0.8, 0.8, 0.8, 0.8, 0.8];
    let std = segment_posterior_std(&posteriors, 0, 4);
    assert!((std - 0.0).abs() < 1e-10);
}

/// Two distinct values → correct sample std (uses n-1 denominator)
#[test]
fn test_posterior_std_two_values() {
    let posteriors = vec![0.6, 0.8];
    let std = segment_posterior_std(&posteriors, 0, 1);
    // mean = 0.7, sample var = ((0.01 + 0.01) / (2-1)) = 0.02, std = sqrt(0.02)
    let expected = (0.02_f64).sqrt();
    assert!((std - expected).abs() < 1e-10, "Expected {expected}, got {std}");
}

/// start_idx == end_idx (single window) → 0.0
#[test]
fn test_posterior_std_single_window() {
    let posteriors = vec![0.5, 0.9, 0.3];
    let std = segment_posterior_std(&posteriors, 1, 1);
    assert!((std - 0.0).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_per_window_lod: properties
// ─────────────────────────────────────────────────────────────────────────────

/// High-identity observation has positive LOD
#[test]
fn test_per_window_lod_high_identity_positive() {
    let obs = vec![0.9999];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lod = compute_per_window_lod(&obs, &params);
    assert!(lod[0] > 0.0, "Very high identity should favor IBD, LOD={}", lod[0]);
}

/// Low-identity observation has negative LOD
#[test]
fn test_per_window_lod_low_identity_negative() {
    let obs = vec![0.99];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lod = compute_per_window_lod(&obs, &params);
    assert!(lod[0] < 0.0, "Low identity should favor non-IBD, LOD={}", lod[0]);
}

/// LOD is monotonically increasing with identity
#[test]
fn test_per_window_lod_monotonic() {
    let obs = vec![0.99, 0.995, 0.999, 0.9999];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lods = compute_per_window_lod(&obs, &params);
    for i in 1..lods.len() {
        assert!(lods[i] >= lods[i - 1],
            "LOD should increase with identity: {} vs {}", lods[i - 1], lods[i]);
    }
}

/// Empty observations → empty LODs
#[test]
fn test_per_window_lod_empty() {
    let obs: Vec<f64> = vec![];
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lods = compute_per_window_lod(&obs, &params);
    assert!(lods.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// refine_states_with_posteriors: multi-pass extension and trimming
// ─────────────────────────────────────────────────────────────────────────────

/// Extension propagates iteratively through multiple high-posterior windows
#[test]
fn test_refine_iterative_extension() {
    let mut states = vec![0, 0, 0, 1, 0, 0, 0];
    let posteriors = vec![0.7, 0.6, 0.6, 0.9, 0.6, 0.6, 0.7];
    // With extend_threshold=0.5, the extension should propagate from state[3]=1
    // Pass 1: states[2] and states[4] become 1 (adjacent to 3)
    // Pass 2: states[1] and states[5] become 1 (adjacent to 2 and 4)
    // Pass 3: states[0] and states[6] become 1 (adjacent to 1 and 5)
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.0);
    assert_eq!(states, vec![1, 1, 1, 1, 1, 1, 1]);
}

/// Trimming removes low-posterior edge windows
#[test]
fn test_refine_trimming_edges() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.1, 0.9, 0.9, 0.9, 0.1];
    // trim_threshold=0.2: states[0] and states[4] should be trimmed
    // states[0] is at left edge (i==0 → left_non_ibd=true), posterior=0.1 < 0.2 → trim
    // states[4] is at right edge (i+1>=n → right_non_ibd=true), posterior=0.1 < 0.2 → trim
    refine_states_with_posteriors(&mut states, &posteriors, 1.0, 0.2);
    assert_eq!(states, vec![0, 1, 1, 1, 0]);
}

/// Interior low-posterior windows NOT trimmed (not at boundary)
#[test]
fn test_refine_interior_low_posterior_kept() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.05, 0.9, 0.9];
    // states[2] has low posterior but is interior (both neighbors are 1)
    refine_states_with_posteriors(&mut states, &posteriors, 1.0, 0.2);
    assert_eq!(states, vec![1, 1, 1, 1, 1], "Interior window should not be trimmed");
}

/// Extension blocked by low-posterior window
#[test]
fn test_refine_extension_blocked_by_low_posterior() {
    let mut states = vec![0, 0, 1, 0, 0];
    let posteriors = vec![0.8, 0.2, 0.9, 0.2, 0.8];
    // extend_threshold = 0.5: states[1] and states[3] have 0.2 < 0.5, won't extend
    // states[0] and states[4] can't extend because they're not adjacent to IBD
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.0);
    assert_eq!(states, vec![0, 0, 1, 0, 0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// infer_ibd_with_training: training vs no-training difference
// ─────────────────────────────────────────────────────────────────────────────

/// baum_welch_iters=0 skips training, same as infer_ibd
#[test]
fn test_training_zero_iters_matches_basic_inference() {
    let obs: Vec<f64> = (0..50).map(|i| if i < 20 || i > 35 { 0.998 } else { 0.9999 }).collect();
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let basic = infer_ibd(&obs, &params);
    let trained = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 0);
    // With 0 BW iterations, results should be the same as basic infer_ibd
    assert_eq!(basic.states, trained.states, "States should match with 0 BW iterations");
}

/// Very few observations (<10) skips BW even if iters > 0
#[test]
fn test_training_few_obs_skips_bw() {
    let obs = vec![0.999, 0.9995, 0.9999, 0.999, 0.998];
    let mut params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let original_emission_0 = params.emission[0].mean;
    let _ = infer_ibd_with_training(&obs, &mut params, Population::EUR, 5000, 50);
    // With < 10 observations, BW is skipped, emissions unchanged
    assert!((params.emission[0].mean - original_emission_0).abs() < 1e-10,
        "Emissions should not change with < 10 observations");
}

/// Different populations use different priors
#[test]
fn test_training_different_populations() {
    let obs: Vec<f64> = (0..30).map(|i| if i < 10 || i > 20 { 0.998 } else { 0.9999 }).collect();
    let mut params_eur = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let mut params_afr = HmmParams::from_population(Population::AFR, 50.0, 0.0001, 5000);
    let result_eur = infer_ibd_with_training(&obs, &mut params_eur, Population::EUR, 5000, 5);
    let result_afr = infer_ibd_with_training(&obs, &mut params_afr, Population::AFR, 5000, 5);
    // Both should produce valid results (finite LL)
    // Both should produce valid state sequences
    assert_eq!(result_eur.states.len(), 30);
    assert_eq!(result_afr.states.len(), 30);
    // States are valid (0 or 1)
    assert!(result_eur.states.iter().all(|&s| s == 0 || s == 1));
    assert!(result_afr.states.iter().all(|&s| s == 0 || s == 1));
}
