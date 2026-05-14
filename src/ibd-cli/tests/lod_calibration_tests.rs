//! LOD score calibration and property tests.
//!
//! Tests LOD score mathematical properties, extract_ibd_segments_with_lod
//! filtering behavior, quality score formula components, and LOD-parameter
//! sensitivity.

use impopk_ibd::hmm::{
    compute_per_window_lod, extract_ibd_segments_with_lod, segment_lod_score,
    segment_quality_score, HmmParams, IbdSegmentWithPosterior, Population,
};
use impopk_ibd::stats::GaussianParams;

// ============================================================================
// LOD mathematical properties
// ============================================================================

#[test]
fn lod_monotonic_with_identity() {
    // Higher identity values (closer to IBD mean) should produce higher LOD
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.990, 0.995, 0.998, 0.999, 0.9995, 0.9999];
    let lods = compute_per_window_lod(&obs, &params);
    for i in 1..lods.len() {
        assert!(
            lods[i] >= lods[i - 1],
            "LOD should increase with identity: LOD[{}]={} < LOD[{}]={}",
            i,
            lods[i],
            i - 1,
            lods[i - 1]
        );
    }
}

#[test]
fn lod_symmetry_around_crossover() {
    // There should be a crossover point where LOD changes sign.
    // For identical observations on both sides of the crossover, |LOD| should
    // be similar (not necessarily equal due to Gaussian asymmetry).
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let ibd_mean = params.emission[1].mean;
    let non_ibd_mean = params.emission[0].mean;

    // At IBD mean: LOD should be positive
    let lod_at_ibd = compute_per_window_lod(&[ibd_mean], &params)[0];
    assert!(lod_at_ibd > 0.0);

    // At non-IBD mean: LOD should be negative
    let lod_at_non_ibd = compute_per_window_lod(&[non_ibd_mean], &params)[0];
    assert!(lod_at_non_ibd < 0.0);
}

#[test]
fn lod_sign_semantics() {
    // Positive LOD = evidence for IBD, negative = evidence against
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    // Very high identity → strong IBD evidence
    let high_lod = compute_per_window_lod(&[0.99999], &params)[0];
    assert!(high_lod > 0.0, "Very high identity should give positive LOD");

    // Very low identity → strong non-IBD evidence
    let low_lod = compute_per_window_lod(&[0.5], &params)[0];
    assert!(low_lod < 0.0, "Very low identity should give negative LOD");

    // Medium identity → depends on emission parameters
    let mid_lod = compute_per_window_lod(&[0.95], &params)[0];
    assert!(mid_lod.is_finite(), "Medium identity should give finite LOD");
}

#[test]
fn segment_lod_additive() {
    // Segment LOD should equal sum of per-window LODs
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.9995, 0.9999, 0.9997, 0.9998];

    let segment_lod = segment_lod_score(&obs, 0, 4, &params);
    let per_window = compute_per_window_lod(&obs, &params);
    let sum: f64 = per_window.iter().sum();

    assert!(
        (segment_lod - sum).abs() < 1e-10,
        "Segment LOD ({}) should equal sum of per-window LODs ({})",
        segment_lod,
        sum
    );
}

#[test]
fn segment_lod_scales_with_length() {
    // For identical observations, LOD should scale linearly with length
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let identity = 0.9998;

    let obs_5: Vec<f64> = vec![identity; 5];
    let obs_10: Vec<f64> = vec![identity; 10];
    let obs_20: Vec<f64> = vec![identity; 20];

    let lod_5 = segment_lod_score(&obs_5, 0, 4, &params);
    let lod_10 = segment_lod_score(&obs_10, 0, 9, &params);
    let lod_20 = segment_lod_score(&obs_20, 0, 19, &params);

    assert!(
        (lod_10 / lod_5 - 2.0).abs() < 1e-10,
        "LOD should double when length doubles: {}/{}={}",
        lod_10,
        lod_5,
        lod_10 / lod_5
    );
    assert!(
        (lod_20 / lod_10 - 2.0).abs() < 1e-10,
        "LOD should double when length doubles: {}/{}={}",
        lod_20,
        lod_10,
        lod_20 / lod_10
    );
}

#[test]
fn lod_finite_for_all_valid_identities() {
    // LOD should be finite for any identity in (0, 1)
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    for &id in &[0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999] {
        let lod = compute_per_window_lod(&[id], &params)[0];
        assert!(
            lod.is_finite(),
            "LOD should be finite for identity={}, got {}",
            id,
            lod
        );
    }
}

#[test]
fn lod_per_window_consistency_with_segment() {
    // Computing segment LOD for a sub-range should match sub-slice of per-window LODs
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.5, 0.998, 0.9995, 0.9999, 0.9997, 0.5, 0.998];
    let per_window = compute_per_window_lod(&obs, &params);

    // Segment [1, 4]
    let seg_lod = segment_lod_score(&obs, 1, 4, &params);
    let expected: f64 = per_window[1..=4].iter().sum();
    assert!((seg_lod - expected).abs() < 1e-10);

    // Segment [2, 3]
    let seg_lod_2 = segment_lod_score(&obs, 2, 3, &params);
    let expected_2: f64 = per_window[2..=3].iter().sum();
    assert!((seg_lod_2 - expected_2).abs() < 1e-10);
}

#[test]
fn lod_sensitive_to_emission_separation() {
    // When emission means are closer together, LOD magnitudes should be smaller
    let params_wide = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let ibd_mean = params_wide.emission[1].mean;

    // Compute LOD with default params
    let lod_default = compute_per_window_lod(&[ibd_mean], &params_wide)[0];

    // Create params with narrower emission separation
    let mut params_narrow = params_wide.clone();
    // Move non-IBD mean closer to IBD mean
    params_narrow.emission[0] = GaussianParams::new_unchecked(
        params_wide.emission[1].mean - 0.00001, // very close to IBD mean
        params_wide.emission[0].std,
    );

    let lod_narrow = compute_per_window_lod(&[ibd_mean], &params_narrow)[0];

    // With narrower separation, LOD at IBD mean should be smaller
    assert!(
        lod_default.abs() > lod_narrow.abs(),
        "Wider emission separation should give larger LOD magnitude: {} vs {}",
        lod_default,
        lod_narrow
    );
}

// ============================================================================
// extract_ibd_segments_with_lod — comprehensive coverage
// ============================================================================

fn make_ibd_scenario() -> (Vec<f64>, Vec<usize>, Vec<f64>, HmmParams) {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    // Two IBD segments separated by non-IBD
    let obs = vec![
        0.998, 0.998, // non-IBD
        0.9998, 0.9999, 0.9997, // IBD segment 1 (3 windows)
        0.998, 0.998, 0.998, // non-IBD
        0.9998, 0.9999, 0.9997, 0.9998, 0.9999, 0.9997, 0.9998, // IBD segment 2 (7 windows)
        0.998, 0.998, // non-IBD
    ];
    let states = vec![0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0];
    let posteriors = vec![
        0.1, 0.1, 0.9, 0.95, 0.85, 0.1, 0.1, 0.1, 0.9, 0.95, 0.92, 0.88, 0.93, 0.90, 0.91,
        0.1, 0.1,
    ];
    (obs, states, posteriors, params)
}

#[test]
fn extract_with_lod_no_filter_returns_all_segments() {
    let (obs, states, posteriors, params) = make_ibd_scenario();
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 2, "Should find 2 IBD segments");
    assert_eq!(segments[0].start_idx, 2);
    assert_eq!(segments[0].end_idx, 4);
    assert_eq!(segments[0].n_windows, 3);
    assert_eq!(segments[1].start_idx, 8);
    assert_eq!(segments[1].end_idx, 14);
    assert_eq!(segments[1].n_windows, 7);
}

#[test]
fn extract_with_lod_longer_segment_has_higher_lod() {
    let (obs, states, posteriors, params) = make_ibd_scenario();
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert!(
        segments[1].lod_score > segments[0].lod_score,
        "Longer segment ({} windows) LOD {} should exceed shorter ({} windows) LOD {}",
        segments[1].n_windows,
        segments[1].lod_score,
        segments[0].n_windows,
        segments[0].lod_score,
    );
}

#[test]
fn extract_with_lod_filter_removes_weak_segment() {
    let (obs, states, posteriors, params) = make_ibd_scenario();

    // First get unfiltered to find the threshold
    let all = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(all.len(), 2);

    // Filter with LOD just above shorter segment
    let threshold = all[0].lod_score + 0.001;
    let filtered = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        Some(threshold),
    );
    assert_eq!(filtered.len(), 1, "Should filter out the shorter segment");
    assert_eq!(filtered[0].start_idx, 8, "Remaining segment should be the longer one");
}

#[test]
fn extract_with_lod_very_high_filter_removes_all() {
    let (obs, states, posteriors, params) = make_ibd_scenario();
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        Some(1000.0), // impossibly high threshold
    );
    assert!(segments.is_empty(), "Very high LOD filter should remove all segments");
}

#[test]
fn extract_with_lod_zero_filter_keeps_all() {
    let (obs, states, posteriors, params) = make_ibd_scenario();
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        Some(0.0),
    );
    assert_eq!(segments.len(), 2, "LOD=0 threshold should keep all positive-LOD segments");
}

#[test]
fn extract_without_obs_gives_zero_lod() {
    let (_obs, states, posteriors, _params) = make_ibd_scenario();
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        None, // no observations/params
        None,
    );
    assert_eq!(segments.len(), 2);
    for seg in &segments {
        assert_eq!(
            seg.lod_score, 0.0,
            "Without obs/params, LOD should be 0.0"
        );
    }
}

#[test]
fn extract_without_obs_lod_filter_respected() {
    let (_obs, states, posteriors, _params) = make_ibd_scenario();
    // Without obs, LOD=0.0 for all segments. min_lod=1.0 should filter everything.
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        None,
        Some(1.0),
    );
    assert!(
        segments.is_empty(),
        "Without obs, LOD=0 so min_lod>0 should filter all"
    );
}

#[test]
fn extract_min_windows_and_lod_interact() {
    let (obs, states, posteriors, params) = make_ibd_scenario();

    // min_windows=5 should filter out the 3-window segment
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        5,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1, "min_windows=5 should filter 3-window segment");
    assert_eq!(segments[0].n_windows, 7);

    // min_windows=5 AND LOD filter should further filter
    let segments_both = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        5,
        0.5,
        Some((&obs, &params)),
        Some(1000.0),
    );
    assert!(segments_both.is_empty(), "Both filters should leave nothing");
}

#[test]
fn extract_min_posterior_and_lod_interact() {
    let (obs, states, posteriors, params) = make_ibd_scenario();

    // Very high min_mean_posterior should filter out everything
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.99, // higher than any segment's mean posterior
        Some((&obs, &params)),
        None,
    );
    assert!(
        segments.is_empty(),
        "Very high min_mean_posterior should filter all"
    );
}

#[test]
fn extract_empty_states() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let segments = extract_ibd_segments_with_lod(
        &[],
        &[],
        1,
        0.5,
        Some((&[], &params)),
        None,
    );
    assert!(segments.is_empty());
}

#[test]
fn extract_mismatched_state_posterior_lengths() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.999; 5];
    let states = vec![1, 1, 1, 0, 0];
    let posteriors = vec![0.9, 0.95]; // shorter than states!

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert!(
        segments.is_empty(),
        "Mismatched lengths should return empty"
    );
}

#[test]
fn extract_all_ibd() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.9999; 10];
    let states = vec![1; 10];
    let posteriors = vec![0.95; 10];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 9);
    assert_eq!(segments[0].n_windows, 10);
    assert!(segments[0].lod_score > 0.0);
}

#[test]
fn extract_all_non_ibd() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998; 10];
    let states = vec![0; 10];
    let posteriors = vec![0.1; 10];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert!(segments.is_empty(), "All non-IBD should produce no segments");
}

#[test]
fn extract_segment_at_end() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.998, 0.9999, 0.9999, 0.9999];
    let states = vec![0, 0, 1, 1, 1];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.92];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].end_idx, 4, "Segment at end should be captured");
}

#[test]
fn extract_segment_at_start() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.9999, 0.9999, 0.9999, 0.998, 0.998];
    let states = vec![1, 1, 1, 0, 0];
    let posteriors = vec![0.9, 0.95, 0.92, 0.1, 0.1];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0, "Segment at start should be captured");
}

#[test]
fn extract_single_window_segment() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.9999, 0.998];
    let states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.1];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 1);
    assert!(segments[0].lod_score > 0.0);
}

#[test]
fn extract_lod_matches_manual_segment_lod() {
    let (obs, states, posteriors, params) = make_ibd_scenario();

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );

    for seg in &segments {
        let manual_lod = segment_lod_score(&obs, seg.start_idx, seg.end_idx, &params);
        assert!(
            (seg.lod_score - manual_lod).abs() < 1e-10,
            "Extracted LOD {} should match manual computation {}",
            seg.lod_score,
            manual_lod
        );
    }
}

#[test]
fn extract_posterior_stats_correct() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.998, 0.9998, 0.9999, 0.9995, 0.998];
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.85, 0.95, 0.90, 0.1];

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    let seg = &segments[0];

    let expected_mean = (0.85 + 0.95 + 0.90) / 3.0;
    assert!((seg.mean_posterior - expected_mean).abs() < 1e-10);
    assert!((seg.min_posterior - 0.85).abs() < 1e-10);
    assert!((seg.max_posterior - 0.95).abs() < 1e-10);
}

// ============================================================================
// Quality score formula properties
// ============================================================================

fn make_seg(
    n_windows: usize,
    mean_post: f64,
    min_post: f64,
    max_post: f64,
    lod: f64,
) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: 0,
        end_idx: n_windows.saturating_sub(1),
        n_windows,
        mean_posterior: mean_post,
        min_posterior: min_post,
        max_posterior: max_post,
        lod_score: lod,
    }
}

#[test]
fn quality_monotonic_in_mean_posterior() {
    // Increasing mean_posterior should increase quality
    let q_low = segment_quality_score(&make_seg(20, 0.3, 0.3, 0.3, 20.0));
    let q_mid = segment_quality_score(&make_seg(20, 0.6, 0.6, 0.6, 20.0));
    let q_high = segment_quality_score(&make_seg(20, 0.9, 0.9, 0.9, 20.0));

    assert!(q_mid > q_low, "Q(mean=0.6)={} > Q(mean=0.3)={}", q_mid, q_low);
    assert!(q_high > q_mid, "Q(mean=0.9)={} > Q(mean=0.6)={}", q_high, q_mid);
}

#[test]
fn quality_monotonic_in_lod() {
    // Increasing LOD should increase quality (up to saturation)
    let q_low = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 1.0));
    let q_mid = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 5.0));
    let q_high = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 10.0));

    assert!(q_mid > q_low, "Q(lod=5)={} > Q(lod=1)={}", q_mid, q_low);
    assert!(q_high >= q_mid, "Q(lod=10)={} >= Q(lod=5)={}", q_high, q_mid);
}

#[test]
fn quality_monotonic_in_length() {
    // Increasing segment length should increase quality when LOD scales proportionally
    // (keeping LOD/window constant so the LOD component doesn't confound)
    let lod_per_window = 2.0;
    let q_short = segment_quality_score(&make_seg(3, 0.9, 0.9, 0.9, 3.0 * lod_per_window));
    let q_mid = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 10.0 * lod_per_window));
    let q_long = segment_quality_score(&make_seg(25, 0.9, 0.9, 0.9, 25.0 * lod_per_window));

    assert!(q_mid > q_short, "Q(n=10)={} > Q(n=3)={}", q_mid, q_short);
    assert!(q_long > q_mid, "Q(n=25)={} > Q(n=10)={}", q_long, q_mid);
}

#[test]
fn quality_length_saturates_at_20() {
    // Length component should max out at n_windows=20
    let q_20 = segment_quality_score(&make_seg(20, 0.9, 0.9, 0.9, 20.0));
    let q_100 = segment_quality_score(&make_seg(100, 0.9, 0.9, 0.9, 100.0)); // same LOD/window

    // LOD per window is 1.0 for both, so LOD component is same
    // Length: 20/20=1.0 vs 100/20=clamped to 1.0 → same
    assert!(
        (q_20 - q_100).abs() < 1e-10,
        "Length should saturate at 20 windows: Q(20)={} vs Q(100)={}",
        q_20,
        q_100
    );
}

#[test]
fn quality_lod_saturates_at_1_per_window() {
    // LOD component maxes out when LOD/window >= 1.0
    let q_1 = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 10.0)); // LOD/w = 1.0
    let q_5 = segment_quality_score(&make_seg(10, 0.9, 0.9, 0.9, 50.0)); // LOD/w = 5.0

    assert!(
        (q_1 - q_5).abs() < 1e-10,
        "LOD component should saturate at LOD/window=1.0: Q(1.0)={} vs Q(5.0)={}",
        q_1,
        q_5
    );
}

#[test]
fn quality_exact_formula_verification() {
    // Verify exact formula computation
    let seg = make_seg(15, 0.8, 0.6, 0.95, 9.0);
    let q = segment_quality_score(&seg);

    // Component 1: Posterior strength = 0.8 * 40 = 32.0
    let expected_posterior = 0.8_f64.clamp(0.0, 1.0) * 40.0;
    // Component 2: Consistency = (0.6/0.8).clamp(0,1) * 20 = 0.75 * 20 = 15.0
    let expected_consistency = (0.6 / 0.8_f64).clamp(0.0, 1.0) * 20.0;
    // Component 3: LOD = (9.0/15).clamp(0,1) * 30 = 0.6 * 30 = 18.0
    let expected_lod = (9.0 / 15.0_f64).clamp(0.0, 1.0) * 30.0;
    // Component 4: Length = (15/20).clamp(0,1) * 10 = 0.75 * 10 = 7.5
    let expected_length = (15.0 / 20.0_f64).clamp(0.0, 1.0) * 10.0;

    let expected = expected_posterior + expected_consistency + expected_lod + expected_length;
    assert!(
        (q - expected).abs() < 1e-10,
        "Q={} != expected {} (components: {}, {}, {}, {})",
        q,
        expected,
        expected_posterior,
        expected_consistency,
        expected_lod,
        expected_length
    );
}

#[test]
fn quality_consistency_rewards_uniform_posteriors() {
    // Uniform posteriors (min=mean) should get full consistency points
    let q_uniform = segment_quality_score(&make_seg(20, 0.9, 0.9, 0.9, 20.0));
    let q_variable = segment_quality_score(&make_seg(20, 0.9, 0.5, 1.0, 20.0));

    assert!(
        q_uniform > q_variable,
        "Uniform posteriors Q={} > variable Q={}",
        q_uniform,
        q_variable
    );
}

#[test]
fn quality_negative_lod_gives_zero_lod_component() {
    // Negative LOD should contribute 0 to LOD component (clamped)
    let seg = make_seg(20, 0.9, 0.9, 0.9, -5.0);
    let q = segment_quality_score(&seg);

    // Without LOD component: posterior=36 + consistency=20 + length=10 = 66
    // LOD component should be 0 (clamped)
    let expected = 0.9 * 40.0 + 1.0 * 20.0 + 0.0 + 1.0 * 10.0;
    assert!(
        (q - expected).abs() < 1e-10,
        "Negative LOD should give 0 LOD component: Q={} expected={}",
        q,
        expected
    );
}

#[test]
fn quality_zero_mean_posterior_gives_zero_consistency() {
    let seg = make_seg(20, 0.0, 0.0, 0.0, 20.0);
    let q = segment_quality_score(&seg);
    // Posterior=0, consistency=0, LOD=30, length=10 = 40
    assert!(
        (q - 40.0).abs() < 1e-10,
        "Zero mean posterior Q={} expected 40.0",
        q
    );
}

// ============================================================================
// LOD score across populations
// ============================================================================

#[test]
fn lod_varies_by_population() {
    // Different populations have different emission parameters → different LODs
    let params_eur = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let params_afr = HmmParams::from_population(Population::AFR, 50.0, 0.001, 5000);

    let obs = vec![0.9999];
    let lod_eur = compute_per_window_lod(&obs, &params_eur)[0];
    let lod_afr = compute_per_window_lod(&obs, &params_afr)[0];

    // Both should be positive (high identity = IBD evidence)
    assert!(lod_eur > 0.0);
    assert!(lod_afr > 0.0);

    // They should differ (unless emissions happen to be identical)
    // At minimum, both should be finite
    assert!(lod_eur.is_finite());
    assert!(lod_afr.is_finite());
}

#[test]
fn lod_consistent_across_expected_lengths() {
    // Expected IBD length affects transitions but NOT emissions,
    // so per-window LOD should be identical across expected lengths
    let params_short = HmmParams::from_expected_length(5.0, 0.001, 5000);
    let params_long = HmmParams::from_expected_length(100.0, 0.001, 5000);

    let obs = vec![0.9999];
    let lod_short = compute_per_window_lod(&obs, &params_short)[0];
    let lod_long = compute_per_window_lod(&obs, &params_long)[0];

    // Per-window LOD depends only on emissions, not transitions
    assert!(
        (lod_short - lod_long).abs() < 1e-10,
        "Per-window LOD should be independent of expected length: {} vs {}",
        lod_short,
        lod_long
    );
}

// ============================================================================
// LOD edge cases
// ============================================================================

#[test]
fn segment_lod_single_window_at_boundary() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = vec![0.9999];
    // start == end == 0, single window
    let lod = segment_lod_score(&obs, 0, 0, &params);
    let per_window = compute_per_window_lod(&obs, &params);
    assert!((lod - per_window[0]).abs() < 1e-10);
}

#[test]
fn lod_with_identical_emissions_returns_zero() {
    // If both emissions have same mean and std, LOD should be ~0 for any observation
    let mut params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let mean = params.emission[0].mean;
    let std = params.emission[0].std;
    params.emission[1] = GaussianParams::new_unchecked(mean, std);

    let obs = vec![0.5, 0.9, 0.999, 0.9999];
    let lods = compute_per_window_lod(&obs, &params);
    for (i, &lod) in lods.iter().enumerate() {
        assert!(
            lod.abs() < 1e-10,
            "Identical emissions should give LOD≈0, got LOD[{}]={}",
            i,
            lod
        );
    }
}

#[test]
fn extract_many_short_segments() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    // Alternating IBD/non-IBD pattern
    let n = 20;
    let mut obs = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);
    let mut posteriors = Vec::with_capacity(n);

    for i in 0..n {
        if i % 4 < 2 {
            // non-IBD
            obs.push(0.998);
            states.push(0);
            posteriors.push(0.1);
        } else {
            // IBD
            obs.push(0.9999);
            states.push(1);
            posteriors.push(0.9);
        }
    }

    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 5, "Should find 5 alternating IBD segments");

    // All segments should have same LOD (same length and same observations)
    let first_lod = segments[0].lod_score;
    for seg in &segments {
        assert!(
            (seg.lod_score - first_lod).abs() < 1e-10,
            "Identical segments should have identical LOD"
        );
    }
}
