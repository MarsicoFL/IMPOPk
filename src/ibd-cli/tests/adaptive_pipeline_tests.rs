//! Tests for adaptive IBD pipeline improvements:
//! - Composite segment scoring (LOD × length coupling)
//! - Adaptive IBD emission variance estimation
//! - Adaptive state refinement (per-segment thresholds)
//! - Adaptive gap bridging (flanking-segment-aware)

use hprc_ibd::hmm::{
    extract_ibd_segments_composite, extract_ibd_segments_with_lod,
    estimate_ibd_emission_std,
    refine_states_adaptive, refine_states_with_posteriors,
    bridge_ibd_gaps, bridge_ibd_gaps_adaptive,
    HmmParams, segment_lod_score,
};

// =============================================================================
// extract_ibd_segments_composite tests
// =============================================================================

#[test]
fn composite_rejects_below_hard_min() {
    // Segment of 3 windows, hard_min=5 → rejected regardless of quality
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.99, 0.99, 0.99, 0.1];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.99, 0.9999, 0.9999, 0.9999, 0.99];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        100, // soft_min_windows
        5,   // hard_min_windows
        0.0, // threshold
    );
    assert!(segs.is_empty(), "3-window segment should be rejected by hard_min=5");
}

#[test]
fn composite_accepts_short_high_quality_segment() {
    // 20 windows with very high posterior and LOD: should pass composite even though
    // it's below soft_min_windows=200
    let n = 20;
    let mut states = vec![0; n + 2];
    let mut posteriors = vec![0.1; n + 2];
    let mut obs = vec![0.997; n + 2];

    for i in 1..=n {
        states[i] = 1;
        posteriors[i] = 0.99;
        obs[i] = 0.9999;
    }

    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        200, // soft_min_windows (20 < 200, so length_factor < 1)
        5,   // hard_min
        0.01, // low threshold → should accept
    );
    assert!(!segs.is_empty(), "Short but high-quality segment should pass composite filter");
    assert_eq!(segs[0].n_windows, n);
}

#[test]
fn composite_rejects_weak_short_segment() {
    // Short segment with low posterior → low composite score
    let states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.3, 0.4, 0.35, 0.3, 0.25, 0.1];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.997, 0.998, 0.998, 0.998, 0.998, 0.998, 0.997];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        200,  // soft_min (5 << 200 → length_factor ≈ 0.025)
        3,    // hard_min
        0.5,  // high threshold → should reject weak segment
    );
    assert!(segs.is_empty(), "Weak short segment should be rejected by composite filter");
}

#[test]
fn composite_soft_min_length_factor() {
    // Segment at exactly soft_min_windows → length_factor = 1.0
    let n = 50;
    let mut states = vec![0; n + 2];
    let mut posteriors = vec![0.1; n + 2];
    let mut obs = vec![0.997; n + 2];

    for i in 1..=n {
        states[i] = 1;
        posteriors[i] = 0.95;
        obs[i] = 0.9999;
    }

    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        50,  // soft_min = n → length_factor = 1.0
        5,
        0.01,
    );
    assert!(!segs.is_empty());
    // The length_factor should be exactly 1.0, so score depends on LOD density and posterior
}

#[test]
fn composite_empty_states() {
    let segs = extract_ibd_segments_composite(
        &[], &[], None, 100, 5, 0.3,
    );
    assert!(segs.is_empty());
}

#[test]
fn composite_mismatched_lengths() {
    let states = vec![0, 1, 1, 0];
    let posteriors = vec![0.1, 0.9]; // wrong length
    let segs = extract_ibd_segments_composite(
        &states, &posteriors, None, 100, 1, 0.0,
    );
    assert!(segs.is_empty());
}

#[test]
fn composite_no_ibd_states() {
    let states = vec![0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1];
    let segs = extract_ibd_segments_composite(
        &states, &posteriors, None, 1, 1, 0.0,
    );
    assert!(segs.is_empty());
}

#[test]
fn composite_all_ibd_single_segment() {
    let states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.98, 0.95, 0.9];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9999, 0.9999, 0.9999, 0.9999, 0.9999];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        5, 1, 0.01,
    );
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 4);
    assert_eq!(segs[0].n_windows, 5);
}

#[test]
fn composite_without_observations() {
    // Without obs/params, LOD = 0, so score = 0 * posterior * length_factor = 0
    // Only threshold=0.0 should accept
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.9, 0.9, 0.1];

    let segs_accept = extract_ibd_segments_composite(
        &states, &posteriors, None,
        3, 1, 0.0, // threshold 0.0 → accept everything
    );
    assert_eq!(segs_accept.len(), 1);

    let segs_reject = extract_ibd_segments_composite(
        &states, &posteriors, None,
        3, 1, 0.01, // threshold > 0 → reject (LOD=0 → score=0)
    );
    assert!(segs_reject.is_empty());
}

#[test]
fn composite_segment_at_end() {
    let states = vec![0, 0, 1, 1, 1];
    let posteriors = vec![0.1, 0.1, 0.9, 0.95, 0.9];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.997, 0.997, 0.9999, 0.9999, 0.9999];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        3, 1, 0.01,
    );
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].end_idx, 4);
}

#[test]
fn composite_multiple_segments() {
    let states = vec![1, 1, 0, 0, 1, 1, 1, 0];
    let posteriors = vec![0.9, 0.9, 0.1, 0.1, 0.95, 0.95, 0.95, 0.1];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.9999, 0.9999, 0.997, 0.997, 0.9999, 0.9999, 0.9999, 0.997];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        2, 1, 0.01,
    );
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].n_windows, 2);
    assert_eq!(segs[1].n_windows, 3);
}

// =============================================================================
// estimate_ibd_emission_std tests
// =============================================================================

#[test]
fn estimate_ibd_std_basic() {
    // Create observations with clear IBD/non-IBD distribution
    let mut obs = Vec::new();
    for _ in 0..100 {
        obs.push(0.997); // non-IBD
    }
    for _ in 0..10 {
        obs.push(0.9998); // IBD
    }

    let std = estimate_ibd_emission_std(&obs, 0.05, 0.0003, 0.002);
    assert!(std.is_some());
    let std = std.unwrap();
    assert!(std >= 0.0003, "std should be >= min_std, got {}", std);
    assert!(std <= 0.002, "std should be <= max_std, got {}", std);
}

#[test]
fn estimate_ibd_std_too_few_observations() {
    let obs = vec![0.999; 15]; // < 20
    assert!(estimate_ibd_emission_std(&obs, 0.05, 0.0003, 0.002).is_none());
}

#[test]
fn estimate_ibd_std_exactly_20_observations() {
    let obs = vec![0.9997; 20];
    assert!(estimate_ibd_emission_std(&obs, 0.05, 0.0003, 0.002).is_some());
}

#[test]
fn estimate_ibd_std_uniform_data() {
    // All same value → variance = 0 → clamped to min_std
    let obs = vec![0.9999; 100];
    let std = estimate_ibd_emission_std(&obs, 0.05, 0.0003, 0.002).unwrap();
    assert!((std - 0.0003).abs() < 1e-10, "Uniform data should give min_std");
}

#[test]
fn estimate_ibd_std_high_variance() {
    // Very noisy data → clamped to max_std
    let mut obs: Vec<f64> = (0..100).map(|i| 0.9 + 0.1 * (i as f64 / 100.0)).collect();
    obs.sort_by(|a, b| b.total_cmp(a));
    let std = estimate_ibd_emission_std(&obs, 0.1, 0.0003, 0.002).unwrap();
    // With high variance in top decile, should clamp to max
    assert!(std <= 0.002);
}

#[test]
fn estimate_ibd_std_respects_quantile() {
    // Only top 5% = 5 observations from 100
    let mut obs = vec![0.997; 95];
    obs.extend(vec![0.9998; 5]);
    let std = estimate_ibd_emission_std(&obs, 0.05, 0.0003, 0.002).unwrap();
    // The top 5% are all 0.9998, so std of those should be 0 → clamped to min
    assert!((std - 0.0003).abs() < 1e-10);
}

#[test]
fn estimate_ibd_std_custom_bounds() {
    let obs = vec![0.9997; 50];
    let std = estimate_ibd_emission_std(&obs, 0.1, 0.001, 0.01).unwrap();
    assert!(std >= 0.001);
    assert!(std <= 0.01);
}

// =============================================================================
// refine_states_adaptive tests
// =============================================================================

#[test]
fn adaptive_refine_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_adaptive(&mut states, &posteriors);
    assert!(states.is_empty());
}

#[test]
fn adaptive_refine_mismatched_lengths() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.5]; // wrong length
    refine_states_adaptive(&mut states, &posteriors);
    // Should be a no-op
    assert_eq!(states, vec![0, 1, 0]);
}

#[test]
fn adaptive_refine_no_ibd() {
    let mut states = vec![0, 0, 0, 0];
    let posteriors = vec![0.1, 0.2, 0.1, 0.05];
    refine_states_adaptive(&mut states, &posteriors);
    assert_eq!(states, vec![0, 0, 0, 0]);
}

#[test]
fn adaptive_refine_extends_high_confidence() {
    // High-confidence segment (mean posterior > 0.8) should extend at threshold 0.3
    let mut states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.35, 0.95, 0.99, 0.95, 0.35, 0.1];

    refine_states_adaptive(&mut states, &posteriors);

    // Windows at idx 1 and 5 have posterior 0.35 > 0.3 (extend_thresh for high-confidence)
    assert_eq!(states[1], 1, "Should extend left into high-posterior window");
    assert_eq!(states[5], 1, "Should extend right into high-posterior window");
    assert_eq!(states[0], 0, "Should not extend into low-posterior window");
    assert_eq!(states[6], 0, "Should not extend into low-posterior window");
}

#[test]
fn adaptive_refine_trims_weak_segment() {
    // Weak segment (mean posterior < 0.5) should trim at threshold 0.15
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.05, 0.1, 0.45, 0.1, 0.05];

    refine_states_adaptive(&mut states, &posteriors);

    // Edge windows with posterior 0.1 < 0.15 (trim_thresh for weak segment) should be trimmed
    assert_eq!(states[1], 0, "Should trim left edge with posterior 0.1");
    assert_eq!(states[3], 0, "Should trim right edge with posterior 0.1");
    assert_eq!(states[2], 1, "Should keep center with posterior 0.45");
}

#[test]
fn adaptive_refine_vs_fixed_more_aggressive_extension() {
    // Fixed thresholds: extend at 0.5, so 0.4 won't extend
    // Adaptive with high-confidence segment: extend at 0.3, so 0.4 extends
    let states_init = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.4, 0.95, 0.99, 0.95, 0.4, 0.1];

    let mut states_fixed = states_init.clone();
    refine_states_with_posteriors(&mut states_fixed, &posteriors, 0.5, 0.2);

    let mut states_adaptive = states_init;
    refine_states_adaptive(&mut states_adaptive, &posteriors);

    // Fixed: windows at 1,5 have posterior 0.4 < 0.5 → NOT extended
    assert_eq!(states_fixed[1], 0);
    assert_eq!(states_fixed[5], 0);

    // Adaptive: windows at 1,5 have posterior 0.4 > 0.3 → extended
    assert_eq!(states_adaptive[1], 1);
    assert_eq!(states_adaptive[5], 1);
}

#[test]
fn adaptive_refine_preserves_confident_segment() {
    // All posteriors well above trim threshold → no trimming
    let mut states = vec![0, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.85, 0.95, 0.95, 0.85, 0.1];
    refine_states_adaptive(&mut states, &posteriors);
    assert_eq!(states[1], 1);
    assert_eq!(states[4], 1);
}

// =============================================================================
// bridge_ibd_gaps_adaptive tests
// =============================================================================

#[test]
fn adaptive_bridge_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert_eq!(n, 0);
}

#[test]
fn adaptive_bridge_disabled() {
    let mut states = vec![0, 1, 0, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.5, 0.9, 0.1];
    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 0, 0.3);
    assert_eq!(n, 0);
}

#[test]
fn adaptive_bridge_high_confidence_flanks() {
    // Gap between two high-confidence segments should bridge more aggressively
    let mut states = vec![1, 1, 1, 0, 1, 1, 1];
    let posteriors = vec![0.95, 0.99, 0.95, 0.2, 0.95, 0.99, 0.95];

    // Standard bridge at 0.3 would reject (gap posterior 0.2 < 0.3)
    let mut states_std = states.clone();
    let n_std = bridge_ibd_gaps(&mut states_std, &posteriors, 1, 0.3);
    assert_eq!(n_std, 0, "Standard bridge should NOT bridge at 0.2 < 0.3");

    // Adaptive bridge: flanks have mean ~0.96, so threshold reduced to ~0.15
    let n_adaptive = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 1, 0.3);
    assert_eq!(n_adaptive, 1, "Adaptive bridge should bridge with high-confidence flanks");
    assert_eq!(states[3], 1, "Gap should be filled");
}

#[test]
fn adaptive_bridge_weak_flanks() {
    // Gap between weak segments: use full base threshold
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.45, 0.45, 0.25, 0.45, 0.45];

    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 1, 0.3);
    // Flank quality ~0.45 (< 0.5) → use full threshold 0.3
    // Gap posterior 0.25 < 0.3 → NOT bridged
    assert_eq!(n, 0, "Weak flanks should NOT reduce bridge threshold");
    assert_eq!(states[2], 0);
}

#[test]
fn adaptive_bridge_gap_too_long() {
    let mut states = vec![1, 1, 0, 0, 0, 1, 1];
    let posteriors = vec![0.95, 0.95, 0.5, 0.5, 0.5, 0.95, 0.95];
    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert_eq!(n, 0, "3-window gap with max_gap=2 should not be bridged");
}

#[test]
fn adaptive_bridge_multiple_gaps() {
    let mut states = vec![1, 1, 0, 1, 1, 0, 1, 1];
    let posteriors = vec![0.95, 0.95, 0.2, 0.95, 0.95, 0.2, 0.95, 0.95];

    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 1, 0.3);
    // Both gaps have high-confidence flanks → both should be bridged
    assert_eq!(n, 2);
    assert_eq!(states[2], 1);
    assert_eq!(states[5], 1);
}

#[test]
fn adaptive_bridge_no_second_segment() {
    // Gap at end, no second IBD segment
    let mut states = vec![1, 1, 0, 0];
    let posteriors = vec![0.95, 0.95, 0.5, 0.5];
    let n = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert_eq!(n, 0, "Gap not followed by IBD should not be bridged");
}

// =============================================================================
// Integration tests: composite vs traditional filtering
// =============================================================================

#[test]
fn composite_vs_traditional_traditional_more_strict_for_short_segments() {
    // Traditional filtering with min_windows=50 rejects a 20-window segment
    // Composite filtering can accept it if LOD and posterior are high
    let n = 22;
    let mut states = vec![0; n];
    let mut posteriors = vec![0.05; n];
    let mut obs = vec![0.997; n];

    for i in 1..=20 {
        states[i] = 1;
        posteriors[i] = 0.98;
        obs[i] = 0.9999;
    }

    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);

    // Traditional: min_windows=50 → rejects 20-window segment
    let trad = extract_ibd_segments_with_lod(
        &states, &posteriors, 50, 0.0,
        Some((&obs, &params)), None,
    );
    assert!(trad.is_empty(), "Traditional filter should reject 20 < 50 windows");

    // Composite: soft_min=50, hard_min=5, low threshold → accepts
    let comp = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        50, 5, 0.01,
    );
    assert!(!comp.is_empty(), "Composite filter should accept high-quality short segment");
}

#[test]
fn composite_preserves_lod_score() {
    let states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.98, 0.99, 0.98, 0.95, 0.1];
    let params = HmmParams::from_expected_length(50.0, 0.001, 5000);
    let obs = vec![0.997, 0.9999, 0.9999, 0.9999, 0.9999, 0.9999, 0.997];

    let segs = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)),
        5, 1, 0.0,
    );
    assert_eq!(segs.len(), 1);

    // LOD should match manual computation
    let expected_lod = segment_lod_score(&obs, 1, 5, &params);
    assert!((segs[0].lod_score - expected_lod).abs() < 1e-10);
}

// =============================================================================
// Edge cases: adaptive refinement with single-window segments
// =============================================================================

#[test]
fn adaptive_refine_single_window_segment() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.6, 0.1];
    refine_states_adaptive(&mut states, &posteriors);
    // Single-window segment with mean_post=0.6 (medium confidence)
    // trim_thresh for 0.5-0.8 range: 0.15 + (0.6-0.5)*0.5 = 0.2
    // posterior 0.6 > 0.2 → not trimmed
    assert_eq!(states[1], 1);
}

#[test]
fn adaptive_refine_two_segments() {
    // Two segments with different quality → different thresholds applied
    let mut states = vec![0, 1, 1, 1, 0, 0, 1, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.95, 0.9, 0.1, 0.1, 0.4, 0.3, 0.1];

    refine_states_adaptive(&mut states, &posteriors);

    // First segment: high quality (mean ~0.92) → extend at 0.3
    // Second segment: low quality (mean ~0.35) → extend at 0.6, trim at 0.15
    // Second segment edge at idx 7 has posterior 0.3 > 0.15 → not trimmed
    assert_eq!(states[1], 1, "First segment should be preserved");
    assert_eq!(states[2], 1);
    assert_eq!(states[3], 1);
}
