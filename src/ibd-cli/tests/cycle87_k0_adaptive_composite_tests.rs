//! Cycle 87: Deep edge-case tests for K0 emissions, adaptive refinement,
//! adaptive gap bridging, composite segment extraction, multi-scale inference,
//! emission std estimation, merge nearby segments, and aggregate observations.
//!
//! These functions had only 1-2 external test files. This adds targeted coverage
//! for boundary conditions, NaN/Inf handling, and algorithmic invariants.

use hprc_ibd::hmm::{
    aggregate_observations, augment_with_k0, bridge_ibd_gaps_adaptive, estimate_k0_emissions,
    extract_ibd_segments_composite, infer_ibd_multi_scale, k0_log_pmf,
    merge_nearby_ibd_segments, estimate_ibd_emission_std, refine_states_adaptive,
    HmmParams, IbdSegmentWithPosterior, Population,
};
use hprc_ibd::stats::GaussianParams;

fn make_params() -> HmmParams {
    HmmParams {
        initial: [0.95, 0.05],
        transition: [[0.99, 0.01], [0.02, 0.98]],
        emission: [
            GaussianParams { mean: 0.85, std: 0.05 },
            GaussianParams { mean: 0.95, std: 0.03 },
        ],
    }
}

fn make_seg(start: usize, end: usize, mean_post: f64, lod: f64) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: start,
        end_idx: end,
        n_windows: end - start + 1,
        mean_posterior: mean_post,
        min_posterior: mean_post * 0.8,
        max_posterior: (mean_post * 1.1).min(1.0),
        lod_score: lod,
    }
}

// ============================================================================
// aggregate_observations
// ============================================================================

#[test]
fn aggregate_empty() {
    let result = aggregate_observations(&[], 2);
    assert!(result.is_empty());
}

#[test]
fn aggregate_factor_zero_passthrough() {
    let obs = vec![1.0, 2.0, 3.0];
    let result = aggregate_observations(&obs, 0);
    assert_eq!(result, obs);
}

#[test]
fn aggregate_factor_one_passthrough() {
    let obs = vec![1.0, 2.0, 3.0];
    let result = aggregate_observations(&obs, 1);
    assert_eq!(result, obs);
}

#[test]
fn aggregate_factor_two() {
    let obs = vec![1.0, 3.0, 5.0, 7.0];
    let result = aggregate_observations(&obs, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-10); // (1+3)/2
    assert!((result[1] - 6.0).abs() < 1e-10); // (5+7)/2
}

#[test]
fn aggregate_factor_two_odd_length() {
    let obs = vec![1.0, 3.0, 5.0];
    let result = aggregate_observations(&obs, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-10); // (1+3)/2
    assert!((result[1] - 5.0).abs() < 1e-10); // just 5/1
}

#[test]
fn aggregate_factor_larger_than_length() {
    let obs = vec![2.0, 4.0, 6.0];
    let result = aggregate_observations(&obs, 10);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 4.0).abs() < 1e-10); // (2+4+6)/3
}

#[test]
fn aggregate_single_element() {
    let obs = vec![42.0];
    let result = aggregate_observations(&obs, 3);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 42.0).abs() < 1e-10);
}

#[test]
fn aggregate_preserves_nan() {
    let obs = vec![1.0, f64::NAN, 3.0, 4.0];
    let result = aggregate_observations(&obs, 2);
    assert!(result[0].is_nan()); // (1+NaN)/2 = NaN
    assert!((result[1] - 3.5).abs() < 1e-10);
}

// ============================================================================
// estimate_k0_emissions
// ============================================================================

#[test]
fn k0_empty_returns_priors() {
    let result = estimate_k0_emissions(&[], &[]);
    assert!((result[0] - 0.015).abs() < 1e-10);
    assert!((result[1] - 0.22).abs() < 1e-10);
}

#[test]
fn k0_all_ibd_posteriors() {
    // All posteriors = 1.0 (fully IBD)
    let indicators = vec![1.0, 1.0, 1.0];
    let posteriors = vec![1.0, 1.0, 1.0];
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // p_nonibd weight = 0 → prior 0.015
    assert!((result[0] - 0.015).abs() < 1e-10);
    // p_ibd: all indicators = 1.0, so w_k0[1] / w_sum[1] = 1.0, clamped to 0.999
    assert!((result[1] - 0.999).abs() < 1e-10);
}

#[test]
fn k0_all_nonibd_posteriors() {
    let indicators = vec![0.0, 0.0, 0.0];
    let posteriors = vec![0.0, 0.0, 0.0];
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // p_nonibd: indicators all 0, so ratio = 0, clamped to 0.001
    assert!((result[0] - 0.001).abs() < 1e-10);
    // p_ibd weight = 0 → prior 0.22
    assert!((result[1] - 0.22).abs() < 1e-10);
}

#[test]
fn k0_mixed_posteriors() {
    let indicators = vec![1.0, 0.0, 1.0, 0.0];
    let posteriors = vec![0.5, 0.5, 0.5, 0.5];
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // Each window: p_ibd=0.5, p_nonibd=0.5
    // w_k0[0] = 0.5*1 + 0.5*0 + 0.5*1 + 0.5*0 = 1.0
    // w_sum[0] = 0.5*4 = 2.0 → p0 = 0.5
    assert!((result[0] - 0.5).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
}

#[test]
fn k0_posteriors_shorter_than_indicators() {
    let indicators = vec![1.0, 0.0, 1.0];
    let posteriors = vec![0.8]; // shorter → defaults to 0.5 for missing
    let result = estimate_k0_emissions(&indicators, &posteriors);
    // Just check bounds
    assert!(result[0] >= 0.001 && result[0] <= 0.999);
    assert!(result[1] >= 0.001 && result[1] <= 0.999);
}

#[test]
fn k0_posteriors_clamped() {
    // Posteriors outside [0,1]
    let indicators = vec![1.0];
    let posteriors = vec![2.0]; // clamped to 1.0
    let result = estimate_k0_emissions(&indicators, &posteriors);
    assert!(result[0] >= 0.001 && result[0] <= 0.999);
    assert!(result[1] >= 0.001 && result[1] <= 0.999);
}

// ============================================================================
// k0_log_pmf
// ============================================================================

#[test]
fn k0_log_pmf_indicator_one() {
    // indicator > 0.5 → ln(p_k0)
    let result = k0_log_pmf(1.0, 0.3);
    assert!((result - 0.3_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_indicator_zero() {
    // indicator <= 0.5 → ln(1 - p_k0)
    let result = k0_log_pmf(0.0, 0.3);
    assert!((result - 0.7_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_boundary_exactly_half() {
    // indicator = 0.5 → <= 0.5, so ln(1 - p_k0)
    let result = k0_log_pmf(0.5, 0.4);
    assert!((result - 0.6_f64.ln()).abs() < 1e-10);
}

#[test]
fn k0_log_pmf_p_one() {
    // p_k0 = 1.0 → ln(1) = 0 for indicator > 0.5
    let result = k0_log_pmf(0.8, 1.0);
    assert!(result.abs() < 1e-10);
}

#[test]
fn k0_log_pmf_p_zero_neg_inf() {
    // p_k0 = 0.0 → ln(0) = -inf for indicator > 0.5
    let result = k0_log_pmf(0.8, 0.0);
    assert!(result.is_infinite() && result < 0.0);
}

#[test]
fn k0_log_pmf_always_negative_or_zero() {
    for &ind in &[0.0, 0.3, 0.5, 0.7, 1.0] {
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            assert!(k0_log_pmf(ind, p) <= 0.0 + 1e-15);
        }
    }
}

// ============================================================================
// augment_with_k0
// ============================================================================

#[test]
fn augment_k0_empty() {
    let mut log_emit: Vec<[f64; 2]> = vec![];
    augment_with_k0(&mut log_emit, &[], &[]);
    assert!(log_emit.is_empty());
}

#[test]
fn augment_k0_nondiscriminative_noop() {
    // If k0_params[1] <= k0_params[0], no augmentation
    // All indicators = 0.0, posteriors = 1.0 (all IBD)
    // → k0_params[0] = prior (0.015), k0_params[1] = 0.001 (clamped, since all indicators are 0)
    // Since 0.001 < 0.015, no augmentation happens
    let mut log_emit = vec![[-1.0, -2.0]];
    let indicators = vec![0.0];
    let posteriors = vec![1.0];
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    assert!((log_emit[0][0] - (-1.0)).abs() < 1e-10);
    assert!((log_emit[0][1] - (-2.0)).abs() < 1e-10);
}

#[test]
fn augment_k0_discriminative_modifies() {
    // To be discriminative, k0 must be more common in IBD than non-IBD.
    // Mix: IBD windows have indicator=1.0 (mutation-free), non-IBD have indicator=0.0.
    let mut log_emit = vec![[0.0, 0.0]; 10];
    // First 5 windows: non-IBD (low posterior), indicator=0
    // Last 5 windows: IBD (high posterior), indicator=1
    let indicators = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9];
    let orig = log_emit.clone();
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    // k0_params[1] (IBD) should be high, k0_params[0] (non-IBD) should be low
    // → discriminative → augmentation happens
    let changed = log_emit.iter().zip(orig.iter()).any(|(a, b)| (a[0] - b[0]).abs() > 1e-10 || (a[1] - b[1]).abs() > 1e-10);
    assert!(changed, "augmentation should modify log_emit when discriminative");
}

#[test]
fn augment_k0_indicators_shorter_than_log_emit() {
    let mut log_emit = vec![[0.0, 0.0]; 5];
    let indicators = vec![1.0; 3];
    let posteriors = vec![0.9; 5];
    // Should only augment first 3 windows
    augment_with_k0(&mut log_emit, &indicators, &posteriors);
    // Just check no panic and values are finite
    for e in &log_emit {
        assert!(e[0].is_finite());
        assert!(e[1].is_finite());
    }
}

// ============================================================================
// estimate_ibd_emission_std
// ============================================================================

#[test]
fn emission_std_too_few_observations() {
    let obs = vec![0.9; 19]; // < 20
    assert!(estimate_ibd_emission_std(&obs, 0.1, 0.01, 0.2).is_none());
}

#[test]
fn emission_std_exactly_20() {
    let obs: Vec<f64> = (0..20).map(|i| 0.85 + i as f64 * 0.005).collect();
    let result = estimate_ibd_emission_std(&obs, 0.1, 0.01, 0.2);
    assert!(result.is_some());
    let std = result.unwrap();
    assert!(std >= 0.01 && std <= 0.2);
}

#[test]
fn emission_std_all_identical() {
    let obs = vec![0.95; 50];
    let result = estimate_ibd_emission_std(&obs, 0.1, 0.005, 0.3);
    assert!(result.is_some());
    // All identical → variance = 0 → clamped to min_std
    assert!((result.unwrap() - 0.005).abs() < 1e-10);
}

#[test]
fn emission_std_high_variance_capped() {
    let mut obs = vec![0.5; 50];
    for (i, v) in obs.iter_mut().enumerate() {
        *v = (i as f64) * 0.02; // range 0..1
    }
    let result = estimate_ibd_emission_std(&obs, 0.5, 0.001, 0.05);
    assert!(result.is_some());
    assert!((result.unwrap() - 0.05).abs() < 1e-10); // clamped to max
}

#[test]
fn emission_std_quantile_fraction_small() {
    let obs: Vec<f64> = (0..100).map(|i| 0.8 + (i as f64) * 0.002).collect();
    let result = estimate_ibd_emission_std(&obs, 0.01, 0.001, 1.0);
    assert!(result.is_some());
    assert!(result.unwrap() > 0.0);
}

#[test]
fn emission_std_empty() {
    assert!(estimate_ibd_emission_std(&[], 0.1, 0.01, 0.2).is_none());
}

// ============================================================================
// merge_nearby_ibd_segments
// ============================================================================

#[test]
fn merge_nearby_empty() {
    let result = merge_nearby_ibd_segments(&[], 5);
    assert!(result.is_empty());
}

#[test]
fn merge_nearby_single() {
    let segs = vec![make_seg(0, 4, 0.9, 5.0)];
    let result = merge_nearby_ibd_segments(&segs, 5);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 4);
}

#[test]
fn merge_nearby_adjacent_merges() {
    let segs = vec![
        make_seg(0, 4, 0.9, 5.0),
        make_seg(5, 9, 0.8, 3.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 9);
    // LOD should be sum
    assert!((result[0].lod_score - 8.0).abs() < 1e-10);
}

#[test]
fn merge_nearby_gap_within_threshold() {
    let segs = vec![
        make_seg(0, 4, 0.9, 5.0),
        make_seg(7, 10, 0.8, 3.0), // gap of 2 windows (5,6)
    ];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 1);
}

#[test]
fn merge_nearby_gap_exceeds_threshold() {
    let segs = vec![
        make_seg(0, 4, 0.9, 5.0),
        make_seg(10, 14, 0.8, 3.0), // gap of 5 windows
    ];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 2);
}

#[test]
fn merge_nearby_three_cascade() {
    let segs = vec![
        make_seg(0, 4, 0.9, 5.0),
        make_seg(6, 9, 0.85, 4.0),
        make_seg(11, 14, 0.8, 3.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1); // all merged
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 14);
}

#[test]
fn merge_nearby_max_gap_zero() {
    let segs = vec![
        make_seg(0, 4, 0.9, 5.0),
        make_seg(6, 9, 0.8, 3.0), // gap of 1
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 2); // gap=1 > 0
}

#[test]
fn merge_nearby_posteriors_weighted_avg() {
    let segs = vec![
        make_seg(0, 3, 0.9, 5.0),  // 4 windows
        make_seg(4, 5, 0.6, 2.0),  // 2 windows
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 1);
    // Weighted avg: (0.9*4 + 0.6*2) / (4+2) = (3.6+1.2)/6 = 0.8
    assert!((result[0].mean_posterior - 0.8).abs() < 1e-10);
}

// ============================================================================
// refine_states_adaptive
// ============================================================================

#[test]
fn refine_adaptive_empty() {
    let mut states: Vec<usize> = vec![];
    refine_states_adaptive(&mut states, &[]);
    assert!(states.is_empty());
}

#[test]
fn refine_adaptive_no_ibd() {
    let mut states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.2, 0.1, 0.05, 0.1];
    refine_states_adaptive(&mut states, &posteriors);
    assert_eq!(states, vec![0, 0, 0, 0, 0]);
}

#[test]
fn refine_adaptive_all_ibd_high_confidence() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.95, 0.9, 0.95, 0.9, 0.85];
    refine_states_adaptive(&mut states, &posteriors);
    // High confidence → conservative trimming (0.3)
    // All posteriors > 0.3, so no trimming
    assert!(states.iter().all(|&s| s == 1));
}

#[test]
fn refine_adaptive_extends_high_confidence_left() {
    // High-confidence segment with moderate posterior to the left
    let mut states = vec![0, 0, 1, 1, 1];
    let posteriors = vec![0.1, 0.4, 0.95, 0.9, 0.95];
    refine_states_adaptive(&mut states, &posteriors);
    // Mean posterior of segment: ~0.93 > 0.8 → extend_thresh=0.3
    // posteriors[1]=0.4 > 0.3 and states[1]=0 → should extend
    assert_eq!(states[1], 1, "should extend left into moderate posterior");
}

#[test]
fn refine_adaptive_trims_low_confidence_edges() {
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.12, 0.4, 0.12, 0.1];
    refine_states_adaptive(&mut states, &posteriors);
    // Mean posterior: (0.12+0.4+0.12)/3 ≈ 0.213 < 0.5
    // → trim_thresh = 0.15
    // Edge posteriors 0.12 < 0.15 → trimmed
    assert_eq!(states[1], 0, "low edge posterior should be trimmed");
    assert_eq!(states[3], 0, "low edge posterior should be trimmed");
}

#[test]
fn refine_adaptive_mismatched_lengths_noop() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.9]; // different length
    let orig = states.clone();
    refine_states_adaptive(&mut states, &posteriors);
    assert_eq!(states, orig);
}

#[test]
fn refine_adaptive_single_window_segment() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.9, 0.1];
    refine_states_adaptive(&mut states, &posteriors);
    // Single window, high posterior → should survive
    assert_eq!(states[1], 1);
}

// ============================================================================
// bridge_ibd_gaps_adaptive
// ============================================================================

#[test]
fn bridge_adaptive_empty() {
    let mut states: Vec<usize> = vec![];
    let result = bridge_ibd_gaps_adaptive(&mut states, &[], 3, 0.4);
    assert_eq!(result, 0);
}

#[test]
fn bridge_adaptive_too_short() {
    let mut states = vec![1, 0];
    let posteriors = vec![0.9, 0.5];
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.4);
    assert_eq!(result, 0); // n < 3
}

#[test]
fn bridge_adaptive_no_gaps() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(result, 0);
}

#[test]
fn bridge_adaptive_max_gap_zero() {
    let mut states = vec![1, 0, 1, 0, 1];
    let posteriors = vec![0.9, 0.5, 0.9, 0.5, 0.9];
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 0, 0.3);
    assert_eq!(result, 0);
}

#[test]
fn bridge_adaptive_high_confidence_flanks_lower_threshold() {
    let mut states = vec![1, 1, 1, 0, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.9, 0.3, 0.9, 0.95, 0.9];
    // Left/right mean posterior > 0.8 → threshold = base * 0.5 = 0.2
    // Gap posterior = 0.3 > 0.2 → bridge
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.4);
    assert_eq!(result, 1);
    assert_eq!(states[3], 1);
}

#[test]
fn bridge_adaptive_low_confidence_flanks_full_threshold() {
    let mut states = vec![1, 0, 1, 0, 0, 0, 1];
    let posteriors = vec![0.45, 0.2, 0.45, 0.35, 0.35, 0.35, 0.45];
    // Flanking mean posteriors < 0.5 → threshold = base (0.4)
    // Gap posterior = 0.35 < 0.4 → no bridge
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 5, 0.4);
    assert_eq!(result, 0);
}

#[test]
fn bridge_adaptive_gap_exceeds_max() {
    let mut states = vec![1, 1, 0, 0, 0, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9];
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert_eq!(result, 0); // gap = 4 > max_gap = 2
}

#[test]
fn bridge_adaptive_mismatched_lengths_noop() {
    let mut states = vec![1, 0, 0, 1];
    let posteriors = vec![0.9, 0.5]; // mismatched
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(result, 0);
}

#[test]
fn bridge_adaptive_cascading_bridges() {
    // Two gaps that can both be bridged
    let mut states = vec![1, 1, 0, 1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.9, 0.9, 0.5, 0.9, 0.9];
    let result = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 1, 0.3);
    assert!(result >= 1);
    assert_eq!(states[2], 1);
}

// ============================================================================
// extract_ibd_segments_composite
// ============================================================================

#[test]
fn composite_empty() {
    let result = extract_ibd_segments_composite(&[], &[], None, 5, 2, 0.3);
    assert!(result.is_empty());
}

#[test]
fn composite_mismatched_lengths() {
    let states = vec![1, 1, 1];
    let posteriors = vec![0.9, 0.8]; // mismatched
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 2, 0.3);
    assert!(result.is_empty());
}

#[test]
fn composite_no_ibd() {
    let states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1, 0.1];
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 2, 0.3);
    assert!(result.is_empty());
}

#[test]
fn composite_segment_below_hard_min() {
    let states = vec![0, 1, 0, 0, 0];
    let posteriors = vec![0.1, 0.9, 0.1, 0.1, 0.1];
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 5, 3, 0.0);
    assert!(result.is_empty()); // 1 window < hard_min=3
}

#[test]
fn composite_long_high_quality_segment() {
    let params = make_params();
    // Strong IBD signal
    let obs = vec![0.85, 0.97, 0.96, 0.97, 0.96, 0.97, 0.85];
    let states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.9, 0.95, 0.9, 0.95, 0.1];
    let result = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)), 3, 2, 0.3,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 1);
    assert_eq!(result[0].end_idx, 5);
}

#[test]
fn composite_with_lod_computation() {
    let params = make_params();
    let obs = vec![0.85, 0.96, 0.97, 0.95, 0.96, 0.97, 0.85];
    let states = vec![0, 1, 1, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.9, 0.95, 0.9, 0.95, 0.1];
    let result = extract_ibd_segments_composite(
        &states, &posteriors, Some((&obs, &params)), 3, 2, 0.3,
    );
    assert_eq!(result.len(), 1);
    assert!(result[0].lod_score > 0.0, "LOD should be positive for IBD");
}

#[test]
fn composite_soft_min_penalizes_short() {
    // Short segment (3 windows) with soft_min=10 → length_factor = 3/10 = 0.3
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.5, 0.5, 0.5, 0.1];
    // composite_score ≈ (lod_density * 0.5 * 0.3). If threshold is high enough, rejected
    let result = extract_ibd_segments_composite(&states, &posteriors, None, 10, 2, 0.5);
    // With no LOD (None params), lod_density = 0 → score = 0 < threshold
    assert!(result.is_empty());
}

// ============================================================================
// infer_ibd_multi_scale
// ============================================================================

#[test]
fn multi_scale_empty() {
    let params = make_params();
    let result = infer_ibd_multi_scale(&[], &params, &[1, 2]);
    assert!(result.states.is_empty());
}

#[test]
fn multi_scale_single_scale() {
    let params = make_params();
    let obs = vec![0.85; 20];
    let result = infer_ibd_multi_scale(&obs, &params, &[1]);
    assert_eq!(result.states.len(), 20);
}

#[test]
fn multi_scale_too_short() {
    let params = make_params();
    let obs = vec![0.85, 0.9, 0.85, 0.9, 0.85]; // < 6
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    assert_eq!(result.states.len(), 5);
}

#[test]
fn multi_scale_clear_ibd_preserved() {
    let params = make_params();
    // Clear IBD signal
    let mut obs = vec![0.85; 30];
    for v in obs[10..20].iter_mut() {
        *v = 0.98;
    }
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    assert_eq!(result.states.len(), 30);
    // At least some windows in the IBD region should be called IBD
    let ibd_count: usize = result.states[10..20].iter().filter(|&&s| s == 1).count();
    assert!(ibd_count > 5, "clear IBD region should be mostly called IBD: {}/10", ibd_count);
}

#[test]
fn multi_scale_posteriors_length() {
    let params = make_params();
    let obs = vec![0.9; 20];
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2]);
    assert_eq!(result.posteriors.len(), 20);
}

#[test]
fn multi_scale_states_valid_indices() {
    let params = make_params();
    let obs = vec![0.88; 30];
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 3, 5]);
    for &s in &result.states {
        assert!(s <= 1, "state should be 0 or 1, got {}", s);
    }
}
