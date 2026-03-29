//! Tests for IBD gap bridging and segment merging functionality.
//!
//! Tests bridge_ibd_gaps() and merge_nearby_ibd_segments() which are
//! critical for IBD F1 improvement by preventing segment splitting
//! from noisy windows within true IBD regions.

use hprc_ibd::hmm::{
    bridge_ibd_gaps, merge_nearby_ibd_segments, IbdSegmentWithPosterior,
    HmmParams, Population, infer_ibd_with_training, extract_ibd_segments_with_lod,
    refine_states_with_posteriors,
};

// ============================================================================
// bridge_ibd_gaps tests
// ============================================================================

#[test]
fn bridge_gaps_disabled_when_max_gap_zero() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 0, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 1, 0, 1, 1]);
}

#[test]
fn bridge_gaps_single_window_gap() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 1);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_two_window_gap() {
    let mut states = vec![1, 1, 0, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.4, 0.5, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 1);
    assert_eq!(states, vec![1, 1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_gap_too_large() {
    let mut states = vec![1, 1, 0, 0, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.5, 0.5, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 1, 0, 0, 0, 1, 1]);
}

#[test]
fn bridge_gaps_posterior_too_low() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.1, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 1, 0, 1, 1]);
}

#[test]
fn bridge_gaps_no_ibd_after_gap() {
    // Gap at end — no IBD segment follows
    let mut states = vec![1, 1, 0, 0];
    let posteriors = vec![0.9, 0.9, 0.5, 0.5];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 1, 0, 0]);
}

#[test]
fn bridge_gaps_no_ibd_before_gap() {
    // Gap at start — no IBD segment precedes
    let mut states = vec![0, 0, 1, 1];
    let posteriors = vec![0.5, 0.5, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![0, 0, 1, 1]);
}

#[test]
fn bridge_gaps_multiple_gaps() {
    let mut states = vec![1, 0, 1, 0, 1];
    let posteriors = vec![0.9, 0.5, 0.9, 0.5, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 2);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_mixed_bridgeable_and_not() {
    // First gap: posterior too low; second gap: bridgeable
    let mut states = vec![1, 0, 1, 1, 0, 1];
    let posteriors = vec![0.9, 0.1, 0.9, 0.9, 0.6, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 1); // only second gap bridged
    assert_eq!(states[4], 1); // second gap filled
    assert_eq!(states[1], 0); // first gap remains
}

#[test]
fn bridge_gaps_empty_input() {
    let mut states: Vec<usize> = Vec::new();
    let posteriors: Vec<f64> = Vec::new();
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
}

#[test]
fn bridge_gaps_too_short() {
    let mut states = vec![1, 0];
    let posteriors = vec![0.9, 0.5];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 0);
}

#[test]
fn bridge_gaps_all_ibd() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_all_non_ibd() {
    let mut states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.1, 0.1, 0.1, 0.1];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![0, 0, 0, 0, 0]);
}

#[test]
fn bridge_gaps_exact_threshold() {
    // Posterior exactly at threshold should bridge
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.3, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 1);
    assert_eq!(states, vec![1, 1, 1]);
}

#[test]
fn bridge_gaps_just_below_threshold() {
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.29, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 0);
    assert_eq!(states, vec![1, 0, 1]);
}

#[test]
fn bridge_gaps_exact_max_gap() {
    // Gap of exactly max_gap windows should be bridged
    let mut states = vec![1, 0, 0, 0, 1];
    let posteriors = vec![0.9, 0.4, 0.5, 0.6, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridged, 1);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_mismatched_lengths() {
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.5]; // too short
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 0);
}

#[test]
fn bridge_gaps_cascading_bridge() {
    // After bridging first gap, the result creates a continuous IBD segment.
    // The second gap should also be bridgeable.
    let mut states = vec![1, 0, 1, 0, 1];
    let posteriors = vec![0.9, 0.4, 0.9, 0.4, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 2);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gaps_large_realistic_scenario() {
    // Simulate a long region with scattered noise
    let mut states = vec![
        0, 0, 0, 0, // non-IBD region
        1, 1, 1, 1, 1, 0, 1, 1, 1, 1, // IBD with 1-window gap
        0, 0, 0, 0, 0, // non-IBD region
        1, 1, 0, 0, 1, 1, // IBD with 2-window gap
        0, 0, 0, // non-IBD region
    ];
    let posteriors = vec![
        0.05, 0.05, 0.05, 0.05,
        0.95, 0.98, 0.97, 0.99, 0.96, 0.4, 0.97, 0.98, 0.95, 0.96,
        0.1, 0.05, 0.05, 0.05, 0.05,
        0.9, 0.95, 0.35, 0.4, 0.92, 0.9,
        0.05, 0.05, 0.05,
    ];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 2);
    // First gap (1 window) bridged
    assert_eq!(states[9], 1);
    // Second gap (2 windows) bridged
    assert_eq!(states[21], 1);
    assert_eq!(states[22], 1);
}

// ============================================================================
// merge_nearby_ibd_segments tests
// ============================================================================

fn make_seg(start: usize, end: usize, mean_post: f64, lod: f64) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: start,
        end_idx: end,
        n_windows: end - start + 1,
        mean_posterior: mean_post,
        min_posterior: mean_post - 0.1,
        max_posterior: mean_post + 0.05,
        lod_score: lod,
    }
}

#[test]
fn merge_segments_empty() {
    let result = merge_nearby_ibd_segments(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn merge_segments_single() {
    let segs = vec![make_seg(0, 10, 0.9, 5.0)];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 10);
}

#[test]
fn merge_segments_no_merge_needed() {
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(20, 30, 0.85, 4.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 2);
}

#[test]
fn merge_segments_close_gap() {
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(12, 20, 0.85, 4.0),
    ];
    // Gap is 12 - 10 - 1 = 1 window
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 20);
    assert_eq!(result[0].n_windows, 21);
}

#[test]
fn merge_segments_lod_additive() {
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(12, 20, 0.85, 4.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    assert!((result[0].lod_score - 9.0).abs() < 1e-10);
}

#[test]
fn merge_segments_weighted_posterior() {
    let seg1 = make_seg(0, 9, 0.9, 5.0); // 10 windows
    let seg2 = make_seg(11, 20, 0.8, 3.0); // 10 windows
    let segs = vec![seg1, seg2];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    // Weighted average: (0.9 * 10 + 0.8 * 10) / 20 = 0.85
    assert!((result[0].mean_posterior - 0.85).abs() < 1e-10);
}

#[test]
fn merge_segments_three_close() {
    let segs = vec![
        make_seg(0, 5, 0.9, 3.0),
        make_seg(7, 12, 0.85, 2.0),
        make_seg(14, 20, 0.88, 4.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 20);
    assert!((result[0].lod_score - 9.0).abs() < 1e-10);
}

#[test]
fn merge_segments_min_max_posterior() {
    let mut seg1 = make_seg(0, 5, 0.9, 3.0);
    seg1.min_posterior = 0.7;
    seg1.max_posterior = 0.95;
    let mut seg2 = make_seg(7, 12, 0.85, 2.0);
    seg2.min_posterior = 0.6;
    seg2.max_posterior = 0.92;
    let segs = vec![seg1, seg2];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    assert!((result[0].min_posterior - 0.6).abs() < 1e-10);
    assert!((result[0].max_posterior - 0.95).abs() < 1e-10);
}

#[test]
fn merge_segments_adjacent() {
    // end_idx=10, start_idx=11 → gap = 0 windows
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(11, 20, 0.85, 4.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 20);
}

#[test]
fn merge_segments_overlapping() {
    // Overlapping segments (start_idx < prev end_idx)
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(8, 20, 0.85, 4.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 20);
}

#[test]
fn merge_segments_gap_exactly_max() {
    // Gap of exactly max_gap_windows should merge
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(14, 20, 0.85, 4.0),
    ];
    // Gap = 14 - 10 - 1 = 3
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 1);
}

#[test]
fn merge_segments_gap_exceeds_max() {
    let segs = vec![
        make_seg(0, 10, 0.9, 5.0),
        make_seg(15, 20, 0.85, 4.0),
    ];
    // Gap = 15 - 10 - 1 = 4
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 2);
}

// ============================================================================
// Integration tests: full pipeline with gap bridging
// ============================================================================

/// Integration: gap bridging on HMM output does not increase segment count.
#[test]
fn integration_gap_bridging_does_not_increase_segments() {
    // Create a signal where BW training should learn emissions
    let mut observations = Vec::with_capacity(50);
    // 15 non-IBD windows
    for i in 0..15 {
        observations.push(0.9970 + (i as f64) * 0.0001);
    }
    // 20 IBD windows
    for _ in 0..20 {
        observations.push(0.9998);
    }
    // 15 non-IBD windows
    for i in 0..15 {
        observations.push(0.9970 + (i as f64) * 0.0001);
    }

    let mut params = HmmParams::from_population(Population::Generic, 50.0, 0.0001, 10000);
    let inference = infer_ibd_with_training(
        &observations, &mut params, Population::Generic, 10000, 20,
    );

    let segs_before = extract_ibd_segments_with_lod(
        &inference.states, &inference.posteriors, 3, 0.0, None, None,
    );

    // Apply bridge_ibd_gaps — should not make things worse
    let mut states = inference.states.clone();
    bridge_ibd_gaps(&mut states, &inference.posteriors, 3, 0.2);

    let segs_after = extract_ibd_segments_with_lod(
        &states, &inference.posteriors, 3, 0.0, None, None,
    );

    // Gap bridging should never increase segment count
    assert!(
        segs_after.len() <= segs_before.len(),
        "Bridging should not increase segment count: {} vs {}",
        segs_after.len(), segs_before.len()
    );

    // Total IBD coverage should not decrease
    let total_before: usize = segs_before.iter().map(|s| s.n_windows).sum();
    let total_after: usize = segs_after.iter().map(|s| s.n_windows).sum();
    assert!(total_after >= total_before);
}

/// Test that gap bridging preserves segments when there are no gaps to bridge.
#[test]
fn integration_no_gaps_preserves_segments() {
    // Clean signal with no noise gaps
    let mut observations = Vec::with_capacity(20);
    observations.extend_from_slice(&[0.9970; 5]); // non-IBD
    observations.extend_from_slice(&[0.9998; 10]); // IBD
    observations.extend_from_slice(&[0.9970; 5]); // non-IBD

    let mut params = HmmParams::from_population(Population::Generic, 50.0, 0.0001, 10000);
    let inference = infer_ibd_with_training(
        &observations, &mut params, Population::Generic, 10000, 20,
    );

    let segs_before = extract_ibd_segments_with_lod(
        &inference.states, &inference.posteriors, 3, 0.0, None, None,
    );

    let mut states = inference.states.clone();
    let bridged = bridge_ibd_gaps(&mut states, &inference.posteriors, 3, 0.3);

    let segs_after = extract_ibd_segments_with_lod(
        &states, &inference.posteriors, 3, 0.0, None, None,
    );

    assert_eq!(bridged, 0, "No gaps to bridge in clean signal");
    assert_eq!(segs_before.len(), segs_after.len());
}

/// Verify gap bridging + refine_states + extract pipeline produces valid segments.
#[test]
fn integration_bridge_after_refine() {
    let observations = vec![
        0.9970, 0.9970, // non-IBD
        0.9998, 0.9999, 0.9997, 0.9998, // IBD segment 1
        0.9985, // gap
        0.9998, 0.9999, 0.9997, 0.9998, // IBD segment 2
        0.9970, 0.9970, // non-IBD
    ];

    let mut params = HmmParams::from_population(Population::Generic, 50.0, 0.0001, 10000);
    let result = infer_ibd_with_training(
        &observations, &mut params, Population::Generic, 10000, 20,
    );

    // refine_states already applied inside infer_ibd_with_training
    let mut states = result.states.clone();
    bridge_ibd_gaps(&mut states, &result.posteriors, 2, 0.2);

    let segments = extract_ibd_segments_with_lod(
        &states, &result.posteriors, 2, 0.0,
        Some((&observations, &params)), None,
    );

    // All segments should have valid properties
    for seg in &segments {
        assert!(seg.n_windows >= 2, "Segment should meet min_windows");
        assert!(seg.mean_posterior >= 0.0 && seg.mean_posterior <= 1.0);
        assert!(seg.min_posterior <= seg.max_posterior);
        assert!(seg.start_idx <= seg.end_idx);
        assert_eq!(seg.n_windows, seg.end_idx - seg.start_idx + 1);
    }
}
