//! Boundary refinement stress tests for IBD segment detection.
//!
//! Tests refine_states_with_posteriors, bridge_ibd_gaps, and
//! refine_segment_boundaries under edge cases and stress scenarios.

use hprc_ibd::hmm::{
    refine_states_with_posteriors, bridge_ibd_gaps, refine_segment_boundaries,
    IbdSegmentWithPosterior, RefinedBoundary,
};

// ── refine_states_with_posteriors tests ──

#[test]
fn refine_extend_single_high_posterior_window() {
    // Non-IBD window with high posterior adjacent to IBD → should extend
    let mut states = vec![0, 1, 1, 0, 0];
    let posteriors = vec![0.7, 0.95, 0.90, 0.3, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 0 has posterior 0.7 >= 0.5 and is adjacent to IBD at index 1
    assert_eq!(states[0], 1, "Window 0 should be extended to IBD");
    // Window 3 has posterior 0.3 < 0.5 → should NOT extend
    assert_eq!(states[3], 0, "Window 3 should remain non-IBD");
}

#[test]
fn refine_extend_chain_propagation() {
    // Multiple adjacent high-posterior windows should all extend iteratively
    let mut states = vec![0, 0, 0, 1, 0, 0, 0];
    let posteriors = vec![0.1, 0.6, 0.7, 0.95, 0.8, 0.6, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Chain: index 2 is adjacent to 3 (IBD) → extends, then index 1 adjacent to 2 → extends
    // Similarly right: index 4 adjacent to 3 → extends, then 5 adjacent to 4 → extends
    assert_eq!(states, vec![0, 1, 1, 1, 1, 1, 0],
        "Chain propagation should extend all high-posterior windows");
}

#[test]
fn refine_trim_low_posterior_boundary() {
    // IBD window at boundary with low posterior → should trim
    let mut states = vec![1, 1, 1, 1, 0];
    let posteriors = vec![0.15, 0.9, 0.95, 0.10, 0.05];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Window 0: IBD, posterior 0.15 < 0.2, at left boundary → trim
    assert_eq!(states[0], 0, "Left boundary low-posterior should be trimmed");
    // Window 3: IBD, posterior 0.10 < 0.2, at right boundary → trim
    assert_eq!(states[3], 0, "Right boundary low-posterior should be trimmed");
    // Interior windows stay IBD
    assert_eq!(states[1], 1);
    assert_eq!(states[2], 1);
}

#[test]
fn refine_no_trim_interior_low_posterior() {
    // Interior IBD windows with low posterior should NOT be trimmed
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.05, 0.9, 0.9];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    // Index 2: IBD, posterior 0.05 < 0.2, but NOT at boundary (both neighbors are IBD)
    assert_eq!(states[2], 1, "Interior low-posterior IBD should NOT be trimmed");
}

#[test]
fn refine_empty_input() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert!(states.is_empty());
}

#[test]
fn refine_length_mismatch_is_noop() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.9, 0.9]; // shorter than states
    let original = states.clone();
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states, original, "Length mismatch should be no-op");
}

#[test]
fn refine_single_window_segment() {
    // Single IBD window: should be trimmable since both sides are boundaries
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.15, 0.1];
    refine_states_with_posteriors(&mut states, &posteriors, 0.5, 0.2);
    assert_eq!(states[1], 0, "Single IBD window below trim threshold should be removed");
}

// ── bridge_ibd_gaps tests ──

#[test]
fn bridge_single_window_gap() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.85, 0.5, 0.88, 0.92];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 1);
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

#[test]
fn bridge_gap_too_long() {
    let mut states = vec![1, 1, 0, 0, 0, 1, 1];
    let posteriors = vec![0.9, 0.85, 0.5, 0.4, 0.5, 0.88, 0.92];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0, "Gap of 3 windows should not be bridged with max_gap=2");
    assert_eq!(states[2], 0);
}

#[test]
fn bridge_low_posterior_gap_rejected() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.85, 0.1, 0.88, 0.92];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridged, 0, "Gap with posterior 0.1 < min_bridge=0.3 should not bridge");
    assert_eq!(states[2], 0);
}

#[test]
fn bridge_gap_not_flanked_both_sides() {
    // Gap at the end, not flanked by IBD on right
    let mut states = vec![1, 1, 0, 0];
    let posteriors = vec![0.9, 0.85, 0.5, 0.5];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridged, 0, "Gap not flanked by IBD on right should not bridge");
}

#[test]
fn bridge_disabled_when_max_gap_zero() {
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.5, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 0, 0.3);
    assert_eq!(bridged, 0, "max_gap=0 should disable bridging");
    assert_eq!(states[1], 0);
}

#[test]
fn bridge_multiple_gaps() {
    let mut states = vec![1, 0, 1, 1, 0, 1];
    let posteriors = vec![0.9, 0.5, 0.9, 0.9, 0.5, 0.9];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 2, "Should bridge both 1-window gaps");
    assert_eq!(states, vec![1, 1, 1, 1, 1, 1]);
}

#[test]
fn bridge_too_few_windows() {
    let mut states = vec![1, 0];
    let posteriors = vec![0.9, 0.5];
    let bridged = bridge_ibd_gaps(&mut states, &posteriors, 1, 0.3);
    assert_eq!(bridged, 0, "n < 3 should return 0 immediately");
}

// ── refine_segment_boundaries tests ──

fn make_segment(start: usize, end: usize) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: start,
        end_idx: end,
        n_windows: end - start + 1,
        mean_posterior: 0.9,
        min_posterior: 0.8,
        max_posterior: 0.95,
        lod_score: 5.0,
    }
}

#[test]
fn refine_boundary_interpolation_start() {
    // Posteriors: 0.2 (non-IBD), 0.8 (IBD) — crossover at 0.5
    // Should interpolate start between window centers
    let segments = vec![make_segment(1, 3)];
    let posteriors = vec![0.2, 0.8, 0.9, 0.85, 0.1];
    let starts = vec![0, 10000, 20000, 30000, 40000];
    let ends = vec![10000, 20000, 30000, 40000, 50000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    assert_eq!(refined.len(), 1);
    // Crossover between window 0 (center=5000, post=0.2) and window 1 (center=15000, post=0.8)
    // t = (0.5 - 0.2) / (0.8 - 0.2) = 0.5
    // pos = 5000 + 0.5 * (15000 - 5000) = 10000
    assert_eq!(refined[0].start_bp, 10000);
}

#[test]
fn refine_boundary_interpolation_end() {
    // Posteriors: IBD segment ending with post=0.8, next window post=0.2
    let segments = vec![make_segment(0, 1)];
    let posteriors = vec![0.9, 0.8, 0.2, 0.1];
    let starts = vec![0, 10000, 20000, 30000];
    let ends = vec![10000, 20000, 30000, 40000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    // End interpolation: p_at=0.8, p_after=0.2, crossover=0.5
    // t = (0.8 - 0.5) / (0.8 - 0.2) = 0.5
    // c_at = 15000, c_after = 25000
    // pos = 15000 + 0.5 * (25000 - 15000) = 20000
    assert_eq!(refined[0].end_bp, 20000);
}

#[test]
fn refine_boundary_no_interpolation_when_both_above_crossover() {
    // Both windows above crossover → no interpolation, use window start/end
    let segments = vec![make_segment(1, 2)];
    let posteriors = vec![0.7, 0.9, 0.85, 0.6];
    let starts = vec![0, 10000, 20000, 30000];
    let ends = vec![10000, 20000, 30000, 40000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    // p_before (0.7) is NOT < crossover (0.5), so no interpolation → use window start
    assert_eq!(refined[0].start_bp, 10000);
}

#[test]
fn refine_boundary_segment_at_first_window() {
    // Segment starts at index 0 → no previous window for interpolation
    let segments = vec![make_segment(0, 2)];
    let posteriors = vec![0.9, 0.85, 0.8, 0.1];
    let starts = vec![0, 10000, 20000, 30000];
    let ends = vec![10000, 20000, 30000, 40000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    assert_eq!(refined[0].start_bp, 0, "Segment at first window should use window_starts[0]");
}

#[test]
fn refine_boundary_segment_at_last_window() {
    // Segment ends at last index → no next window for interpolation
    let segments = vec![make_segment(1, 3)];
    let posteriors = vec![0.1, 0.85, 0.9, 0.8];
    let starts = vec![0, 10000, 20000, 30000];
    let ends = vec![10000, 20000, 30000, 40000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    assert_eq!(refined[0].end_bp, 40000, "Segment at last window should use window_ends[last]");
}

#[test]
fn refine_boundary_empty_segments() {
    let refined = refine_segment_boundaries(&[], &[0.5, 0.6], &[0, 10000], &[10000, 20000], 0.5);
    assert!(refined.is_empty());
}

#[test]
fn refine_boundary_empty_posteriors() {
    let segments = vec![make_segment(0, 0)];
    let refined = refine_segment_boundaries(&segments, &[], &[], &[], 0.5);
    // Empty posteriors → fallback path
    assert_eq!(refined.len(), 1);
    assert_eq!(refined[0].start_bp, 0);
    assert_eq!(refined[0].end_bp, 0);
}

#[test]
fn refine_boundary_multiple_segments() {
    let segments = vec![make_segment(0, 1), make_segment(3, 4)];
    let posteriors = vec![0.9, 0.8, 0.1, 0.85, 0.9, 0.05];
    let starts = vec![0, 10000, 20000, 30000, 40000, 50000];
    let ends = vec![10000, 20000, 30000, 40000, 50000, 60000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    assert_eq!(refined.len(), 2);
    // First segment ends with interpolation between idx 1 (post=0.8) and idx 2 (post=0.1)
    // t = (0.8 - 0.5) / (0.8 - 0.1) ≈ 0.4286
    // c1 = 15000, c2 = 25000 → pos = 15000 + 0.4286 * 10000 ≈ 19286
    assert!(refined[0].end_bp > 15000 && refined[0].end_bp < 25000,
        "First segment end should be interpolated: {}", refined[0].end_bp);

    // Second segment starts with interpolation between idx 2 (post=0.1) and idx 3 (post=0.85)
    assert!(refined[1].start_bp > 20000 && refined[1].start_bp < 40000,
        "Second segment start should be interpolated: {}", refined[1].start_bp);
}

#[test]
fn refine_boundary_ensures_start_le_end() {
    // Regression guard: refined start should never exceed refined end
    let segments = vec![make_segment(2, 2)]; // single-window segment
    let posteriors = vec![0.1, 0.2, 0.6, 0.3, 0.1];
    let starts = vec![0, 10000, 20000, 30000, 40000];
    let ends = vec![10000, 20000, 30000, 40000, 50000];
    let refined = refine_segment_boundaries(&segments, &posteriors, &starts, &ends, 0.5);

    assert!(
        refined[0].start_bp <= refined[0].end_bp,
        "start_bp ({}) must be <= end_bp ({})",
        refined[0].start_bp, refined[0].end_bp
    );
}
