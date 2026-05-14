//! Tests for posterior-based boundary refinement (refine_segment_boundaries)

use impopk_ibd::hmm::{IbdSegmentWithPosterior, refine_segment_boundaries};

fn make_segment(start: usize, end: usize, mean_post: f64) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx: start,
        end_idx: end,
        n_windows: end - start + 1,
        mean_posterior: mean_post,
        min_posterior: mean_post - 0.05,
        max_posterior: mean_post + 0.05,
        lod_score: 5.0,
    }
}

// 10kb windows: [1, 10000], [10001, 20000], ...
fn windows_10kb(n: usize) -> (Vec<u64>, Vec<u64>) {
    let starts: Vec<u64> = (0..n).map(|i| i as u64 * 10000 + 1).collect();
    let ends: Vec<u64> = (0..n).map(|i| (i as u64 + 1) * 10000).collect();
    (starts, ends)
}

#[test]
fn basic_boundary_refinement() {
    // 7 windows, segment at windows 2-4
    // Posteriors: low, low, high, high, high, low, low
    let posteriors = vec![0.05, 0.2, 0.9, 0.98, 0.85, 0.15, 0.03];
    let seg = make_segment(2, 4, 0.91);
    let (starts, ends) = windows_10kb(7);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    assert_eq!(refined.len(), 1);

    // Start: crossover between window 1 (P=0.2) and window 2 (P=0.9)
    // t = (0.5 - 0.2) / (0.9 - 0.2) ≈ 0.4286
    // center[1] = 15000.5, center[2] = 25000.5
    // boundary ≈ 15000.5 + 0.4286 * 10000 = 19286
    assert!(refined[0].start_bp > 15000, "start should be past center of window 1");
    assert!(refined[0].start_bp < 25000, "start should be before center of window 2");

    // End: crossover between window 4 (P=0.85) and window 5 (P=0.15)
    // t = (0.85 - 0.5) / (0.85 - 0.15) = 0.5
    // center[4] = 45000.5, center[5] = 55000.5
    // boundary ≈ 45000.5 + 0.5 * 10000 = 50001
    assert!(refined[0].end_bp > 45000, "end should be past center of window 4");
    assert!(refined[0].end_bp < 55001, "end should be before center of window 5");
}

#[test]
fn no_refinement_at_region_edges() {
    // Segment starts at window 0, ends at last window — no adjacent windows to interpolate
    let posteriors = vec![0.9, 0.95, 0.88];
    let seg = make_segment(0, 2, 0.91);
    let (starts, ends) = windows_10kb(3);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    assert_eq!(refined[0].start_bp, 1); // window 0 start
    assert_eq!(refined[0].end_bp, 30000); // window 2 end
}

#[test]
fn no_refinement_when_posteriors_dont_cross() {
    // Both sides have high posteriors (no clear crossover)
    let posteriors = vec![0.6, 0.9, 0.95, 0.88, 0.7];
    let seg = make_segment(1, 3, 0.91);
    let (starts, ends) = windows_10kb(5);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    // Window 0 has P=0.6 > 0.5, so no crossover at start → use window start
    assert_eq!(refined[0].start_bp, starts[1]);
    // Window 4 has P=0.7 > 0.5, so no crossover at end → use window end
    assert_eq!(refined[0].end_bp, ends[3]);
}

#[test]
fn sharp_transition_refined_near_boundary() {
    // Very sharp transition: P jumps from 0.01 to 0.99 between adjacent windows
    let posteriors = vec![0.01, 0.99, 0.99, 0.01];
    let seg = make_segment(1, 2, 0.99);
    let (starts, ends) = windows_10kb(4);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);

    // t_start = (0.5 - 0.01) / (0.99 - 0.01) = 0.5
    // center[0] = 5000.5, center[1] = 15000.5
    // boundary ≈ 5000.5 + 0.5 * 10000 = 10001 (right at window boundary)
    let expected_center = 10001;
    assert!((refined[0].start_bp as i64 - expected_center as i64).abs() < 100,
        "sharp transition should give boundary near window edge, got {}", refined[0].start_bp);

    // Same for end
    let expected_end_center = 30001;
    assert!((refined[0].end_bp as i64 - expected_end_center as i64).abs() < 100,
        "sharp transition should give boundary near window edge, got {}", refined[0].end_bp);
}

#[test]
fn gradual_transition_shifts_boundary() {
    // Gradual rise: P=0.4 → 0.8 — crossover shifted toward the low-posterior side
    let posteriors = vec![0.1, 0.4, 0.8, 0.95, 0.8, 0.4, 0.1];
    let seg = make_segment(2, 4, 0.85);
    let (starts, ends) = windows_10kb(7);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);

    // Start: t = (0.5 - 0.4) / (0.8 - 0.4) = 0.25
    // center[1] = 15000.5, center[2] = 25000.5
    // boundary ≈ 15000.5 + 0.25 * 10000 = 17501
    assert!(refined[0].start_bp < 20000, "gradual transition should shift start left");
    assert!(refined[0].start_bp > 15000);

    // End: t = (0.8 - 0.5) / (0.8 - 0.4) = 0.75
    // center[4] = 45000.5, center[5] = 55000.5
    // boundary ≈ 45000.5 + 0.75 * 10000 = 52501
    assert!(refined[0].end_bp > 50000, "gradual transition should shift end right");
    assert!(refined[0].end_bp < 55001);
}

#[test]
fn multiple_segments_refined_independently() {
    let posteriors = vec![0.05, 0.9, 0.95, 0.1, 0.05, 0.85, 0.92, 0.1];
    let seg1 = make_segment(1, 2, 0.93);
    let seg2 = make_segment(5, 6, 0.89);
    let (starts, ends) = windows_10kb(8);

    let refined = refine_segment_boundaries(&[seg1, seg2], &posteriors, &starts, &ends, 0.5);
    assert_eq!(refined.len(), 2);

    // Each segment should be independently refined
    assert!(refined[0].start_bp > starts[0]); // shifted from window boundary
    assert!(refined[1].start_bp > starts[4]); // shifted from window boundary
}

#[test]
fn empty_segments() {
    let posteriors = vec![0.1, 0.5, 0.9];
    let (starts, ends) = windows_10kb(3);
    let refined = refine_segment_boundaries(&[], &posteriors, &starts, &ends, 0.5);
    assert!(refined.is_empty());
}

#[test]
fn empty_posteriors() {
    let seg = make_segment(0, 0, 0.9);
    let refined = refine_segment_boundaries(&[seg], &[], &[], &[], 0.5);
    assert_eq!(refined.len(), 1);
    // Should fall back to 0 when arrays are empty
}

#[test]
fn mismatched_lengths_fallback() {
    let posteriors = vec![0.5, 0.9];
    let starts = vec![1, 10001, 20001]; // 3 elements vs 2 posteriors
    let ends = vec![10000, 20000, 30000];
    let seg = make_segment(0, 1, 0.7);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    assert_eq!(refined.len(), 1);
    // Mismatched lengths → fallback to window positions
}

#[test]
fn single_window_segment() {
    let posteriors = vec![0.1, 0.95, 0.1];
    let seg = make_segment(1, 1, 0.95);
    let (starts, ends) = windows_10kb(3);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);

    // Both start and end should be refined
    assert!(refined[0].start_bp > starts[0]); // interpolated
    assert!(refined[0].end_bp < ends[2]); // interpolated
    assert!(refined[0].end_bp >= refined[0].start_bp, "end must be >= start");
}

#[test]
fn custom_crossover_threshold() {
    let posteriors = vec![0.1, 0.9, 0.95, 0.1];
    let seg = make_segment(1, 2, 0.93);
    let (starts, ends) = windows_10kb(4);

    // With crossover = 0.3 (lower threshold)
    let refined_low = refine_segment_boundaries(&[seg.clone()], &posteriors, &starts, &ends, 0.3);
    // With crossover = 0.7 (higher threshold)
    let refined_high = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.7);

    // Lower crossover should give earlier (smaller) start
    assert!(refined_low[0].start_bp < refined_high[0].start_bp,
        "lower crossover should place boundary earlier: {} vs {}",
        refined_low[0].start_bp, refined_high[0].start_bp);

    // Lower crossover should give later (larger) end
    assert!(refined_low[0].end_bp > refined_high[0].end_bp,
        "lower crossover should place boundary later: {} vs {}",
        refined_low[0].end_bp, refined_high[0].end_bp);
}

#[test]
fn refinement_preserves_segment_ordering() {
    // end_bp must always be >= start_bp even with extreme posteriors
    let posteriors = vec![0.49, 0.51, 0.49]; // barely crossing threshold
    let seg = make_segment(1, 1, 0.51);
    let (starts, ends) = windows_10kb(3);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    assert!(refined[0].end_bp >= refined[0].start_bp,
        "refined end {} must be >= start {}", refined[0].end_bp, refined[0].start_bp);
}

#[test]
fn refinement_with_5kb_windows() {
    // 5kb windows instead of 10kb
    let n = 6;
    let starts: Vec<u64> = (0..n).map(|i| i as u64 * 5000 + 1).collect();
    let ends: Vec<u64> = (0..n).map(|i| (i as u64 + 1) * 5000).collect();

    let posteriors = vec![0.1, 0.3, 0.9, 0.95, 0.2, 0.05];
    let seg = make_segment(2, 3, 0.93);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);

    // Start interpolation between windows 1 and 2
    // t = (0.5 - 0.3) / (0.9 - 0.3) = 0.333
    // center[1] = 7500.5, center[2] = 12500.5
    // boundary ≈ 7500.5 + 0.333 * 5000 = 9167
    assert!(refined[0].start_bp > 7500);
    assert!(refined[0].start_bp < 12500);
}

#[test]
fn posterior_exactly_at_crossover() {
    // Edge case: posterior is exactly 0.5 at boundary window
    let posteriors = vec![0.5, 0.9, 0.5];
    let seg = make_segment(1, 1, 0.9);
    let (starts, ends) = windows_10kb(3);

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);
    // P_before = 0.5 = crossover, so condition p_before < crossover is false → no refinement
    assert_eq!(refined[0].start_bp, starts[1]);
    assert_eq!(refined[0].end_bp, ends[1]);
}

#[test]
fn refinement_improves_resolution() {
    // Verify that refined boundaries are within the inter-window gap
    // (narrower than full window-snapped boundaries)
    let posteriors = vec![0.1, 0.3, 0.9, 0.95, 0.3, 0.1];
    let seg = make_segment(2, 3, 0.93);
    let (starts, ends) = windows_10kb(6);

    let unrefined_start = starts[2]; // 20001
    let unrefined_end = ends[3]; // 40000

    let refined = refine_segment_boundaries(&[seg], &posteriors, &starts, &ends, 0.5);

    // Refined start should be before the unrefined start (moved into previous window)
    assert!(refined[0].start_bp < unrefined_start,
        "refined start {} should be < unrefined {}", refined[0].start_bp, unrefined_start);

    // Refined end should be after the unrefined end (moved into next window)
    assert!(refined[0].end_bp > unrefined_end,
        "refined end {} should be > unrefined {}", refined[0].end_bp, unrefined_end);
}
