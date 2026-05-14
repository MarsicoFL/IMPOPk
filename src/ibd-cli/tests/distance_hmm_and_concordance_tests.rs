//! Tests for distance-aware HMM algorithms, merge_segments proportional
//! overlap, covered_bp/intersection_bp, and concordance edge cases.
//!
//! Cycle 23: targets uncovered edge cases in forward/backward_with_distances,
//! GeneticMap, merge_segments proportional identity, and concordance interval
//! helpers.

use impopk_ibd::hmm::{
    backward_with_distances, distance_dependent_log_transition, forward_backward_with_distances,
    forward_with_distances, GeneticMap, HmmParams,
    recombination_aware_log_transition,
};

use impopk_ibd::segment::{
    detect_segments_rle, merge_segments, segment_length_distribution, segment_length_histogram,
    IdentityTrack, RleParams, Segment,
};

use impopk_ibd::concordance::{
    boundary_accuracy, f1_score, haplotype_level_concordance, length_correlation,
    matched_segments, per_window_concordance, segment_overlap_bp, segments_jaccard,
    segments_precision_recall,
};

// ============================================================================
// forward_with_distances / backward_with_distances edge cases
// ============================================================================

#[test]
fn test_forward_with_distances_empty() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let (alpha, ll) = forward_with_distances(&[], &params, &[]);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn test_backward_with_distances_empty() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let beta = backward_with_distances(&[], &params, &[]);
    assert!(beta.is_empty());
}

#[test]
fn test_forward_backward_with_distances_empty() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let (posteriors, ll) = forward_backward_with_distances(&[], &params, &[]);
    assert!(posteriors.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn test_forward_with_distances_single_obs() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.999];
    let positions = [(0u64, 4999u64)];
    let (alpha, ll) = forward_with_distances(&obs, &params, &positions);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

#[test]
fn test_backward_with_distances_single_obs() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.999];
    let positions = [(0u64, 4999u64)];
    let beta = backward_with_distances(&obs, &params, &positions);
    assert_eq!(beta.len(), 1);
    // Last element of backward should be [0.0, 0.0] (log(1) = 0)
    assert_eq!(beta[0], [0.0, 0.0]);
}

#[test]
fn test_forward_with_distances_mismatched_positions_falls_back() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.999, 0.998, 0.997];
    let positions = [(0u64, 4999u64)]; // Wrong length — only 1 position for 3 obs

    // Should fall back to regular forward algorithm (not crash)
    let (alpha, ll) = forward_with_distances(&obs, &params, &positions);
    assert_eq!(alpha.len(), 3);
    assert!(ll.is_finite());
}

#[test]
fn test_backward_with_distances_mismatched_positions_falls_back() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.999, 0.998];
    let positions = [(0u64, 4999u64), (5000, 9999), (10000, 14999)]; // 3 positions for 2 obs

    let beta = backward_with_distances(&obs, &params, &positions);
    assert_eq!(beta.len(), 2);
}

#[test]
fn test_forward_backward_with_distances_posteriors_valid() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.5, 0.6, 0.999, 0.9995, 0.9998, 0.5, 0.4];
    let positions: Vec<(u64, u64)> = (0..7)
        .map(|i| (i * 5000, i * 5000 + 4999))
        .collect();

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert_eq!(posteriors.len(), 7);
    assert!(ll.is_finite());

    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior {} out of [0,1]", p);
    }
}

#[test]
fn test_forward_backward_with_distances_large_gap() {
    // Two observations with a huge gap between them — should increase transition probability
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.9999, 0.5]; // First IBD-like, second non-IBD-like
    let positions = [(0u64, 4999u64), (1_000_000, 1_004_999)]; // 1Mb gap

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert_eq!(posteriors.len(), 2);
    assert!(ll.is_finite());
}

#[test]
fn test_forward_backward_with_distances_adjacent_windows() {
    // Adjacent windows (no gap) — results should be similar to standard forward-backward
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let obs = [0.9999, 0.9998, 0.9997, 0.9996, 0.9999];
    let positions: Vec<(u64, u64)> = (0..5)
        .map(|i| (i * 5000, (i + 1) * 5000 - 1))
        .collect();

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert_eq!(posteriors.len(), 5);
    assert!(ll.is_finite());

    // All high identity — posteriors should lean toward IBD
    for &p in &posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

// ============================================================================
// distance_dependent_log_transition additional edge cases
// ============================================================================

#[test]
fn test_distance_dependent_very_small_distance() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 1, 5000);

    // Very small distance → very small transition probability
    let p_enter = trans[0][1].exp();
    assert!(p_enter > 0.0 && p_enter < 0.001,
        "Expected very small p_enter for tiny distance, got {}", p_enter);

    // Rows should sum to ~1.0 in probability space
    let row0_sum = trans[0][0].exp() + trans[0][1].exp();
    assert!((row0_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_distance_dependent_extremely_large_distance() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let trans = distance_dependent_log_transition(&params, 100_000_000, 5000);

    // Huge distance: probability of state change approaches stationary distribution
    // Both rows should still sum to 1.0
    for row in 0..2 {
        let row_sum = trans[row][0].exp() + trans[row][1].exp();
        assert!((row_sum - 1.0).abs() < 1e-6,
            "Row {} sum = {}, expected ~1.0", row, row_sum);
    }
}

#[test]
fn test_distance_dependent_monotonic_with_distance() {
    // Transition probability should increase monotonically with distance
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    let distances = [1000u64, 5000, 10000, 50000, 200000, 1_000_000];
    let mut prev_p_enter = 0.0;
    for &d in &distances {
        let trans = distance_dependent_log_transition(&params, d, 5000);
        let p_enter = trans[0][1].exp();
        assert!(p_enter >= prev_p_enter,
            "p_enter should increase with distance: {} < {} at d={}", p_enter, prev_p_enter, d);
        prev_p_enter = p_enter;
    }
}

// ============================================================================
// GeneticMap edge cases
// ============================================================================

#[test]
fn test_genetic_map_uniform_distance() {
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0); // 1 cM/Mb
    let d = gmap.genetic_distance_cm(0, 500_000);
    assert!((d - 0.5).abs() < 1e-6, "Expected 0.5 cM, got {}", d);
}

#[test]
fn test_genetic_map_uniform_full_distance() {
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
    let d = gmap.genetic_distance_cm(0, 1_000_000);
    assert!((d - 1.0).abs() < 1e-6, "Expected 1.0 cM, got {}", d);
}

#[test]
fn test_genetic_map_uniform_zero_distance() {
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
    let d = gmap.genetic_distance_cm(500_000, 500_000);
    assert_eq!(d, 0.0);
}

#[test]
fn test_genetic_map_uniform_reversed_positions() {
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
    let d = gmap.genetic_distance_cm(800_000, 200_000);
    // abs() is used internally
    assert!((d - 0.6).abs() < 1e-6, "Expected 0.6 cM for reversed, got {}", d);
}

#[test]
fn test_genetic_map_len_and_is_empty() {
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
    assert_eq!(gmap.len(), 2);
    assert!(!gmap.is_empty());
}

#[test]
fn test_genetic_map_different_rates() {
    let slow = GeneticMap::uniform(0, 1_000_000, 0.5); // 0.5 cM/Mb
    let fast = GeneticMap::uniform(0, 1_000_000, 2.0); // 2.0 cM/Mb
    let d_slow = slow.genetic_distance_cm(0, 1_000_000);
    let d_fast = fast.genetic_distance_cm(0, 1_000_000);
    assert!((d_slow - 0.5).abs() < 1e-6);
    assert!((d_fast - 2.0).abs() < 1e-6);
}

// ============================================================================
// recombination_aware_log_transition: more scenarios
// ============================================================================

#[test]
fn test_recombination_aware_rows_sum_to_one() {
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);
    let gmap = GeneticMap::uniform(0, 10_000_000, 1.0);

    for &(pos1, pos2) in &[
        (100_000, 200_000),
        (0, 5_000_000),
        (1_000_000, 1_005_000),
    ] {
        let trans = recombination_aware_log_transition(&params, pos1, pos2, &gmap, 5000);
        for row in 0..2 {
            let row_sum = trans[row][0].exp() + trans[row][1].exp();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} at ({},{}) sums to {}", row, pos1, pos2, row_sum
            );
        }
    }
}

#[test]
fn test_recombination_aware_higher_rate_increases_exit() {
    // Higher recombination rate region should have higher exit probability
    let params = HmmParams::from_expected_length(10.0, 0.001, 5000);

    let slow_map = GeneticMap::uniform(0, 10_000_000, 0.5); // low recomb rate
    let fast_map = GeneticMap::uniform(0, 10_000_000, 3.0); // high recomb rate

    let trans_slow = recombination_aware_log_transition(&params, 0, 100_000, &slow_map, 5000);
    let trans_fast = recombination_aware_log_transition(&params, 0, 100_000, &fast_map, 5000);

    let p_exit_slow = trans_slow[1][0].exp();
    let p_exit_fast = trans_fast[1][0].exp();

    assert!(
        p_exit_fast > p_exit_slow,
        "Higher recomb rate should increase exit prob: fast={} vs slow={}",
        p_exit_fast, p_exit_slow
    );
}

// ============================================================================
// merge_segments: proportional identity estimation
// ============================================================================

fn make_segment(
    chrom: &str,
    start: u64,
    end: u64,
    hap_a: &str,
    hap_b: &str,
    start_idx: usize,
    end_idx: usize,
    mean_identity: f64,
    min_identity: f64,
) -> Segment {
    let n_windows = end_idx - start_idx + 1;
    Segment {
        chrom: chrom.to_string(),
        start,
        end,
        hap_a: hap_a.to_string(),
        hap_b: hap_b.to_string(),
        n_windows,
        mean_identity,
        min_identity,
        identity_sum: mean_identity * n_windows as f64,
        n_called: n_windows,
        start_idx,
        end_idx,
    }
}

#[test]
fn test_merge_segments_proportional_identity_estimation() {
    // seg1: indices [0-9] (10 windows), mean=0.999
    // seg2: indices [8-14] (7 windows), mean=0.998
    // overlap: [8-9] = 2 windows
    // non-overlap of seg2: 5 out of 7 windows
    // Expected after merge:
    //   identity_sum = 0.999*10 + 0.998*7*(5/7) = 9.99 + 4.99 = 14.98
    //   n_called = 10 + round(7 * 5/7) = 10 + 5 = 15
    //   mean_identity = 14.98 / 15 ≈ 0.99867
    let mut segments = vec![
        make_segment("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998),
        make_segment("chr1", 8000, 14999, "A", "B", 8, 14, 0.998, 0.997),
    ];

    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 14);
    assert_eq!(segments[0].n_windows, 15); // indices 0-14

    // Check identity_sum is reasonable
    assert!(segments[0].identity_sum > 14.0 && segments[0].identity_sum < 15.5);
    assert!(segments[0].min_identity <= 0.997);
}

#[test]
fn test_merge_segments_complete_containment_identity() {
    // seg1: indices [0-20] (21 windows), mean=0.999
    // seg2: indices [5-15] (11 windows), mean=0.998
    // overlap: [5-15] = 11 windows (all of seg2)
    // non-overlap of seg2: 0 windows → no identity contribution
    let mut segments = vec![
        make_segment("chr1", 0, 20000, "A", "B", 0, 20, 0.999, 0.998),
        make_segment("chr1", 5000, 15000, "A", "B", 5, 15, 0.998, 0.997),
    ];

    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 21); // unchanged
    assert_eq!(segments[0].end, 20000); // kept larger end
    // Identity should barely change since seg2 contributes 0 non-overlap windows
    assert!((segments[0].mean_identity - 0.999).abs() < 0.002);
}

#[test]
fn test_merge_segments_different_n_called() {
    // Test with segments where n_called != n_windows (partial data)
    let mut seg1 = make_segment("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998);
    seg1.n_called = 8; // Only 8 of 10 windows have data
    seg1.identity_sum = 0.999 * 8.0;

    let seg2 = make_segment("chr1", 5000, 14999, "A", "B", 5, 14, 0.998, 0.997);

    let mut segments = vec![seg1, seg2];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 15);
    assert!(segments[0].n_called > 0);
}

#[test]
fn test_merge_segments_three_overlapping_same_pair() {
    // Three segments with cascading overlaps, same haplotype pair
    let mut segments = vec![
        make_segment("chr1", 0, 5000, "A", "B", 0, 5, 0.999, 0.999),
        make_segment("chr1", 4000, 9000, "A", "B", 4, 9, 0.998, 0.998),
        make_segment("chr1", 8000, 13000, "A", "B", 8, 13, 0.997, 0.997),
    ];

    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 13);
    assert_eq!(segments[0].n_windows, 14);
    assert_eq!(segments[0].min_identity, 0.997);
}

#[test]
fn test_merge_segments_interleaved_pairs() {
    // Different haplotype pairs that overlap in position but should NOT merge
    let mut segments = vec![
        make_segment("chr1", 0, 10000, "A", "B", 0, 10, 0.999, 0.999),
        make_segment("chr1", 5000, 15000, "C", "D", 5, 15, 0.998, 0.998),
        make_segment("chr1", 8000, 18000, "A", "B", 8, 18, 0.997, 0.997),
    ];

    merge_segments(&mut segments);
    // A-B should merge (overlap at indices 8-10), C-D stays separate
    assert_eq!(segments.len(), 2);
    // Find the A-B merged segment
    let ab = segments.iter().find(|s| s.hap_a == "A" && s.hap_b == "B").unwrap();
    assert_eq!(ab.start_idx, 0);
    assert_eq!(ab.end_idx, 18);
}

// ============================================================================
// detect_segments_rle: gap bridging scenarios
// ============================================================================

#[test]
fn test_detect_segments_rle_max_gap_zero() {
    // max_gap=0: no gaps tolerated
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9999), (3, 0.9999), (4, 0.9999), (5, 0.9999)],
        n_total_windows: 6,
    };
    let positions: Vec<(u64, u64)> = (0..6)
        .map(|i| (i * 2000, i * 2000 + 1999))
        .collect();
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 2,
        min_length_bp: 1000,
        drop_tolerance: 0.0,
    };

    let segments = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // Gap at index 2 should split: [0-1] and [3-5]
    assert_eq!(segments.len(), 2);
}

#[test]
fn test_detect_segments_rle_max_gap_two() {
    // max_gap=2: can bridge 2 missing windows
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9999), (4, 0.9999), (5, 0.9999)],
        n_total_windows: 6,
    };
    let positions: Vec<(u64, u64)> = (0..6)
        .map(|i| (i * 2000, i * 2000 + 1999))
        .collect();
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 2,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segments = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // Gaps at indices 2-3 (2 gaps), should bridge
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 6); // indices 0-5
}

#[test]
fn test_detect_segments_rle_low_followed_by_restart() {
    // A low-identity window should end the segment, then a new high-identity
    // window should start a new segment
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999), (1, 0.9999), (2, 0.9999),
            (3, 0.5), // Low identity — ends segment
            (4, 0.9999), (5, 0.9999), (6, 0.9999),
        ],
        n_total_windows: 7,
    };
    let positions: Vec<(u64, u64)> = (0..7)
        .map(|i| (i * 2000, i * 2000 + 1999))
        .collect();
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segments = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].n_called, 3); // windows 0-2
    assert_eq!(segments[1].n_called, 3); // windows 4-6
}

// ============================================================================
// segment_length_distribution: more edge cases
// ============================================================================

#[test]
fn test_segment_length_distribution_three_same_length() {
    let segments = vec![
        make_segment("chr1", 0, 4999, "A", "B", 0, 4, 0.999, 0.999),
        make_segment("chr1", 0, 4999, "C", "D", 0, 4, 0.999, 0.999),
        make_segment("chr1", 0, 4999, "E", "F", 0, 4, 0.999, 0.999),
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 3);
    assert_eq!(stats.mean_bp, 5000.0);
    assert_eq!(stats.median_bp, 5000.0);
    assert_eq!(stats.std_bp, 0.0);
    assert_eq!(stats.min_bp, 5000);
    assert_eq!(stats.max_bp, 5000);
    assert_eq!(stats.total_bp, 15000);
}

#[test]
fn test_segment_length_histogram_single_segment_exact_bin() {
    let segments = vec![
        make_segment("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.999), // 10000 bp
    ];
    let hist = segment_length_histogram(&segments, 10000);
    // 10000 / 10000 = bin 1 → (10000, 1)
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (10000, 1));
}

// ============================================================================
// concordance: covered_bp and intersection edge cases
// ============================================================================

#[test]
fn test_segments_jaccard_overlapping_segments_in_same_set() {
    // Both sets have overlapping segments within themselves
    let ours = vec![(100, 300), (200, 400)]; // overlapping → covers 100-400 = 300bp
    let theirs = vec![(150, 350)]; // 200bp
    // Intersection: 150-350 within 100-400 → 200bp
    // Union: 100-400 = 300bp
    let j = segments_jaccard(&ours, &theirs, (0, 1000));
    let expected = 200.0 / 300.0;
    assert!(
        (j - expected).abs() < 1e-9,
        "Expected Jaccard={}, got {}",
        expected,
        j
    );
}

#[test]
fn test_segments_precision_recall_superset() {
    // Our segments are a superset of theirs
    let ours = vec![(100, 500)]; // 400bp
    let theirs = vec![(200, 300)]; // 100bp
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
    // Precision: 100/400 = 0.25
    // Recall: 100/100 = 1.0
    assert!((p - 0.25).abs() < 1e-9, "Expected precision=0.25, got {}", p);
    assert!((r - 1.0).abs() < 1e-9, "Expected recall=1.0, got {}", r);
}

#[test]
fn test_per_window_concordance_1bp_windows() {
    // Very small window size
    let ours = vec![(0, 5)];
    let theirs = vec![(0, 5)];
    let c = per_window_concordance(&ours, &theirs, (0, 10), 1);
    // Windows: [0,1), [1,2), ... [9,10) — 10 windows
    // ours covers [0,5), theirs covers [0,5)
    // Windows 0-4: both have it (half=0, coverage=1 > 0) → agree
    // Windows 5-9: neither → agree
    assert!((c - 1.0).abs() < 1e-9);
}

#[test]
fn test_per_window_concordance_region_equals_window() {
    // Single window covering the whole region
    let ours = vec![(0, 100)];
    let theirs = vec![(50, 100)];
    let c = per_window_concordance(&ours, &theirs, (0, 100), 100);
    // One window [0,100), half = 50
    // ours covers 100bp > 50 → yes
    // theirs covers 50bp > 50 → no (50 is NOT > 50, it's equal)
    // Actually: theirs overlap with [0,100) is [50,100) = 50bp. half = (100-0)/2 = 50. 50 > 50 is false.
    // So ours_call=true, theirs_call=false → disagree → c=0.0
    assert!((c - 0.0).abs() < 1e-9);
}

// ============================================================================
// matched_segments: multi-match scenarios
// ============================================================================

#[test]
fn test_matched_segments_many_to_many() {
    // Multiple segments on each side with cross-matching
    let ours = vec![(100, 200), (300, 400), (500, 600)];
    let theirs = vec![(110, 190), (310, 390), (510, 590)];
    let m = matched_segments(&ours, &theirs, 0.5);
    // Each pair should match (80bp overlap / 80bp shorter = 1.0)
    assert_eq!(m.len(), 3);
}

#[test]
fn test_matched_segments_one_ours_matches_multiple_theirs() {
    // One large ours matches multiple small theirs
    let ours = vec![(100, 500)]; // 400bp
    let theirs = vec![(100, 200), (300, 400)]; // 100bp each
    // Overlap: (100,200) → 100bp, shorter=100, frac=1.0
    // Overlap: (300,400) → 100bp, shorter=100, frac=1.0
    let m = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m.len(), 2);
    assert!(m.contains(&(0, 0)));
    assert!(m.contains(&(0, 1)));
}

// ============================================================================
// length_correlation: specific scenarios
// ============================================================================

#[test]
fn test_length_correlation_zero_length_pair() {
    // One pair with zero-length intervals
    let matches = vec![
        ((100u64, 100u64), (200u64, 200u64)), // both length 0
        ((0, 300), (0, 300)),                   // both length 300
    ];
    let r = length_correlation(&matches);
    assert!(r.is_finite(), "Correlation should be finite even with zero-length pairs");
}

#[test]
fn test_length_correlation_three_pairs_known_value() {
    // ours lengths: [100, 200, 300], theirs: [150, 250, 350]
    // Perfect linear relationship y = x + 50 → r = 1.0
    let matches = vec![
        ((0u64, 100u64), (0u64, 150u64)),
        ((0, 200), (0, 250)),
        ((0, 300), (0, 350)),
    ];
    let r = length_correlation(&matches);
    assert!(
        (r - 1.0).abs() < 1e-6,
        "Expected r=1.0 for perfect linear relationship, got {}",
        r
    );
}

// ============================================================================
// boundary_accuracy: more scenarios
// ============================================================================

#[test]
fn test_boundary_accuracy_even_count_median() {
    // 4 matches → even count → median is average of middle two
    let matches = vec![
        ((100u64, 1000u64), (110u64, 1010u64)), // start diff=10, end diff=10
        ((200, 2000), (250, 2100)),               // start diff=50, end diff=100
        ((300, 3000), (310, 3030)),               // start diff=10, end diff=30
        ((400, 4000), (480, 4200)),               // start diff=80, end diff=200
    ];
    let acc = boundary_accuracy(&matches, 1000).unwrap();
    assert_eq!(acc.n_matched, 4);

    // Sorted start distances: [10, 10, 50, 80] → median = (10+50)/2 = 30
    assert!((acc.median_start_distance_bp - 30.0).abs() < 1e-9);
    // Sorted end distances: [10, 30, 100, 200] → median = (30+100)/2 = 65
    assert!((acc.median_end_distance_bp - 65.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_all_within_threshold() {
    let matches = vec![
        ((100u64, 1000u64), (105u64, 1003u64)), // 5, 3
        ((200, 2000), (202, 2001)),               // 2, 1
    ];
    let acc = boundary_accuracy(&matches, 10).unwrap();
    assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
    assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

// ============================================================================
// haplotype_level_concordance: edge cases
// ============================================================================

#[test]
fn test_haplotype_concordance_only_our_data() {
    // We have segments but hap-ibd doesn't for this pair
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 1000u64, 5000u64),
    ];
    let hapibd_segs: Vec<(String, u8, String, u8, u64, u64)> = vec![];

    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    ).unwrap();

    assert_eq!(result.n_our_hap_combos, 1);
    assert_eq!(result.n_hapibd_hap_combos, 0);
    assert!((result.best_jaccard - 0.0).abs() < 1e-9);
}

#[test]
fn test_haplotype_concordance_only_hapibd_data() {
    // hap-ibd has segments but we don't for this pair
    let our_segs: Vec<(String, String, u64, u64)> = vec![];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 1000u64, 5000u64),
    ];

    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    ).unwrap();

    assert_eq!(result.n_our_hap_combos, 0);
    assert_eq!(result.n_hapibd_hap_combos, 1);
    assert!((result.best_jaccard - 0.0).abs() < 1e-9);
}

#[test]
fn test_haplotype_concordance_unrelated_pair() {
    // Segments exist but for a completely different sample pair
    let our_segs = vec![
        ("HG003#1#C".to_string(), "HG004#2#C".to_string(), 1000u64, 5000u64),
    ];
    let hapibd_segs = vec![
        ("HG005".to_string(), 1u8, "HG006".to_string(), 2u8, 1000u64, 5000u64),
    ];

    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    );
    assert!(result.is_none(), "Should be None for unrelated pair");
}

#[test]
fn test_haplotype_concordance_all_four_combos() {
    // Both samples have IBD on all 4 haplotype combinations (1-1, 1-2, 2-1, 2-2)
    let our_segs = vec![
        ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 0u64, 5000u64),
        ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 0u64, 5000u64),
        ("HG001#2#C".to_string(), "HG002#1#C".to_string(), 0u64, 5000u64),
        ("HG001#2#C".to_string(), "HG002#2#C".to_string(), 0u64, 5000u64),
    ];
    let hapibd_segs = vec![
        ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 0u64, 5000u64),
        ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 0u64, 5000u64),
        ("HG001".to_string(), 2u8, "HG002".to_string(), 1u8, 0u64, 5000u64),
        ("HG001".to_string(), 2u8, "HG002".to_string(), 2u8, 0u64, 5000u64),
    ];

    let result = haplotype_level_concordance(
        &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
    ).unwrap();

    assert_eq!(result.n_our_hap_combos, 4);
    assert_eq!(result.n_hapibd_hap_combos, 4);
    assert!((result.best_jaccard - 1.0).abs() < 1e-9);
    assert!((result.sample_level_jaccard - 1.0).abs() < 1e-9);
}

// ============================================================================
// f1_score: boundary cases
// ============================================================================

#[test]
fn test_f1_score_one_zero() {
    assert!((f1_score(1.0, 0.0) - 0.0).abs() < 1e-9);
    assert!((f1_score(0.0, 1.0) - 0.0).abs() < 1e-9);
}

#[test]
fn test_f1_score_known_value() {
    // precision=0.9, recall=0.6 → F1 = 2*0.54/1.5 = 0.72
    let f = f1_score(0.9, 0.6);
    assert!((f - 0.72).abs() < 1e-6);
}

// ============================================================================
// segment_overlap_bp: more edge cases
// ============================================================================

#[test]
fn test_segment_overlap_bp_reversed() {
    // If intervals are reversed (start > end), overlap should be 0
    assert_eq!(segment_overlap_bp((200, 100), (100, 200)), 0);
}

#[test]
fn test_segment_overlap_bp_large_intervals() {
    let a = (0, 1_000_000_000);
    let b = (500_000_000, 2_000_000_000);
    assert_eq!(segment_overlap_bp(a, b), 500_000_000);
}

#[test]
fn test_segment_overlap_bp_nested() {
    // b completely inside a
    assert_eq!(segment_overlap_bp((0, 1000), (100, 200)), 100);
    // a completely inside b
    assert_eq!(segment_overlap_bp((100, 200), (0, 1000)), 100);
}
