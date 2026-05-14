//! Tests for detect_segments_rle() and merge_segments() — the two major
//! untested public functions in segment.rs.
//!
//! detect_segments_rle: RLE-based segment detection from identity tracks
//! merge_segments: overlapping segment merging with proportional identity estimation

use impopk_ibd::segment::{detect_segments_rle, merge_segments, IdentityTrack, RleParams, Segment};

// ============================================================================
// Helper
// ============================================================================

fn make_positions(n: usize, window_size: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|i| {
            let start = i as u64 * window_size;
            (start, start + window_size - 1)
        })
        .collect()
}

fn make_segment(
    chrom: &str,
    start: u64,
    end: u64,
    hap_a: &str,
    hap_b: &str,
    start_idx: usize,
    end_idx: usize,
    mean_identity: f64,
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
        min_identity: mean_identity,
        identity_sum: mean_identity * n_windows as f64,
        n_called: n_windows,
        start_idx,
        end_idx,
    }
}

// ============================================================================
// detect_segments_rle tests
// ============================================================================

#[test]
fn test_detect_rle_empty_track() {
    let track = IdentityTrack {
        windows: vec![],
        n_total_windows: 0,
    };
    let positions = make_positions(0, 5000);
    let params = RleParams::default();
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert!(segs.is_empty());
}

#[test]
fn test_detect_rle_all_below_threshold() {
    let track = IdentityTrack {
        windows: vec![(0, 0.5), (1, 0.6), (2, 0.7), (3, 0.8)],
        n_total_windows: 4,
    };
    let positions = make_positions(4, 5000);
    let params = RleParams::default(); // min_identity = 0.9995
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert!(segs.is_empty());
}

#[test]
fn test_detect_rle_all_above_threshold() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997), (3, 0.9996)],
        n_total_windows: 4,
    };
    let positions = make_positions(4, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 4);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 19999);
    assert_eq!(segs[0].n_called, 4);
}

#[test]
fn test_detect_rle_segment_too_few_windows() {
    // Only 2 windows above threshold, but min_windows = 3
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998)],
        n_total_windows: 2,
    };
    let positions = make_positions(2, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert!(segs.is_empty());
}

#[test]
fn test_detect_rle_segment_too_short_bp() {
    // 3 windows * 1000bp = 3000bp, but min_length_bp = 5000
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)],
        n_total_windows: 3,
    };
    let positions = make_positions(3, 1000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert!(segs.is_empty());
}

#[test]
fn test_detect_rle_two_separate_segments() {
    // Windows 0-3 high, 4 low, 5-8 high
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999),
            (1, 0.9998),
            (2, 0.9997),
            (3, 0.9996),
            (4, 0.5), // break
            (5, 0.9999),
            (6, 0.9998),
            (7, 0.9997),
            (8, 0.9996),
        ],
        n_total_windows: 9,
    };
    let positions = make_positions(9, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 3);
    assert_eq!(segs[1].start_idx, 5);
    assert_eq!(segs[1].end_idx, 8);
}

#[test]
fn test_detect_rle_gap_bridging() {
    // Windows 0-2 high, 3 missing, 4-6 high — max_gap=1 should bridge
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999),
            (1, 0.9998),
            (2, 0.9997),
            // 3 is missing
            (4, 0.9999),
            (5, 0.9998),
            (6, 0.9997),
        ],
        n_total_windows: 7,
    };
    let positions = make_positions(7, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 6);
    // Only 6 called (window 3 was missing)
    assert_eq!(segs[0].n_called, 6);
    assert_eq!(segs[0].n_windows, 7);
}

#[test]
fn test_detect_rle_gap_too_large() {
    // Windows 0-2 high, 3-4 missing, 5-7 high — max_gap=1 should NOT bridge 2 gaps
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999),
            (1, 0.9998),
            (2, 0.9997),
            // 3 missing
            // 4 missing
            (5, 0.9999),
            (6, 0.9998),
            (7, 0.9997),
        ],
        n_total_windows: 8,
    };
    let positions = make_positions(8, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert_eq!(segs.len(), 2);
}

#[test]
fn test_detect_rle_drop_tolerance() {
    // One window slightly below min_identity but within drop_tolerance
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999),
            (1, 0.9990), // below 0.9995 but above 0.998
            (2, 0.9998),
            (3, 0.9997),
        ],
        n_total_windows: 4,
    };
    let positions = make_positions(4, 5000);
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.002, // effective threshold = 0.9975
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 4);
}

#[test]
fn test_detect_rle_haplotype_normalization() {
    // Haplotypes should be normalized (hap_a <= hap_b)
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)],
        n_total_windows: 3,
    };
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    // Pass hap_b < hap_a (reversed order)
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "ZZZ#1", "AAA#2");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].hap_a, "AAA#2");
    assert_eq!(segs[0].hap_b, "ZZZ#1");
}

#[test]
fn test_detect_rle_single_window_above_threshold() {
    // Only 1 window above threshold, min_windows=3 → no segment
    let track = IdentityTrack {
        windows: vec![(0, 0.5), (1, 0.9999), (2, 0.5)],
        n_total_windows: 3,
    };
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn test_detect_rle_min_identity_at_exact_threshold() {
    // All windows exactly at threshold should be detected
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (1, 0.999), (2, 0.999)],
        n_total_windows: 3,
    };
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
}

#[test]
fn test_detect_rle_tracks_min_identity() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9990), (2, 0.9998)],
        n_total_windows: 3,
    };
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert!((segs[0].min_identity - 0.9990).abs() < 1e-10);
}

#[test]
fn test_detect_rle_segment_at_end_of_track() {
    // Segment running to the last window
    let track = IdentityTrack {
        windows: vec![
            (0, 0.5),
            (1, 0.5),
            (2, 0.9999),
            (3, 0.9998),
            (4, 0.9997),
        ],
        n_total_windows: 5,
    };
    let positions = make_positions(5, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 2);
    assert_eq!(segs[0].end_idx, 4);
    assert_eq!(segs[0].start, 10000);
    assert_eq!(segs[0].end, 24999);
}

#[test]
fn test_detect_rle_low_window_restarts_segment() {
    // When a low-identity window is itself above threshold after ending a segment,
    // it should start a new segment at that window.
    // Actually test the case: high-segment, then a low window (but still above threshold),
    // which should be part of the same segment.
    let track = IdentityTrack {
        windows: vec![
            (0, 0.9999),
            (1, 0.9998),
            (2, 0.9997),
            (3, 0.5),     // break: below threshold
            (4, 0.9999),  // new start
            (5, 0.9998),
            (6, 0.9997),
        ],
        n_total_windows: 7,
    };
    let positions = make_positions(7, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 2);
    // Second segment should start at window 4
    assert_eq!(segs[1].start_idx, 4);
}

// ============================================================================
// merge_segments tests
// ============================================================================

#[test]
fn test_merge_empty() {
    let mut segs: Vec<Segment> = vec![];
    merge_segments(&mut segs);
    assert!(segs.is_empty());
}

#[test]
fn test_merge_single_segment() {
    let mut segs = vec![make_segment("chr1", 0, 9999, "A", "B", 0, 1, 0.999)];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
}

#[test]
fn test_merge_non_overlapping_same_pair() {
    let mut segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 0, 1, 0.999),
        make_segment("chr1", 20000, 29999, "A", "B", 4, 5, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn test_merge_overlapping_same_pair() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 24999);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 4);
    assert_eq!(segs[0].n_windows, 5);
}

#[test]
fn test_merge_different_haplotype_pairs_no_merge() {
    // Different haplotype pairs should never merge even if overlapping
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 5000, 19999, "C", "D", 1, 3, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn test_merge_different_chromosomes_no_merge() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr2", 0, 14999, "A", "B", 0, 2, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn test_merge_contained_segment() {
    // Second segment is fully contained within first
    let mut segs = vec![
        make_segment("chr1", 0, 24999, "A", "B", 0, 4, 0.999),
        make_segment("chr1", 5000, 14999, "A", "B", 1, 2, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 24999);
}

#[test]
fn test_merge_three_overlapping() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.998),
        make_segment("chr1", 20000, 34999, "A", "B", 4, 6, 0.997),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 34999);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 6);
}

#[test]
fn test_merge_preserves_min_identity() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.990),
    ];
    // Set different min_identity values
    segs[0].min_identity = 0.998;
    segs[1].min_identity = 0.985;
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert!((segs[0].min_identity - 0.985).abs() < 1e-10);
}

#[test]
fn test_merge_adjacent_by_index() {
    // Segments touching at index boundary (end_idx of first == start_idx of second)
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
}

#[test]
fn test_merge_unsorted_input() {
    // merge_segments should sort before merging
    let mut segs = vec![
        make_segment("chr1", 20000, 34999, "A", "B", 4, 6, 0.997),
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.998),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 34999);
}

#[test]
fn test_merge_mixed_pairs_and_chroms() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 0.998),
        make_segment("chr1", 0, 14999, "C", "D", 0, 2, 0.997),
        make_segment("chr2", 0, 14999, "A", "B", 0, 2, 0.996),
    ];
    merge_segments(&mut segs);
    // chr1 A-B: merged into 1
    // chr1 C-D: separate (1)
    // chr2 A-B: separate (1)
    assert_eq!(segs.len(), 3);
}

#[test]
fn test_merge_updates_mean_identity() {
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 0.999),
        make_segment("chr1", 15000, 29999, "A", "B", 3, 5, 0.990),
    ];
    // These have start_idx 3 > end_idx 2, so they won't overlap by index
    // Actually start_idx=3 > end_idx=2, so they should NOT merge
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn test_merge_identity_sum_accounting() {
    // Two overlapping segments — check that identity_sum is updated
    let mut segs = vec![
        make_segment("chr1", 0, 14999, "A", "B", 0, 2, 1.0),
        make_segment("chr1", 10000, 24999, "A", "B", 2, 4, 1.0),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    // n_called should account for overlap
    assert!(segs[0].n_called >= 3); // at least the non-overlapping windows
    assert!((segs[0].mean_identity - 1.0).abs() < 1e-6); // all identity=1.0
}
