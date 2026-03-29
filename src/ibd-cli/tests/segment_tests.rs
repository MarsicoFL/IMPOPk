//! Edge case tests for IBD segment utilities:
//! - SegmentLengthStats and segment_length_distribution
//! - segment_length_histogram
//!
//! These complement the unit tests in segment.rs by covering additional
//! edge cases and boundary conditions.

use hprc_ibd::segment::{segment_length_distribution, segment_length_histogram, Segment};

// ===========================================================================
// Helper
// ===========================================================================

fn make_segment(start: u64, end: u64) -> Segment {
    let n_win = ((end.saturating_sub(start)) / 5000 + 1) as usize;
    Segment {
        chrom: "chr1".to_string(),
        start,
        end,
        hap_a: "hap_a".to_string(),
        hap_b: "hap_b".to_string(),
        n_windows: n_win,
        mean_identity: 0.999,
        min_identity: 0.998,
        identity_sum: 0.999 * n_win as f64,
        n_called: n_win,
        start_idx: 0,
        end_idx: n_win.saturating_sub(1),
    }
}

// ===========================================================================
// 1. segment_length_distribution edge cases
// ===========================================================================

/// Two segments: even count median should be average of two middle values.
#[test]
fn test_segment_length_stats_two_segments_median() {
    // Segment 1: length = 10001, Segment 2: length = 20001
    let segments = vec![make_segment(1000, 11000), make_segment(1000, 21000)];
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.count, 2);
    // median = (10001 + 20001) / 2 = 15001.0
    assert!(
        (stats.median_bp - 15001.0).abs() < 0.01,
        "Median should be 15001.0 (avg of 10001 and 20001), got {}",
        stats.median_bp
    );
    // std with n-1 denominator
    let mean = 15001.0;
    let expected_var = ((10001.0_f64 - mean).powi(2) + (20001.0_f64 - mean).powi(2)) / 1.0;
    assert!(
        (stats.std_bp - expected_var.sqrt()).abs() < 0.01,
        "Std should be {}, got {}",
        expected_var.sqrt(),
        stats.std_bp
    );
}

/// All segments have the same length: std should be 0.
#[test]
fn test_segment_length_stats_all_same_length() {
    let segments = vec![
        make_segment(1000, 11000),
        make_segment(2000, 12000),
        make_segment(3000, 13000),
    ];
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.count, 3);
    assert!((stats.mean_bp - 10001.0).abs() < 0.01);
    assert!((stats.median_bp - 10001.0).abs() < 0.01);
    assert!(
        stats.std_bp < 1e-10,
        "Std should be 0 for identical lengths, got {}",
        stats.std_bp
    );
    assert_eq!(stats.min_bp, stats.max_bp);
}

/// Segment with start == end: length_bp() returns 1 (saturating_sub + 1).
#[test]
fn test_segment_length_stats_zero_length_segment() {
    let segments = vec![make_segment(5000, 5000)]; // start == end → length = 1
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.count, 1);
    assert!((stats.mean_bp - 1.0).abs() < 0.01, "Length should be 1");
    assert_eq!(stats.min_bp, 1);
    assert_eq!(stats.max_bp, 1);
    assert_eq!(stats.total_bp, 1);
}

/// Odd count: median is the middle element.
#[test]
fn test_segment_length_stats_odd_count_median() {
    // Lengths: 1001, 5001, 20001 → median = 5001
    let segments = vec![
        make_segment(1000, 2000),   // 1001
        make_segment(1000, 6000),   // 5001
        make_segment(1000, 21000),  // 20001
    ];
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.count, 3);
    assert!(
        (stats.median_bp - 5001.0).abs() < 0.01,
        "Median should be 5001, got {}",
        stats.median_bp
    );
    assert_eq!(stats.min_bp, 1001);
    assert_eq!(stats.max_bp, 20001);
}

/// Large number of segments: verify total_bp is correct.
#[test]
fn test_segment_length_stats_many_segments() {
    let segments: Vec<Segment> = (0..100)
        .map(|i| make_segment(i * 100_000, i * 100_000 + 10_000))
        .collect();

    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 100);
    // Each segment: length = 10001
    assert_eq!(stats.total_bp, 100 * 10001);
    assert!((stats.mean_bp - 10001.0).abs() < 0.01);
}

/// Unsorted input: distribution should still compute correctly.
#[test]
fn test_segment_length_stats_unsorted_input() {
    // Provide in reverse order of length
    let segments = vec![
        make_segment(1000, 31000),  // 30001
        make_segment(1000, 11000),  // 10001
        make_segment(1000, 21000),  // 20001
    ];
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.min_bp, 10001);
    assert_eq!(stats.max_bp, 30001);
    assert!((stats.median_bp - 20001.0).abs() < 0.01);
}

// ===========================================================================
// 2. segment_length_histogram edge cases
// ===========================================================================

/// Single segment: should produce exactly one bin.
#[test]
fn test_histogram_single_segment() {
    let segments = vec![make_segment(1000, 11000)]; // length = 10001
    let hist = segment_length_histogram(&segments, 10000);

    assert_eq!(hist.len(), 1);
    // 10001 / 10000 = bin 1 → (10000, 1)
    assert_eq!(hist[0], (10000, 1));
}

/// All segments in same bin.
#[test]
fn test_histogram_all_same_bin() {
    let segments = vec![
        make_segment(1000, 6000),   // 5001
        make_segment(1000, 7000),   // 6001
        make_segment(1000, 8000),   // 7001
    ];
    let hist = segment_length_histogram(&segments, 10000);

    // All lengths < 10000, so all in bin 0
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 3));
}

/// Bin size larger than max segment: all in bin 0.
#[test]
fn test_histogram_bin_larger_than_segments() {
    let segments = vec![
        make_segment(1000, 2000),   // 1001
        make_segment(1000, 3000),   // 2001
    ];
    let hist = segment_length_histogram(&segments, 1_000_000);

    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 2));
}

/// Bin size of 1: each unique length gets its own bin.
#[test]
fn test_histogram_bin_size_one() {
    let segments = vec![
        make_segment(1000, 1001),   // 2
        make_segment(1000, 1002),   // 3
        make_segment(1000, 1001),   // 2 (duplicate)
    ];
    let hist = segment_length_histogram(&segments, 1);

    // Lengths 2, 3, 2 → bins: (2, 2) and (3, 1)
    assert_eq!(hist.len(), 2);
    let map: std::collections::HashMap<u64, usize> = hist.into_iter().collect();
    assert_eq!(map[&2], 2);
    assert_eq!(map[&3], 1);
}

/// Segments exactly at bin boundaries.
#[test]
fn test_histogram_exact_bin_boundaries() {
    // bin_size = 10000
    // length 10000 → bin 10000/10000 = 1 → (10000, count)
    // length 20000 → bin 20000/10000 = 2 → (20000, count)
    // length 9999 → bin 9999/10000 = 0 → (0, count)
    let segments = vec![
        make_segment(1000, 10999),  // length = 10000, bin 1
        make_segment(1000, 20999),  // length = 20000, bin 2
        make_segment(1000, 10998),  // length = 9999, bin 0
    ];
    let hist = segment_length_histogram(&segments, 10000);

    let map: std::collections::HashMap<u64, usize> = hist.into_iter().collect();
    assert_eq!(map.get(&0), Some(&1), "9999 should be in bin 0");
    assert_eq!(map.get(&10000), Some(&1), "10000 should be in bin 10000");
    assert_eq!(map.get(&20000), Some(&1), "20000 should be in bin 20000");
}

/// Multiple segments spanning many bins with gaps.
#[test]
fn test_histogram_sparse_bins() {
    let segments = vec![
        make_segment(0, 999),           // 1000, bin 0
        make_segment(0, 99_999),        // 100000, bin 10
        make_segment(0, 999_999),       // 1000000, bin 100
    ];
    let hist = segment_length_histogram(&segments, 10000);

    // Should have exactly 3 non-empty bins (0, 10, 100)
    assert_eq!(
        hist.len(),
        3,
        "Expected 3 non-empty bins, got {:?}",
        hist
    );
}

/// Histogram preserves sorted order by bin_start.
#[test]
fn test_histogram_sorted_output() {
    let segments = vec![
        make_segment(0, 99_999),  // 100000
        make_segment(0, 999),     // 1000
        make_segment(0, 49_999),  // 50000
    ];
    let hist = segment_length_histogram(&segments, 10000);

    for window in hist.windows(2) {
        assert!(
            window[0].0 < window[1].0,
            "Bins not sorted: {:?}",
            hist
        );
    }
}

// ===========================================================================
// 3. format_segment_bed edge cases
// ===========================================================================

use hprc_ibd::segment::format_segment_bed;

/// BED line with LOD exactly 0: score should be 0.
#[test]
fn test_format_bed_lod_zero() {
    let seg = make_segment(1000, 5000);
    let bed = format_segment_bed(&seg, 0.0);
    // score = 0 since lod = 0
    assert!(bed.contains("\t0\t."), "Score should be 0 for LOD=0: {}", bed);
}

/// BED line with very large LOD: score capped at 1000.
#[test]
fn test_format_bed_lod_very_large() {
    let seg = make_segment(1000, 5000);
    let bed = format_segment_bed(&seg, 999.9);
    // score = min(999.9 * 100, 1000) = 1000
    assert!(bed.contains("\t1000\t."), "Score should be capped at 1000: {}", bed);
}

/// BED line with LOD just above 0: score should be a small integer.
#[test]
fn test_format_bed_lod_small_positive() {
    let seg = make_segment(1000, 5000);
    let bed = format_segment_bed(&seg, 0.005);
    // score = round(0.005 * 100) = 1 (after rounding)
    // 0.005 * 100 = 0.5, round = 1
    assert!(bed.contains("\t1\t.") || bed.contains("\t0\t."),
        "Small LOD should produce small score: {}", bed);
}

/// BED line with start=1: 0-based start should be 0.
#[test]
fn test_format_bed_start_one_converts_to_zero() {
    let mut seg = make_segment(1, 5000);
    seg.start = 1;
    let bed = format_segment_bed(&seg, 5.0);
    // BED start = 1 - 1 = 0
    let fields: Vec<&str> = bed.split('\t').collect();
    assert_eq!(fields[1], "0", "BED start should be 0 when segment start is 1");
}

/// BED line with start=0: saturating_sub(1) should give 0 (not underflow).
#[test]
fn test_format_bed_start_zero_no_underflow() {
    let mut seg = make_segment(0, 5000);
    seg.start = 0;
    let bed = format_segment_bed(&seg, 5.0);
    let fields: Vec<&str> = bed.split('\t').collect();
    // saturating_sub(1) on 0 gives 0
    assert_eq!(fields[1], "0", "BED start should handle 0 with saturating_sub");
}

// ===========================================================================
// 4. IdentityTrack edge cases
// ===========================================================================

use hprc_ibd::segment::IdentityTrack;

/// IdentityTrack::to_map with duplicate window indices: last value wins.
#[test]
fn test_identity_track_to_map_duplicates() {
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (0, 0.888), (1, 0.997)],
        n_total_windows: 2,
    };
    let map = track.to_map();
    // HashMap::from_iter with duplicates: last value wins
    // But actually .cloned().collect() behavior for HashMaps gives arbitrary last-insert-wins
    assert_eq!(map.len(), 2, "Should have 2 unique keys");
    assert!(map.contains_key(&0));
    assert!(map.contains_key(&1));
}

/// IdentityTrack::get with out-of-bounds index returns None.
#[test]
fn test_identity_track_get_out_of_bounds() {
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (5, 0.998)],
        n_total_windows: 10,
    };
    assert!(track.get(100).is_none(), "Out-of-range index should return None");
    assert!(track.get(usize::MAX).is_none(), "usize::MAX should return None");
}

// ===========================================================================
// 5. segment_length_distribution with 4 segments (even count, non-trivial median)
// ===========================================================================

/// Four segments: median should be average of 2nd and 3rd values.
#[test]
fn test_segment_length_stats_four_segments_median() {
    // Lengths: 1001, 5001, 10001, 50001
    let segments = vec![
        make_segment(0, 1000),     // 1001
        make_segment(0, 5000),     // 5001
        make_segment(0, 10000),    // 10001
        make_segment(0, 50000),    // 50001
    ];
    let stats = segment_length_distribution(&segments);

    assert_eq!(stats.count, 4);
    // Sorted: [1001, 5001, 10001, 50001]
    // Median = (5001 + 10001) / 2 = 7501.0
    assert!(
        (stats.median_bp - 7501.0).abs() < 0.01,
        "Median of 4 segments should be 7501.0, got {}",
        stats.median_bp
    );
}
