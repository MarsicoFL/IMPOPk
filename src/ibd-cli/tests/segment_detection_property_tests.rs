//! Property-based tests for RLE segment detection, merge, and related utilities.
//!
//! Tests detect_segments_rle with various gap bridging scenarios, drop tolerance,
//! boundary conditions. Tests merge_segments overlap calculations.
//! Tests format_segment_bed, segment_length_distribution, and segment_length_histogram.

use hprc_ibd::segment::{
    detect_segments_rle, merge_segments, format_segment_bed,
    segment_length_distribution, segment_length_histogram,
    IdentityTrack, RleParams, Segment,
};

fn make_positions(n: usize, window_size: u64) -> Vec<(u64, u64)> {
    (0..n).map(|i| {
        let start = i as u64 * window_size;
        (start, start + window_size - 1)
    }).collect()
}

fn make_segment(
    chrom: &str, start: u64, end: u64, hap_a: &str, hap_b: &str,
    start_idx: usize, end_idx: usize, mean_identity: f64, min_identity: f64,
) -> Segment {
    let n_windows = end_idx - start_idx + 1;
    Segment {
        chrom: chrom.into(), start, end,
        hap_a: hap_a.into(), hap_b: hap_b.into(),
        n_windows, mean_identity, min_identity,
        identity_sum: mean_identity * n_windows as f64,
        n_called: n_windows, start_idx, end_idx,
    }
}

// ── detect_segments_rle: basic detection ─────────────────────────────────

#[test]
fn rle_detects_high_identity_run() {
    let track = IdentityTrack {
        windows: (0..10).map(|i| (i, 0.9999)).collect(),
        n_total_windows: 10,
    };
    let positions = make_positions(10, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A#1", "B#1");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 10);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 49999);
}

#[test]
fn rle_no_detection_below_threshold() {
    let track = IdentityTrack {
        windows: (0..10).map(|i| (i, 0.90)).collect(),
        n_total_windows: 10,
    };
    let positions = make_positions(10, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A#1", "B#1");
    assert!(segs.is_empty());
}

#[test]
fn rle_two_disjoint_segments() {
    let mut windows: Vec<(usize, f64)> = Vec::new();
    // Segment 1: windows 0-4
    for i in 0..5 { windows.push((i, 0.9999)); }
    // Gap at 5-6
    for i in 5..7 { windows.push((i, 0.50)); }
    // Segment 2: windows 7-11
    for i in 7..12 { windows.push((i, 0.9998)); }

    let track = IdentityTrack { windows, n_total_windows: 12 };
    let positions = make_positions(12, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].n_windows, 5);
    assert_eq!(segs[1].n_windows, 5);
}

// ── Gap bridging ────────────────────────────────────────────────────────

#[test]
fn rle_bridges_single_missing_window() {
    let mut windows: Vec<(usize, f64)> = Vec::new();
    for i in 0..5 { windows.push((i, 0.9999)); }
    // Window 5 is missing (not in track)
    for i in 6..11 { windows.push((i, 0.9998)); }

    let track = IdentityTrack { windows, n_total_windows: 11 };
    let positions = make_positions(11, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1, // Bridge 1 missing window
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1); // Bridged into one segment
    assert_eq!(segs[0].n_windows, 11);
}

#[test]
fn rle_does_not_bridge_two_missing_with_max_gap_1() {
    let mut windows: Vec<(usize, f64)> = Vec::new();
    for i in 0..5 { windows.push((i, 0.9999)); }
    // Windows 5-6 are missing
    for i in 7..12 { windows.push((i, 0.9998)); }

    let track = IdentityTrack { windows, n_total_windows: 12 };
    let positions = make_positions(12, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 1, // Can only bridge 1
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 2); // Not bridged
}

#[test]
fn rle_bridges_two_missing_with_max_gap_2() {
    let mut windows: Vec<(usize, f64)> = Vec::new();
    for i in 0..5 { windows.push((i, 0.9999)); }
    // Windows 5-6 are missing
    for i in 7..12 { windows.push((i, 0.9998)); }

    let track = IdentityTrack { windows, n_total_windows: 12 };
    let positions = make_positions(12, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 2,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1); // Bridged
}

// ── Drop tolerance ──────────────────────────────────────────────────────

#[test]
fn rle_drop_tolerance_includes_slightly_below_threshold() {
    let mut windows: Vec<(usize, f64)> = Vec::new();
    for i in 0..5 { windows.push((i, 0.9999)); }
    // Window 5 is slightly below threshold
    windows.push((5, 0.9990));
    for i in 6..11 { windows.push((i, 0.9998)); }

    let track = IdentityTrack { windows, n_total_windows: 11 };
    let positions = make_positions(11, 5000);

    // Without drop tolerance: breaks into 2 segments
    let params_no_tol = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params_no_tol, "chr1", "A", "B");
    assert_eq!(segs.len(), 2);

    // With drop tolerance: single segment (0.9990 >= 0.9995 - 0.001)
    let params_tol = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 3,
        min_length_bp: 5000,
        drop_tolerance: 0.001,
    };
    let segs = detect_segments_rle(&track, &positions, &params_tol, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
}

// ── Minimum windows and length filtering ────────────────────────────────

#[test]
fn rle_filters_short_segments_by_min_windows() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998)], // Only 2 windows
        n_total_windows: 10,
    };
    let positions = make_positions(10, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 3, // Requires 3+
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn rle_filters_short_segments_by_min_length_bp() {
    let track = IdentityTrack {
        windows: (0..5).map(|i| (i, 0.9999)).collect(),
        n_total_windows: 5,
    };
    let positions = make_positions(5, 1000); // 5 * 1000 = 5000 bp total
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 10000, // Requires 10kb+
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

// ── Empty and edge inputs ───────────────────────────────────────────────

#[test]
fn rle_empty_track() {
    let track = IdentityTrack { windows: vec![], n_total_windows: 0 };
    let positions = vec![];
    let params = RleParams::default();
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn rle_single_window_with_min_windows_1() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999)],
        n_total_windows: 1,
    };
    let positions = vec![(0, 9999)];
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 1);
}

#[test]
fn rle_normalizes_haplotype_order() {
    let track = IdentityTrack {
        windows: (0..5).map(|i| (i, 0.9999)).collect(),
        n_total_windows: 5,
    };
    let positions = make_positions(5, 5000);
    let params = RleParams {
        min_identity: 0.999,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };

    // hap_a="Z", hap_b="A" → normalized to hap_a="A", hap_b="Z"
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "Z#1", "A#1");
    assert_eq!(segs[0].hap_a, "A#1");
    assert_eq!(segs[0].hap_b, "Z#1");
}

// ── Segment statistics ──────────────────────────────────────────────────

#[test]
fn rle_mean_identity_correct() {
    // Windows with varying identity
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (1, 0.998), (2, 0.997), (3, 0.996)],
        n_total_windows: 4,
    };
    let positions = make_positions(4, 5000);
    let params = RleParams {
        min_identity: 0.995,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 1,
        drop_tolerance: 0.0,
    };

    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    let expected_mean = (0.999 + 0.998 + 0.997 + 0.996) / 4.0;
    assert!((segs[0].mean_identity - expected_mean).abs() < 1e-10);
    assert!((segs[0].min_identity - 0.996).abs() < 1e-10);
}

// ── merge_segments property tests ───────────────────────────────────────

#[test]
fn merge_segments_empty() {
    let mut segments: Vec<Segment> = vec![];
    merge_segments(&mut segments);
    assert!(segments.is_empty());
}

#[test]
fn merge_segments_single() {
    let mut segments = vec![make_segment("chr1", 0, 100, "A", "B", 0, 10, 0.999, 0.998)];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
}

#[test]
fn merge_segments_non_overlapping_same_pair() {
    let mut segments = vec![
        make_segment("chr1", 0, 100, "A", "B", 0, 5, 0.999, 0.998),
        make_segment("chr1", 200, 300, "A", "B", 20, 25, 0.998, 0.997),
    ];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 2);
}

#[test]
fn merge_segments_different_chrom_not_merged() {
    let mut segments = vec![
        make_segment("chr1", 0, 100, "A", "B", 0, 10, 0.999, 0.998),
        make_segment("chr2", 0, 100, "A", "B", 0, 10, 0.999, 0.998),
    ];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 2);
}

#[test]
fn merge_segments_preserves_min_identity() {
    let mut segments = vec![
        make_segment("chr1", 0, 100, "A", "B", 0, 10, 0.999, 0.998),
        make_segment("chr1", 50, 150, "A", "B", 5, 15, 0.997, 0.990),
    ];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].min_identity - 0.990).abs() < 1e-10);
}

#[test]
fn merge_segments_contained_segment() {
    // Second segment is entirely contained within first
    let mut segments = vec![
        make_segment("chr1", 0, 200, "A", "B", 0, 20, 0.999, 0.998),
        make_segment("chr1", 50, 100, "A", "B", 5, 10, 0.998, 0.997),
    ];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].end, 200); // Should keep the larger end
}

// ── format_segment_bed ──────────────────────────────────────────────────

#[test]
fn format_bed_basic() {
    let seg = make_segment("chr1", 1000, 2000, "HapA", "HapB", 0, 10, 0.999, 0.998);
    let bed = format_segment_bed(&seg, 5.0);
    // 1-based start (1000) → 0-based (999)
    assert!(bed.starts_with("chr1\t999\t2000\tHapA_HapB\t500\t."));
}

#[test]
fn format_bed_lod_capped_at_1000() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 0, 5, 0.999, 0.998);
    let bed = format_segment_bed(&seg, 15.0); // 15 * 100 = 1500, capped to 1000
    assert!(bed.contains("\t1000\t."));
}

#[test]
fn format_bed_negative_lod() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 0, 5, 0.999, 0.998);
    let bed = format_segment_bed(&seg, -2.0);
    assert!(bed.contains("\t0\t.")); // Negative LOD → score 0
}

#[test]
fn format_bed_zero_lod() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 0, 5, 0.999, 0.998);
    let bed = format_segment_bed(&seg, 0.0);
    assert!(bed.contains("\t0\t."));
}

#[test]
fn format_bed_start_at_1_becomes_0() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 0, 5, 0.999, 0.998);
    let bed = format_segment_bed(&seg, 1.0);
    assert!(bed.starts_with("chr1\t0\t100\t")); // 1 → 0
}

// ── segment_length_distribution ─────────────────────────────────────────

#[test]
fn length_distribution_empty() {
    let stats = segment_length_distribution(&[]);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.total_bp, 0);
}

#[test]
fn length_distribution_single_segment() {
    let segments = vec![make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998)];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min_bp, 1000);
    assert_eq!(stats.max_bp, 1000);
    assert_eq!(stats.total_bp, 1000);
    assert!((stats.std_bp - 0.0).abs() < 1e-10);
}

#[test]
fn length_distribution_median_even_count() {
    // Two segments: lengths 1000 and 2000 → median = 1500
    let segments = vec![
        make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998),
        make_segment("chr1", 0, 1999, "A", "B", 0, 19, 0.999, 0.998),
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 2);
    assert!((stats.median_bp - 1500.0).abs() < 1e-10);
}

#[test]
fn length_distribution_median_odd_count() {
    // Three segments: lengths 1000, 2000, 3000 → median = 2000
    let segments = vec![
        make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998),
        make_segment("chr1", 0, 1999, "C", "D", 0, 19, 0.999, 0.998),
        make_segment("chr1", 0, 2999, "E", "F", 0, 29, 0.999, 0.998),
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 3);
    assert!((stats.median_bp - 2000.0).abs() < 1e-10);
    assert_eq!(stats.min_bp, 1000);
    assert_eq!(stats.max_bp, 3000);
}

#[test]
fn length_distribution_min_le_median_le_max() {
    let segments = vec![
        make_segment("chr1", 0, 499, "A", "B", 0, 4, 0.999, 0.998),
        make_segment("chr1", 0, 1999, "C", "D", 0, 19, 0.999, 0.998),
        make_segment("chr1", 0, 4999, "E", "F", 0, 49, 0.999, 0.998),
    ];
    let stats = segment_length_distribution(&segments);
    assert!(stats.min_bp as f64 <= stats.median_bp);
    assert!(stats.median_bp <= stats.max_bp as f64);
}

// ── segment_length_histogram ────────────────────────────────────────────

#[test]
fn histogram_empty() {
    assert!(segment_length_histogram(&[], 1000).is_empty());
}

#[test]
fn histogram_zero_bin_size() {
    let segments = vec![make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998)];
    assert!(segment_length_histogram(&segments, 0).is_empty());
}

#[test]
fn histogram_single_segment_single_bin() {
    let segments = vec![make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998)];
    let hist = segment_length_histogram(&segments, 5000);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0].0, 0); // Bin starting at 0
    assert_eq!(hist[0].1, 1); // Count of 1
}

#[test]
fn histogram_segments_in_different_bins() {
    let segments = vec![
        make_segment("chr1", 0, 999, "A", "B", 0, 9, 0.999, 0.998),    // 1000 bp → bin 0
        make_segment("chr1", 0, 4999, "C", "D", 0, 49, 0.999, 0.998),   // 5000 bp → bin 1
        make_segment("chr1", 0, 9999, "E", "F", 0, 99, 0.999, 0.998),   // 10000 bp → bin 2
    ];
    let hist = segment_length_histogram(&segments, 5000);
    assert_eq!(hist.len(), 3);
    // Each bin has exactly 1 segment
    for (_, count) in &hist {
        assert_eq!(*count, 1);
    }
}

// ── Segment methods ─────────────────────────────────────────────────────

#[test]
fn segment_fraction_called_full() {
    let seg = make_segment("chr1", 0, 100, "A", "B", 0, 10, 0.999, 0.998);
    assert!((seg.fraction_called() - 1.0).abs() < 1e-10);
}

#[test]
fn segment_fraction_called_partial() {
    let mut seg = make_segment("chr1", 0, 100, "A", "B", 0, 10, 0.999, 0.998);
    seg.n_called = 5; // Only 5 of 11 windows had data
    assert!((seg.fraction_called() - 5.0 / 11.0).abs() < 1e-10);
}

#[test]
fn segment_fraction_called_zero_windows() {
    let mut seg = make_segment("chr1", 0, 100, "A", "B", 0, 0, 0.999, 0.998);
    seg.n_windows = 0;
    assert!((seg.fraction_called() - 0.0).abs() < 1e-10);
}

#[test]
fn segment_length_bp_start_gt_end_saturates() {
    let mut seg = make_segment("chr1", 200, 100, "A", "B", 0, 5, 0.999, 0.998);
    seg.start = 200;
    seg.end = 100;
    // saturating_sub: 100 - 200 → 0, + 1 = 1
    assert_eq!(seg.length_bp(), 1);
}
