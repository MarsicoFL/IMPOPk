//! Segment merge and distribution invariant tests.
//!
//! Cycle 48: Properties that must hold after merging and distribution analysis.

use impopk_ibd::segment::*;

// ============================================================================
// Helpers
// ============================================================================

fn make_segment(
    chrom: &str,
    start: u64,
    end: u64,
    hap_a: &str,
    hap_b: &str,
    n_windows: usize,
    mean_id: f64,
    start_idx: usize,
    end_idx: usize,
) -> Segment {
    Segment {
        chrom: chrom.to_string(),
        start,
        end,
        hap_a: hap_a.to_string(),
        hap_b: hap_b.to_string(),
        n_windows,
        mean_identity: mean_id,
        min_identity: mean_id - 0.001,
        identity_sum: mean_id * n_windows as f64,
        n_called: n_windows,
        start_idx,
        end_idx,
    }
}

// ============================================================================
// Merge segment properties
// ============================================================================

#[test]
fn merge_empty_segments() {
    let mut segs: Vec<Segment> = vec![];
    merge_segments(&mut segs);
    assert!(segs.is_empty());
}

#[test]
fn merge_single_segment() {
    let mut segs = vec![make_segment("chr1", 0, 5000, "A", "B", 1, 0.999, 0, 0)];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
}

#[test]
fn merge_non_overlapping_segments_stay_separate() {
    let mut segs = vec![
        make_segment("chr1", 0, 5000, "A", "B", 1, 0.999, 0, 0),
        make_segment("chr1", 10000, 15000, "A", "B", 1, 0.998, 2, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn merge_overlapping_segments() {
    let mut segs = vec![
        make_segment("chr1", 0, 10000, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 15000, "A", "B", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 15000);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 2);
}

#[test]
fn merge_different_chroms_stay_separate() {
    let mut segs = vec![
        make_segment("chr1", 0, 5000, "A", "B", 1, 0.999, 0, 0),
        make_segment("chr2", 0, 5000, "A", "B", 1, 0.998, 0, 0),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn merge_different_haplotype_pairs_stay_separate() {
    let mut segs = vec![
        make_segment("chr1", 0, 10000, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 15000, "C", "D", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn merge_preserves_min_identity() {
    let mut seg1 = make_segment("chr1", 0, 10000, "A", "B", 2, 0.999, 0, 1);
    seg1.min_identity = 0.990;
    let mut seg2 = make_segment("chr1", 5000, 15000, "A", "B", 2, 0.998, 1, 2);
    seg2.min_identity = 0.985;

    let mut segs = vec![seg1, seg2];
    merge_segments(&mut segs);

    assert_eq!(segs.len(), 1);
    assert!(
        (segs[0].min_identity - 0.985).abs() < 1e-10,
        "Min identity should be the minimum of both: {}",
        segs[0].min_identity
    );
}

#[test]
fn merge_sorts_unsorted_input() {
    let mut segs = vec![
        make_segment("chr1", 10000, 15000, "A", "B", 1, 0.998, 2, 2),
        make_segment("chr1", 0, 5000, "A", "B", 1, 0.999, 0, 0),
    ];
    merge_segments(&mut segs);
    // Should be sorted by position after merge
    assert!(segs[0].start <= segs.last().unwrap().start);
}

#[test]
fn merge_adjacent_segments_touching() {
    // Merge condition: seg.start_idx <= last.end_idx
    // For touching: seg2.start_idx == seg1.end_idx (both use same window index at boundary)
    let mut segs = vec![
        make_segment("chr1", 0, 5000, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 9999, "A", "B", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    // seg2.start_idx(1) <= seg1.end_idx(1) → merge
    assert_eq!(segs.len(), 1);
}

#[test]
fn merge_contained_segment() {
    // Second segment is entirely contained within first
    let mut segs = vec![
        make_segment("chr1", 0, 20000, "A", "B", 4, 0.999, 0, 3),
        make_segment("chr1", 5000, 10000, "A", "B", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 20000);
    assert_eq!(segs[0].end_idx, 3);
}

#[test]
fn merge_multiple_overlapping_segments() {
    let mut segs = vec![
        make_segment("chr1", 0, 10000, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 15000, "A", "B", 2, 0.998, 1, 2),
        make_segment("chr1", 10000, 20000, "A", "B", 2, 0.997, 2, 3),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 20000);
}

// ============================================================================
// Segment length properties
// ============================================================================

#[test]
fn segment_length_bp_correct() {
    // length_bp() = end - start + 1 (inclusive)
    let seg = make_segment("chr1", 1000, 4999, "A", "B", 1, 0.999, 0, 0);
    assert_eq!(seg.length_bp(), 4000);
}

#[test]
fn segment_length_bp_same_start_end() {
    // Single-base segment: length_bp() = 1000 - 1000 + 1 = 1
    let seg = make_segment("chr1", 1000, 1000, "A", "B", 1, 0.999, 0, 0);
    assert_eq!(seg.length_bp(), 1);
}

#[test]
fn segment_fraction_called_all() {
    let seg = make_segment("chr1", 0, 5000, "A", "B", 10, 0.999, 0, 9);
    assert!((seg.fraction_called() - 1.0).abs() < 1e-10);
}

#[test]
fn segment_fraction_called_partial() {
    let mut seg = make_segment("chr1", 0, 5000, "A", "B", 10, 0.999, 0, 9);
    seg.n_called = 5;
    assert!((seg.fraction_called() - 0.5).abs() < 1e-10);
}

#[test]
fn segment_fraction_called_zero_windows() {
    let mut seg = make_segment("chr1", 0, 5000, "A", "B", 0, 0.0, 0, 0);
    seg.n_windows = 0;
    assert_eq!(seg.fraction_called(), 0.0);
}

// ============================================================================
// Segment length distribution
// ============================================================================

#[test]
fn segment_length_distribution_empty() {
    let stats = segment_length_distribution(&[]);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean_bp, 0.0);
    assert_eq!(stats.total_bp, 0);
}

#[test]
fn segment_length_distribution_single() {
    // length_bp() = end - start + 1 = 5999 - 1000 + 1 = 5000
    let segs = vec![make_segment("chr1", 1000, 5999, "A", "B", 1, 0.999, 0, 0)];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min_bp, 5000);
    assert_eq!(stats.max_bp, 5000);
    assert_eq!(stats.total_bp, 5000);
}

#[test]
fn segment_length_distribution_multiple() {
    // length_bp() = end - start + 1, so use end = start + desired_length - 1
    let segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),        // 5000 bp
        make_segment("chr1", 10000, 19999, "A", "B", 2, 0.998, 2, 3),    // 10000 bp
        make_segment("chr1", 30000, 49999, "A", "B", 4, 0.997, 6, 9),    // 20000 bp
    ];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 3);
    assert_eq!(stats.min_bp, 5000);
    assert_eq!(stats.max_bp, 20000);
    assert_eq!(stats.total_bp, 35000);
    assert!(stats.std_bp >= 0.0, "Std should be non-negative");
}

#[test]
fn segment_length_distribution_min_leq_median_leq_max() {
    let segs: Vec<Segment> = (1..=10)
        .map(|i| {
            make_segment(
                "chr1",
                0,
                i * 1000,
                "A",
                "B",
                1,
                0.999,
                0,
                0,
            )
        })
        .collect();
    let stats = segment_length_distribution(&segs);
    assert!(stats.min_bp as f64 <= stats.median_bp);
    assert!(stats.median_bp <= stats.max_bp as f64);
    assert!(stats.mean_bp >= stats.min_bp as f64);
    assert!(stats.mean_bp <= stats.max_bp as f64);
}

// ============================================================================
// Segment histogram
// ============================================================================

#[test]
fn segment_histogram_empty() {
    let hist = segment_length_histogram(&[], 5000);
    assert!(hist.is_empty());
}

#[test]
fn segment_histogram_single_bin() {
    let segs = vec![
        make_segment("chr1", 0, 3000, "A", "B", 1, 0.999, 0, 0),
        make_segment("chr1", 0, 4000, "C", "D", 1, 0.998, 0, 0),
    ];
    let hist = segment_length_histogram(&segs, 10000);
    // Both segments < 10000, so they should be in the same bin
    assert!(!hist.is_empty());
    let total: usize = hist.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 2);
}

#[test]
fn segment_histogram_bin_boundaries() {
    let segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),  // 4999 bp
        make_segment("chr1", 0, 9999, "C", "D", 1, 0.998, 0, 0),  // 9999 bp
        make_segment("chr1", 0, 14999, "E", "F", 1, 0.997, 0, 0), // 14999 bp
    ];
    let hist = segment_length_histogram(&segs, 5000);
    let total: usize = hist.iter().map(|(_, c)| c).sum();
    assert_eq!(total, 3, "All segments should be counted");
}

// ============================================================================
// Format BED
// ============================================================================

#[test]
fn format_segment_bed_contains_all_fields() {
    let seg = make_segment("chr1", 1000, 5000, "HG001", "HG002", 10, 0.999, 0, 9);
    let bed = format_segment_bed(&seg, 5.0);
    // BED format: chrom \t start_0based \t end \t name \t score \t strand
    // start_0based = 1000 - 1 = 999, score = (5.0 * 100).round().min(1000) = 500
    assert!(bed.contains("chr1"));
    assert!(bed.contains("999"));  // 0-based start
    assert!(bed.contains("5000"));
    assert!(bed.contains("HG001"));
    assert!(bed.contains("HG002"));
    assert!(bed.contains("500")); // LOD score as integer
}

#[test]
fn format_segment_bed_tab_separated() {
    let seg = make_segment("chr1", 0, 5000, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 5.0);
    let fields: Vec<&str> = bed.split('\t').collect();
    assert!(fields.len() >= 4, "BED should have at least 4 tab-separated fields");
}

// ============================================================================
// IdentityTrack
// ============================================================================

#[test]
fn identity_track_get_valid_index() {
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (2, 0.998), (5, 0.997)],
        n_total_windows: 10,
    };
    assert!((track.get(0).unwrap() - 0.999).abs() < 1e-10);
    assert!((track.get(2).unwrap() - 0.998).abs() < 1e-10);
    assert!((track.get(5).unwrap() - 0.997).abs() < 1e-10);
}

#[test]
fn identity_track_get_missing_index() {
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (2, 0.998)],
        n_total_windows: 10,
    };
    assert!(track.get(1).is_none());
    assert!(track.get(3).is_none());
}

#[test]
fn identity_track_to_map() {
    let track = IdentityTrack {
        windows: vec![(0, 0.999), (5, 0.998)],
        n_total_windows: 10,
    };
    let map = track.to_map();
    assert_eq!(map.len(), 2);
    assert!((map[&0] - 0.999).abs() < 1e-10);
    assert!((map[&5] - 0.998).abs() < 1e-10);
}

// ============================================================================
// RLE detection properties
// ============================================================================

#[test]
fn detect_segments_rle_empty_track() {
    let track = IdentityTrack {
        windows: vec![],
        n_total_windows: 0,
    };
    let params = RleParams::default();
    let positions = vec![];
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn detect_segments_rle_all_below_threshold() {
    let track = IdentityTrack {
        windows: vec![(0, 0.5), (1, 0.6), (2, 0.55)],
        n_total_windows: 3,
    };
    let params = RleParams::default(); // default threshold is high
    let positions = vec![(0, 4999), (5000, 9999), (10000, 14999)];
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // With default params, low identity values should not form segments
    assert!(
        segs.is_empty(),
        "Low-identity windows should not form segments"
    );
}

#[test]
fn detect_segments_rle_single_high_window() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999)],
        n_total_windows: 1,
    };
    let params = RleParams {
        min_identity: 0.999,
        min_windows: 1,
        max_gap: 0,
        min_length_bp: 0,
        drop_tolerance: 0.0,
    };
    let positions = vec![(0, 4999)];
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
}

#[test]
fn detect_segments_rle_normalized_haplotypes() {
    let track = IdentityTrack {
        windows: vec![(0, 0.9999)],
        n_total_windows: 1,
    };
    let params = RleParams {
        min_identity: 0.999,
        min_windows: 1,
        max_gap: 0,
        min_length_bp: 0,
        drop_tolerance: 0.0,
    };
    let positions = vec![(0, 4999)];

    // B < A, so they should be normalized to (A, B)
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "Z_hap", "A_hap");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].hap_a, "A_hap");
    assert_eq!(segs[0].hap_b, "Z_hap");
}

#[test]
fn detect_segments_rle_gap_bridging() {
    // Gap bridging works on MISSING data (no entry in track), not low-identity windows
    let track = IdentityTrack {
        // Window 1 is missing (not in track) — this is the gap
        windows: vec![(0, 0.9999), (2, 0.9998)],
        n_total_windows: 3,
    };
    let positions = vec![(0, 4999), (5000, 9999), (10000, 14999)];

    // With max_gap=0, missing window triggers split
    let params_no_gap = RleParams {
        min_identity: 0.999,
        min_windows: 1,
        max_gap: 0,
        min_length_bp: 0,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params_no_gap, "chr1", "A", "B");
    assert_eq!(segs.len(), 2, "Without gap bridging, should have 2 segments");

    // With max_gap=1, missing window is bridged
    let params_gap = RleParams {
        min_identity: 0.999,
        min_windows: 1,
        max_gap: 1,
        min_length_bp: 0,
        drop_tolerance: 0.0,
    };
    let segs = detect_segments_rle(&track, &positions, &params_gap, "chr1", "A", "B");
    assert_eq!(segs.len(), 1, "With gap bridging, should have 1 segment");
}
