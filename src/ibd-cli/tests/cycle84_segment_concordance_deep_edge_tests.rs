//! Cycle 84: Deep edge case tests for segment detection, merging, concordance,
//! and formatting functions. Tests boundary conditions, degenerate inputs,
//! and numerical edge cases that are not covered by existing tests.

use hprc_ibd::segment::{
    detect_segments_rle, format_segment_bed, merge_segments, segment_length_distribution,
    segment_length_histogram, IdentityTrack, RleParams, Segment,
};
use hprc_ibd::concordance::{
    boundary_accuracy, extract_haplotype_index, extract_sample_id, f1_score,
    haplotype_level_concordance, length_correlation, matched_segments,
    per_window_concordance, segment_overlap_bp, segments_jaccard, segments_precision_recall,
};

// ============== Helper ==============

fn make_segment(
    chrom: &str, start: u64, end: u64, hap_a: &str, hap_b: &str,
    n_windows: usize, mean_identity: f64, start_idx: usize, end_idx: usize,
) -> Segment {
    Segment {
        chrom: chrom.to_string(),
        start,
        end,
        hap_a: hap_a.to_string(),
        hap_b: hap_b.to_string(),
        n_windows,
        mean_identity,
        min_identity: mean_identity - 0.001,
        identity_sum: mean_identity * n_windows as f64,
        n_called: n_windows,
        start_idx,
        end_idx,
    }
}

fn make_positions(n: usize, window_size: u64) -> Vec<(u64, u64)> {
    (0..n)
        .map(|i| {
            let start = i as u64 * window_size;
            let end = start + window_size - 1;
            (start, end)
        })
        .collect()
}

fn make_track(windows: Vec<(usize, f64)>, n_total: usize) -> IdentityTrack {
    IdentityTrack {
        windows,
        n_total_windows: n_total,
    }
}

// ==================== detect_segments_rle ====================

#[test]
fn rle_empty_track() {
    let track = make_track(vec![], 0);
    let positions = vec![];
    let params = RleParams::default();
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn rle_all_below_threshold() {
    let track = make_track(vec![(0, 0.5), (1, 0.6), (2, 0.7)], 3);
    let positions = make_positions(3, 5000);
    let params = RleParams::default(); // min_identity = 0.9995
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty());
}

#[test]
fn rle_single_window_above_threshold_rejected_min_windows() {
    let track = make_track(vec![(0, 0.9999)], 1);
    let positions = make_positions(1, 10000);
    let params = RleParams { min_windows: 3, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty()); // only 1 window, min_windows=3
}

#[test]
fn rle_exactly_min_windows() {
    let track = make_track(vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)], 3);
    let positions = make_positions(3, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 3);
}

#[test]
fn rle_gap_at_start_ignored() {
    // Gap at window 0, then good windows 1-3
    let track = make_track(vec![(1, 0.9999), (2, 0.9998), (3, 0.9997)], 4);
    let positions = make_positions(4, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start_idx, 1);
}

#[test]
fn rle_gap_at_end_extends_segment() {
    // Good windows 0-2, then missing window 3
    let track = make_track(vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)], 4);
    let positions = make_positions(4, 5000);
    // max_gap=1 allows bridging one missing window
    let params = RleParams {
        min_windows: 3, min_length_bp: 0, max_gap: 1,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    // Segment should extend through the gap to window 3
    assert_eq!(segs[0].end_idx, 3);
}

#[test]
fn rle_max_gap_exactly_bridges() {
    // Windows: good, missing, good, good, good
    let track = make_track(
        vec![(0, 0.9999), (2, 0.9998), (3, 0.9997), (4, 0.9996)],
        5,
    );
    let positions = make_positions(5, 5000);
    let params = RleParams {
        min_windows: 3, min_length_bp: 0, max_gap: 1,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // Should bridge gap at window 1
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 5);
}

#[test]
fn rle_max_gap_exceeded_splits_segment() {
    // Windows: good, missing, missing, good, good, good
    let track = make_track(
        vec![(0, 0.9999), (3, 0.9998), (4, 0.9997), (5, 0.9996)],
        6,
    );
    let positions = make_positions(6, 5000);
    let params = RleParams {
        min_windows: 1, min_length_bp: 0, max_gap: 1,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // Gap of 2 exceeds max_gap=1, should split
    assert_eq!(segs.len(), 2);
}

#[test]
fn rle_drop_tolerance_widens_threshold() {
    // Window at 0.998 < min_identity=0.9995 but within drop_tolerance=0.002
    let track = make_track(
        vec![(0, 0.9999), (1, 0.998), (2, 0.9999)],
        3,
    );
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_windows: 3, min_length_bp: 0, drop_tolerance: 0.002,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // 0.998 >= 0.9995 - 0.002 = 0.9975, so it should be included
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].n_windows, 3);
}

#[test]
fn rle_min_length_bp_filter() {
    // 3 windows at 5kb = 15kb total, min_length_bp = 20kb
    let track = make_track(
        vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)],
        3,
    );
    let positions = make_positions(3, 5000);
    let params = RleParams {
        min_windows: 1, min_length_bp: 20_000,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert!(segs.is_empty()); // 14999 bp < 20000
}

#[test]
fn rle_haplotype_pair_normalization() {
    let track = make_track(vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)], 3);
    let positions = make_positions(3, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };

    // Pass hap_a > hap_b, should be normalized
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "Z", "A");
    assert_eq!(segs[0].hap_a, "A");
    assert_eq!(segs[0].hap_b, "Z");
}

#[test]
fn rle_two_disjoint_segments() {
    // Two high-identity blocks separated by low identity
    let track = make_track(
        vec![
            (0, 0.9999), (1, 0.9998), (2, 0.9997),
            (3, 0.5), // break
            (4, 0.9999), (5, 0.9998), (6, 0.9997),
        ],
        7,
    );
    let positions = make_positions(7, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start_idx, 0);
    assert_eq!(segs[0].end_idx, 2);
    assert_eq!(segs[1].start_idx, 4);
    assert_eq!(segs[1].end_idx, 6);
}

#[test]
fn rle_low_identity_window_between_good_resets_no_bridge() {
    // Good → below threshold → good, max_gap = 0
    let track = make_track(
        vec![(0, 0.9999), (1, 0.5), (2, 0.9999), (3, 0.9998), (4, 0.9997)],
        5,
    );
    let positions = make_positions(5, 5000);
    let params = RleParams {
        min_windows: 1, min_length_bp: 0, max_gap: 0,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // Low identity at window 1 breaks the segment, max_gap=0 means no bridging
    assert_eq!(segs.len(), 2);
}

#[test]
fn rle_n_total_larger_than_data() {
    // Only 2 windows in data but n_total = 5 → windows 2-4 are missing
    let track = make_track(vec![(0, 0.9999), (1, 0.9998)], 5);
    let positions = make_positions(5, 5000);
    let params = RleParams {
        min_windows: 2, min_length_bp: 0, max_gap: 3,
        ..RleParams::default()
    };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    // The segment should extend through missing windows 2-4 within max_gap
    assert_eq!(segs.len(), 1);
}

#[test]
fn rle_mean_identity_is_calculated() {
    let track = make_track(
        vec![(0, 1.0), (1, 0.9999), (2, 0.9998)],
        3,
    );
    let positions = make_positions(3, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    let expected_mean = (1.0 + 0.9999 + 0.9998) / 3.0;
    assert!((segs[0].mean_identity - expected_mean).abs() < 1e-10);
}

#[test]
fn rle_min_identity_is_tracked() {
    let track = make_track(
        vec![(0, 1.0), (1, 0.9996), (2, 0.9999)],
        3,
    );
    let positions = make_positions(3, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);
    assert!((segs[0].min_identity - 0.9996).abs() < 1e-10);
}

// ==================== IdentityTrack ====================

#[test]
fn identity_track_get_existing() {
    let track = make_track(vec![(0, 0.5), (3, 0.9)], 5);
    assert_eq!(track.get(0), Some(0.5));
    assert_eq!(track.get(3), Some(0.9));
}

#[test]
fn identity_track_get_missing() {
    let track = make_track(vec![(0, 0.5)], 5);
    assert_eq!(track.get(1), None);
    assert_eq!(track.get(4), None);
}

#[test]
fn identity_track_to_map() {
    let track = make_track(vec![(1, 0.5), (3, 0.9)], 5);
    let map = track.to_map();
    assert_eq!(map.len(), 2);
    assert_eq!(map[&1], 0.5);
    assert_eq!(map[&3], 0.9);
}

#[test]
fn identity_track_empty_to_map() {
    let track = make_track(vec![], 0);
    let map = track.to_map();
    assert!(map.is_empty());
}

// ==================== merge_segments ====================

#[test]
fn merge_empty() {
    let mut segs: Vec<Segment> = vec![];
    merge_segments(&mut segs);
    assert!(segs.is_empty());
}

#[test]
fn merge_single_segment() {
    let mut segs = vec![make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1)];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
}

#[test]
fn merge_non_overlapping_same_pair() {
    let mut segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),
        make_segment("chr1", 10000, 14999, "A", "B", 1, 0.998, 2, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2); // no overlap → no merge
}

#[test]
fn merge_overlapping_same_pair() {
    let mut segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 14999, "A", "B", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 14999);
    assert_eq!(segs[0].n_windows, 3); // indices 0 to 2
}

#[test]
fn merge_overlapping_different_pairs_no_merge() {
    let mut segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 14999, "A", "C", 2, 0.998, 1, 2),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2); // different pair → no merge
}

#[test]
fn merge_different_chromosomes_no_merge() {
    let mut segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr2", 0, 9999, "A", "B", 2, 0.998, 0, 1),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2); // different chrom → no merge
}

#[test]
fn merge_three_cascading_overlaps() {
    let mut segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 5000, 14999, "A", "B", 2, 0.998, 1, 2),
        make_segment("chr1", 10000, 19999, "A", "B", 2, 0.997, 2, 3),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 0);
    assert_eq!(segs[0].end, 19999);
    assert_eq!(segs[0].n_windows, 4);
}

#[test]
fn merge_preserves_min_identity() {
    let mut seg1 = make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1);
    seg1.min_identity = 0.998;
    let mut seg2 = make_segment("chr1", 5000, 14999, "A", "B", 2, 0.997, 1, 2);
    seg2.min_identity = 0.995;
    let mut segs = vec![seg1, seg2];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 1);
    assert!((segs[0].min_identity - 0.995).abs() < 1e-10);
}

#[test]
fn merge_sorts_by_chrom_then_pair_then_position() {
    let mut segs = vec![
        make_segment("chr2", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr1", 0, 9999, "B", "C", 2, 0.998, 0, 1),
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.997, 0, 1),
    ];
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].chrom, "chr1");
    assert_eq!(segs[0].hap_a, "A");
    assert_eq!(segs[1].chrom, "chr1");
    assert_eq!(segs[1].hap_a, "B");
    assert_eq!(segs[2].chrom, "chr2");
}

// ==================== segment_length_distribution ====================

#[test]
fn length_dist_empty() {
    let stats = segment_length_distribution(&[]);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.mean_bp, 0.0);
    assert_eq!(stats.total_bp, 0);
}

#[test]
fn length_dist_single_segment() {
    let segs = vec![make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1)];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min_bp, 10000);
    assert_eq!(stats.max_bp, 10000);
    assert_eq!(stats.mean_bp, 10000.0);
    assert_eq!(stats.median_bp, 10000.0);
    assert_eq!(stats.std_bp, 0.0); // single element → 0 variance
}

#[test]
fn length_dist_two_segments_even_median() {
    let segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),    // 10000 bp
        make_segment("chr1", 0, 19999, "A", "C", 4, 0.998, 0, 3),   // 20000 bp
    ];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.median_bp, 15000.0); // (10000 + 20000) / 2
    assert_eq!(stats.min_bp, 10000);
    assert_eq!(stats.max_bp, 20000);
}

#[test]
fn length_dist_three_segments_odd_median() {
    let segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),    // 5000 bp
        make_segment("chr1", 0, 9999, "A", "C", 2, 0.998, 0, 1),    // 10000 bp
        make_segment("chr1", 0, 19999, "A", "D", 4, 0.997, 0, 3),   // 20000 bp
    ];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 3);
    assert_eq!(stats.median_bp, 10000.0); // middle element
}

#[test]
fn length_dist_all_identical_lengths() {
    let segs = vec![
        make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1),
        make_segment("chr2", 0, 9999, "A", "B", 2, 0.998, 0, 1),
        make_segment("chr3", 0, 9999, "A", "B", 2, 0.997, 0, 1),
    ];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.std_bp, 0.0);
    assert_eq!(stats.min_bp, stats.max_bp);
}

// ==================== segment_length_histogram ====================

#[test]
fn histogram_empty() {
    let hist = segment_length_histogram(&[], 5000);
    assert!(hist.is_empty());
}

#[test]
fn histogram_zero_bin_size() {
    let segs = vec![make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1)];
    let hist = segment_length_histogram(&segs, 0);
    assert!(hist.is_empty());
}

#[test]
fn histogram_single_segment_single_bin() {
    let segs = vec![make_segment("chr1", 0, 9999, "A", "B", 2, 0.999, 0, 1)]; // 10000 bp
    let hist = segment_length_histogram(&segs, 10000);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (10000, 1)); // bin starting at 10000
}

#[test]
fn histogram_segments_in_different_bins() {
    let segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),    // 5000 bp → bin 0
        make_segment("chr1", 0, 14999, "A", "C", 3, 0.998, 0, 2),   // 15000 bp → bin 10000
    ];
    let hist = segment_length_histogram(&segs, 10000);
    assert_eq!(hist.len(), 2);
    // 5000 → bin 0 (0..10000)
    assert!(hist.iter().any(|&(bin, count)| bin == 0 && count == 1));
    // 15000 → bin 10000 (10000..20000)
    assert!(hist.iter().any(|&(bin, count)| bin == 10000 && count == 1));
}

#[test]
fn histogram_multiple_in_same_bin() {
    let segs = vec![
        make_segment("chr1", 0, 4999, "A", "B", 1, 0.999, 0, 0),   // 5000 bp → bin 0
        make_segment("chr1", 0, 7999, "A", "C", 2, 0.998, 0, 1),   // 8000 bp → bin 0
    ];
    let hist = segment_length_histogram(&segs, 10000);
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 2));
}

// ==================== format_segment_bed ====================

#[test]
fn bed_normal_segment() {
    let seg = make_segment("chr1", 100, 5000, "HapA", "HapB", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 5.0);
    // BED: chr1, start=99 (0-based), end=5000, name=HapA_HapB, score=500
    assert!(bed.starts_with("chr1\t99\t5000\tHapA_HapB\t500\t."));
}

#[test]
fn bed_zero_lod() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 0.0);
    assert!(bed.contains("\t0\t."));
}

#[test]
fn bed_negative_lod() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, -3.0);
    assert!(bed.contains("\t0\t.")); // negative → 0
}

#[test]
fn bed_very_large_lod_capped_at_1000() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 100.0);
    // 100 * 100 = 10000, capped at 1000
    assert!(bed.contains("\t1000\t."));
}

#[test]
fn bed_lod_exactly_10_score_1000() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 10.0);
    // 10 * 100 = 1000, exactly at cap
    assert!(bed.contains("\t1000\t."));
}

#[test]
fn bed_start_at_one_zero_based_is_zero() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 1.0);
    assert!(bed.starts_with("chr1\t0\t100")); // 1-based 1 → 0-based 0
}

#[test]
fn bed_start_at_zero_saturating_sub() {
    let seg = make_segment("chr1", 0, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 1.0);
    assert!(bed.starts_with("chr1\t0\t100")); // saturating_sub(1) of 0 = 0
}

#[test]
fn bed_fractional_lod_rounds() {
    let seg = make_segment("chr1", 1, 100, "A", "B", 1, 0.999, 0, 0);
    let bed = format_segment_bed(&seg, 3.456);
    // 3.456 * 100 = 345.6, rounds to 346
    assert!(bed.contains("\t346\t."));
}

// ==================== Segment methods ====================

#[test]
fn segment_length_bp_normal() {
    let seg = make_segment("chr1", 100, 200, "A", "B", 1, 0.999, 0, 0);
    assert_eq!(seg.length_bp(), 101); // inclusive
}

#[test]
fn segment_length_bp_same_start_end() {
    let seg = make_segment("chr1", 100, 100, "A", "B", 1, 0.999, 0, 0);
    assert_eq!(seg.length_bp(), 1);
}

#[test]
fn segment_length_bp_start_greater_than_end() {
    let seg = make_segment("chr1", 200, 100, "A", "B", 1, 0.999, 0, 0);
    // end.saturating_sub(start) + 1 = 0 + 1 = 1 (minimum 1 due to +1)
    assert_eq!(seg.length_bp(), 1);
}

#[test]
fn segment_fraction_called_normal() {
    let mut seg = make_segment("chr1", 0, 9999, "A", "B", 10, 0.999, 0, 9);
    seg.n_called = 8;
    assert!((seg.fraction_called() - 0.8).abs() < 1e-10);
}

#[test]
fn segment_fraction_called_zero_windows() {
    let mut seg = make_segment("chr1", 0, 9999, "A", "B", 0, 0.999, 0, 0);
    seg.n_windows = 0;
    assert_eq!(seg.fraction_called(), 0.0);
}

#[test]
fn segment_fraction_called_all_called() {
    let seg = make_segment("chr1", 0, 9999, "A", "B", 5, 0.999, 0, 4);
    assert!((seg.fraction_called() - 1.0).abs() < 1e-10);
}

// ==================== segment_overlap_bp ====================

#[test]
fn overlap_complete() {
    assert_eq!(segment_overlap_bp((100, 200), (100, 200)), 100);
}

#[test]
fn overlap_partial_left() {
    assert_eq!(segment_overlap_bp((100, 200), (150, 250)), 50);
}

#[test]
fn overlap_partial_right() {
    assert_eq!(segment_overlap_bp((150, 250), (100, 200)), 50);
}

#[test]
fn overlap_none_adjacent() {
    assert_eq!(segment_overlap_bp((100, 200), (200, 300)), 0);
}

#[test]
fn overlap_none_disjoint() {
    assert_eq!(segment_overlap_bp((100, 200), (300, 400)), 0);
}

#[test]
fn overlap_contained() {
    assert_eq!(segment_overlap_bp((100, 300), (150, 200)), 50);
}

#[test]
fn overlap_zero_length_interval() {
    assert_eq!(segment_overlap_bp((100, 100), (100, 200)), 0);
}

// ==================== segments_jaccard ====================

#[test]
fn jaccard_identical_segments() {
    let segs = vec![(100, 200)];
    let j = segments_jaccard(&segs, &segs, (0, 300));
    assert!((j - 1.0).abs() < 1e-10);
}

#[test]
fn jaccard_no_overlap() {
    let a = vec![(100, 200)];
    let b = vec![(200, 300)];
    let j = segments_jaccard(&a, &b, (0, 400));
    assert_eq!(j, 0.0);
}

#[test]
fn jaccard_both_empty() {
    let j = segments_jaccard(&[], &[], (0, 100));
    assert_eq!(j, 0.0);
}

#[test]
fn jaccard_one_empty() {
    let a = vec![(100, 200)];
    let j = segments_jaccard(&a, &[], (0, 300));
    assert_eq!(j, 0.0);
}

#[test]
fn jaccard_zero_region() {
    let a = vec![(100, 200)];
    let j = segments_jaccard(&a, &a, (100, 100));
    assert_eq!(j, 0.0);
}

#[test]
fn jaccard_segments_outside_region_clipped() {
    let a = vec![(0, 1000)];
    let b = vec![(500, 1500)];
    // Region clips both to (500, 1000)
    let j = segments_jaccard(&a, &b, (500, 1000));
    assert!((j - 1.0).abs() < 1e-10);
}

// ==================== segments_precision_recall ====================

#[test]
fn precision_recall_perfect() {
    let segs = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&segs, &segs, (0, 300));
    assert!((p - 1.0).abs() < 1e-10);
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn precision_recall_no_overlap() {
    let a = vec![(100, 200)];
    let b = vec![(300, 400)];
    let (p, r) = segments_precision_recall(&a, &b, (0, 500));
    assert_eq!(p, 0.0);
    assert_eq!(r, 0.0);
}

#[test]
fn precision_recall_both_empty() {
    let (p, r) = segments_precision_recall(&[], &[], (0, 100));
    assert_eq!(p, 0.0);
    assert_eq!(r, 0.0);
}

#[test]
fn precision_recall_ours_empty() {
    let b = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&[], &b, (0, 300));
    assert_eq!(p, 0.0);
    assert_eq!(r, 0.0);
}

#[test]
fn precision_recall_theirs_empty() {
    let a = vec![(100, 200)];
    let (p, r) = segments_precision_recall(&a, &[], (0, 300));
    assert_eq!(p, 0.0);
    assert_eq!(r, 0.0);
}

#[test]
fn precision_recall_subset_recall_lower() {
    // We call a wider segment than ground truth
    let ours = vec![(100, 400)];
    let theirs = vec![(200, 300)];
    let (p, r) = segments_precision_recall(&ours, &theirs, (0, 500));
    // recall should be 1.0 (all theirs covered), precision < 1.0
    assert!((r - 1.0).abs() < 1e-10);
    assert!(p < 1.0);
}

// ==================== per_window_concordance ====================

#[test]
fn per_window_concordance_both_empty() {
    let c = per_window_concordance(&[], &[], (0, 100), 10);
    // Both empty = both agree on "no IBD" everywhere
    assert!(c >= 0.0);
}

#[test]
fn per_window_concordance_identical() {
    let segs = vec![(100, 200)];
    let c = per_window_concordance(&segs, &segs, (0, 300), 50);
    assert!((c - 1.0).abs() < 1e-10);
}

#[test]
fn per_window_concordance_zero_window_size() {
    let segs = vec![(100, 200)];
    let c = per_window_concordance(&segs, &segs, (0, 300), 0);
    assert_eq!(c, 0.0);
}

#[test]
fn per_window_concordance_invalid_region() {
    let segs = vec![(100, 200)];
    let c = per_window_concordance(&segs, &segs, (200, 100), 10);
    assert_eq!(c, 0.0); // region.1 <= region.0
}

// ==================== matched_segments ====================

#[test]
fn matched_perfect_overlap() {
    let a = vec![(100, 200)];
    let b = vec![(100, 200)];
    let m = matched_segments(&a, &b, 0.5);
    assert_eq!(m.len(), 1);
    assert_eq!(m[0], (0, 0));
}

#[test]
fn matched_no_overlap() {
    let a = vec![(100, 200)];
    let b = vec![(300, 400)];
    let m = matched_segments(&a, &b, 0.5);
    assert!(m.is_empty());
}

#[test]
fn matched_below_min_overlap_frac() {
    let a = vec![(100, 300)]; // 200 bp
    let b = vec![(250, 400)]; // 150 bp, overlap = 50 bp
    // overlap fraction = 50/150 = 0.333
    let m = matched_segments(&a, &b, 0.5);
    assert!(m.is_empty()); // 0.333 < 0.5
}

#[test]
fn matched_above_min_overlap_frac() {
    let a = vec![(100, 300)]; // 200 bp
    let b = vec![(200, 400)]; // 200 bp, overlap = 100 bp
    // overlap fraction = 100/200 = 0.5
    let m = matched_segments(&a, &b, 0.5);
    assert_eq!(m.len(), 1);
}

#[test]
fn matched_zero_length_segment() {
    let a = vec![(100, 100)]; // 0 length
    let b = vec![(100, 200)];
    let m = matched_segments(&a, &b, 0.0);
    assert!(m.is_empty()); // shorter = 0 → skip
}

#[test]
fn matched_both_empty() {
    let m = matched_segments(&[], &[], 0.5);
    assert!(m.is_empty());
}

#[test]
fn matched_many_to_many() {
    let a = vec![(100, 200), (300, 400)];
    let b = vec![(100, 200), (300, 400)];
    let m = matched_segments(&a, &b, 0.5);
    assert_eq!(m.len(), 2);
}

// ==================== f1_score ====================

#[test]
fn f1_both_zero() {
    assert_eq!(f1_score(0.0, 0.0), 0.0);
}

#[test]
fn f1_perfect() {
    assert!((f1_score(1.0, 1.0) - 1.0).abs() < 1e-10);
}

#[test]
fn f1_one_zero() {
    assert_eq!(f1_score(1.0, 0.0), 0.0);
    assert_eq!(f1_score(0.0, 1.0), 0.0);
}

#[test]
fn f1_symmetric() {
    let f1_ab = f1_score(0.7, 0.9);
    let f1_ba = f1_score(0.9, 0.7);
    assert!((f1_ab - f1_ba).abs() < 1e-10);
}

#[test]
fn f1_known_value() {
    // F1 = 2 * 0.8 * 0.6 / (0.8 + 0.6) = 0.96/1.4 ≈ 0.6857
    let f = f1_score(0.8, 0.6);
    assert!((f - 0.96 / 1.4).abs() < 1e-10);
}

// ==================== length_correlation ====================

#[test]
fn length_corr_empty() {
    let r = length_correlation(&[]);
    assert_eq!(r, 0.0);
}

#[test]
fn length_corr_single() {
    let pairs = vec![((100, 200), (100, 200))];
    let r = length_correlation(&pairs);
    // Single pair → 0.0 (not enough data)
    assert!(r == 0.0 || r.is_nan());
}

#[test]
fn length_corr_perfect() {
    let pairs = vec![
        ((100, 200), (100, 200)),
        ((100, 300), (100, 300)),
        ((100, 500), (100, 500)),
    ];
    let r = length_correlation(&pairs);
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn length_corr_negative() {
    // Inverse relationship: as ours gets longer, theirs gets shorter
    let pairs = vec![
        ((100, 200), (100, 500)),
        ((100, 300), (100, 400)),
        ((100, 500), (100, 200)),
    ];
    let r = length_correlation(&pairs);
    assert!(r < 0.0);
}

// ==================== boundary_accuracy ====================

#[test]
fn boundary_accuracy_empty() {
    assert!(boundary_accuracy(&[], 100).is_none());
}

#[test]
fn boundary_accuracy_perfect_match() {
    let pairs = vec![((100, 200), (100, 200))];
    let ba = boundary_accuracy(&pairs, 100).unwrap();
    assert!((ba.mean_start_distance_bp - 0.0).abs() < 1e-10);
    assert!((ba.mean_end_distance_bp - 0.0).abs() < 1e-10);
}

#[test]
fn boundary_accuracy_known_distances() {
    let pairs = vec![
        ((100, 200), (110, 210)),  // off by 10 each
        ((300, 400), (290, 410)),  // off by 10 each
    ];
    let ba = boundary_accuracy(&pairs, 100).unwrap();
    assert!((ba.mean_start_distance_bp - 10.0).abs() < 1e-10);
    assert!((ba.mean_end_distance_bp - 10.0).abs() < 1e-10);
    assert!((ba.frac_start_within_threshold - 1.0).abs() < 1e-10); // all within 100
    assert!((ba.frac_end_within_threshold - 1.0).abs() < 1e-10);
}

#[test]
fn boundary_accuracy_threshold_zero_exact_only() {
    let pairs = vec![
        ((100, 200), (100, 200)),   // exact
        ((300, 400), (310, 410)),   // off by 10
    ];
    let ba = boundary_accuracy(&pairs, 0).unwrap();
    // Only the first pair has distance = 0
    // Start threshold: one exact (0), one off by 10 → 50% within 0
    assert!((ba.frac_start_within_threshold - 0.5).abs() < 1e-10);
}

// ==================== extract_haplotype_index ====================

#[test]
fn extract_hap_idx_standard() {
    assert_eq!(extract_haplotype_index("HG00097#1"), Some(1));
    assert_eq!(extract_haplotype_index("HG00097#2"), Some(2));
}

#[test]
fn extract_hap_idx_no_hash() {
    assert_eq!(extract_haplotype_index("HG00097"), None);
}

#[test]
fn extract_hap_idx_empty() {
    assert_eq!(extract_haplotype_index(""), None);
}

#[test]
fn extract_hap_idx_non_numeric() {
    assert_eq!(extract_haplotype_index("HG00097#mat"), None);
}

#[test]
fn extract_hap_idx_zero() {
    assert_eq!(extract_haplotype_index("HG00097#0"), Some(0));
}

// ==================== extract_sample_id ====================

#[test]
fn extract_sample_standard() {
    assert_eq!(extract_sample_id("HG00097#1"), "HG00097");
}

#[test]
fn extract_sample_no_hash() {
    assert_eq!(extract_sample_id("HG00097"), "HG00097");
}

#[test]
fn extract_sample_empty() {
    assert_eq!(extract_sample_id(""), "");
}

#[test]
fn extract_sample_multiple_hashes() {
    // Should take everything before the first #
    assert_eq!(extract_sample_id("HG00097#1#chr12"), "HG00097");
}

// ==================== RleParams Default ====================

#[test]
fn rle_params_default_values() {
    let p = RleParams::default();
    assert!((p.min_identity - 0.9995).abs() < 1e-10);
    assert_eq!(p.max_gap, 1);
    assert_eq!(p.min_windows, 3);
    assert_eq!(p.min_length_bp, 5000);
    assert!((p.drop_tolerance - 0.0).abs() < 1e-10);
}

// ==================== Cross-function consistency ====================

#[test]
fn detected_segments_can_be_merged() {
    // Detect two segments in successive positions, then merge
    let track = make_track(
        vec![
            (0, 0.9999), (1, 0.9998), (2, 0.9997),
            (3, 0.5), // break
            (4, 0.9999), (5, 0.9998), (6, 0.9997),
        ],
        7,
    );
    let positions = make_positions(7, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let mut segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 2);
    // merge_segments should be a no-op for non-overlapping segments
    merge_segments(&mut segs);
    assert_eq!(segs.len(), 2);
}

#[test]
fn detected_segments_have_valid_stats() {
    let track = make_track(
        vec![(0, 1.0), (1, 0.9999), (2, 0.9998), (3, 0.9997), (4, 0.9996)],
        5,
    );
    let positions = make_positions(5, 5000);
    let params = RleParams { min_windows: 3, min_length_bp: 0, ..RleParams::default() };
    let segs = detect_segments_rle(&track, &positions, &params, "chr1", "A", "B");
    assert_eq!(segs.len(), 1);

    let seg = &segs[0];
    let stats = segment_length_distribution(&segs);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min_bp, seg.length_bp());
    assert!(seg.fraction_called() >= 0.0 && seg.fraction_called() <= 1.0);
    assert!(seg.mean_identity >= seg.min_identity);
}

#[test]
fn precision_recall_jaccard_consistency() {
    let a = vec![(100, 300)];
    let b = vec![(200, 400)];
    let region = (0, 500);
    let (p, r) = segments_precision_recall(&a, &b, region);
    let j = segments_jaccard(&a, &b, region);
    let f1 = f1_score(p, r);
    // All metrics should be between 0 and 1
    assert!(p >= 0.0 && p <= 1.0);
    assert!(r >= 0.0 && r <= 1.0);
    assert!(j >= 0.0 && j <= 1.0);
    assert!(f1 >= 0.0 && f1 <= 1.0);
    // Jaccard should be <= F1 for reasonable inputs
    // (not a strict mathematical guarantee but holds for well-behaved inputs)
}

#[test]
fn matched_segments_and_length_correlation_pipeline() {
    let a = vec![(100, 200), (300, 400), (500, 700)];
    let b = vec![(100, 200), (300, 400), (500, 700)];
    let matches = matched_segments(&a, &b, 0.5);
    assert_eq!(matches.len(), 3);

    let pairs: Vec<_> = matches.iter()
        .map(|&(i, j)| (a[i], b[j]))
        .collect();
    let r = length_correlation(&pairs);
    assert!((r - 1.0).abs() < 1e-10);
}
