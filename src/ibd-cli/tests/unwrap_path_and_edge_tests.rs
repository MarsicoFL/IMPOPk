//! Tests targeting unwrap()-bearing code paths and edge cases in segment.rs,
//! concordance.rs, and stats.rs that haven't been exercised yet.

use impopk_ibd::segment::{
    detect_segments_rle, format_segment_bed, merge_segments, segment_length_distribution,
    segment_length_histogram, IdentityTrack, RleParams, Segment,
};

use impopk_ibd::concordance::{
    boundary_accuracy, extract_haplotype_index, extract_sample_id, f1_score, length_correlation,
    matched_segments, per_window_concordance, segment_overlap_bp, segments_jaccard,
    segments_precision_recall, MatchedInterval,
};

use impopk_ibd::stats::{
    bic_model_selection, em_two_component, em_two_component_map, gaussian_to_logit_space,
    inv_logit, kmeans_1d, logit, logit_transform_observations, trimmed_mean, GaussianParams,
    OnlineStats, LOGIT_CAP,
};

/// Helper to create a test segment
fn make_seg(
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

// =============================================================================
// format_segment_bed edge cases
// =============================================================================

#[test]
fn test_format_segment_bed_zero_lod() {
    let seg = make_seg("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, 0.0);
    // LOD = 0.0 → score = 0
    assert!(bed.contains("\t0\t."));
}

#[test]
fn test_format_segment_bed_very_high_lod_capped_at_1000() {
    let seg = make_seg("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, 50.0); // 50.0 * 100 = 5000 → capped at 1000
    assert!(bed.contains("\t1000\t."));
}

#[test]
fn test_format_segment_bed_fractional_lod() {
    let seg = make_seg("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, 3.456); // 3.456 * 100 = 345.6 → rounds to 346
    assert!(bed.contains("\t346\t."));
}

#[test]
fn test_format_segment_bed_start_is_one_based_to_zero_based() {
    let seg = make_seg("chr5", 1, 100, "X", "Y", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, 1.0);
    // start 1 → bed_start 0 (1-based to 0-based), end stays as-is
    assert!(bed.starts_with("chr5\t0\t100\t"));
}

#[test]
fn test_format_segment_bed_start_zero_saturates() {
    let seg = make_seg("chr1", 0, 100, "A", "B", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, 1.0);
    // start 0 → saturating_sub(1) = 0 (not underflow)
    assert!(bed.starts_with("chr1\t0\t100\t"));
}

#[test]
fn test_format_segment_bed_nan_lod() {
    let seg = make_seg("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
    // NaN LOD: NaN > 0.0 is false, so score = 0
    let bed = format_segment_bed(&seg, f64::NAN);
    assert!(bed.contains("\t0\t."));
}

#[test]
fn test_format_segment_bed_negative_infinity_lod() {
    let seg = make_seg("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
    let bed = format_segment_bed(&seg, f64::NEG_INFINITY);
    assert!(bed.contains("\t0\t."));
}

// =============================================================================
// segment_length_distribution edge cases
// =============================================================================

#[test]
fn test_segment_length_distribution_two_segments_even_median() {
    // Even number: median is average of two middle values
    let segments = vec![
        make_seg("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998), // 10000 bp
        make_seg("chr1", 0, 19999, "C", "D", 0, 19, 0.999, 0.998), // 20000 bp
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 2);
    assert!((stats.median_bp - 15000.0).abs() < 1e-6);
    assert_eq!(stats.min_bp, 10000);
    assert_eq!(stats.max_bp, 20000);
    assert_eq!(stats.total_bp, 30000);
}

#[test]
fn test_segment_length_distribution_four_segments_even_median() {
    // 4 segments: median = avg of 2nd and 3rd
    let segments = vec![
        make_seg("chr1", 0, 999, "A", "B", 0, 0, 0.999, 0.999),    // 1000
        make_seg("chr1", 0, 2999, "C", "D", 0, 2, 0.999, 0.999),   // 3000
        make_seg("chr1", 0, 4999, "E", "F", 0, 4, 0.999, 0.999),   // 5000
        make_seg("chr1", 0, 9999, "G", "H", 0, 9, 0.999, 0.999),   // 10000
    ];
    let stats = segment_length_distribution(&segments);
    assert_eq!(stats.count, 4);
    // median = (3000 + 5000) / 2 = 4000
    assert!((stats.median_bp - 4000.0).abs() < 1e-6);
}

// =============================================================================
// segment_length_histogram edge cases
// =============================================================================

#[test]
fn test_segment_length_histogram_single_segment() {
    let segments = vec![make_seg("chr1", 0, 4999, "A", "B", 0, 4, 0.999, 0.998)]; // 5000 bp
    let hist = segment_length_histogram(&segments, 10000);
    // Bin 0: [0, 10000) → contains 5000 → 1 segment
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (0, 1));
}

#[test]
fn test_segment_length_histogram_bin_size_equals_length() {
    let segments = vec![make_seg("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998)]; // 10000 bp
    let hist = segment_length_histogram(&segments, 10000);
    // 10000 / 10000 = bin 1
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (10000, 1));
}

#[test]
fn test_segment_length_histogram_very_small_bin_size() {
    let segments = vec![make_seg("chr1", 0, 99, "A", "B", 0, 0, 0.999, 0.998)]; // 100 bp
    let hist = segment_length_histogram(&segments, 1);
    // Single segment of 100 bp in bin 100
    assert_eq!(hist.len(), 1);
    assert_eq!(hist[0], (100, 1));
}

// =============================================================================
// detect_segments_rle: specific unwrap() paths
// =============================================================================

#[test]
fn test_detect_rle_single_good_window_below_min_windows() {
    // Single good window when min_windows=1 — exercises the unwrap at line 188/189
    let track = IdentityTrack {
        windows: vec![(0, 0.9999)],
        n_total_windows: 1,
    };
    let window_positions = vec![(0, 5999)];
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 5000,
        drop_tolerance: 0.0,
    };
    let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_called, 1);
}

#[test]
fn test_detect_rle_low_window_between_good_windows() {
    // Good, bad, good pattern — exercises the unwrap at line 231/232 where
    // a new segment is started after flushing
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.5), (2, 0.9998), (3, 0.9999), (4, 0.9997)],
        n_total_windows: 5,
    };
    let window_positions = vec![
        (0, 1999),
        (2000, 3999),
        (4000, 5999),
        (6000, 7999),
        (8000, 9999),
    ];
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 1000,
        drop_tolerance: 0.0,
    };
    let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
    // Window 0: good → start segment
    // Window 1: 0.5 is not good and not missing → flush segment [0..0] (1 window, 2000bp),
    //          then 0.5 is not good, so no restart
    // Window 2: good → start new segment
    // Windows 3-4: good → extend
    // End: flush segment [2..4]
    // [0] length = 1999-0+1 = 2000 >= 1000 ✓, n_windows=1 >= 1 ✓
    // [2..4] length = 9999-4000+1 = 6000 ✓, n_windows=3 >= 1 ✓
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].n_called, 1);
    assert_eq!(segments[1].n_called, 3);
}

#[test]
fn test_detect_rle_window_positions_shorter_than_n_total() {
    // If window_positions is shorter than n_total_windows, finalize_segment returns None
    // for out-of-range indices via window_positions.get(idx)?
    let track = IdentityTrack {
        windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997), (3, 0.9996), (4, 0.9995)],
        n_total_windows: 5,
    };
    let window_positions = vec![(0, 999), (1000, 1999)]; // only 2 positions for 5 windows
    let params = RleParams {
        min_identity: 0.9995,
        max_gap: 0,
        min_windows: 1,
        min_length_bp: 0,
        drop_tolerance: 0.0,
    };
    let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
    // finalize will use window_positions.get(4) → None → segment is None
    // Only the first finalized sub-range that fits within window_positions can succeed
    assert!(segments.is_empty() || segments[0].end <= 1999);
}

// =============================================================================
// merge_segments: identity tracking after merge
// =============================================================================

#[test]
fn test_merge_segments_min_identity_preserved_across_chain() {
    // Three overlapping segments with decreasing min_identity
    let mut segments = vec![
        make_seg("chr1", 0, 5000, "A", "B", 0, 5, 0.999, 0.998),
        make_seg("chr1", 3000, 8000, "A", "B", 3, 8, 0.998, 0.990),
        make_seg("chr1", 6000, 11000, "A", "B", 6, 11, 0.997, 0.950),
    ];
    merge_segments(&mut segments);
    assert_eq!(segments.len(), 1);
    assert!((segments[0].min_identity - 0.950).abs() < 1e-10);
}

#[test]
fn test_merge_segments_five_segments_mixed_pairs() {
    // Mix of overlapping and non-overlapping, different haplotype pairs
    let mut segments = vec![
        make_seg("chr1", 0, 5000, "A", "B", 0, 5, 0.999, 0.998),
        make_seg("chr1", 3000, 8000, "A", "B", 3, 8, 0.998, 0.997),
        make_seg("chr1", 0, 5000, "C", "D", 0, 5, 0.999, 0.998),
        make_seg("chr2", 0, 5000, "A", "B", 0, 5, 0.999, 0.998),
        make_seg("chr1", 3000, 8000, "C", "D", 3, 8, 0.998, 0.997),
    ];
    merge_segments(&mut segments);
    // AB on chr1 merges → 1, CD on chr1 merges → 1, AB on chr2 stays → 1 = 3 total
    assert_eq!(segments.len(), 3);
}

// =============================================================================
// concordance edge cases: inverted intervals, boundary conditions
// =============================================================================

#[test]
fn test_segment_overlap_bp_inverted_first() {
    // start > end on first interval — saturating_sub handles this
    assert_eq!(segment_overlap_bp((300, 100), (50, 200)), 0);
}

#[test]
fn test_segment_overlap_bp_inverted_both() {
    assert_eq!(segment_overlap_bp((300, 100), (400, 200)), 0);
}

#[test]
fn test_segment_overlap_bp_max_u64() {
    // Very large intervals near u64 max
    let a = (0, u64::MAX);
    let b = (0, u64::MAX);
    assert_eq!(segment_overlap_bp(a, b), u64::MAX);
}

#[test]
fn test_segments_jaccard_identical_overlapping_segments() {
    // Multiple overlapping segments in same set
    let ours = vec![(100, 300), (200, 400)]; // merged: 100-400 = 300bp
    let theirs = vec![(100, 400)]; // 300bp
    let j = segments_jaccard(&ours, &theirs, (0, 500));
    assert!((j - 1.0).abs() < 1e-9, "Overlapping same-extent should give J=1.0, got {}", j);
}

#[test]
fn test_per_window_concordance_single_bp_window() {
    let ours = vec![(0, 10)];
    let theirs = vec![(0, 10)];
    // window_size=1: each bp is a window
    let c = per_window_concordance(&ours, &theirs, (0, 10), 1);
    assert!((c - 1.0).abs() < 1e-9, "1-bp windows, same segments → 1.0, got {}", c);
}

#[test]
fn test_per_window_concordance_region_barely_larger_than_window() {
    // Region of 11bp, window of 10bp → 2 windows: [0,10) and [10,11)
    let ours = vec![(0, 11)];
    let theirs = vec![(0, 11)];
    let c = per_window_concordance(&ours, &theirs, (0, 11), 10);
    assert!((c - 1.0).abs() < 1e-9);
}

// =============================================================================
// matched_segments and length_correlation edge cases
// =============================================================================

#[test]
fn test_matched_segments_many_to_many() {
    // 3 ours, 3 theirs with various overlaps
    let ours = vec![(0, 100), (200, 300), (400, 500)];
    let theirs = vec![(50, 150), (250, 350), (450, 550)];
    // overlap fractions: 50/100=0.5 for each pair
    let m = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m.len(), 3);
}

#[test]
fn test_matched_segments_threshold_exactly_at_frac() {
    // overlap = 50, shorter = 100, frac = 0.5 — should match at threshold 0.5
    let ours = vec![(0, 100)];
    let theirs = vec![(50, 150)];
    let m = matched_segments(&ours, &theirs, 0.5);
    assert_eq!(m.len(), 1);
    // But at 0.5 + epsilon, should not match
    let m2 = matched_segments(&ours, &theirs, 0.500001);
    assert!(m2.is_empty());
}

#[test]
fn test_length_correlation_three_pairs_zero_variance_one_side() {
    // All "ours" same length but "theirs" differ → correlation = 0
    let matches: Vec<MatchedInterval> = vec![
        ((0, 100), (0, 50)),
        ((0, 100), (0, 200)),
        ((0, 100), (0, 150)),
    ];
    let r = length_correlation(&matches);
    assert!((r - 0.0).abs() < 1e-9, "Constant on one side → r=0, got {}", r);
}

// =============================================================================
// boundary_accuracy edge cases
// =============================================================================

#[test]
fn test_boundary_accuracy_large_distances() {
    let matches: Vec<MatchedInterval> = vec![
        ((0, 1_000_000), (500_000, 2_000_000)), // start diff=500000, end diff=1000000
    ];
    let acc = boundary_accuracy(&matches, 1_000_000).unwrap();
    assert_eq!(acc.n_matched, 1);
    assert!((acc.mean_start_distance_bp - 500_000.0).abs() < 1e-9);
    assert!((acc.mean_end_distance_bp - 1_000_000.0).abs() < 1e-9);
    // 500000 within 1000000 → yes
    assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
    // 1000000 within 1000000 → yes (<=)
    assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
}

#[test]
fn test_boundary_accuracy_four_matches_even_median() {
    // 4 matches: even count, median = avg of 2nd and 3rd
    let matches: Vec<MatchedInterval> = vec![
        ((1000, 10000), (1000, 10000)),   // start=0, end=0
        ((2000, 20000), (2100, 20200)),   // start=100, end=200
        ((3000, 30000), (3500, 30500)),   // start=500, end=500
        ((4000, 40000), (5000, 42000)),   // start=1000, end=2000
    ];
    let acc = boundary_accuracy(&matches, 10000).unwrap();
    assert_eq!(acc.n_matched, 4);
    // Sorted start distances: [0, 100, 500, 1000] → median = (100+500)/2 = 300
    assert!((acc.median_start_distance_bp - 300.0).abs() < 1e-9);
    // Sorted end distances: [0, 200, 500, 2000] → median = (200+500)/2 = 350
    assert!((acc.median_end_distance_bp - 350.0).abs() < 1e-9);
}

// =============================================================================
// extract_haplotype_index and extract_sample_id edge cases
// =============================================================================

#[test]
fn test_extract_haplotype_index_large_value() {
    assert_eq!(extract_haplotype_index("sample#255#contig"), Some(255));
}

#[test]
fn test_extract_haplotype_index_too_large_for_u8() {
    // 256 doesn't fit in u8
    assert_eq!(extract_haplotype_index("sample#256#contig"), None);
}

#[test]
fn test_extract_haplotype_index_multiple_hashes() {
    assert_eq!(extract_haplotype_index("sample#1#scaffold#extra"), Some(1));
}

#[test]
fn test_extract_sample_id_with_only_hash() {
    assert_eq!(extract_sample_id("#"), "");
}

#[test]
fn test_extract_sample_id_hash_at_end() {
    assert_eq!(extract_sample_id("sample#"), "sample");
}

// =============================================================================
// stats: logit/inv_logit numerical edge cases
// =============================================================================

#[test]
fn test_logit_monotonically_increasing() {
    let vals: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
    let logits: Vec<f64> = vals.iter().map(|&x| logit(x)).collect();
    for i in 1..logits.len() {
        assert!(
            logits[i] >= logits[i - 1],
            "logit should be monotone: logit({})={} < logit({})={}",
            vals[i - 1], logits[i - 1], vals[i], logits[i]
        );
    }
}

#[test]
fn test_inv_logit_at_zero() {
    assert!((inv_logit(0.0) - 0.5).abs() < 1e-10);
}

#[test]
fn test_inv_logit_symmetry() {
    // inv_logit(x) + inv_logit(-x) = 1
    for x in [0.1, 1.0, 5.0, 10.0, 50.0] {
        let sum = inv_logit(x) + inv_logit(-x);
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "inv_logit({}) + inv_logit({}) = {}, expected 1.0",
            x, -x, sum
        );
    }
}

#[test]
fn test_inv_logit_extreme_positive() {
    // Very large x → inv_logit → 1.0
    assert!((inv_logit(100.0) - 1.0).abs() < 1e-15);
    assert!((inv_logit(700.0) - 1.0).abs() < 1e-15);
}

#[test]
fn test_inv_logit_extreme_negative() {
    // Very negative x → inv_logit → 0.0
    assert!(inv_logit(-100.0) < 1e-15);
    assert!(inv_logit(-700.0) < 1e-15);
}

#[test]
fn test_logit_transform_single_observation() {
    let transformed = logit_transform_observations(&[0.5]);
    assert_eq!(transformed.len(), 1);
    assert!((transformed[0]).abs() < 1e-10); // logit(0.5) = 0
}

#[test]
fn test_logit_transform_all_same_value() {
    let obs = vec![0.999; 10];
    let transformed = logit_transform_observations(&obs);
    // All should be the same
    for i in 1..transformed.len() {
        assert!((transformed[i] - transformed[0]).abs() < 1e-15);
    }
}

// =============================================================================
// gaussian_to_logit_space edge cases
// =============================================================================

#[test]
fn test_gaussian_to_logit_space_mean_at_half() {
    // mean=0.5 → logit(0.5)=0, denominator = 0.5*0.5 = 0.25
    let params = gaussian_to_logit_space(0.5, 0.1);
    assert!((params.mean - 0.0).abs() < 1e-10);
    // logit_std = 0.1 / 0.25 = 0.4
    assert!((params.std - 0.4).abs() < 1e-6);
}

#[test]
fn test_gaussian_to_logit_space_mean_very_near_one() {
    // mean near 1.0: denominator very small → std gets capped
    let params = gaussian_to_logit_space(0.99999, 0.001);
    // logit(0.99999) ≈ 11.51, should be within [-LOGIT_CAP, LOGIT_CAP]
    assert!(params.mean > 11.0 && params.mean <= LOGIT_CAP,
        "logit(0.99999) should be ~11.51, got {}", params.mean);
    // std should be capped at LOGIT_CAP * 0.5 or at minimum 0.01
    assert!(params.std >= 0.01);
    assert!(params.std <= LOGIT_CAP * 0.5 + 0.001);
}

#[test]
fn test_gaussian_to_logit_space_zero_std() {
    // Zero std → logit_std should be max(0.0 / denom, 0.01) = 0.01
    let params = gaussian_to_logit_space(0.5, 0.0);
    assert!((params.std - 0.01).abs() < 1e-10);
}

// =============================================================================
// trimmed_mean additional edge cases
// =============================================================================

#[test]
fn test_trimmed_mean_single_element() {
    let data = vec![42.0];
    // trim_fraction=0 → trim_count=0 → start=0, end=1 → mean of [42.0]
    let tm = trimmed_mean(&data, 0.0).unwrap();
    assert!((tm - 42.0).abs() < 1e-10);
}

#[test]
fn test_trimmed_mean_single_element_with_trim() {
    let data = vec![42.0];
    // trim_fraction=0.49 → trim_count=floor(1*0.49)=0 → still works
    let tm = trimmed_mean(&data, 0.49).unwrap();
    assert!((tm - 42.0).abs() < 1e-10);
}

#[test]
fn test_trimmed_mean_two_elements_half_trim() {
    let data = vec![1.0, 100.0];
    // trim_fraction=0.49 → trim_count=floor(2*0.49)=0 → no trimming
    let tm = trimmed_mean(&data, 0.49).unwrap();
    assert!((tm - 50.5).abs() < 1e-10);
}

#[test]
fn test_trimmed_mean_with_nan() {
    // NaN in data: total_cmp sorts NaN to the end, so high-trim removes it
    let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN];
    let tm = trimmed_mean(&data, 0.2); // trim 1 from each end
    // After sorting with total_cmp: [1.0, 2.0, 3.0, 4.0, NaN]
    // Trim 1 from each: [2.0, 3.0, 4.0] → mean = 3.0
    if let Some(val) = tm {
        assert!((val - 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_trimmed_mean_with_infinity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, f64::INFINITY];
    let tm = trimmed_mean(&data, 0.2); // trim 1 from each end
    // After sorting: [1.0, 2.0, 3.0, 4.0, INF]
    // Trim 1 from each: [2.0, 3.0, 4.0] → mean = 3.0
    if let Some(val) = tm {
        assert!((val - 3.0).abs() < 1e-10);
    }
}

// =============================================================================
// kmeans_1d edge cases
// =============================================================================

#[test]
fn test_kmeans_zero_iterations() {
    // 0 iterations: should return initial centers (no refinement loop)
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = kmeans_1d(&data, 2, 0);
    assert!(result.is_some());
    let (centers, _) = result.unwrap();
    assert_eq!(centers.len(), 2);
}

#[test]
fn test_kmeans_all_nan() {
    // All NaN values — shouldn't panic
    let data = vec![f64::NAN; 5];
    let result = kmeans_1d(&data, 2, 10);
    assert!(result.is_some()); // total_cmp handles NaN
}

// =============================================================================
// em_two_component edge cases
// =============================================================================

#[test]
fn test_em_two_component_converges_in_one_iter() {
    // Already well-separated, max_iter=1
    let mut data = vec![0.0; 50];
    data.extend(vec![10.0; 50]);
    let init_low = GaussianParams::new_unchecked(0.0, 1.0);
    let init_high = GaussianParams::new_unchecked(10.0, 1.0);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 1, 1e-6);
    assert!(result.is_some());
    let (low, high, _, _) = result.unwrap();
    assert!(low.mean < 2.0);
    assert!(high.mean > 8.0);
}

#[test]
fn test_em_map_strong_prior() {
    // Very strong prior should keep means near initial values
    let data = vec![5.0; 100]; // All data at 5.0
    let init_low = GaussianParams::new_unchecked(0.0, 1.0);
    let init_high = GaussianParams::new_unchecked(10.0, 1.0);
    let result = em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 1000.0);
    if let Some((low, high, _, _)) = result {
        // With very strong prior, means should stay close to 0 and 10
        assert!(low.mean < 3.0, "Strong prior should keep low mean near 0, got {}", low.mean);
        assert!(high.mean > 7.0, "Strong prior should keep high mean near 10, got {}", high.mean);
    }
}

// =============================================================================
// bic_model_selection edge cases
// =============================================================================

#[test]
fn test_bic_two_data_points() {
    let data = vec![0.998, 0.999];
    let low = GaussianParams::new_unchecked(0.998, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // With only 2 data points, both should be finite
    assert!(bic_1.is_finite());
    assert!(bic_2.is_finite());
}

#[test]
fn test_bic_identical_components() {
    // When both components are identical, 2-component model wastes params
    let data: Vec<f64> = (0..100).map(|i| 0.999 + 0.0001 * (i as f64 / 100.0)).collect();
    let same = GaussianParams::new_unchecked(0.9995, 0.0001);
    let (bic_1, bic_2) = bic_model_selection(&data, &same, &same, 0.5);
    // 1-component should be better because 2-component uses 3 extra params for no gain
    assert!(
        bic_1 < bic_2,
        "Identical components: BIC_1={} should be < BIC_2={}",
        bic_1, bic_2
    );
}

// =============================================================================
// f1_score edge cases
// =============================================================================

#[test]
fn test_f1_score_negative_values() {
    // Negative precision/recall — shouldn't panic
    let f = f1_score(-0.5, 0.5);
    // 2 * (-0.5) * 0.5 / (-0.5 + 0.5) = -0.5 / 0 → formula uses < 1e-15 check
    // -0.5 + 0.5 = 0 < 1e-15 → returns 0.0
    assert!((f - 0.0).abs() < 1e-9);
}

#[test]
fn test_f1_score_very_small() {
    let f = f1_score(1e-16, 1e-16);
    // sum = 2e-16 < 1e-15 → returns 0.0
    assert!((f - 0.0).abs() < 1e-9);
}

// =============================================================================
// OnlineStats edge cases
// =============================================================================

#[test]
fn test_online_stats_add_nan() {
    let mut stats = OnlineStats::new();
    stats.add(1.0);
    stats.add(f64::NAN);
    assert_eq!(stats.count(), 2);
    // Mean should be NaN after adding NaN
    assert!(stats.mean().is_nan());
}

#[test]
fn test_online_stats_add_infinity() {
    let mut stats = OnlineStats::new();
    stats.add(1.0);
    stats.add(f64::INFINITY);
    assert_eq!(stats.count(), 2);
    // Mean should be inf
    assert!(stats.mean().is_infinite());
}
