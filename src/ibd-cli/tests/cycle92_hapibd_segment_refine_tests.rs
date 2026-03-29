//! Cycle 92 edge-case tests targeting low-coverage functions.
//!
//! Functions targeted (test-file refs before this file):
//! - HapIbdSegment::involves_pair (2 refs)
//! - HapIbdSegment::involves_sample (2 refs)
//! - HapIbdSegment::as_interval (3 refs)
//! - parse_hapibd_file (2 refs)
//! - hapibd_segments_for_pair (3 refs)
//! - hapibd_segments_for_chr (3 refs)
//! - hapibd_segments_above_lod (3 refs)
//! - unique_pairs (3 refs)
//! - smooth_log_emissions (3 refs)
//! - aggregate_observations (3 refs)
//! - merge_nearby_ibd_segments (3 refs)
//! - refine_segment_boundaries (3 refs)
//! - refine_states_adaptive (3 refs)
//! - bridge_ibd_gaps_adaptive (3 refs)
//! - estimate_ibd_emission_std (3 refs)
//! - forward_backward_with_genetic_map_from_log_emit (2 refs)

use hprc_ibd::hapibd::{
    hapibd_segments_above_lod, hapibd_segments_for_chr, hapibd_segments_for_pair,
    parse_hapibd_content, parse_hapibd_file, unique_pairs, HapIbdSegment,
};
use hprc_ibd::hmm::{
    aggregate_observations, bridge_ibd_gaps_adaptive, estimate_ibd_emission_std,
    forward_backward_with_genetic_map_from_log_emit, merge_nearby_ibd_segments,
    refine_segment_boundaries, refine_states_adaptive, smooth_log_emissions, GeneticMap, HmmParams,
    IbdSegmentWithPosterior,
};

fn make_seg(s1: &str, s2: &str, chr: &str, start: u64, end: u64, lod: f64) -> HapIbdSegment {
    HapIbdSegment {
        sample1: s1.into(),
        hap1: 1,
        sample2: s2.into(),
        hap2: 1,
        chr: chr.into(),
        start,
        end,
        lod,
    }
}

fn make_ibd_seg(
    start_idx: usize,
    end_idx: usize,
    mean_post: f64,
    lod: f64,
) -> IbdSegmentWithPosterior {
    IbdSegmentWithPosterior {
        start_idx,
        end_idx,
        n_windows: end_idx - start_idx + 1,
        mean_posterior: mean_post,
        min_posterior: mean_post * 0.9,
        max_posterior: (mean_post * 1.1).min(1.0),
        lod_score: lod,
    }
}

// ===================== involves_pair =====================

#[test]
fn involves_pair_exact_order() {
    let seg = make_seg("A", "B", "chr1", 100, 200, 5.0);
    assert!(seg.involves_pair("A", "B"));
}

#[test]
fn involves_pair_reversed_order() {
    let seg = make_seg("A", "B", "chr1", 100, 200, 5.0);
    assert!(seg.involves_pair("B", "A"));
}

#[test]
fn involves_pair_no_match_first() {
    let seg = make_seg("A", "B", "chr1", 100, 200, 5.0);
    assert!(!seg.involves_pair("C", "B"));
}

#[test]
fn involves_pair_no_match_second() {
    let seg = make_seg("A", "B", "chr1", 100, 200, 5.0);
    assert!(!seg.involves_pair("A", "C"));
}

#[test]
fn involves_pair_neither_match() {
    let seg = make_seg("A", "B", "chr1", 100, 200, 5.0);
    assert!(!seg.involves_pair("X", "Y"));
}

#[test]
fn involves_pair_same_sample() {
    let seg = make_seg("A", "A", "chr1", 100, 200, 5.0);
    assert!(seg.involves_pair("A", "A"));
}

#[test]
fn involves_pair_empty_strings() {
    let seg = make_seg("", "", "chr1", 100, 200, 5.0);
    assert!(seg.involves_pair("", ""));
}

// ===================== involves_sample =====================

#[test]
fn involves_sample_matches_sample1() {
    let seg = make_seg("HG001", "HG002", "chr1", 0, 100, 3.0);
    assert!(seg.involves_sample("HG001"));
}

#[test]
fn involves_sample_matches_sample2() {
    let seg = make_seg("HG001", "HG002", "chr1", 0, 100, 3.0);
    assert!(seg.involves_sample("HG002"));
}

#[test]
fn involves_sample_no_match() {
    let seg = make_seg("HG001", "HG002", "chr1", 0, 100, 3.0);
    assert!(!seg.involves_sample("HG003"));
}

#[test]
fn involves_sample_partial_match() {
    let seg = make_seg("HG001", "HG002", "chr1", 0, 100, 3.0);
    assert!(!seg.involves_sample("HG00"));
}

#[test]
fn involves_sample_empty() {
    let seg = make_seg("HG001", "HG002", "chr1", 0, 100, 3.0);
    assert!(!seg.involves_sample(""));
}

// ===================== as_interval =====================

#[test]
fn as_interval_basic() {
    let seg = make_seg("A", "B", "chr1", 1000, 5000, 3.0);
    assert_eq!(seg.as_interval(), (1000, 5000));
}

#[test]
fn as_interval_zero_start() {
    let seg = make_seg("A", "B", "chr1", 0, 100, 1.0);
    assert_eq!(seg.as_interval(), (0, 100));
}

#[test]
fn as_interval_same_start_end() {
    let seg = make_seg("A", "B", "chr1", 500, 500, 1.0);
    assert_eq!(seg.as_interval(), (500, 500));
}

#[test]
fn as_interval_large_coords() {
    let seg = make_seg("A", "B", "chr1", 100_000_000, 200_000_000, 10.0);
    assert_eq!(seg.as_interval(), (100_000_000, 200_000_000));
}

// ===================== parse_hapibd_file =====================

#[test]
fn parse_hapibd_file_nonexistent() {
    let result = parse_hapibd_file("/nonexistent/path.ibd");
    assert!(result.is_err());
}

#[test]
fn parse_hapibd_file_from_tmpfile() {
    let content = "HG001\t1\tHG002\t2\tchr1\t1000\t5000\t12.5\n";
    let path = "/tmp/test_hapibd_cycle92.ibd";
    std::fs::write(path, content).unwrap();
    let segs = parse_hapibd_file(path).unwrap();
    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].sample1, "HG001");
    assert_eq!(segs[0].lod, 12.5);
    std::fs::remove_file(path).ok();
}

#[test]
fn parse_hapibd_file_empty_file() {
    let path = "/tmp/test_hapibd_empty_cycle92.ibd";
    std::fs::write(path, "").unwrap();
    let segs = parse_hapibd_file(path).unwrap();
    assert!(segs.is_empty());
    std::fs::remove_file(path).ok();
}

#[test]
fn parse_hapibd_file_comments_only() {
    let path = "/tmp/test_hapibd_comments_cycle92.ibd";
    std::fs::write(path, "# header1\n# header2\n").unwrap();
    let segs = parse_hapibd_file(path).unwrap();
    assert!(segs.is_empty());
    std::fs::remove_file(path).ok();
}

// ===================== hapibd_segments_for_pair =====================

#[test]
fn segments_for_pair_empty_input() {
    let segs: Vec<HapIbdSegment> = vec![];
    let result = hapibd_segments_for_pair(&segs, "A", "B");
    assert!(result.is_empty());
}

#[test]
fn segments_for_pair_multiple_hits() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 5.0),
        make_seg("C", "D", "chr1", 300, 400, 3.0),
        make_seg("B", "A", "chr2", 500, 600, 7.0),
    ];
    let result = hapibd_segments_for_pair(&segs, "A", "B");
    assert_eq!(result.len(), 2);
}

#[test]
fn segments_for_pair_no_match() {
    let segs = vec![make_seg("A", "B", "chr1", 100, 200, 5.0)];
    let result = hapibd_segments_for_pair(&segs, "X", "Y");
    assert!(result.is_empty());
}

// ===================== hapibd_segments_for_chr =====================

#[test]
fn segments_for_chr_empty() {
    let segs: Vec<HapIbdSegment> = vec![];
    let result = hapibd_segments_for_chr(&segs, "chr1");
    assert!(result.is_empty());
}

#[test]
fn segments_for_chr_filters_correctly() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 5.0),
        make_seg("A", "B", "chr2", 300, 400, 5.0),
        make_seg("C", "D", "chr1", 500, 600, 5.0),
    ];
    let result = hapibd_segments_for_chr(&segs, "chr1");
    assert_eq!(result.len(), 2);
}

#[test]
fn segments_for_chr_no_match() {
    let segs = vec![make_seg("A", "B", "chr1", 100, 200, 5.0)];
    let result = hapibd_segments_for_chr(&segs, "chrX");
    assert!(result.is_empty());
}

// ===================== hapibd_segments_above_lod =====================

#[test]
fn segments_above_lod_empty() {
    let segs: Vec<HapIbdSegment> = vec![];
    let result = hapibd_segments_above_lod(&segs, 3.0);
    assert!(result.is_empty());
}

#[test]
fn segments_above_lod_all_pass() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 10.0),
        make_seg("A", "B", "chr1", 300, 400, 5.0),
    ];
    let result = hapibd_segments_above_lod(&segs, 3.0);
    assert_eq!(result.len(), 2);
}

#[test]
fn segments_above_lod_none_pass() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 1.0),
        make_seg("A", "B", "chr1", 300, 400, 2.0),
    ];
    let result = hapibd_segments_above_lod(&segs, 5.0);
    assert!(result.is_empty());
}

#[test]
fn segments_above_lod_boundary() {
    let segs = vec![make_seg("A", "B", "chr1", 100, 200, 5.0)];
    let result = hapibd_segments_above_lod(&segs, 5.0);
    assert_eq!(result.len(), 1); // >= threshold
}

#[test]
fn segments_above_lod_negative() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, -1.0),
        make_seg("A", "B", "chr1", 300, 400, 3.0),
    ];
    let result = hapibd_segments_above_lod(&segs, 0.0);
    assert_eq!(result.len(), 1);
}

// ===================== unique_pairs =====================

#[test]
fn unique_pairs_empty() {
    let segs: Vec<HapIbdSegment> = vec![];
    let pairs = unique_pairs(&segs);
    assert!(pairs.is_empty());
}

#[test]
fn unique_pairs_deduplicates() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 5.0),
        make_seg("B", "A", "chr2", 300, 400, 3.0), // same pair reversed
        make_seg("A", "B", "chr1", 500, 600, 7.0), // same pair again
    ];
    let pairs = unique_pairs(&segs);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("A".to_string(), "B".to_string()));
}

#[test]
fn unique_pairs_canonical_order() {
    let segs = vec![make_seg("Z", "A", "chr1", 100, 200, 5.0)];
    let pairs = unique_pairs(&segs);
    assert_eq!(pairs[0].0, "A");
    assert_eq!(pairs[0].1, "Z");
}

#[test]
fn unique_pairs_multiple_distinct() {
    let segs = vec![
        make_seg("A", "B", "chr1", 100, 200, 5.0),
        make_seg("C", "D", "chr1", 300, 400, 3.0),
        make_seg("A", "C", "chr1", 500, 600, 7.0),
    ];
    let pairs = unique_pairs(&segs);
    assert_eq!(pairs.len(), 3);
}

// ===================== smooth_log_emissions =====================

#[test]
fn smooth_log_emissions_empty() {
    let result = smooth_log_emissions(&[], 2);
    assert!(result.is_empty());
}

#[test]
fn smooth_log_emissions_context_zero() {
    let log_emit = vec![[0.1, -0.1], [-0.5, 0.5]];
    let result = smooth_log_emissions(&log_emit, 0);
    assert_eq!(result.len(), 2);
    assert!((result[0][0] - 0.1).abs() < 1e-10);
}

#[test]
fn smooth_log_emissions_single_window() {
    let log_emit = vec![[-1.0, -2.0]];
    let result = smooth_log_emissions(&log_emit, 3);
    assert_eq!(result.len(), 1);
    assert!((result[0][0] - (-1.0)).abs() < 1e-10);
}

#[test]
fn smooth_log_emissions_uniform_unchanged() {
    let log_emit = vec![[-1.0, -1.0]; 5];
    let result = smooth_log_emissions(&log_emit, 2);
    for r in &result {
        assert!((r[0] - (-1.0)).abs() < 1e-10);
        assert!((r[1] - (-1.0)).abs() < 1e-10);
    }
}

#[test]
fn smooth_log_emissions_averages_neighbors() {
    let log_emit = vec![[0.0, 0.0], [3.0, 3.0], [0.0, 0.0]];
    let result = smooth_log_emissions(&log_emit, 1);
    // Middle window: avg of all 3 = 1.0
    assert!((result[1][0] - 1.0).abs() < 1e-10);
    // Edge window 0: avg of [0] and [1] = 1.5
    assert!((result[0][0] - 1.5).abs() < 1e-10);
}

#[test]
fn smooth_log_emissions_large_context() {
    let log_emit = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = smooth_log_emissions(&log_emit, 100);
    // All windows should be the global mean (context covers everything)
    let mean0 = (1.0 + 3.0 + 5.0) / 3.0;
    let mean1 = (2.0 + 4.0 + 6.0) / 3.0;
    for r in &result {
        assert!((r[0] - mean0).abs() < 1e-10);
        assert!((r[1] - mean1).abs() < 1e-10);
    }
}

// ===================== aggregate_observations =====================

#[test]
fn aggregate_empty() {
    let result = aggregate_observations(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn aggregate_factor_zero() {
    let obs = vec![1.0, 2.0, 3.0];
    let result = aggregate_observations(&obs, 0);
    assert_eq!(result, obs);
}

#[test]
fn aggregate_factor_one() {
    let obs = vec![1.0, 2.0, 3.0];
    let result = aggregate_observations(&obs, 1);
    assert_eq!(result, obs);
}

#[test]
fn aggregate_factor_two() {
    let obs = vec![1.0, 3.0, 5.0, 7.0];
    let result = aggregate_observations(&obs, 2);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-10); // avg(1,3)
    assert!((result[1] - 6.0).abs() < 1e-10); // avg(5,7)
}

#[test]
fn aggregate_factor_not_divisible() {
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 elements, factor 3
    let result = aggregate_observations(&obs, 3);
    assert_eq!(result.len(), 2);
    assert!((result[0] - 2.0).abs() < 1e-10); // avg(1,2,3)
    assert!((result[1] - 4.5).abs() < 1e-10); // avg(4,5)
}

#[test]
fn aggregate_factor_larger_than_len() {
    let obs = vec![2.0, 4.0, 6.0];
    let result = aggregate_observations(&obs, 10);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 4.0).abs() < 1e-10); // avg(2,4,6)
}

#[test]
fn aggregate_single_element() {
    let obs = vec![42.0];
    let result = aggregate_observations(&obs, 5);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 42.0).abs() < 1e-10);
}

// ===================== merge_nearby_ibd_segments =====================

#[test]
fn merge_nearby_empty() {
    let result = merge_nearby_ibd_segments(&[], 5);
    assert!(result.is_empty());
}

#[test]
fn merge_nearby_single_segment() {
    let segs = vec![make_ibd_seg(0, 5, 0.9, 10.0)];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 5);
}

#[test]
fn merge_nearby_gap_within_threshold() {
    let segs = vec![
        make_ibd_seg(0, 3, 0.9, 5.0),
        make_ibd_seg(5, 8, 0.8, 4.0), // gap = 1 window
    ];
    let result = merge_nearby_ibd_segments(&segs, 2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].start_idx, 0);
    assert_eq!(result[0].end_idx, 8);
}

#[test]
fn merge_nearby_gap_exceeds_threshold() {
    let segs = vec![
        make_ibd_seg(0, 3, 0.9, 5.0),
        make_ibd_seg(10, 15, 0.8, 4.0), // gap = 6 windows
    ];
    let result = merge_nearby_ibd_segments(&segs, 3);
    assert_eq!(result.len(), 2);
}

#[test]
fn merge_nearby_adjacent_segments() {
    let segs = vec![
        make_ibd_seg(0, 3, 0.9, 5.0),
        make_ibd_seg(4, 7, 0.8, 4.0), // gap = 0
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].end_idx, 7);
}

#[test]
fn merge_nearby_lod_additive() {
    let segs = vec![
        make_ibd_seg(0, 3, 0.9, 5.0),
        make_ibd_seg(4, 7, 0.8, 3.0),
    ];
    let result = merge_nearby_ibd_segments(&segs, 0);
    assert!((result[0].lod_score - 8.0).abs() < 1e-10);
}

#[test]
fn merge_nearby_three_segments_chain() {
    let segs = vec![
        make_ibd_seg(0, 2, 0.9, 3.0),
        make_ibd_seg(4, 6, 0.8, 2.0), // gap 1
        make_ibd_seg(8, 10, 0.7, 1.0), // gap 1
    ];
    let result = merge_nearby_ibd_segments(&segs, 1);
    assert_eq!(result.len(), 1);
}

// ===================== refine_segment_boundaries =====================

#[test]
fn refine_boundaries_empty_segments() {
    let result = refine_segment_boundaries(&[], &[0.1, 0.9], &[0, 10000], &[9999, 19999], 0.5);
    assert!(result.is_empty());
}

#[test]
fn refine_boundaries_empty_posteriors() {
    let segs = vec![make_ibd_seg(0, 1, 0.9, 5.0)];
    let result = refine_segment_boundaries(&segs, &[], &[], &[], 0.5);
    assert_eq!(result.len(), 1);
}

#[test]
fn refine_boundaries_interpolates_start() {
    let segs = vec![make_ibd_seg(2, 4, 0.9, 5.0)];
    let posteriors = vec![0.1, 0.3, 0.92, 0.98, 0.91, 0.2, 0.05];
    let starts = vec![1, 10001, 20001, 30001, 40001, 50001, 60001];
    let ends = vec![10000, 20000, 30000, 40000, 50000, 60000, 70000];
    let result = refine_segment_boundaries(&segs, &posteriors, &starts, &ends, 0.5);
    assert_eq!(result.len(), 1);
    // Start refined between window 1 (P=0.3) and window 2 (P=0.92)
    assert!(result[0].start_bp > 15000 && result[0].start_bp < 25000);
}

#[test]
fn refine_boundaries_interpolates_end() {
    let segs = vec![make_ibd_seg(2, 4, 0.9, 5.0)];
    let posteriors = vec![0.1, 0.3, 0.92, 0.98, 0.91, 0.2, 0.05];
    let starts = vec![1, 10001, 20001, 30001, 40001, 50001, 60001];
    let ends = vec![10000, 20000, 30000, 40000, 50000, 60000, 70000];
    let result = refine_segment_boundaries(&segs, &posteriors, &starts, &ends, 0.5);
    // End refined between window 4 (P=0.91) and window 5 (P=0.2)
    assert!(result[0].end_bp > 45000 && result[0].end_bp < 55000);
}

#[test]
fn refine_boundaries_no_crossover_uses_window_edge() {
    // All posteriors above crossover — no interpolation possible
    let segs = vec![make_ibd_seg(0, 2, 0.9, 5.0)];
    let posteriors = vec![0.8, 0.9, 0.85];
    let starts = vec![0, 10000, 20000];
    let ends = vec![9999, 19999, 29999];
    let result = refine_segment_boundaries(&segs, &posteriors, &starts, &ends, 0.5);
    assert_eq!(result[0].start_bp, 0);
    assert_eq!(result[0].end_bp, 29999);
}

#[test]
fn refine_boundaries_mismatched_lengths_fallback() {
    let segs = vec![make_ibd_seg(0, 1, 0.9, 5.0)];
    let posteriors = vec![0.8, 0.9];
    let starts = vec![0, 10000, 20000]; // 3 vs 2 posteriors
    let ends = vec![9999, 19999, 29999];
    let result = refine_segment_boundaries(&segs, &posteriors, &starts, &ends, 0.5);
    assert_eq!(result.len(), 1);
}

// ===================== refine_states_adaptive =====================

#[test]
fn refine_states_adaptive_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    refine_states_adaptive(&mut states, &posteriors);
    assert!(states.is_empty());
}

#[test]
fn refine_states_adaptive_mismatched_lengths() {
    let mut states = vec![0, 1, 0];
    let posteriors = vec![0.5, 0.9]; // shorter
    refine_states_adaptive(&mut states, &posteriors);
    // Should return without modification
    assert_eq!(states, vec![0, 1, 0]);
}

#[test]
fn refine_states_adaptive_no_ibd() {
    let mut states = vec![0, 0, 0, 0];
    let posteriors = vec![0.1, 0.2, 0.1, 0.05];
    refine_states_adaptive(&mut states, &posteriors);
    assert_eq!(states, vec![0, 0, 0, 0]);
}

#[test]
fn refine_states_adaptive_high_confidence_extends() {
    let mut states = vec![0, 0, 1, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.4, 0.95, 0.98, 0.93, 0.35, 0.05];
    refine_states_adaptive(&mut states, &posteriors);
    // High confidence segment (mean > 0.8) should extend left to window 1 (P=0.4 >= 0.3)
    assert_eq!(states[1], 1);
    // Should extend right to window 5 (P=0.35 >= 0.3)
    assert_eq!(states[5], 1);
}

#[test]
fn refine_states_adaptive_low_confidence_trims() {
    let mut states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.05, 0.12, 0.45, 0.40, 0.05];
    refine_states_adaptive(&mut states, &posteriors);
    // Low confidence segment (mean ~0.32) should trim windows below 0.15
    assert_eq!(states[1], 0); // P=0.12 < 0.15
}

#[test]
fn refine_states_adaptive_all_ibd() {
    let mut states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.98, 0.93, 0.91];
    refine_states_adaptive(&mut states, &posteriors);
    // All high confidence — no trimming
    assert_eq!(states, vec![1, 1, 1, 1, 1]);
}

// ===================== bridge_ibd_gaps_adaptive =====================

#[test]
fn bridge_gaps_empty() {
    let mut states: Vec<usize> = vec![];
    let posteriors: Vec<f64> = vec![];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridges, 0);
}

#[test]
fn bridge_gaps_too_short() {
    let mut states = vec![1, 0];
    let posteriors = vec![0.9, 0.1];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridges, 0);
}

#[test]
fn bridge_gaps_mismatched() {
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.5]; // shorter
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridges, 0);
}

#[test]
fn bridge_gaps_max_gap_zero() {
    let mut states = vec![1, 0, 1];
    let posteriors = vec![0.9, 0.5, 0.9];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 0, 0.3);
    assert_eq!(bridges, 0);
}

#[test]
fn bridge_gaps_short_gap_bridged() {
    let mut states = vec![1, 1, 0, 1, 1];
    let posteriors = vec![0.9, 0.9, 0.5, 0.9, 0.9];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert!(bridges >= 1);
    assert_eq!(states[2], 1); // gap bridged
}

#[test]
fn bridge_gaps_long_gap_not_bridged() {
    let mut states = vec![1, 0, 0, 0, 0, 0, 1];
    let posteriors = vec![0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 2, 0.3);
    assert_eq!(bridges, 0);
    assert_eq!(states[3], 0); // gap not bridged
}

#[test]
fn bridge_gaps_no_second_segment() {
    let mut states = vec![1, 1, 0, 0, 0];
    let posteriors = vec![0.9, 0.9, 0.5, 0.5, 0.1];
    let bridges = bridge_ibd_gaps_adaptive(&mut states, &posteriors, 3, 0.3);
    assert_eq!(bridges, 0);
}

// ===================== estimate_ibd_emission_std =====================

#[test]
fn estimate_emission_std_too_few() {
    let obs: Vec<f64> = (0..19).map(|i| i as f64 * 0.01).collect();
    assert!(estimate_ibd_emission_std(&obs, 0.1, 0.01, 0.5).is_none());
}

#[test]
fn estimate_emission_std_exactly_20() {
    let obs: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
    let result = estimate_ibd_emission_std(&obs, 0.1, 0.01, 0.5);
    assert!(result.is_some());
    let std = result.unwrap();
    assert!(std >= 0.01 && std <= 0.5);
}

#[test]
fn estimate_emission_std_uniform_low_variance() {
    let obs = vec![0.5; 100];
    let result = estimate_ibd_emission_std(&obs, 0.1, 0.01, 0.5).unwrap();
    assert_eq!(result, 0.01); // clamped to min
}

#[test]
fn estimate_emission_std_clamped_to_max() {
    // Observations where the top quantile has high variance
    // Use a range so the top fraction isn't all the same value
    let obs: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let result = estimate_ibd_emission_std(&obs, 0.5, 0.01, 0.5).unwrap();
    // Top 50% of 0..99 = 50..99, std is large → clamped to max
    assert_eq!(result, 0.5);
}

#[test]
fn estimate_emission_std_quantile_fraction_full() {
    let obs: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
    let result = estimate_ibd_emission_std(&obs, 1.0, 0.001, 10.0).unwrap();
    assert!(result > 0.0);
}

// ===================== forward_backward_with_genetic_map_from_log_emit =====================

#[test]
fn fb_genmap_log_emit_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let gm = GeneticMap::uniform(0, 100_000, 1.0);
    let (posteriors, ll) =
        forward_backward_with_genetic_map_from_log_emit(&[], &params, &[], &gm, 10000);
    assert!(posteriors.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn fb_genmap_log_emit_single() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let gm = GeneticMap::uniform(0, 100_000, 1.0);
    let log_emit = vec![[-0.5, -1.0]];
    let positions = vec![(0u64, 9999u64)];
    let (posteriors, ll) =
        forward_backward_with_genetic_map_from_log_emit(&log_emit, &params, &positions, &gm, 10000);
    assert_eq!(posteriors.len(), 1);
    assert!(posteriors[0] >= 0.0 && posteriors[0] <= 1.0);
    assert!(ll.is_finite());
}

#[test]
fn fb_genmap_log_emit_strong_ibd_signal() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let gm = GeneticMap::uniform(0, 200_000, 1.0);
    // Strong IBD signal: state 1 heavily favored
    let log_emit: Vec<[f64; 2]> = (0..10)
        .map(|_| [-5.0, -0.1]) // state 1 strongly preferred
        .collect();
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 10000, (i + 1) * 10000 - 1)).collect();
    let (posteriors, ll) =
        forward_backward_with_genetic_map_from_log_emit(&log_emit, &params, &positions, &gm, 10000);
    assert_eq!(posteriors.len(), 10);
    // All posteriors should be high (close to 1.0)
    for p in &posteriors {
        assert!(*p > 0.5, "IBD posterior {} should be > 0.5", p);
    }
    assert!(ll.is_finite());
}

#[test]
fn fb_genmap_log_emit_posteriors_bounded() {
    let params = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let gm = GeneticMap::uniform(0, 100_000, 1.0);
    let log_emit: Vec<[f64; 2]> = (0..5)
        .map(|i| if i < 3 { [-0.5, -2.0] } else { [-2.0, -0.5] })
        .collect();
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 10000, (i + 1) * 10000 - 1)).collect();
    let (posteriors, _ll) =
        forward_backward_with_genetic_map_from_log_emit(&log_emit, &params, &positions, &gm, 10000);
    for p in &posteriors {
        assert!(*p >= 0.0 && *p <= 1.0, "Posterior {} out of [0,1]", p);
    }
}
