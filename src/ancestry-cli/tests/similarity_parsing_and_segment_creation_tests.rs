//! Tests for parse_similarity_data_column edge cases, create_segment discriminability,
//! rfmix_to_windows segment boundary paths, smooth_states end-of-sequence behavior,
//! extract_ancestry_segments state transition patterns, and estimate_admixture/coverage.

use std::collections::HashMap;

use impopk_ancestry_cli::ancestry::{
    coverage_ratio, count_smoothing_changes, estimate_admixture_proportions,
    extract_ancestry_segments, filter_segments_by_min_lod, parse_similarity_data,
    parse_similarity_data_column, parse_similarity_data_with_coverage, smooth_states,
    AdmixtureProportions,
};
use impopk_ancestry_cli::hmm::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
};
use impopk_ancestry_cli::rfmix::{
    parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts, RfmixResult, RfmixSegment,
};

// ─────────────────────────────────────────────────────────────────────────────
// parse_similarity_data_column: deduplication and reversed query/ref
// ─────────────────────────────────────────────────────────────────────────────

fn make_sim_header() -> String {
    "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string()
}

/// Multiple alignments for same query-reference pair → take max similarity
#[test]
fn test_parse_sim_data_max_dedup() {
    let lines = vec![
        make_sim_header(),
        "chr10\t1000\t2000\tQ#1#scaffold1:0-1000\tR#1#scaffold1:0-1000\t0.95".to_string(),
        "chr10\t1000\t2000\tQ#1#scaffold2:0-1000\tR#1#scaffold2:0-1000\t0.98".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &ref_haps).unwrap();
    let obs = &result["Q#1"];
    assert_eq!(obs.len(), 1);
    // Max similarity should be 0.98
    assert!((obs[0].similarities["R#1"] - 0.98).abs() < 1e-6);
}

/// Query in group.b and reference in group.a (reversed) → still parsed correctly
#[test]
fn test_parse_sim_data_reversed_query_ref() {
    let lines = vec![
        make_sim_header(),
        "chr10\t1000\t2000\tR#1#scaffold:0-1000\tQ#1#scaffold:0-1000\t0.97".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &ref_haps).unwrap();
    assert!(result.contains_key("Q#1"));
    assert!((result["Q#1"][0].similarities["R#1"] - 0.97).abs() < 1e-6);
}

/// Neither query nor reference → row skipped
#[test]
fn test_parse_sim_data_non_query_ref_skipped() {
    let lines = vec![
        make_sim_header(),
        "chr10\t1000\t2000\tA#1\tB#1\t0.99".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &ref_haps).unwrap();
    assert!(result.is_empty());
}

/// Multiple windows sorted by position
#[test]
fn test_parse_sim_data_sorted_by_position() {
    let lines = vec![
        make_sim_header(),
        "chr10\t3000\t4000\tQ#1\tR#1\t0.95".to_string(),
        "chr10\t1000\t2000\tQ#1\tR#1\t0.97".to_string(),
        "chr10\t2000\t3000\tQ#1\tR#1\t0.96".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &ref_haps).unwrap();
    let obs = &result["Q#1"];
    assert_eq!(obs.len(), 3);
    assert_eq!(obs[0].start, 1000); // Sorted by position
    assert_eq!(obs[1].start, 2000);
    assert_eq!(obs[2].start, 3000);
}

/// Custom similarity column name
#[test]
fn test_parse_sim_data_custom_column() {
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\tjaccard.similarity".to_string(),
        "chr10\t1000\t2000\tQ#1\tR#1\t0.85".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data_column(
        lines.into_iter(), &query, &ref_haps, "jaccard.similarity",
    ).unwrap();
    assert!((result["Q#1"][0].similarities["R#1"] - 0.85).abs() < 1e-6);
}

/// Missing custom column → error
#[test]
fn test_parse_sim_data_missing_custom_column() {
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data_column(
        lines.into_iter(), &query, &ref_haps, "nonexistent_column",
    );
    assert!(result.is_err());
}

/// Multiple query samples in same file
#[test]
fn test_parse_sim_data_multiple_queries() {
    let lines = vec![
        make_sim_header(),
        "chr10\t1000\t2000\tQ1#1\tR#1\t0.95".to_string(),
        "chr10\t1000\t2000\tQ2#1\tR#1\t0.97".to_string(),
    ];
    let query = vec!["Q1#1".to_string(), "Q2#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &ref_haps).unwrap();
    assert_eq!(result.len(), 2);
    assert!(result.contains_key("Q1#1"));
    assert!(result.contains_key("Q2#1"));
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_similarity_data_with_coverage: edge cases
// ─────────────────────────────────────────────────────────────────────────────

fn make_cov_header() -> String {
    "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length".to_string()
}

/// Both lengths equal → coverage ratio = 1.0
#[test]
fn test_parse_sim_with_cov_equal_lengths() {
    let lines = vec![
        make_cov_header(),
        "chr10\t1000\t2000\tQ#1\tR#1\t0.95\t1000\t1000".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(), &query, &ref_haps, "estimated.identity",
    ).unwrap();
    let obs = &result["Q#1"];
    let cov = obs[0].coverage_ratios.as_ref().unwrap();
    assert!((cov["R#1"] - 1.0).abs() < 1e-6);
}

/// a_len < b_len → coverage ratio = a/b < 1.0
#[test]
fn test_parse_sim_with_cov_asymmetric() {
    let lines = vec![
        make_cov_header(),
        "chr10\t1000\t2000\tQ#1\tR#1\t0.95\t500\t1000".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(), &query, &ref_haps, "estimated.identity",
    ).unwrap();
    let cov = result["Q#1"][0].coverage_ratios.as_ref().unwrap();
    assert!((cov["R#1"] - 0.5).abs() < 1e-6);
}

/// Both lengths zero → coverage ratio = 0.0
#[test]
fn test_parse_sim_with_cov_both_zero() {
    let lines = vec![
        make_cov_header(),
        "chr10\t1000\t2000\tQ#1\tR#1\t0.95\t0\t0".to_string(),
    ];
    let query = vec!["Q#1".to_string()];
    let ref_haps = vec!["R#1".to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(), &query, &ref_haps, "estimated.identity",
    ).unwrap();
    let cov = result["Q#1"][0].coverage_ratios.as_ref().unwrap();
    assert!((cov["R#1"] - 0.0).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// rfmix_to_windows: segment boundary edge cases
// ─────────────────────────────────────────────────────────────────────────────

fn make_rfmix_result(segments: Vec<RfmixSegment>, n_haps: usize) -> RfmixResult {
    let hap_names: Vec<String> = (0..n_haps).map(|i| format!("HAP.{}", i)).collect();
    RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: hap_names,
        segments,
    }
}

/// Segment starts before computed min_pos → w_start = 0 branch
#[test]
fn test_rfmix_to_windows_seg_start_before_min() {
    // Two segments: first at 100-200, second at 300-400
    // min_pos = 100
    // If we manually craft a segment starting before min_pos we need segments with
    // different min. Actually all segments define the min.
    // Let's use three segments where the min is from one, and another segment's start
    // is also at min, exercising the `seg.start >= min_pos` branch (true path)
    let segs = vec![
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 100,
            end: 200,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        },
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 200,
            end: 400,
            start_cm: 1.0,
            end_cm: 3.0,
            n_snps: 20,
            hap_ancestries: vec![1],
        },
    ];
    let result = make_rfmix_result(segs, 1);
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1); // 1 haplotype
    // Region: 100-400, window_size=100 → 3 windows
    assert_eq!(windows[0].len(), 3);
    assert_eq!(windows[0][0], Some(0)); // First segment covers window 0
    assert_eq!(windows[0][1], Some(1)); // Second segment covers windows 1-2
    assert_eq!(windows[0][2], Some(1));
}

/// Exact window fit: (max_pos - min_pos) % window_size == 0
#[test]
fn test_rfmix_to_windows_exact_fit() {
    let segs = vec![
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 0,
            end: 300,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        },
    ];
    let result = make_rfmix_result(segs, 1);
    let windows = rfmix_to_windows(&result, 100);
    // 0-300, window_size=100 → div_ceil(300/100)=3 windows
    assert_eq!(windows[0].len(), 3);
}

/// Large window_size > region → single window
#[test]
fn test_rfmix_to_windows_huge_window() {
    let segs = vec![
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 100,
            end: 200,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        },
    ];
    let result = make_rfmix_result(segs, 1);
    let windows = rfmix_to_windows(&result, 1_000_000);
    // 100-200 = 100bp, window_size=1M → div_ceil(100/1M)=1 window
    assert_eq!(windows[0].len(), 1);
    assert_eq!(windows[0][0], Some(0));
}

/// Multiple haplotypes with different ancestries per segment
#[test]
fn test_rfmix_to_windows_multi_hap() {
    let segs = vec![
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 0,
            end: 200,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0, 1],
        },
    ];
    let result = make_rfmix_result(segs, 2);
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[0][0], Some(0)); // hap 0 = AFR
    assert_eq!(windows[1][0], Some(1)); // hap 1 = EUR
}

/// rfmix_window_starts: verify correct start positions
#[test]
fn test_rfmix_window_starts_basic() {
    let segs = vec![
        RfmixSegment {
            chrom: "chr10".to_string(),
            start: 100,
            end: 500,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        },
    ];
    let result = make_rfmix_result(segs, 1);
    let starts = rfmix_window_starts(&result, 100);
    assert_eq!(starts, vec![100, 200, 300, 400]);
}

// ─────────────────────────────────────────────────────────────────────────────
// smooth_states: end-of-sequence behavior
// ─────────────────────────────────────────────────────────────────────────────

/// Short run at the very end → NOT smoothed (run_end == smoothed.len())
#[test]
fn test_smooth_states_short_run_at_end() {
    let states = vec![0, 0, 0, 0, 1, 0]; // Short run of 1 at index 4
    let smoothed = smooth_states(&states, 2);
    // The run [1] at index 4 is length 1 < min_run=2
    // i=4, run_end=5, prev=0, next=0 → prev == next, so it IS smoothed to 0
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0]);
}

/// Short run at very end without lookahead → NOT smoothed
#[test]
fn test_smooth_states_final_short_run() {
    let states = vec![0, 0, 0, 1]; // Single 1 at end
    let smoothed = smooth_states(&states, 2);
    // run_end=4 == smoothed.len(), so condition `run_end < smoothed.len()` fails
    // The 1 at the end should NOT be smoothed
    assert_eq!(smoothed, vec![0, 0, 0, 1]);
}

/// Short run at the start → NOT smoothed (i == 0)
#[test]
fn test_smooth_states_short_run_at_start() {
    let states = vec![1, 0, 0, 0, 0]; // Single 1 at start
    let smoothed = smooth_states(&states, 2);
    // i=0, so condition `i > 0` fails → NOT smoothed
    assert_eq!(smoothed, vec![1, 0, 0, 0, 0]);
}

/// Two short runs between same-state boundaries → both smoothed
#[test]
fn test_smooth_states_two_short_runs() {
    let states = vec![0, 0, 1, 0, 0, 1, 0, 0];
    let smoothed = smooth_states(&states, 2);
    // Both [1] runs at index 2 and 5 are short (length 1), surrounded by 0s
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0, 0]);
}

/// Different neighbors → short run NOT smoothed
#[test]
fn test_smooth_states_different_neighbors() {
    let states = vec![0, 0, 2, 1, 1]; // Short run of 2 at index 2
    let smoothed = smooth_states(&states, 2);
    // prev=0, next=1 → different, so NOT smoothed
    assert_eq!(smoothed, vec![0, 0, 2, 1, 1]);
}

/// Length exactly min_run → NOT smoothed (only < min_run is smoothed)
#[test]
fn test_smooth_states_exact_min_run() {
    let states = vec![0, 0, 1, 1, 0, 0]; // Run of 2 ones, min_run=2
    let smoothed = smooth_states(&states, 2);
    // run_len(2) < min_run(2) is false → NOT smoothed
    assert_eq!(smoothed, vec![0, 0, 1, 1, 0, 0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// extract_ancestry_segments: state patterns
// ─────────────────────────────────────────────────────────────────────────────

fn make_obs(n: usize) -> Vec<AncestryObservation> {
    (0..n).map(|i| AncestryObservation {
        chrom: "chr10".to_string(),
        start: i as u64 * 10000,
        end: (i + 1) as u64 * 10000,
        sample: "Q#1".to_string(),
        similarities: HashMap::new(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }).collect()
}

fn make_params(n_states: usize) -> AncestryHmmParams {
    let pops: Vec<AncestralPopulation> = (0..n_states).map(|i| AncestralPopulation {
        name: format!("POP{}", i),
        haplotypes: vec![format!("hap{}", i)],
    }).collect();
    AncestryHmmParams::new(pops, 0.01)
}

/// All same state → single segment
#[test]
fn test_extract_ancestry_segments_uniform() {
    let obs = make_obs(5);
    let states = vec![0usize; 5];
    let params = make_params(2);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 5);
    assert_eq!(segments[0].ancestry_idx, 0);
}

/// Alternating states → one segment per window
#[test]
fn test_extract_ancestry_segments_alternating() {
    let obs = make_obs(4);
    let states = vec![0, 1, 0, 1];
    let params = make_params(2);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 4);
    for (i, seg) in segments.iter().enumerate() {
        assert_eq!(seg.ancestry_idx, i % 2);
        assert_eq!(seg.n_windows, 1);
    }
}

/// Three consecutive state changes
#[test]
fn test_extract_ancestry_segments_three_states() {
    let obs = make_obs(6);
    let states = vec![0, 0, 1, 1, 2, 2];
    let params = make_params(3);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].ancestry_idx, 0);
    assert_eq!(segments[0].n_windows, 2);
    assert_eq!(segments[1].ancestry_idx, 1);
    assert_eq!(segments[1].n_windows, 2);
    assert_eq!(segments[2].ancestry_idx, 2);
    assert_eq!(segments[2].n_windows, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// estimate_admixture_proportions: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// Two equal-length segments of different ancestry → 50/50
#[test]
fn test_admixture_two_equal_segments() {
    let obs = make_obs(4);
    let states = vec![0, 0, 1, 1];
    let params = make_params(2);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    let pop_names = vec!["POP0".to_string(), "POP1".to_string()];
    let result = estimate_admixture_proportions(&segments, "Q#1", &pop_names);
    assert_eq!(result.proportions.len(), 2);
    assert!((result.proportions["POP0"] - 0.5).abs() < 0.01);
    assert!((result.proportions["POP1"] - 0.5).abs() < 0.01);
    assert_eq!(result.n_switches, 1);
}

/// Three ancestries with unequal proportions
#[test]
fn test_admixture_three_unequal() {
    let obs = make_obs(10);
    let states = vec![0, 0, 0, 0, 0, 1, 1, 1, 2, 2]; // 50% pop0, 30% pop1, 20% pop2
    let params = make_params(3);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    let pop_names = vec!["POP0".to_string(), "POP1".to_string(), "POP2".to_string()];
    let result = estimate_admixture_proportions(&segments, "Q#1", &pop_names);
    assert_eq!(result.proportions.len(), 3);
    assert!((result.proportions["POP0"] - 0.5).abs() < 0.01);
    assert!((result.proportions["POP1"] - 0.3).abs() < 0.01);
    assert!((result.proportions["POP2"] - 0.2).abs() < 0.01);
    assert_eq!(result.n_switches, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// filter_segments_by_min_lod: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// All segments pass → same output
#[test]
fn test_filter_segments_all_pass() {
    let obs = make_obs(4);
    let states = vec![0, 0, 1, 1];
    let params = make_params(2);
    let segments = extract_ancestry_segments(&obs, &states, &params, None);
    let n_segments = segments.len();
    let filtered = filter_segments_by_min_lod(segments, 0.0);
    assert_eq!(filtered.len(), n_segments);
}

// ─────────────────────────────────────────────────────────────────────────────
// coverage_ratio: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// Symmetric coverage
#[test]
fn test_coverage_ratio_symmetric() {
    assert!((coverage_ratio(100, 200) - coverage_ratio(200, 100)).abs() < 1e-10);
}

/// Equal values → 1.0
#[test]
fn test_coverage_ratio_equal() {
    assert!((coverage_ratio(500, 500) - 1.0).abs() < 1e-10);
}

/// One very large, one small
#[test]
fn test_coverage_ratio_extreme() {
    let r = coverage_ratio(1, 1_000_000);
    assert!(r > 0.0 && r < 0.001);
}

// ─────────────────────────────────────────────────────────────────────────────
// count_smoothing_changes: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// Identical sequences → 0 changes
#[test]
fn test_count_smoothing_no_changes() {
    assert_eq!(count_smoothing_changes(&[0, 1, 0], &[0, 1, 0]), 0);
}

/// All changed → length of shorter
#[test]
fn test_count_smoothing_all_changed() {
    assert_eq!(count_smoothing_changes(&[0, 0, 0], &[1, 1, 1]), 3);
}

/// Empty sequences → 0 changes
#[test]
fn test_count_smoothing_empty() {
    assert_eq!(count_smoothing_changes(&[], &[]), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_rfmix_msp_content: edge cases
// ─────────────────────────────────────────────────────────────────────────────

/// RFMix with single segment
#[test]
fn test_parse_rfmix_single_segment() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\n\
                   chr10\t100\t200\t0.0\t1.0\t50\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.segments[0].hap_ancestries, vec![0]);
}

/// RFMix with multiple chromosomes
#[test]
fn test_parse_rfmix_multi_chrom() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\n\
                   chr10\t100\t200\t0.0\t1.0\t50\t0\n\
                   chr11\t100\t200\t0.0\t1.0\t50\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.segments[0].chrom, "chr10");
    assert_eq!(result.segments[1].chrom, "chr11");
}

/// 3-way ancestry
#[test]
fn test_parse_rfmix_three_way() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\tAMR=2\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1\n\
                   chr10\t100\t200\t0.0\t1.0\t50\t2\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "AMR"]);
    assert_eq!(result.segments[0].hap_ancestries, vec![2, 0]);
}
