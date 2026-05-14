//! Tests for ancestry-cli edge cases: coverage_ratio, smooth_states,
//! parse_similarity_data header paths, rfmix parsing, filter_segments_by_min_lod,
//! estimate_admixture_proportions, and rfmix_to_windows/rfmix_window_starts.

use impopk_ancestry_cli::ancestry::{
    coverage_ratio, count_smoothing_changes, estimate_admixture_proportions,
    filter_segments_by_min_lod, smooth_states, AncestrySegment,
};
use impopk_ancestry_cli::rfmix::{
    parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts, RfmixResult, RfmixSegment,
};

// =============================================================================
// coverage_ratio edge cases
// =============================================================================

#[test]
fn test_coverage_ratio_both_zero() {
    assert!((coverage_ratio(0, 0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_equal() {
    assert!((coverage_ratio(100, 100) - 1.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_one_zero() {
    assert!((coverage_ratio(0, 100) - 0.0).abs() < 1e-10);
    assert!((coverage_ratio(100, 0) - 0.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_half() {
    assert!((coverage_ratio(50, 100) - 0.5).abs() < 1e-10);
    // Symmetric
    assert!((coverage_ratio(100, 50) - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_large_values() {
    let ratio = coverage_ratio(1_000_000, 2_000_000);
    assert!((ratio - 0.5).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_one_vs_max() {
    let ratio = coverage_ratio(1, u64::MAX);
    assert!(ratio > 0.0);
    assert!(ratio < 1e-10); // Very small
}

// =============================================================================
// smooth_states edge cases
// =============================================================================

#[test]
fn test_smooth_states_empty() {
    let result = smooth_states(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn test_smooth_states_single_element() {
    let result = smooth_states(&[0], 3);
    assert_eq!(result, vec![0]);
}

#[test]
fn test_smooth_states_two_elements() {
    let result = smooth_states(&[0, 1], 3);
    assert_eq!(result, vec![0, 1]); // < 3 elements → returned as-is
}

#[test]
fn test_smooth_states_min_run_zero() {
    // min_run < 2 → returned as-is
    let result = smooth_states(&[0, 0, 1, 1, 0, 0], 0);
    assert_eq!(result, vec![0, 0, 1, 1, 0, 0]);
}

#[test]
fn test_smooth_states_min_run_one() {
    let result = smooth_states(&[0, 0, 1, 1, 0, 0], 1);
    assert_eq!(result, vec![0, 0, 1, 1, 0, 0]); // min_run < 2 → no change
}

#[test]
fn test_smooth_states_short_run_between_same() {
    // Short run of 1s between 0s, min_run=3
    let input = vec![0, 0, 0, 1, 1, 0, 0, 0];
    let result = smooth_states(&input, 3);
    // The run of 1s (length 2) is < min_run=3, and it's between 0s → smoothed to 0
    assert_eq!(result, vec![0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_smooth_states_short_run_between_different() {
    // Short run of 1s between 0 and 2 → not smoothed (surrounding states differ)
    let input = vec![0, 0, 0, 1, 2, 2, 2];
    let result = smooth_states(&input, 3);
    assert_eq!(result, vec![0, 0, 0, 1, 2, 2, 2]);
}

#[test]
fn test_smooth_states_all_same() {
    let result = smooth_states(&[0, 0, 0, 0, 0], 3);
    assert_eq!(result, vec![0, 0, 0, 0, 0]);
}

#[test]
fn test_smooth_states_long_enough_run_not_smoothed() {
    // Run of 1s is length 3 >= min_run=3 → not smoothed
    let input = vec![0, 0, 0, 1, 1, 1, 0, 0, 0];
    let result = smooth_states(&input, 3);
    assert_eq!(result, vec![0, 0, 0, 1, 1, 1, 0, 0, 0]);
}

#[test]
fn test_smooth_states_at_boundaries() {
    // Short run at the start — not smoothed (no preceding state)
    let input = vec![1, 0, 0, 0, 0];
    let result = smooth_states(&input, 3);
    // Run of 1 at start: i=0, run_len=1 < min_run=3
    // But i > 0 is false → no smoothing
    assert_eq!(result, vec![1, 0, 0, 0, 0]);
}

#[test]
fn test_smooth_states_at_end() {
    // Short run at the end — not smoothed (no following state)
    let input = vec![0, 0, 0, 0, 1];
    let result = smooth_states(&input, 3);
    // Run of 1 at end: run_end = len → no next state
    assert_eq!(result, vec![0, 0, 0, 0, 1]);
}

// =============================================================================
// count_smoothing_changes
// =============================================================================

#[test]
fn test_count_smoothing_changes_identical() {
    assert_eq!(count_smoothing_changes(&[0, 1, 2], &[0, 1, 2]), 0);
}

#[test]
fn test_count_smoothing_changes_all_different() {
    assert_eq!(count_smoothing_changes(&[0, 0, 0], &[1, 1, 1]), 3);
}

#[test]
fn test_count_smoothing_changes_empty() {
    assert_eq!(count_smoothing_changes(&[], &[]), 0);
}

#[test]
fn test_count_smoothing_changes_partial() {
    assert_eq!(count_smoothing_changes(&[0, 1, 0, 1], &[0, 0, 0, 1]), 1);
}

// =============================================================================
// filter_segments_by_min_lod
// =============================================================================

fn make_ancestry_segment(name: &str, start: u64, end: u64, lod: f64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "test".to_string(),
        ancestry_idx: 0,
        ancestry_name: name.to_string(),
        n_windows: 1,
        mean_similarity: 0.999,
        mean_posterior: None,
        discriminability: 0.5,
        lod_score: lod,
    }
}

#[test]
fn test_filter_segments_empty() {
    let result = filter_segments_by_min_lod(vec![], 0.0);
    assert!(result.is_empty());
}

#[test]
fn test_filter_segments_all_above() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, 5.0),
        make_ancestry_segment("AFR", 1000, 2000, 10.0),
    ];
    let result = filter_segments_by_min_lod(segs, 3.0);
    assert_eq!(result.len(), 2);
}

#[test]
fn test_filter_segments_all_below() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, 1.0),
        make_ancestry_segment("AFR", 1000, 2000, 2.0),
    ];
    let result = filter_segments_by_min_lod(segs, 5.0);
    assert!(result.is_empty());
}

#[test]
fn test_filter_segments_mixed() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, 1.0),
        make_ancestry_segment("AFR", 1000, 2000, 5.0),
        make_ancestry_segment("AMR", 2000, 3000, 10.0),
    ];
    let result = filter_segments_by_min_lod(segs, 3.0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].ancestry_name, "AFR");
    assert_eq!(result[1].ancestry_name, "AMR");
}

#[test]
fn test_filter_segments_zero_min_lod_keeps_all() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, -5.0),
        make_ancestry_segment("AFR", 1000, 2000, 0.0),
    ];
    let result = filter_segments_by_min_lod(segs, 0.0);
    // 0.0 >= 0.0 → kept, -5.0 >= 0.0 → filtered
    assert_eq!(result.len(), 1);
}

#[test]
fn test_filter_segments_negative_lod() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, -5.0),
        make_ancestry_segment("AFR", 1000, 2000, -1.0),
    ];
    let result = filter_segments_by_min_lod(segs, -3.0);
    // -1.0 >= -3.0 ✓, -5.0 >= -3.0 ✗
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].ancestry_name, "AFR");
}

// =============================================================================
// estimate_admixture_proportions edge cases
// =============================================================================

#[test]
fn test_admixture_empty_segments() {
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let result = estimate_admixture_proportions(&[], "test", &pop_names);
    assert_eq!(result.total_length_bp, 0);
    assert_eq!(result.n_switches, 0);
    assert!((result.mean_tract_length_bp - 0.0).abs() < 1e-10);
    for name in &pop_names {
        assert!((result.proportions[name] - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_admixture_single_ancestry() {
    let segs = vec![make_ancestry_segment("EUR", 0, 10000, 5.0)];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let result = estimate_admixture_proportions(&segs, "test", &pop_names);
    assert!((result.proportions["EUR"] - 1.0).abs() < 1e-10);
    assert!((result.proportions["AFR"] - 0.0).abs() < 1e-10);
    assert_eq!(result.n_switches, 0);
}

#[test]
fn test_admixture_two_ancestries_equal() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 5000, 5.0),
        make_ancestry_segment("AFR", 5000, 10000, 5.0),
    ];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let result = estimate_admixture_proportions(&segs, "test", &pop_names);
    assert!((result.proportions["EUR"] - 0.5).abs() < 1e-10);
    assert!((result.proportions["AFR"] - 0.5).abs() < 1e-10);
    assert_eq!(result.n_switches, 1);
}

#[test]
fn test_admixture_switches_only_at_ancestry_changes() {
    let segs = vec![
        make_ancestry_segment("EUR", 0, 1000, 5.0),
        make_ancestry_segment("EUR", 1000, 2000, 5.0),
        make_ancestry_segment("AFR", 2000, 3000, 5.0),
        make_ancestry_segment("EUR", 3000, 4000, 5.0),
    ];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let result = estimate_admixture_proportions(&segs, "test", &pop_names);
    // EUR→EUR (no switch), EUR→AFR (switch), AFR→EUR (switch) = 2
    assert_eq!(result.n_switches, 2);
}

#[test]
fn test_admixture_segment_with_start_equals_end() {
    // Zero-length segment
    let segs = vec![make_ancestry_segment("EUR", 5000, 5000, 5.0)];
    let pop_names = vec!["EUR".to_string()];
    let result = estimate_admixture_proportions(&segs, "test", &pop_names);
    assert_eq!(result.total_length_bp, 0);
}

// =============================================================================
// rfmix parsing edge cases
// =============================================================================

#[test]
fn test_parse_rfmix_empty_content() {
    let result = parse_rfmix_msp_content("");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Empty"));
}

#[test]
fn test_parse_rfmix_only_header() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_ok());
    let r = result.unwrap();
    assert_eq!(r.population_names, vec!["AFR", "EUR"]);
    assert_eq!(r.haplotype_names, vec!["HG00733.0", "HG00733.1"]);
    assert!(r.segments.is_empty());
}

#[test]
fn test_parse_rfmix_minimal_valid() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1\n\
                   chr20\t82590\t728330\t0.00\t2.67\t339\t1\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.segments[0].chrom, "chr20");
    assert_eq!(result.segments[0].start, 82590);
    assert_eq!(result.segments[0].end, 728330);
    assert_eq!(result.segments[0].hap_ancestries, vec![1, 0]);
}

#[test]
fn test_parse_rfmix_whitespace_separated_pops() {
    // Format 2: whitespace-separated names
    let content = "#reference_panel_population:  AFR  EUR  AMR\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\n\
                   chr10\t100\t200\t0.01\t0.02\t10\t2\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "AMR"]);
    assert_eq!(result.segments[0].hap_ancestries, vec![2]);
}

#[test]
fn test_parse_rfmix_comment_lines_skipped() {
    let content = "#Subpopulation order/codes: AFR=0\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tSample.0\n\
                   # This is a comment\n\
                   chr1\t100\t200\t0.0\t0.1\t5\t0\n\
                   \n\
                   chr1\t200\t300\t0.1\t0.2\t5\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    // Comments and empty lines should be skipped
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn test_parse_rfmix_invalid_start_position() {
    let content = "#Subpopulation order/codes: AFR=0\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tSample.0\n\
                   chr1\tNOT_A_NUMBER\t200\t0.0\t0.1\t5\t0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("invalid start"));
}

#[test]
fn test_parse_rfmix_too_few_fields() {
    let content = "#Subpopulation order/codes: AFR=0\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tSample.0\n\
                   chr1\t100\t200\n"; // missing fields
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn test_parse_rfmix_missing_column_header() {
    let content = "#Subpopulation order/codes: AFR=0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing column header"));
}

#[test]
fn test_parse_rfmix_non_contiguous_pop_indices() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=2\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tSample.0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not contiguous"));
}

// =============================================================================
// rfmix_to_windows edge cases
// =============================================================================

fn make_rfmix_result(
    pop_names: Vec<&str>,
    hap_names: Vec<&str>,
    segments: Vec<(u64, u64, Vec<usize>)>,
) -> RfmixResult {
    RfmixResult {
        population_names: pop_names.into_iter().map(String::from).collect(),
        haplotype_names: hap_names.into_iter().map(String::from).collect(),
        segments: segments
            .into_iter()
            .map(|(start, end, hap_anc)| RfmixSegment {
                chrom: "chr1".to_string(),
                start,
                end,
                start_cm: 0.0,
                end_cm: 0.0,
                n_snps: 10,
                hap_ancestries: hap_anc,
            })
            .collect(),
    }
}

#[test]
fn test_rfmix_to_windows_empty_segments() {
    let result = make_rfmix_result(vec!["AFR"], vec!["S.0", "S.1"], vec![]);
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 2);
    assert!(windows[0].is_empty());
    assert!(windows[1].is_empty());
}

#[test]
fn test_rfmix_to_windows_zero_window_size() {
    let result = make_rfmix_result(
        vec!["AFR"],
        vec!["S.0"],
        vec![(0, 10000, vec![0])],
    );
    let windows = rfmix_to_windows(&result, 0);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn test_rfmix_to_windows_single_segment() {
    let result = make_rfmix_result(
        vec!["AFR", "EUR"],
        vec!["S.0", "S.1"],
        vec![(0, 20000, vec![0, 1])],
    );
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 2);
    // 2 windows: [0, 10000) and [10000, 20000)
    assert_eq!(windows[0].len(), 2);
    assert_eq!(windows[0][0], Some(0)); // AFR
    assert_eq!(windows[0][1], Some(0)); // AFR
    assert_eq!(windows[1][0], Some(1)); // EUR
    assert_eq!(windows[1][1], Some(1)); // EUR
}

#[test]
fn test_rfmix_to_windows_multiple_segments() {
    let result = make_rfmix_result(
        vec!["AFR", "EUR"],
        vec!["S.0"],
        vec![
            (0, 10000, vec![0]),     // AFR for first 10kb
            (10000, 20000, vec![1]), // EUR for next 10kb
        ],
    );
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 2);
    assert_eq!(windows[0][0], Some(0)); // AFR
    assert_eq!(windows[0][1], Some(1)); // EUR
}

// =============================================================================
// rfmix_window_starts edge cases
// =============================================================================

#[test]
fn test_rfmix_window_starts_empty() {
    let result = make_rfmix_result(vec!["AFR"], vec!["S.0"], vec![]);
    let starts = rfmix_window_starts(&result, 10000);
    assert!(starts.is_empty());
}

#[test]
fn test_rfmix_window_starts_zero_window_size() {
    let result = make_rfmix_result(
        vec!["AFR"],
        vec!["S.0"],
        vec![(0, 10000, vec![0])],
    );
    let starts = rfmix_window_starts(&result, 0);
    assert!(starts.is_empty());
}

#[test]
fn test_rfmix_window_starts_basic() {
    let result = make_rfmix_result(
        vec!["AFR"],
        vec!["S.0"],
        vec![(100, 30100, vec![0])],
    );
    // Range: 100 to 30100, window_size=10000
    // n_windows = ceil(30000 / 10000) = 3
    let starts = rfmix_window_starts(&result, 10000);
    assert_eq!(starts.len(), 3);
    assert_eq!(starts[0], 100);
    assert_eq!(starts[1], 10100);
    assert_eq!(starts[2], 20100);
}

#[test]
fn test_rfmix_window_starts_same_start_and_end() {
    let result = make_rfmix_result(
        vec!["AFR"],
        vec!["S.0"],
        vec![(1000, 1000, vec![0])], // zero-length segment
    );
    let starts = rfmix_window_starts(&result, 10000);
    assert!(starts.is_empty()); // max_pos <= min_pos
}

// =============================================================================
// parse_similarity_data edge cases (header parsing)
// =============================================================================

#[test]
fn test_parse_similarity_data_empty_input() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines: Vec<String> = vec![];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_parse_similarity_data_header_only() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_parse_similarity_data_missing_column() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        // Missing "estimated.identity" column
        "chrom\tstart\tend\tgroup.a\tgroup.b".to_string(),
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing column"));
}

#[test]
fn test_parse_similarity_data_valid_row() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
        "chr1\t0\t10000\tsample#1#scaffold:0-10000\tref#1#scaffold:0-10000\t0.999".to_string(),
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains_key("sample#1"));
    assert_eq!(result["sample#1"].len(), 1);
    assert!((result["sample#1"][0].similarities["ref#1"] - 0.999).abs() < 1e-10);
}

#[test]
fn test_parse_similarity_data_skips_non_query_ref() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
        // Both are refs, neither is query → should be skipped
        "chr1\t0\t10000\tref#1#scaffold:0\tref#2#scaffold:0\t0.999".to_string(),
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string(), "ref#2".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_parse_similarity_data_reversed_query_ref_order() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
        // group.a is ref, group.b is query (reversed)
        "chr1\t0\t10000\tref#1#scaffold:0\tsample#1#scaffold:0\t0.998".to_string(),
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result["sample#1"][0].similarities["ref#1"] - 0.998).abs() < 1e-10);
}

#[test]
fn test_parse_similarity_data_duplicate_keeps_max() {
    use impopk_ancestry_cli::ancestry::parse_similarity_data;
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string(),
        "chr1\t0\t10000\tsample#1#s1:0\tref#1#s1:0\t0.990".to_string(),
        "chr1\t0\t10000\tsample#1#s1:0\tref#1#s2:0\t0.999".to_string(), // same window, same ref, higher → kept
    ];
    let query = vec!["sample#1".to_string()];
    let refs = vec!["ref#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    // Should keep the max (0.999)
    assert!((result["sample#1"][0].similarities["ref#1"] - 0.999).abs() < 1e-10);
}
