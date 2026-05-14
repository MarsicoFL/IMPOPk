//! Robustness tests for similarity data parsing functions.
//!
//! Tests parse_similarity_data, parse_similarity_data_column,
//! parse_similarity_data_with_coverage, extract_sample_id,
//! coverage_ratio, and admixture proportion estimation.

use impopk_ancestry_cli::ancestry::{
    parse_similarity_data, parse_similarity_data_column, parse_similarity_data_with_coverage,
    coverage_ratio, estimate_admixture_proportions, AncestrySegment,
    load_population_samples, load_populations_from_dir,
    smooth_states, count_smoothing_changes, filter_segments_by_min_lod,
};
use tempfile::TempDir;
use std::fs;

// ── extract_sample_id logic (tested indirectly via parsing) ─────────────

fn make_header() -> String {
    "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string()
}

fn make_header_with_coverage() -> String {
    "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\testimated.identity"
        .to_string()
}

fn make_data_line(chrom: &str, start: u64, end: u64, ga: &str, gb: &str, ident: f64) -> String {
    format!("{}\t{}\t{}\t{}\t{}\t{}", chrom, start, end, ga, gb, ident)
}

fn make_data_line_with_lengths(
    chrom: &str, start: u64, end: u64, ga: &str, gb: &str, a_len: u64, b_len: u64, ident: f64,
) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        chrom, start, end, ga, gb, a_len, b_len, ident
    )
}

// ── parse_similarity_data basic correctness ─────────────────────────────

#[test]
fn parse_similarity_basic_two_refs() {
    let lines = vec![
        make_header(),
        make_data_line("chr1", 0, 5000, "query#1#scaffold:0-5000", "ref_a#1#scaffold:0-5000", 0.95),
        make_data_line("chr1", 0, 5000, "query#1#scaffold:0-5000", "ref_b#1#scaffold:0-5000", 0.88),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string(), "ref_b#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert!(result.contains_key("query#1"));

    let obs = &result["query#1"];
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].similarities.len(), 2);
    assert!((obs[0].similarities["ref_a#1"] - 0.95).abs() < 1e-10);
    assert!((obs[0].similarities["ref_b#1"] - 0.88).abs() < 1e-10);
}

#[test]
fn parse_similarity_query_in_group_b_position() {
    // Query appears as group.b instead of group.a
    let lines = vec![
        make_header(),
        make_data_line("chr1", 0, 5000, "ref_a#1#scaffold:0-5000", "query#1#scaffold:0-5000", 0.92),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert!(result.contains_key("query#1"));
    assert!((result["query#1"][0].similarities["ref_a#1"] - 0.92).abs() < 1e-10);
}

#[test]
fn parse_similarity_skips_ref_vs_ref_comparisons() {
    let lines = vec![
        make_header(),
        make_data_line("chr1", 0, 5000, "ref_a#1#s:0-5000", "ref_b#1#s:0-5000", 0.85),
        make_data_line("chr1", 0, 5000, "query#1#s:0-5000", "ref_a#1#s:0-5000", 0.95),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string(), "ref_b#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert_eq!(result["query#1"].len(), 1);
    assert_eq!(result["query#1"][0].similarities.len(), 1);
}

#[test]
fn parse_similarity_skips_query_vs_query_comparisons() {
    let lines = vec![
        make_header(),
        make_data_line("chr1", 0, 5000, "q1#1#s:0-5000", "q2#1#s:0-5000", 0.90),
    ];

    let query_samples = vec!["q1#1".to_string(), "q2#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert!(result.is_empty());
}

#[test]
fn parse_similarity_duplicate_alignment_keeps_max() {
    let lines = vec![
        make_header(),
        make_data_line("chr1", 0, 5000, "query#1#s:0-5000", "ref_a#1#s:0-5000", 0.85),
        make_data_line("chr1", 0, 5000, "query#1#s:0-5000", "ref_a#1#s:0-5000", 0.92),
        make_data_line("chr1", 0, 5000, "query#1#s:0-5000", "ref_a#1#s:0-5000", 0.88),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert!((result["query#1"][0].similarities["ref_a#1"] - 0.92).abs() < 1e-10);
}

#[test]
fn parse_similarity_multiple_windows_sorted_by_position() {
    let lines = vec![
        make_header(),
        make_data_line("chr1", 10000, 15000, "query#1#s:10000-15000", "ref_a#1#s:10000-15000", 0.90),
        make_data_line("chr1", 0, 5000, "query#1#s:0-5000", "ref_a#1#s:0-5000", 0.95),
        make_data_line("chr1", 5000, 10000, "query#1#s:5000-10000", "ref_a#1#s:5000-10000", 0.92),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    let obs = &result["query#1"];
    assert_eq!(obs.len(), 3);
    assert_eq!(obs[0].start, 0);
    assert_eq!(obs[1].start, 5000);
    assert_eq!(obs[2].start, 10000);
}

#[test]
fn parse_similarity_empty_lines_empty_result() {
    let lines: Vec<String> = vec![make_header()];
    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).unwrap();
    assert!(result.is_empty());
}

#[test]
fn parse_similarity_no_header_errors() {
    let lines = vec!["not\ta\theader".to_string()];
    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    assert!(parse_similarity_data(lines.into_iter(), &query_samples, &ref_haps).is_err());
}

// ── parse_similarity_data_column with custom column ─────────────────────

#[test]
fn parse_similarity_custom_column() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tjaccard.similarity".to_string();
    let lines = vec![
        header,
        "chr1\t0\t5000\tquery#1#s:0\tref_a#1#s:0\t0.75".to_string(),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data_column(
        lines.into_iter(), &query_samples, &ref_haps, "jaccard.similarity",
    ).unwrap();
    assert!((result["query#1"][0].similarities["ref_a#1"] - 0.75).abs() < 1e-10);
}

#[test]
fn parse_similarity_missing_custom_column_errors() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string();
    let lines = vec![header];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    assert!(parse_similarity_data_column(
        lines.into_iter(), &query_samples, &ref_haps, "nonexistent.column",
    ).is_err());
}

// ── parse_similarity_data_with_coverage ──────────────────────────────────

#[test]
fn parse_similarity_with_coverage_basic() {
    let lines = vec![
        make_header_with_coverage(),
        make_data_line_with_lengths("chr1", 0, 5000, "query#1#s:0", "ref_a#1#s:0", 4500, 5000, 0.95),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data_with_coverage(
        lines.into_iter(), &query_samples, &ref_haps, "estimated.identity",
    ).unwrap();

    let obs = &result["query#1"][0];
    assert!(obs.coverage_ratios.is_some());
    let cov = obs.coverage_ratios.as_ref().unwrap();
    assert!((cov["ref_a#1"] - 0.9).abs() < 1e-10);
}

#[test]
fn parse_similarity_with_coverage_zero_lengths() {
    let lines = vec![
        make_header_with_coverage(),
        make_data_line_with_lengths("chr1", 0, 5000, "query#1#s:0", "ref_a#1#s:0", 0, 0, 0.50),
    ];

    let query_samples = vec!["query#1".to_string()];
    let ref_haps = vec!["ref_a#1".to_string()];

    let result = parse_similarity_data_with_coverage(
        lines.into_iter(), &query_samples, &ref_haps, "estimated.identity",
    ).unwrap();

    let cov = result["query#1"][0].coverage_ratios.as_ref().unwrap();
    assert!((cov["ref_a#1"] - 0.0).abs() < 1e-10);
}

// ── coverage_ratio ──────────────────────────────────────────────────────

#[test]
fn coverage_ratio_equal_lengths() {
    assert!((coverage_ratio(100, 100) - 1.0).abs() < 1e-10);
}

#[test]
fn coverage_ratio_asymmetric() {
    assert!((coverage_ratio(50, 100) - 0.5).abs() < 1e-10);
    assert!((coverage_ratio(100, 50) - 0.5).abs() < 1e-10);
}

#[test]
fn coverage_ratio_zero_both() {
    assert!((coverage_ratio(0, 0) - 0.0).abs() < 1e-10);
}

#[test]
fn coverage_ratio_one_zero() {
    assert!((coverage_ratio(100, 0) - 0.0).abs() < 1e-10);
    assert!((coverage_ratio(0, 100) - 0.0).abs() < 1e-10);
}

// ── smooth_states edge cases ────────────────────────────────────────────

#[test]
fn smooth_states_min_run_1_noop() {
    let states = vec![0, 1, 0, 1, 0];
    assert_eq!(smooth_states(&states, 1), states);
}

#[test]
fn smooth_states_two_elements_noop() {
    let states = vec![0, 1];
    assert_eq!(smooth_states(&states, 3), states);
}

#[test]
fn smooth_states_all_same() {
    let states = vec![2, 2, 2, 2, 2];
    assert_eq!(smooth_states(&states, 3), states);
}

#[test]
fn smooth_states_short_run_between_different_states_not_merged() {
    let states = vec![0, 0, 0, 1, 2, 2, 2];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed[3], 1);
}

#[test]
fn smooth_states_preserves_count() {
    let states = vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed.len(), states.len());
}

// ── count_smoothing_changes ─────────────────────────────────────────────

#[test]
fn count_smoothing_changes_no_changes() {
    let a = vec![0, 1, 2];
    assert_eq!(count_smoothing_changes(&a, &a), 0);
}

#[test]
fn count_smoothing_changes_all_changed() {
    let a = vec![0, 0, 0];
    let b = vec![1, 1, 1];
    assert_eq!(count_smoothing_changes(&a, &b), 3);
}

// ── filter_segments_by_min_lod ──────────────────────────────────────────

fn make_ancestry_segment(pop: &str, lod: f64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test_sample".to_string(),
        ancestry_idx: 0,
        ancestry_name: pop.to_string(),
        n_windows: 10,
        mean_similarity: 0.95,
        mean_posterior: Some(0.9),
        discriminability: 0.1,
        lod_score: lod,
    }
}

#[test]
fn filter_segments_by_min_lod_keeps_above() {
    let segments = vec![
        make_ancestry_segment("AFR", 5.0),
        make_ancestry_segment("EUR", 2.0),
        make_ancestry_segment("NAT", 8.0),
    ];
    let filtered = filter_segments_by_min_lod(segments, 3.0);
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].ancestry_name, "AFR");
    assert_eq!(filtered[1].ancestry_name, "NAT");
}

#[test]
fn filter_segments_by_min_lod_exact_threshold() {
    let segments = vec![make_ancestry_segment("AFR", 3.0)];
    let filtered = filter_segments_by_min_lod(segments, 3.0);
    assert_eq!(filtered.len(), 1);
}

#[test]
fn filter_segments_by_min_lod_all_below() {
    let segments = vec![make_ancestry_segment("AFR", 1.0), make_ancestry_segment("EUR", 0.5)];
    let filtered = filter_segments_by_min_lod(segments, 5.0);
    assert!(filtered.is_empty());
}

#[test]
fn filter_segments_by_min_lod_negative_lod() {
    let segments = vec![make_ancestry_segment("AFR", -2.0), make_ancestry_segment("EUR", 5.0)];
    let filtered = filter_segments_by_min_lod(segments, 0.0);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].ancestry_name, "EUR");
}

// ── estimate_admixture_proportions ──────────────────────────────────────

#[test]
fn admixture_proportions_single_ancestry() {
    let segments = vec![AncestrySegment {
        chrom: "chr1".to_string(), start: 0, end: 100000,
        sample: "test".to_string(), ancestry_idx: 0, ancestry_name: "EUR".to_string(),
        n_windows: 10, mean_similarity: 0.95, mean_posterior: Some(0.95),
        discriminability: 0.1, lod_score: 5.0,
    }];
    let pop_names = vec!["EUR".to_string()];

    let result = estimate_admixture_proportions(&segments, "test", &pop_names);
    assert!((result.proportions["EUR"] - 1.0).abs() < 1e-6);
    assert_eq!(result.n_switches, 0);
}

#[test]
fn admixture_proportions_two_way_admixture() {
    let segments = vec![
        AncestrySegment {
            chrom: "chr1".to_string(), start: 0, end: 50000,
            sample: "test".to_string(), ancestry_idx: 0, ancestry_name: "AFR".to_string(),
            n_windows: 5, mean_similarity: 0.95, mean_posterior: Some(0.9),
            discriminability: 0.1, lod_score: 3.0,
        },
        AncestrySegment {
            chrom: "chr1".to_string(), start: 50000, end: 100000,
            sample: "test".to_string(), ancestry_idx: 1, ancestry_name: "EUR".to_string(),
            n_windows: 5, mean_similarity: 0.93, mean_posterior: Some(0.85),
            discriminability: 0.1, lod_score: 2.5,
        },
    ];
    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];

    let result = estimate_admixture_proportions(&segments, "test", &pop_names);
    assert!(result.proportions["AFR"] > 0.4);
    assert!(result.proportions["EUR"] > 0.4);
    let total: f64 = result.proportions.values().sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert_eq!(result.n_switches, 1);
}

#[test]
fn admixture_proportions_empty_segments() {
    let segments: Vec<AncestrySegment> = vec![];
    let pop_names = vec!["EUR".to_string()];

    let result = estimate_admixture_proportions(&segments, "test", &pop_names);
    assert_eq!(result.total_length_bp, 0);
}

// ── load_population_samples ─────────────────────────────────────────────

#[test]
fn load_population_samples_skips_comments() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pop.txt");
    fs::write(&path, "# comment\nHG00733\n# another comment\nNA19239\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples, vec!["HG00733", "NA19239"]);
}

#[test]
fn load_population_samples_trims_whitespace() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pop.txt");
    fs::write(&path, "  HG00733  \n  NA19239\t\n").unwrap();

    let samples = load_population_samples(&path).unwrap();
    assert_eq!(samples[0], "HG00733");
    assert_eq!(samples[1], "NA19239");
}

// ── load_populations_from_dir ───────────────────────────────────────────

#[test]
fn load_populations_from_dir_sorted_order() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("EUR.txt"), "sample1\nsample2\n").unwrap();
    fs::write(dir.path().join("AFR.txt"), "sample3\nsample4\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 2);
    assert_eq!(pops[0].name, "AFR");
    assert_eq!(pops[1].name, "EUR");
}

#[test]
fn load_populations_from_dir_ignores_non_txt() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("EUR.txt"), "sample1\n").unwrap();
    fs::write(dir.path().join("notes.md"), "not a pop file\n").unwrap();
    fs::write(dir.path().join("data.csv"), "also not\n").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "EUR");
}

#[test]
fn load_populations_from_dir_skips_empty_files() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("EUR.txt"), "sample1\n").unwrap();
    fs::write(dir.path().join("EMPTY.txt"), "").unwrap();

    let pops = load_populations_from_dir(dir.path()).unwrap();
    assert_eq!(pops.len(), 1);
    assert_eq!(pops[0].name, "EUR");
}

#[test]
fn load_populations_from_dir_nonexistent_errors() {
    assert!(load_populations_from_dir(std::path::Path::new("/no/such/dir")).is_err());
}
