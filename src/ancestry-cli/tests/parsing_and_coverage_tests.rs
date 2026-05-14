//! Tests for parse_similarity_data() (wrapper),
//! parse_similarity_data_with_coverage edge cases, and coverage_ratio
//! boundary conditions.

use impopk_ancestry_cli::ancestry::{
    parse_similarity_data, parse_similarity_data_with_coverage,
    coverage_ratio,
};

// ============================================================================
// parse_similarity_data tests (wrapper around parse_similarity_data_column)
// ============================================================================

fn make_similarity_lines(rows: &[&str]) -> Vec<String> {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let mut lines = vec![header.to_string()];
    for row in rows {
        lines.push(row.to_string());
    }
    lines
}

#[test]
fn test_parse_similarity_data_basic() {
    let lines = make_similarity_lines(&[
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_A#1#scaffold:0-5000\t0.95",
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_B#1#scaffold:0-5000\t0.85",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string(), "REF_B#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert!(result.contains_key("QUERY#1"));
    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].similarities.len(), 2);
    assert!((obs[0].similarities["REF_A#1"] - 0.95).abs() < 1e-10);
    assert!((obs[0].similarities["REF_B#1"] - 0.85).abs() < 1e-10);
}

#[test]
fn test_parse_similarity_data_skips_non_query_ref() {
    let lines = make_similarity_lines(&[
        "chr1\t0\t5000\tREF_A#1#scaffold:0-5000\tREF_B#1#scaffold:0-5000\t0.95",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string(), "REF_B#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert!(result.is_empty(), "ref-vs-ref comparison should be skipped");
}

#[test]
fn test_parse_similarity_data_reversed_order() {
    // group.b is query, group.a is ref — should still work
    let lines = make_similarity_lines(&[
        "chr1\t0\t5000\tREF_A#1#scaffold:0-5000\tQUERY#1#scaffold:0-5000\t0.95",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert!(result.contains_key("QUERY#1"));
}

#[test]
fn test_parse_similarity_data_empty_lines() {
    let lines = vec!["chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity".to_string()];
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_parse_similarity_data_multiple_windows_sorted() {
    let lines = make_similarity_lines(&[
        "chr1\t5000\t10000\tQUERY#1#scaffold:5000-10000\tREF_A#1#scaffold:5000-10000\t0.90",
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_A#1#scaffold:0-5000\t0.95",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 2);
    // Should be sorted by position
    assert!(obs[0].start <= obs[1].start);
}

#[test]
fn test_parse_similarity_data_max_identity_on_duplicates() {
    // Two lines for same query-ref-window: should keep max identity
    let lines = make_similarity_lines(&[
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_A#1#scaffold:0-5000\t0.85",
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_A#1#scaffold:0-5000\t0.95",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 1);
    assert!((obs[0].similarities["REF_A#1"] - 0.95).abs() < 1e-10);
}

// ============================================================================
// parse_similarity_data_with_coverage tests
// ============================================================================

fn make_coverage_lines(rows: &[&str]) -> Vec<String> {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\testimated.identity";
    let mut lines = vec![header.to_string()];
    for row in rows {
        lines.push(row.to_string());
    }
    lines
}

#[test]
fn test_parse_with_coverage_basic() {
    let lines = make_coverage_lines(&[
        "chr1\t0\t5000\tQUERY#1#scaffold:0-5000\tREF_A#1#scaffold:0-5000\t4500\t5000\t0.95",
    ]);
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &query,
        &refs,
        "estimated.identity",
    )
    .unwrap();
    assert!(result.contains_key("QUERY#1"));
    let obs = &result["QUERY#1"];
    assert_eq!(obs.len(), 1);
    // Coverage ratios should be present
    assert!(obs[0].coverage_ratios.is_some());
    let ratios = obs[0].coverage_ratios.as_ref().unwrap();
    // coverage_ratio(4500, 5000) = 4500/5000 = 0.9
    assert!((ratios["REF_A#1"] - 0.9).abs() < 1e-6);
}

#[test]
fn test_parse_with_coverage_empty() {
    let lines = vec![
        "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\testimated.identity"
            .to_string(),
    ];
    let query = vec!["QUERY#1".to_string()];
    let refs = vec!["REF_A#1".to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &query,
        &refs,
        "estimated.identity",
    )
    .unwrap();
    assert!(result.is_empty());
}

// ============================================================================
// coverage_ratio additional edge cases
// ============================================================================

#[test]
fn test_coverage_ratio_u64_max() {
    // Very large values should not overflow
    let r = coverage_ratio(u64::MAX / 2, u64::MAX);
    assert!(r > 0.0 && r <= 1.0);
    assert!((r - 0.5).abs() < 0.01);
}

#[test]
fn test_coverage_ratio_one_vs_one() {
    assert!((coverage_ratio(1, 1) - 1.0).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_one_vs_large() {
    let r = coverage_ratio(1, 1_000_000);
    assert!(r > 0.0);
    assert!(r < 0.001);
}
