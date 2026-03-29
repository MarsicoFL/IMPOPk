//! Edge case tests for parsing, temperature/switch estimation, and rfmix windowing.
//!
//! Covers untested paths:
//! - rfmix: non-contiguous indices, space-separated format1 assignments, window_size=0,
//!   rfmix_window_starts with identical start/end, extra fields in data lines
//! - ancestry: parse_header with reordered columns, parse_similarity_data with NaN identity
//! - hmm: estimate_temperature with empty/constant/single obs, estimate_switch_prob fallback
//! - smooth_states: property tests, idempotency, exact min_run boundary

use hprc_ancestry_cli::rfmix::*;
use hprc_ancestry_cli::ancestry::*;
use hprc_ancestry_cli::hmm::{AncestralPopulation, AncestryObservation,
    estimate_temperature, estimate_switch_prob};

// =============================================
// rfmix: parse_population_header edge cases
// =============================================

#[test]
fn test_population_header_format1_spaces_between_assignments() {
    // Format 1 with space-separated (instead of tab) assignments — should still work
    let content = "#Subpopulation order/codes: AFR=0 EUR=1 NAT=2\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\tHG.1\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
}

#[test]
fn test_population_header_non_contiguous_indices_error() {
    // Index gap: 0, 2 (missing 1)
    let content = "#Subpopulation order/codes: AFR=0\tEUR=2\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Non-contiguous indices should fail");
    assert!(result.unwrap_err().contains("not contiguous"));
}

#[test]
fn test_population_header_reversed_indices() {
    // Indices in reverse order — should be sorted correctly
    let content = "#Subpopulation order/codes: EUR=1\tAFR=0\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]); // sorted by index
}

#[test]
fn test_population_header_format1_no_colon_error() {
    // Format 1 (has '=') but no colon separator
    let content = "#AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content);
    // splitn(2, ':') → parts.len()=1 → error
    assert!(result.is_err());
}

#[test]
fn test_population_header_format2_many_populations() {
    // Format 2 with many populations
    let content = "#reference_panel_population: AFR EUR EAS SAS AMR\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names.len(), 5);
    assert_eq!(result.population_names[0], "AFR");
    assert_eq!(result.population_names[4], "AMR");
}

#[test]
fn test_population_header_invalid_index_error() {
    let content = "#Subpopulation order/codes: AFR=abc\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid population index"));
}

// =============================================
// rfmix: data line parsing edge cases
// =============================================

#[test]
fn test_data_line_too_few_fields() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\tHG.1\n\
                   chr1\t100\t200\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn test_data_line_invalid_position() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\tHG.1\n\
                   chr1\tNaN\t200\t0.0\t1.0\t10\t0\t1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn test_data_line_invalid_ancestry_index() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\tHG.1\n\
                   chr1\t100\t200\t0.0\t1.0\t10\tabc\t1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn test_data_lines_with_comments_and_blank_lines() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\tHG.1\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\t1\n\
                   \n\
                   # some comment\n\
                   chr1\t200\t300\t1.0\t2.0\t15\t1\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn test_data_line_extra_trailing_fields() {
    let content = "#Subpopulation order/codes: AFR=0\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG.0\n\
                   chr1\t100\t200\t0.0\t1.0\t10\t0\textra\tmore\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.segments[0].hap_ancestries, vec![0]);
}

// =============================================
// rfmix: rfmix_to_windows / rfmix_window_starts edge cases
// =============================================

#[test]
fn test_rfmix_to_windows_window_size_zero() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 100, end: 200,
            start_cm: 0.0, end_cm: 1.0, n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 0);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn test_rfmix_window_starts_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 100, end: 200,
            start_cm: 0.0, end_cm: 1.0, n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 0);
    assert!(starts.is_empty());
}

#[test]
fn test_rfmix_to_windows_same_start_end() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 100, end: 100,
            start_cm: 0.0, end_cm: 0.0, n_snps: 0,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 10);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn test_rfmix_to_windows_large_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 100, end: 200,
            start_cm: 0.0, end_cm: 1.0, n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 1_000_000);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 1);
    assert_eq!(windows[0][0], Some(0));
}

#[test]
fn test_rfmix_to_windows_three_haplotypes() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string(), "NAT".to_string()],
        haplotype_names: vec!["HG.0".to_string(), "HG.1".to_string(), "NA.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 0, end: 30000,
            start_cm: 0.0, end_cm: 1.0, n_snps: 100,
            hap_ancestries: vec![0, 1, 2],
        }],
    };
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 3);
    assert_eq!(windows[0][0], Some(0));
    assert_eq!(windows[1][0], Some(1));
    assert_eq!(windows[2][0], Some(2));
}

#[test]
fn test_rfmix_window_starts_alignment() {
    // Verify window starts are properly aligned
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["HG.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr1".to_string(), start: 1000, end: 5500,
            start_cm: 0.0, end_cm: 1.0, n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 1000);
    // range = 4500, div_ceil(4500, 1000) = 5 windows
    assert_eq!(starts.len(), 5);
    assert_eq!(starts[0], 1000);
    assert_eq!(starts[4], 5000);
}

// =============================================
// ancestry: parse_similarity_data edge cases
// =============================================

#[test]
fn test_parse_similarity_data_no_header() {
    let lines: Vec<String> = vec![];
    let query = vec!["Q#1".to_string()];
    let refs = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_parse_similarity_data_reordered_columns() {
    let header = "group.b\tgroup.a\tend\tstart\testimated.identity\tchrom";
    let line = "R#1#s:0-10\tQ#1#s:0-10\t10000\t0\t0.95\tchr1";
    let lines = vec![header.to_string(), line.to_string()];
    let query = vec!["Q#1".to_string()];
    let refs = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result["Q#1"][0].similarities["R#1"] - 0.95).abs() < 1e-6);
}

#[test]
fn test_parse_similarity_data_nonexistent_similarity_column() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\tjaccard.similarity";
    let lines = vec![header.to_string()];
    let query = vec!["Q#1".to_string()];
    let refs = vec!["R#1".to_string()];
    // Requesting "estimated.identity" but only "jaccard.similarity" exists
    let result = parse_similarity_data(lines.into_iter(), &query, &refs);
    assert!(result.is_err());
}

#[test]
fn test_parse_similarity_data_multiple_queries() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let line1 = "chr1\t0\t10000\tQ1#1#s:0-10\tR#1#s:0-10\t0.95";
    let line2 = "chr1\t0\t10000\tQ2#1#s:0-10\tR#1#s:0-10\t0.88";
    let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
    let query = vec!["Q1#1".to_string(), "Q2#1".to_string()];
    let refs = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    assert_eq!(result.len(), 2);
    assert!(result.contains_key("Q1#1"));
    assert!(result.contains_key("Q2#1"));
}

#[test]
fn test_parse_similarity_data_multi_chromosome() {
    let header = "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity";
    let line1 = "chr1\t0\t10000\tQ#1#s:0-10\tR#1#s:0-10\t0.95";
    let line2 = "chr2\t0\t10000\tQ#1#s:0-10\tR#1#s:0-10\t0.88";
    let lines = vec![header.to_string(), line1.to_string(), line2.to_string()];
    let query = vec!["Q#1".to_string()];
    let refs = vec!["R#1".to_string()];
    let result = parse_similarity_data(lines.into_iter(), &query, &refs).unwrap();
    let obs = &result["Q#1"];
    assert_eq!(obs.len(), 2);
    // Sorted by (chrom, start)
    assert_eq!(obs[0].chrom, "chr1");
    assert_eq!(obs[1].chrom, "chr2");
}

// =============================================
// ancestry: coverage_ratio edge cases
// =============================================

#[test]
fn test_coverage_ratio_one_zero() {
    assert_eq!(coverage_ratio(0, 100), 0.0);
    assert_eq!(coverage_ratio(100, 0), 0.0);
}

#[test]
fn test_coverage_ratio_large_values() {
    let ratio = coverage_ratio(1_000_000_000, 2_000_000_000);
    assert!((ratio - 0.5).abs() < 1e-10);
}

// =============================================
// hmm: estimate_temperature edge cases
// =============================================

#[test]
fn test_estimate_temperature_empty_observations() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let temp = estimate_temperature(&[], &pops);
    assert!((temp - 0.03).abs() < 1e-10, "Empty obs should return fallback 0.03, got {}", temp);
}

#[test]
fn test_estimate_temperature_single_population_in_obs() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(), start: 0, end: 5000,
        sample: "test".to_string(),
        similarities: [("A#1".to_string(), 0.95)].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    assert!((temp - 0.03).abs() < 1e-10, "Should return fallback, got {}", temp);
}

#[test]
fn test_estimate_temperature_constant_sims() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(), start: 0, end: 5000,
        sample: "test".to_string(),
        similarities: [
            ("A#1".to_string(), 0.90),
            ("B#1".to_string(), 0.90),
        ].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    assert!((temp - 0.03).abs() < 1e-10, "Equal sims should give fallback, got {}", temp);
}

#[test]
fn test_estimate_temperature_large_diff_clamped() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(), start: 0, end: 5000,
        sample: "test".to_string(),
        similarities: [
            ("A#1".to_string(), 0.95),
            ("B#1".to_string(), 0.50),
        ].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    assert!((temp - 0.15).abs() < 1e-10, "Large diff should clamp to 0.15, got {}", temp);
}

#[test]
fn test_estimate_temperature_small_diff_clamped() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(), start: 0, end: 5000,
        sample: "test".to_string(),
        similarities: [
            ("A#1".to_string(), 0.9001),
            ("B#1".to_string(), 0.9000),
        ].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    // Single observation: upper-Q mean of [0.0001] = 0.0001 → clamped to 0.0005
    assert!((temp - 0.0005).abs() < 1e-10, "Tiny diff should clamp to 0.0005, got {}", temp);
}

#[test]
fn test_estimate_temperature_normal_range() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![AncestryObservation {
        chrom: "chr1".to_string(), start: 0, end: 5000,
        sample: "test".to_string(),
        similarities: [
            ("A#1".to_string(), 0.95),
            ("B#1".to_string(), 0.90),
        ].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }];
    let temp = estimate_temperature(&obs, &pops);
    // Single observation: upper-Q mean = 0.05 (passes through)
    assert!((temp - 0.05).abs() < 1e-10, "Normal diff should pass through, got {}", temp);
}

#[test]
fn test_estimate_temperature_median_selection() {
    // With multiple observations, temperature should be median of diffs
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs = vec![
        AncestryObservation {
            chrom: "chr1".to_string(), start: 0, end: 5000,
            sample: "test".to_string(),
            similarities: [("A#1".to_string(), 0.95), ("B#1".to_string(), 0.90)].into(), // diff=0.05
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        },
        AncestryObservation {
            chrom: "chr1".to_string(), start: 5000, end: 10000,
            sample: "test".to_string(),
            similarities: [("A#1".to_string(), 0.92), ("B#1".to_string(), 0.85)].into(), // diff=0.07
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        },
        AncestryObservation {
            chrom: "chr1".to_string(), start: 10000, end: 15000,
            sample: "test".to_string(),
            similarities: [("A#1".to_string(), 0.98), ("B#1".to_string(), 0.88)].into(), // diff=0.10
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        },
    ];
    let temp = estimate_temperature(&obs, &pops);
    // Sorted diffs: [0.05, 0.07, 0.10], upper-Q-mean (above P75) = 0.10
    assert!((temp - 0.10).abs() < 1e-10, "Upper-Q mean should be 0.10, got {}", temp);
}

// =============================================
// hmm: estimate_switch_prob edge cases
// =============================================

#[test]
fn test_estimate_switch_prob_few_observations() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs: Vec<AncestryObservation> = (0..5).map(|i| AncestryObservation {
        chrom: "chr1".to_string(), start: i * 1000, end: (i + 1) * 1000,
        sample: "test".to_string(),
        similarities: [("A#1".to_string(), 0.95), ("B#1".to_string(), 0.80)].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }).collect();
    let prob = estimate_switch_prob(&obs, &pops, 0.05);
    assert!((prob - 0.001).abs() < 1e-10, "Few obs should return fallback 0.001, got {}", prob);
}

#[test]
fn test_estimate_switch_prob_no_switches() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs: Vec<AncestryObservation> = (0..20).map(|i| AncestryObservation {
        chrom: "chr1".to_string(), start: i * 1000, end: (i + 1) * 1000,
        sample: "test".to_string(),
        similarities: [("A#1".to_string(), 0.99), ("B#1".to_string(), 0.50)].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }).collect();
    let prob = estimate_switch_prob(&obs, &pops, 0.05);
    assert!(prob >= 0.0001, "Should be >= 0.0001, got {}", prob);
    assert!(prob <= 0.01, "No switches should give low rate, got {}", prob);
}

#[test]
fn test_estimate_switch_prob_clamped_range() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["A#1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["B#1".to_string()] },
    ];
    let obs: Vec<AncestryObservation> = (0..50).map(|i| AncestryObservation {
        chrom: "chr1".to_string(), start: i * 1000, end: (i + 1) * 1000,
        sample: "test".to_string(),
        similarities: [
            ("A#1".to_string(), if i % 2 == 0 { 0.95 } else { 0.70 }),
            ("B#1".to_string(), if i % 2 == 0 { 0.70 } else { 0.95 }),
        ].into(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }).collect();
    let prob = estimate_switch_prob(&obs, &pops, 0.05);
    assert!(prob >= 0.0001, "Should be >= 0.0001, got {}", prob);
    assert!(prob <= 0.05, "Should be <= 0.05, got {}", prob);
}

// =============================================
// ancestry: smooth_states property tests
// =============================================

#[test]
fn test_smooth_states_preserves_length() {
    let states = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed.len(), states.len());
}

#[test]
fn test_smooth_states_idempotent() {
    let states = vec![0, 0, 0, 0, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states);
    let double_smoothed = smooth_states(&smoothed, 3);
    assert_eq!(double_smoothed, smoothed);
}

#[test]
fn test_smooth_states_exact_min_run_not_smoothed() {
    let states = vec![0, 0, 0, 1, 1, 1, 0, 0, 0];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states);
}

#[test]
fn test_smooth_states_consecutive_short_runs() {
    // Two consecutive short runs between same state
    let states = vec![0, 0, 0, 1, 0, 0, 0]; // single 1 between 0s
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, vec![0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn test_smooth_states_three_states() {
    // Short run of state 1 between state 0 and state 2 — should NOT be smoothed
    // (neighbors differ)
    let states = vec![0, 0, 0, 1, 2, 2, 2];
    let smoothed = smooth_states(&states, 3);
    assert_eq!(smoothed, states); // unchanged since neighbors differ
}

