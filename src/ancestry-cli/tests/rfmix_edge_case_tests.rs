//! Edge case and robustness tests for RFMix parser and windowing functions.
//!
//! Tests parse_population_header format variations, malformed data,
//! rfmix_to_windows boundary conditions, and rfmix_window_starts edge cases.

use impopk_ancestry_cli::rfmix::{
    parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts, RfmixResult, RfmixSegment,
};

// ── Population header parsing edge cases ────────────────────────────────

#[test]
fn rfmix_population_header_reversed_indices() {
    // Indices out of order: EUR=1, AFR=0 (reversed in listing)
    let content = "\
#Subpopulation order/codes: EUR=1\tAFR=0
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t0";

    let result = parse_rfmix_msp_content(content).unwrap();
    // Sorted by index: AFR=0, EUR=1
    assert_eq!(result.population_names[0], "AFR");
    assert_eq!(result.population_names[1], "EUR");
}

#[test]
fn rfmix_population_header_five_way() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tEAS=2\tCSA=3\tAMR=4
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t3";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names.len(), 5);
    assert_eq!(result.population_names[4], "AMR");
    assert_eq!(result.segments[0].hap_ancestries, vec![3]); // CSA
}

#[test]
fn rfmix_population_header_format2_whitespace() {
    // Format 2: whitespace-separated names after colon
    let content = "\
#reference_panel_population:   AFR   EUR   NAT
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t2";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
}

#[test]
fn rfmix_population_header_space_separated_format1() {
    // Some RFMix versions use spaces instead of tabs between assignments
    let content = "\
#Subpopulation order/codes: AFR=0 EUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

#[test]
fn rfmix_missing_population_header_errors() {
    // Empty content
    assert!(parse_rfmix_msp_content("").is_err());
}

#[test]
fn rfmix_missing_column_header_errors() {
    // Only population header, no column header
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1";
    assert!(parse_rfmix_msp_content(content).is_err());
}

#[test]
fn rfmix_population_header_non_contiguous_indices_errors() {
    // Indices 0, 2 (skipping 1)
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t0";

    assert!(parse_rfmix_msp_content(content).is_err());
}

#[test]
fn rfmix_population_header_no_hash_errors() {
    let content = "\
Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSAMPLE.0
chr1\t100\t200\t0.0\t1.0\t10\t0";

    assert!(parse_rfmix_msp_content(content).is_err());
}

// ── Column header edge cases ────────────────────────────────────────────

#[test]
fn rfmix_column_header_many_haplotypes() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS1.0\tS1.1\tS2.0\tS2.1
chr1\t100\t200\t0.0\t1.0\t10\t0\t1\t1\t0";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.haplotype_names.len(), 4);
    assert_eq!(result.segments[0].hap_ancestries.len(), 4);
}

#[test]
fn rfmix_column_header_too_few_fields_errors() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos";

    assert!(parse_rfmix_msp_content(content).is_err());
}

// ── Data line edge cases ────────────────────────────────────────────────

#[test]
fn rfmix_data_line_too_few_fields_errors() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0\tS.1
chr1\t100\t200\t0.0\t1.0";

    assert!(parse_rfmix_msp_content(content).is_err());
}

#[test]
fn rfmix_data_line_invalid_position_errors() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\tabc\t200\t0.0\t1.0\t10\t0";

    assert!(parse_rfmix_msp_content(content).is_err());
}

#[test]
fn rfmix_data_line_invalid_ancestry_index_errors() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t100\t200\t0.0\t1.0\t10\tXX";

    assert!(parse_rfmix_msp_content(content).is_err());
}

#[test]
fn rfmix_skips_comments_in_data_lines() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t100\t200\t0.0\t1.0\t10\t0
# This is a comment
chr1\t200\t300\t1.0\t2.0\t15\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn rfmix_skips_blank_lines_in_data() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t100\t200\t0.0\t1.0\t10\t0

chr1\t200\t300\t1.0\t2.0\t15\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn rfmix_no_data_lines_ok() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0\tS.1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 0);
    assert_eq!(result.population_names.len(), 2);
    assert_eq!(result.haplotype_names.len(), 2);
}

// ── rfmix_to_windows edge cases ─────────────────────────────────────────

#[test]
fn rfmix_to_windows_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["A".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![RfmixSegment {
            chrom: "chr1".into(),
            start: 100,
            end: 200,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };

    let windows = rfmix_to_windows(&result, 0);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn rfmix_to_windows_single_bp_segment() {
    let result = RfmixResult {
        population_names: vec!["A".into(), "B".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![RfmixSegment {
            chrom: "chr1".into(),
            start: 100,
            end: 101,
            start_cm: 0.0,
            end_cm: 0.01,
            n_snps: 1,
            hap_ancestries: vec![1],
        }],
    };

    let windows = rfmix_to_windows(&result, 10);
    assert_eq!(windows.len(), 1);
    // Only 1 window needed (1 bp range / 10 bp window = 1)
    assert!(windows[0].len() >= 1);
    assert_eq!(windows[0][0], Some(1));
}

#[test]
fn rfmix_to_windows_large_window_covers_all_segments() {
    let result = RfmixResult {
        population_names: vec!["AFR".into(), "EUR".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![
            RfmixSegment {
                chrom: "chr1".into(),
                start: 1000,
                end: 2000,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 50,
                hap_ancestries: vec![0],
            },
            RfmixSegment {
                chrom: "chr1".into(),
                start: 2000,
                end: 3000,
                start_cm: 1.0,
                end_cm: 2.0,
                n_snps: 50,
                hap_ancestries: vec![1],
            },
        ],
    };

    // Window larger than entire range → single window, last segment wins
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 1);
    // The second segment overwrites the first since they share the same single window
    assert!(windows[0][0].is_some());
}

#[test]
fn rfmix_to_windows_consistency_with_window_starts() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t0\t50000\t0.0\t1.0\t100\t0
chr1\t50000\t100000\t1.0\t2.0\t100\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    let ws = 10000;
    let windows = rfmix_to_windows(&result, ws);
    let starts = rfmix_window_starts(&result, ws);

    // Number of windows should match
    assert_eq!(windows[0].len(), starts.len());

    // Starts should be evenly spaced
    for i in 1..starts.len() {
        assert_eq!(starts[i] - starts[i - 1], ws);
    }
}

#[test]
fn rfmix_to_windows_max_pos_equals_min_pos() {
    // A segment where start == end
    let result = RfmixResult {
        population_names: vec!["A".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![RfmixSegment {
            chrom: "chr1".into(),
            start: 500,
            end: 500,
            start_cm: 0.0,
            end_cm: 0.0,
            n_snps: 0,
            hap_ancestries: vec![0],
        }],
    };

    // max_pos <= min_pos → empty windows
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

// ── rfmix_window_starts edge cases ──────────────────────────────────────

#[test]
fn rfmix_window_starts_empty_segments() {
    let result = RfmixResult {
        population_names: vec!["A".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![],
    };
    assert!(rfmix_window_starts(&result, 10000).is_empty());
}

#[test]
fn rfmix_window_starts_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["A".into()],
        haplotype_names: vec!["S.0".into()],
        segments: vec![RfmixSegment {
            chrom: "chr1".into(),
            start: 100,
            end: 200,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    assert!(rfmix_window_starts(&result, 0).is_empty());
}

#[test]
fn rfmix_window_starts_first_equals_min_pos() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t82590\t100000\t0.0\t1.0\t50\t0";

    let result = parse_rfmix_msp_content(content).unwrap();
    let starts = rfmix_window_starts(&result, 10000);
    assert_eq!(starts[0], 82590);
}

// ── Segment field preservation ──────────────────────────────────────────

#[test]
fn rfmix_segment_preserves_genetic_map_positions() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t82590\t728330\t0.123\t2.678\t339\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    let seg = &result.segments[0];
    assert!((seg.start_cm - 0.123).abs() < 1e-6);
    assert!((seg.end_cm - 2.678).abs() < 1e-6);
    assert_eq!(seg.n_snps, 339);
}

#[test]
fn rfmix_segment_multiple_chromosomes() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr1\t100\t200\t0.0\t1.0\t10\t0
chr20\t500\t600\t1.5\t2.5\t20\t1";

    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.segments[0].chrom, "chr1");
    assert_eq!(result.segments[1].chrom, "chr20");
}

#[test]
fn rfmix_three_haplotype_ancestry_independence() {
    // Verify each haplotype gets independent ancestry calls
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS1.0\tS1.1\tS2.0
chr1\t100\t200\t0.0\t1.0\t10\t0\t1\t2";

    let result = parse_rfmix_msp_content(content).unwrap();
    let seg = &result.segments[0];
    assert_eq!(seg.hap_ancestries[0], 0); // AFR
    assert_eq!(seg.hap_ancestries[1], 1); // EUR
    assert_eq!(seg.hap_ancestries[2], 2); // NAT
}
