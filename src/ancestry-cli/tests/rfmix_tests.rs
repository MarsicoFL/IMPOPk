//! Edge case tests for the RFMix .msp.tsv parser module.
//!
//! Tests cover:
//! - Empty file → proper error
//! - 2-way ancestry (EUR/AFR) → correct parsing
//! - 3-way ancestry (EUR/AFR/NAT) → correct parsing
//! - Header parsing: both formats
//! - Coordinate conversion to window indices
//! - Single-segment chromosome
//! - Malformed data lines
//! - Window conversion edge cases

use impopk_ancestry_cli::rfmix::*;

// =============================================
// Empty and minimal inputs
// =============================================

#[test]
fn test_parse_empty_content() {
    let result = parse_rfmix_msp_content("");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Empty"));
}

#[test]
fn test_parse_only_population_header() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Missing column header should error");
}

#[test]
fn test_parse_headers_no_data() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
    assert_eq!(result.haplotype_names, vec!["HG00733.0", "HG00733.1"]);
    assert!(result.segments.is_empty());
}

// =============================================
// Population header parsing
// =============================================

#[test]
fn test_population_header_format1_two_way() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

#[test]
fn test_population_header_format1_three_way() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t2
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
    assert_eq!(result.segments[0].hap_ancestries, vec![0, 2]); // AFR, NAT
}

#[test]
fn test_population_header_format2() {
    let content = "\
#reference_panel_population:  AFR  EUR
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

#[test]
fn test_population_header_format1_reversed_indices() {
    // Indices out of order but contiguous
    let content = "\
#Subpopulation order/codes: EUR=1\tAFR=0
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    // Should sort by index: 0=AFR, 1=EUR
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

#[test]
fn test_population_header_non_contiguous_indices() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=2\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\n\
                   chr20\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Non-contiguous indices should error");
}

#[test]
fn test_population_header_no_hash() {
    let content = "Subpopulation order/codes: AFR=0\tEUR=1\n\
                   #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\n\
                   chr20\t100\t200\t0.0\t1.0\t10\t0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Missing # in population header should error");
}

// =============================================
// Column header parsing
// =============================================

#[test]
fn test_column_header_many_haplotypes() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS1.0\tS1.1\tS2.0\tS2.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1\t0\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.haplotype_names.len(), 4);
    assert_eq!(result.segments[0].hap_ancestries.len(), 4);
}

#[test]
fn test_column_header_single_haplotype() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0
chr20\t100\t200\t0.0\t1.0\t10\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.haplotype_names.len(), 1);
    assert_eq!(result.segments[0].hap_ancestries, vec![1]);
}

// =============================================
// Data line parsing edge cases
// =============================================

#[test]
fn test_single_segment_chromosome() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t82590\t64444167\t0.00\t80.05\t5000\t1\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.segments[0].start, 82590);
    assert_eq!(result.segments[0].end, 64444167);
    assert_eq!(result.segments[0].n_snps, 5000);
}

#[test]
fn test_malformed_data_line_too_few_fields() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Data line with too few fields should error");
}

#[test]
fn test_malformed_data_line_bad_position() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\tabc\t200\t0.0\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Invalid start position should error");
}

#[test]
fn test_malformed_data_line_bad_ancestry_index() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\tX\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Non-numeric ancestry index should error");
}

#[test]
fn test_data_with_blank_lines_and_comments() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1

# This is a comment in the middle
chr20\t200\t300\t1.0\t2.0\t20\t1\t0
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

// =============================================
// rfmix_to_windows edge cases
// =============================================

#[test]
fn test_to_windows_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 0,
            end: 1000,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let windows = rfmix_to_windows(&result, 0);
    assert!(windows[0].is_empty());
}

#[test]
fn test_to_windows_empty_segments() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![],
    };
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn test_to_windows_single_window_segment() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 0,
            end: 100,
            start_cm: 0.0,
            end_cm: 0.1,
            n_snps: 5,
            hap_ancestries: vec![1],
        }],
    };
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 1);
    assert_eq!(windows[0][0], Some(1)); // EUR
}

#[test]
fn test_to_windows_ancestry_switch() {
    // Two segments: first AFR, then EUR
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr20".into(),
                start: 0,
                end: 500,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 50,
                hap_ancestries: vec![0], // AFR
            },
            RfmixSegment {
                chrom: "chr20".into(),
                start: 500,
                end: 1000,
                start_cm: 1.0,
                end_cm: 2.0,
                n_snps: 50,
                hap_ancestries: vec![1], // EUR
            },
        ],
    };
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 10);

    // First 5 windows should be AFR
    for (i, w) in windows[0].iter().enumerate().take(5) {
        assert_eq!(*w, Some(0), "Window {} should be AFR", i);
    }
    // Last 5 windows should be EUR
    for (i, w) in windows[0].iter().enumerate().take(10).skip(5) {
        assert_eq!(*w, Some(1), "Window {} should be EUR", i);
    }
}

#[test]
fn test_to_windows_large_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr20".into(),
                start: 0,
                end: 500,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 50,
                hap_ancestries: vec![0],
            },
            RfmixSegment {
                chrom: "chr20".into(),
                start: 500,
                end: 1000,
                start_cm: 1.0,
                end_cm: 2.0,
                n_snps: 50,
                hap_ancestries: vec![1],
            },
        ],
    };
    // Window size = 1000 → single window covers both segments
    let windows = rfmix_to_windows(&result, 1000);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 1);
    // Last segment overwrites → EUR
    assert_eq!(windows[0][0], Some(1));
}

// =============================================
// rfmix_window_starts edge cases
// =============================================

#[test]
fn test_window_starts_empty() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![],
    };
    let starts = rfmix_window_starts(&result, 10000);
    assert!(starts.is_empty());
}

#[test]
fn test_window_starts_zero_window_size() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 0,
            end: 1000,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 0);
    assert!(starts.is_empty());
}

#[test]
fn test_window_starts_alignment() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 100,
            end: 500,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 100);
    // Range: 100 to 500 → 4 windows: [100,200), [200,300), [300,400), [400,500)
    assert_eq!(starts.len(), 4);
    assert_eq!(starts[0], 100);
    assert_eq!(starts[1], 200);
    assert_eq!(starts[2], 300);
    assert_eq!(starts[3], 400);
}

// =============================================
// 3-way ancestry parsing
// =============================================

#[test]
fn test_three_way_ancestry_full() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0\t1
chr20\t200\t300\t1.0\t2.0\t15\t2\t0
chr20\t300\t400\t2.0\t3.0\t20\t1\t2
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
    assert_eq!(result.segments.len(), 3);

    // Check all ancestry assignments
    assert_eq!(result.segments[0].hap_ancestries, vec![0, 1]); // AFR, EUR
    assert_eq!(result.segments[1].hap_ancestries, vec![2, 0]); // NAT, AFR
    assert_eq!(result.segments[2].hap_ancestries, vec![1, 2]); // EUR, NAT

    // Verify all ancestry indices are valid
    for seg in &result.segments {
        for &anc in &seg.hap_ancestries {
            assert!(anc < 3, "Ancestry index {} should be < 3", anc);
        }
    }
}

// =============================================
// Segment field validation
// =============================================

#[test]
fn test_genetic_map_coordinates_preserved() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t82590\t728330\t0.123\t2.678\t339\t1\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    let seg = &result.segments[0];
    assert!((seg.start_cm - 0.123).abs() < 1e-6);
    assert!((seg.end_cm - 2.678).abs() < 1e-6);
    assert_eq!(seg.n_snps, 339);
    assert_eq!(seg.chrom, "chr20");
}

#[test]
fn test_consecutive_segments_no_gap() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t100\t200\t0.0\t1.0\t10\t0
chr20\t200\t300\t1.0\t2.0\t10\t1
chr20\t300\t400\t2.0\t3.0\t10\t0
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 3);

    // Verify segments are contiguous
    for i in 1..result.segments.len() {
        assert_eq!(
            result.segments[i].start,
            result.segments[i - 1].end,
            "Segments should be contiguous"
        );
    }
}

// =============================================
// File-based parsing
// =============================================

#[test]
fn test_parse_nonexistent_file() {
    let path = std::path::Path::new("/nonexistent/rfmix.msp.tsv");
    let result = parse_rfmix_msp(path);
    assert!(result.is_err());
}

// =============================================
// Stress test
// =============================================

#[test]
fn test_many_segments() {
    let mut content = String::new();
    content.push_str("#Subpopulation order/codes: AFR=0\tEUR=1\n");
    content.push_str("#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0\tS.1\n");

    for i in 0..500 {
        let start = i * 1000;
        let end = (i + 1) * 1000;
        let ancestry0 = (i % 2) as usize;
        let ancestry1 = ((i + 1) % 2) as usize;
        content.push_str(&format!(
            "chr20\t{}\t{}\t{:.2}\t{:.2}\t{}\t{}\t{}\n",
            start,
            end,
            i as f64 * 0.01,
            (i + 1) as f64 * 0.01,
            10 + i,
            ancestry0,
            ancestry1,
        ));
    }

    let result = parse_rfmix_msp_content(&content).unwrap();
    assert_eq!(result.segments.len(), 500);
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

// =============================================
// Additional edge cases: malformed data lines
// =============================================

#[test]
fn test_malformed_data_line_bad_end_position() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\tnope\t0.0\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Invalid end position should error");
}

#[test]
fn test_malformed_data_line_bad_genetic_position() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\tnotafloat\t1.0\t10\t0\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Invalid genetic position should error");
}

#[test]
fn test_malformed_data_line_bad_nsnps() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t-5\t0\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Negative n_snps should error (u32 parse)");
}

#[test]
fn test_malformed_data_line_negative_ancestry_index() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t-1\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Negative ancestry index should error (usize parse)");
}

#[test]
fn test_malformed_data_line_float_ancestry_index() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t100\t200\t0.0\t1.0\t10\t0.5\t1
";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err(), "Float ancestry index should error (usize parse)");
}

// =============================================
// Window conversion with gaps between segments
// =============================================

#[test]
fn test_to_windows_gap_between_segments() {
    // Two segments with a gap: [100,200) and [500,600)
    // Gap region [200,500) should have None windows
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr20".into(),
                start: 100,
                end: 200,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 10,
                hap_ancestries: vec![0], // AFR
            },
            RfmixSegment {
                chrom: "chr20".into(),
                start: 500,
                end: 600,
                start_cm: 4.0,
                end_cm: 5.0,
                n_snps: 10,
                hap_ancestries: vec![1], // EUR
            },
        ],
    };
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1); // 1 haplotype

    // Range: 100 to 600 → 5 windows of 100bp
    assert_eq!(windows[0].len(), 5);

    // Window 0: [100,200) → AFR
    assert_eq!(windows[0][0], Some(0));
    // Windows 1-3: [200,500) → gap → None
    assert_eq!(windows[0][1], None);
    assert_eq!(windows[0][2], None);
    assert_eq!(windows[0][3], None);
    // Window 4: [500,600) → EUR
    assert_eq!(windows[0][4], Some(1));
}

#[test]
fn test_to_windows_overlapping_segments() {
    // Two overlapping segments — later one should overwrite
    let result = RfmixResult {
        population_names: vec!["AFR".to_string(), "EUR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![
            RfmixSegment {
                chrom: "chr20".into(),
                start: 0,
                end: 500,
                start_cm: 0.0,
                end_cm: 1.0,
                n_snps: 50,
                hap_ancestries: vec![0], // AFR
            },
            RfmixSegment {
                chrom: "chr20".into(),
                start: 300,
                end: 800,
                start_cm: 0.6,
                end_cm: 1.6,
                n_snps: 50,
                hap_ancestries: vec![1], // EUR
            },
        ],
    };
    let windows = rfmix_to_windows(&result, 100);
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].len(), 8); // range 0..800

    // Windows 0-2 are covered by first segment (AFR) then overwritten by second (EUR) for 3-4
    // Window 0,1,2: initially AFR from seg0
    // Window 3,4: overwritten by seg1 (EUR)
    // Window 5,6,7: only seg1 (EUR)
    assert_eq!(windows[0][0], Some(0)); // seg0 only
    assert_eq!(windows[0][3], Some(1)); // overlap → seg1 overwrites
    assert_eq!(windows[0][7], Some(1)); // seg1 only
}

// =============================================
// Population header: additional edge cases
// =============================================

#[test]
fn test_population_header_spaces_in_format1() {
    // Some RFMix versions use spaces instead of tabs between assignments
    let content = "\
#Subpopulation order/codes: AFR=0 EUR=1 NAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t100\t200\t0.0\t1.0\t10\t2
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
    assert_eq!(result.segments[0].hap_ancestries, vec![2]); // NAT
}

#[test]
fn test_population_header_single_population() {
    let content = "\
#reference_panel_population: SINGLE
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t100\t200\t0.0\t1.0\t10\t0
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["SINGLE"]);
}

// =============================================
// Scientific notation and edge float values
// =============================================

#[test]
fn test_scientific_notation_genetic_positions() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t100\t200\t1.5e-2\t3.0e1\t10\t0
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert!((result.segments[0].start_cm - 0.015).abs() < 1e-9);
    assert!((result.segments[0].end_cm - 30.0).abs() < 1e-9);
}

#[test]
fn test_zero_genetic_positions() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t100\t200\t0.0\t0.0\t10\t0
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert!((result.segments[0].start_cm).abs() < 1e-9);
    assert!((result.segments[0].end_cm).abs() < 1e-9);
}

// =============================================
// Window starts with non-aligned ranges
// =============================================

#[test]
fn test_window_starts_non_aligned_range() {
    // Segments from 150 to 550 with window_size=200
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 150,
            end: 550,
            start_cm: 0.0,
            end_cm: 1.0,
            n_snps: 10,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 200);
    // Range: 150 to 550 = 400bp → ceil(400/200) = 2 windows
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0], 150);
    assert_eq!(starts[1], 350);
}

#[test]
fn test_window_starts_single_bp_range() {
    // Segment from 100 to 101 — 1bp range
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["S.0".to_string()],
        segments: vec![RfmixSegment {
            chrom: "chr20".into(),
            start: 100,
            end: 101,
            start_cm: 0.0,
            end_cm: 0.001,
            n_snps: 1,
            hap_ancestries: vec![0],
        }],
    };
    let starts = rfmix_window_starts(&result, 10000);
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0], 100);
}

// =============================================
// Multiple samples (4+ haplotypes)
// =============================================

#[test]
fn test_three_way_four_haplotypes_windows() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS1.0\tS1.1\tS2.0\tS2.1
chr20\t0\t1000\t0.0\t1.0\t100\t0\t1\t2\t0
chr20\t1000\t2000\t1.0\t2.0\t100\t1\t2\t0\t1
";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
    assert_eq!(result.haplotype_names.len(), 4);

    let windows = rfmix_to_windows(&result, 1000);
    assert_eq!(windows.len(), 4);

    // Window 0 (0-1000): S1.0=AFR, S1.1=EUR, S2.0=NAT, S2.1=AFR
    assert_eq!(windows[0][0], Some(0));
    assert_eq!(windows[1][0], Some(1));
    assert_eq!(windows[2][0], Some(2));
    assert_eq!(windows[3][0], Some(0));

    // Window 1 (1000-2000): S1.0=EUR, S1.1=NAT, S2.0=AFR, S2.1=EUR
    assert_eq!(windows[0][1], Some(1));
    assert_eq!(windows[1][1], Some(2));
    assert_eq!(windows[2][1], Some(0));
    assert_eq!(windows[3][1], Some(1));
}
