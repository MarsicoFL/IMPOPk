//! Cycle 92 edge-case tests for parse_rfmix_msp (1 test-file ref before this).
//!
//! Also targets parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts
//! with additional boundary conditions.

use hprc_ancestry_cli::rfmix::{
    parse_rfmix_msp, parse_rfmix_msp_content, rfmix_to_windows, rfmix_window_starts, RfmixResult,
};

// ===================== parse_rfmix_msp (file-based) =====================

#[test]
fn parse_rfmix_msp_nonexistent_file() {
    let path = std::path::Path::new("/nonexistent/path.msp.tsv");
    let result = parse_rfmix_msp(path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

#[test]
fn parse_rfmix_msp_from_tmpfile() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0\tHG001.1
chr1\t1000\t5000\t0.00\t1.50\t50\t0\t1\n";
    let path = std::path::Path::new("/tmp/test_rfmix_cycle92.msp.tsv");
    std::fs::write(path, content).unwrap();
    let result = parse_rfmix_msp(path).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
    assert_eq!(result.segments.len(), 1);
    assert_eq!(result.haplotype_names, vec!["HG001.0", "HG001.1"]);
    std::fs::remove_file(path).ok();
}

#[test]
fn parse_rfmix_msp_empty_file() {
    let path = std::path::Path::new("/tmp/test_rfmix_empty_cycle92.msp.tsv");
    std::fs::write(path, "").unwrap();
    let result = parse_rfmix_msp(path);
    assert!(result.is_err());
    std::fs::remove_file(path).ok();
}

// ===================== parse_rfmix_msp_content edge cases =====================

#[test]
fn parse_rfmix_content_empty() {
    let result = parse_rfmix_msp_content("");
    assert!(result.is_err());
}

#[test]
fn parse_rfmix_content_missing_header() {
    let content = "not_a_header_line\n#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn parse_rfmix_content_missing_column_header() {
    let content = "#Subpopulation order/codes: AFR=0\tEUR=1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn parse_rfmix_content_three_populations() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0\tHG001.1
chr1\t1000\t5000\t0.00\t1.50\t50\t2\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR", "NAT"]);
    assert_eq!(result.segments[0].hap_ancestries, vec![2, 0]);
}

#[test]
fn parse_rfmix_content_whitespace_format() {
    let content = "\
#reference_panel_population:  AFR  EUR
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG001.0\tHG001.1
chr1\t1000\t5000\t0.00\t1.50\t50\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.population_names, vec!["AFR", "EUR"]);
}

#[test]
fn parse_rfmix_content_data_with_comments() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t1000\t2000\t0.00\t0.50\t20\t0\t0
# inline comment
chr1\t2000\t3000\t0.50\t1.00\t25\t1\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn parse_rfmix_content_empty_data_lines() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1

chr1\t1000\t2000\t0.00\t0.50\t20\t0\t1

\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.segments.len(), 1);
}

#[test]
fn parse_rfmix_content_too_few_data_fields() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t1000\t2000\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn parse_rfmix_content_invalid_position() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\tNOT_A_NUMBER\t2000\t0.00\t0.50\t20\t0\t1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err());
}

#[test]
fn parse_rfmix_content_noncontiguous_pop_indices() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t1000\t2000\t0.00\t0.50\t20\t0\t1\n";
    let result = parse_rfmix_msp_content(content);
    assert!(result.is_err()); // non-contiguous indices
}

#[test]
fn parse_rfmix_content_four_haplotypes() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tA.0\tA.1\tB.0\tB.1
chr1\t1000\t2000\t0.00\t0.50\t20\t0\t1\t1\t0\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    assert_eq!(result.haplotype_names.len(), 4);
    assert_eq!(result.segments[0].hap_ancestries, vec![0, 1, 1, 0]);
}

// ===================== rfmix_to_windows edge cases =====================

#[test]
fn rfmix_to_windows_empty_segments() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["H.0".to_string()],
        segments: vec![],
    };
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 1);
    assert!(windows[0].is_empty());
}

#[test]
fn rfmix_to_windows_zero_window_size() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t1000\t5000\t0.00\t1.50\t50\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    let windows = rfmix_to_windows(&result, 0);
    assert!(windows[0].is_empty());
}

#[test]
fn rfmix_to_windows_single_segment() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t0\t30000\t0.00\t1.00\t100\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    let windows = rfmix_to_windows(&result, 10000);
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[0].len(), 3); // 0-30k in 10k windows
    for w in &windows[0] {
        assert_eq!(*w, Some(0)); // AFR
    }
    for w in &windows[1] {
        assert_eq!(*w, Some(1)); // EUR
    }
}

// ===================== rfmix_window_starts edge cases =====================

#[test]
fn rfmix_window_starts_empty() {
    let result = RfmixResult {
        population_names: vec!["AFR".to_string()],
        haplotype_names: vec!["H.0".to_string()],
        segments: vec![],
    };
    let starts = rfmix_window_starts(&result, 10000);
    assert!(starts.is_empty());
}

#[test]
fn rfmix_window_starts_zero_window_size() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t1000\t5000\t0.00\t1.50\t50\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    let starts = rfmix_window_starts(&result, 0);
    assert!(starts.is_empty());
}

#[test]
fn rfmix_window_starts_correct_offsets() {
    let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tH.0\tH.1
chr1\t100000\t150000\t0.00\t1.50\t50\t0\t1\n";
    let result = parse_rfmix_msp_content(content).unwrap();
    let starts = rfmix_window_starts(&result, 10000);
    assert_eq!(starts[0], 100000);
    assert_eq!(starts[1], 110000);
}
