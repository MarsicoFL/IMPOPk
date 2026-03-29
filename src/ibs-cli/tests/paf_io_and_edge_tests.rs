//! Edge case tests for ibs-cli PAF I/O functions and parsing edge cases.
//!
//! Covers:
//! - read_paf_alignments: file not found, empty file, gzipped, sorting, subset filter, min_aligned_length
//! - read_paf_alignments_multi: multi-chrom, empty chroms, no matching chroms, subset filter
//! - extract_sample_from_hap: edge cases (empty, multi-hash, no hash)
//! - extract_hap_id: additional edge cases
//! - extract_target_chrom: additional edge cases
//! - parse_paf_line: malformed lines, edge field values
//! - parse_cigar_mismatches: edge cases (M ops, unknown ops, large counts)
//! - compute_window_pairwise: empty input, single haplotype, same-sample, zero-length window

use hprc_ibs::paf::{
    compute_window_pairwise, extract_hap_id, extract_sample_from_hap, extract_target_chrom,
    parse_cigar_mismatches, parse_paf_line, read_paf_alignments, read_paf_alignments_multi,
    PafAlignment,
};
use std::collections::HashSet;
use std::io::Write;

// -----------------------------------------------------------------------
// Helper
// -----------------------------------------------------------------------

fn make_alignment(hap: &str, start: u64, end: u64, mismatches: Vec<u64>) -> PafAlignment {
    PafAlignment {
        hap_id: hap.to_string(),
        target_start: start,
        target_end: end,
        gap_identity: 1.0 - (mismatches.len() as f64 / (end - start).max(1) as f64),
        mismatch_positions: mismatches,
        aligned_bases: end - start,
    }
}

/// Build a minimal valid PAF line targeting the given chrom, from given hap.
fn paf_line(hap: &str, chrom: &str, tstart: u64, tend: u64, cigar: &str) -> String {
    let block_len = tend - tstart;
    format!(
        "{hap}#contig.1\t{block_len}\t0\t{block_len}\t+\tCHM13#0#{chrom}\t248387328\t{tstart}\t{tend}\t{block_len}\t{block_len}\t60\tgi:f:0.99\tcg:Z:{cigar}"
    )
}

/// Write lines to a temp file and return its path.
fn write_temp_paf(lines: &[String]) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.paf");
    let mut f = std::fs::File::create(&path).unwrap();
    for line in lines {
        writeln!(f, "{}", line).unwrap();
    }
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

/// Write lines to a gzipped temp file and return its path.
fn write_temp_paf_gz(lines: &[String]) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.paf.gz");
    let f = std::fs::File::create(&path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
    for line in lines {
        writeln!(gz, "{}", line).unwrap();
    }
    gz.finish().unwrap();
    let path_str = path.to_str().unwrap().to_string();
    (dir, path_str)
}

// =======================================================================
// read_paf_alignments
// =======================================================================

#[test]
fn read_paf_alignments_file_not_found() {
    let result = read_paf_alignments("/nonexistent/path.paf", "chr1", None, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn read_paf_alignments_empty_file() {
    let (_dir, path) = write_temp_paf(&[]);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert!(alns.is_empty());
}

#[test]
fn read_paf_alignments_single_alignment() {
    let lines = vec![paf_line("HG00097#1", "chr12", 5000, 6000, "1000=")];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 1);
    assert_eq!(alns[0].hap_id, "HG00097#1");
    assert_eq!(alns[0].target_start, 5000);
    assert_eq!(alns[0].target_end, 6000);
}

#[test]
fn read_paf_alignments_wrong_chrom_filtered() {
    let lines = vec![paf_line("HG00097#1", "chr1", 0, 1000, "1000=")];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert!(alns.is_empty());
}

#[test]
fn read_paf_alignments_sorted_by_target_start() {
    let lines = vec![
        paf_line("HG00097#1", "chr12", 50000, 60000, "10000="),
        paf_line("HG00098#1", "chr12", 10000, 20000, "10000="),
        paf_line("HG00099#1", "chr12", 30000, 40000, "10000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 3);
    // Must be sorted by target_start
    assert!(alns[0].target_start <= alns[1].target_start);
    assert!(alns[1].target_start <= alns[2].target_start);
    assert_eq!(alns[0].target_start, 10000);
    assert_eq!(alns[1].target_start, 30000);
    assert_eq!(alns[2].target_start, 50000);
}

#[test]
fn read_paf_alignments_subset_filter() {
    let lines = vec![
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00099#1", "chr12", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let mut subset = HashSet::new();
    subset.insert("HG00097#1".to_string());
    subset.insert("HG00099#1".to_string());
    let alns = read_paf_alignments(&path, "chr12", Some(&subset), 0).unwrap();
    assert_eq!(alns.len(), 2);
    let haps: Vec<&str> = alns.iter().map(|a| a.hap_id.as_str()).collect();
    assert!(haps.contains(&"HG00097#1"));
    assert!(haps.contains(&"HG00099#1"));
    assert!(!haps.contains(&"HG00098#1"));
}

#[test]
fn read_paf_alignments_min_length_filter() {
    let lines = vec![
        paf_line("HG00097#1", "chr12", 0, 500, "500="),
        paf_line("HG00098#1", "chr12", 0, 5000, "5000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 1000).unwrap();
    assert_eq!(alns.len(), 1);
    assert_eq!(alns[0].hap_id, "HG00098#1");
}

#[test]
fn read_paf_alignments_skips_chm13_and_grch38() {
    let lines = vec![
        paf_line("CHM13#0", "chr12", 0, 1000, "1000="),
        paf_line("GRCh38#0", "chr12", 0, 1000, "1000="),
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 1);
    assert_eq!(alns[0].hap_id, "HG00097#1");
}

#[test]
fn read_paf_alignments_gzipped_file() {
    let lines = vec![
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 1000, 2000, "1000="),
    ];
    let (_dir, path) = write_temp_paf_gz(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 2);
    assert_eq!(alns[0].hap_id, "HG00097#1");
    assert_eq!(alns[1].hap_id, "HG00098#1");
}

#[test]
fn read_paf_alignments_malformed_lines_skipped() {
    let lines = vec![
        "this is not a paf line".to_string(),
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        "too\tfew\tfields".to_string(),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 1);
}

#[test]
fn read_paf_alignments_multiple_chroms_in_file() {
    let lines = vec![
        paf_line("HG00097#1", "chr1", 0, 1000, "1000="),
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 1000, 2000, "1000="),
        paf_line("HG00097#1", "chr20", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 2);
}

#[test]
fn read_paf_alignments_with_cigar_mismatches() {
    let lines = vec![paf_line("HG00097#1", "chr12", 1000, 1100, "50=2X48=")];
    let (_dir, path) = write_temp_paf(&lines);
    let alns = read_paf_alignments(&path, "chr12", None, 0).unwrap();
    assert_eq!(alns.len(), 1);
    assert_eq!(alns[0].mismatch_positions.len(), 2);
    assert_eq!(alns[0].mismatch_positions[0], 1050);
    assert_eq!(alns[0].mismatch_positions[1], 1051);
}

#[test]
fn read_paf_alignments_subset_by_sample_name() {
    // Subset filter also matches sample name (without haplotype number)
    let lines = vec![
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00097#2", "chr12", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let mut subset = HashSet::new();
    subset.insert("HG00097".to_string()); // sample name, not hap ID
    let alns = read_paf_alignments(&path, "chr12", Some(&subset), 0).unwrap();
    assert_eq!(alns.len(), 2); // both HG00097#1 and HG00097#2
}

// =======================================================================
// read_paf_alignments_multi
// =======================================================================

#[test]
fn read_paf_alignments_multi_file_not_found() {
    let chroms: HashSet<String> = ["chr1".to_string()].into_iter().collect();
    let result = read_paf_alignments_multi("/nonexistent/path.paf", &chroms, None, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn read_paf_alignments_multi_empty_file() {
    let (_dir, path) = write_temp_paf(&[]);
    let chroms: HashSet<String> = ["chr1".to_string(), "chr12".to_string()].into_iter().collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn read_paf_alignments_multi_multiple_chroms() {
    let lines = vec![
        paf_line("HG00097#1", "chr1", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 0, 2000, "2000="),
        paf_line("HG00099#1", "chr12", 2000, 3000, "1000="),
        paf_line("HG00097#1", "chr20", 0, 500, "500="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr1".to_string(), "chr12".to_string(), "chr20".to_string()]
        .into_iter()
        .collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result["chr1"].len(), 1);
    assert_eq!(result["chr12"].len(), 2);
    assert_eq!(result["chr20"].len(), 1);
}

#[test]
fn read_paf_alignments_multi_no_matching_chroms() {
    let lines = vec![paf_line("HG00097#1", "chr1", 0, 1000, "1000=")];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr22".to_string()].into_iter().collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn read_paf_alignments_multi_sorted_per_chrom() {
    let lines = vec![
        paf_line("HG00097#1", "chr12", 50000, 60000, "10000="),
        paf_line("HG00098#1", "chr12", 10000, 20000, "10000="),
        paf_line("HG00099#1", "chr1", 30000, 40000, "10000="),
        paf_line("HG00097#1", "chr1", 5000, 6000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr1".to_string(), "chr12".to_string()]
        .into_iter()
        .collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    // chr12 should be sorted
    assert!(result["chr12"][0].target_start <= result["chr12"][1].target_start);
    // chr1 should be sorted
    assert!(result["chr1"][0].target_start <= result["chr1"][1].target_start);
}

#[test]
fn read_paf_alignments_multi_with_subset_filter() {
    let lines = vec![
        paf_line("HG00097#1", "chr1", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr1", 0, 1000, "1000="),
        paf_line("HG00097#1", "chr12", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr1".to_string(), "chr12".to_string()]
        .into_iter()
        .collect();
    let mut subset = HashSet::new();
    subset.insert("HG00097#1".to_string());
    let result = read_paf_alignments_multi(&path, &chroms, Some(&subset), 0).unwrap();
    assert_eq!(result["chr1"].len(), 1);
    assert_eq!(result["chr12"].len(), 1);
    assert_eq!(result["chr1"][0].hap_id, "HG00097#1");
}

#[test]
fn read_paf_alignments_multi_with_min_length() {
    let lines = vec![
        paf_line("HG00097#1", "chr1", 0, 500, "500="),
        paf_line("HG00098#1", "chr1", 0, 5000, "5000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr1".to_string()].into_iter().collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 1000).unwrap();
    assert_eq!(result["chr1"].len(), 1);
    assert_eq!(result["chr1"][0].hap_id, "HG00098#1");
}

#[test]
fn read_paf_alignments_multi_gzipped() {
    let lines = vec![
        paf_line("HG00097#1", "chr1", 0, 1000, "1000="),
        paf_line("HG00098#1", "chr12", 0, 2000, "2000="),
    ];
    let (_dir, path) = write_temp_paf_gz(&lines);
    let chroms: HashSet<String> = ["chr1".to_string(), "chr12".to_string()]
        .into_iter()
        .collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn read_paf_alignments_multi_empty_chroms_set() {
    let lines = vec![paf_line("HG00097#1", "chr1", 0, 1000, "1000=")];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = HashSet::new();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert!(result.is_empty());
}

#[test]
fn read_paf_alignments_multi_skips_ref_haplotypes() {
    let lines = vec![
        paf_line("CHM13#0", "chr1", 0, 1000, "1000="),
        paf_line("GRCh38#0", "chr1", 0, 1000, "1000="),
        paf_line("HG00097#1", "chr1", 0, 1000, "1000="),
    ];
    let (_dir, path) = write_temp_paf(&lines);
    let chroms: HashSet<String> = ["chr1".to_string()].into_iter().collect();
    let result = read_paf_alignments_multi(&path, &chroms, None, 0).unwrap();
    assert_eq!(result["chr1"].len(), 1);
    assert_eq!(result["chr1"][0].hap_id, "HG00097#1");
}

// =======================================================================
// extract_sample_from_hap — edge cases
// =======================================================================

#[test]
fn extract_sample_from_hap_standard() {
    assert_eq!(extract_sample_from_hap("HG00097#1"), "HG00097");
}

#[test]
fn extract_sample_from_hap_no_hash() {
    assert_eq!(extract_sample_from_hap("HG00097"), "HG00097");
}

#[test]
fn extract_sample_from_hap_empty() {
    assert_eq!(extract_sample_from_hap(""), "");
}

#[test]
fn extract_sample_from_hap_multiple_hashes() {
    // Should return everything before first '#'
    assert_eq!(extract_sample_from_hap("HG00097#1#extra"), "HG00097");
}

#[test]
fn extract_sample_from_hap_hash_at_start() {
    assert_eq!(extract_sample_from_hap("#1"), "");
}

#[test]
fn extract_sample_from_hap_only_hash() {
    assert_eq!(extract_sample_from_hap("#"), "");
}

// =======================================================================
// extract_hap_id — additional edge cases
// =======================================================================

#[test]
fn extract_hap_id_single_char_segments() {
    assert_eq!(extract_hap_id("A#1#B"), "A#1");
}

#[test]
fn extract_hap_id_consecutive_hashes() {
    assert_eq!(extract_hap_id("HG##contig"), "HG#");
}

#[test]
fn extract_hap_id_trailing_hash() {
    assert_eq!(extract_hap_id("HG00097#1#"), "HG00097#1");
}

// =======================================================================
// extract_target_chrom — additional edge cases
// =======================================================================

#[test]
fn extract_target_chrom_empty() {
    assert_eq!(extract_target_chrom(""), "");
}

#[test]
fn extract_target_chrom_multiple_hashes() {
    assert_eq!(extract_target_chrom("CHM13#0#chr12"), "chr12");
}

#[test]
fn extract_target_chrom_single_hash() {
    assert_eq!(extract_target_chrom("ref#chr1"), "chr1");
}

// =======================================================================
// parse_cigar_mismatches — additional edge cases
// =======================================================================

#[test]
fn parse_cigar_m_ops_treated_as_match() {
    // M operations should be treated like = (match)
    let (mm, aligned) = parse_cigar_mismatches("100M", 0);
    assert!(mm.is_empty());
    assert_eq!(aligned, 100);
}

#[test]
fn parse_cigar_unknown_ops_skipped() {
    // S, H, N, P operations should be ignored
    let (mm, aligned) = parse_cigar_mismatches("10S50=5H", 0);
    assert!(mm.is_empty());
    assert_eq!(aligned, 50);
}

#[test]
fn parse_cigar_single_base_ops() {
    // Operations without explicit count default to 1
    let (mm, aligned) = parse_cigar_mismatches("=X=", 100);
    assert_eq!(mm, vec![101]);
    assert_eq!(aligned, 3);
}

#[test]
fn parse_cigar_large_count() {
    let (mm, aligned) = parse_cigar_mismatches("1000000=", 0);
    assert!(mm.is_empty());
    assert_eq!(aligned, 1000000);
}

#[test]
fn parse_cigar_all_mismatches() {
    let (mm, aligned) = parse_cigar_mismatches("5X", 10);
    assert_eq!(mm, vec![10, 11, 12, 13, 14]);
    assert_eq!(aligned, 5);
}

#[test]
fn parse_cigar_alternating_match_mismatch() {
    let (mm, aligned) = parse_cigar_mismatches("1=1X1=1X1=", 0);
    assert_eq!(mm, vec![1, 3]);
    assert_eq!(aligned, 5);
}

#[test]
fn parse_cigar_deletion_at_start() {
    let (mm, aligned) = parse_cigar_mismatches("3D10=", 100);
    assert_eq!(mm, vec![100, 101, 102]);
    assert_eq!(aligned, 13);
}

#[test]
fn parse_cigar_insertion_doesnt_advance_target() {
    let (mm, aligned) = parse_cigar_mismatches("5=10I5=", 0);
    assert!(mm.is_empty());
    assert_eq!(aligned, 10);
}

// =======================================================================
// parse_paf_line — additional edge cases
// =======================================================================

#[test]
fn parse_paf_line_too_few_fields() {
    assert!(parse_paf_line("a\tb\tc", "chr1", None, 0).is_none());
}

#[test]
fn parse_paf_line_exactly_12_fields_no_tags() {
    // Minimal valid line: 12 fields but no gi:f or cg:Z tags
    let line = "HG00097#1#contig.1\t1000\t0\t1000\t+\tCHM13#0#chr12\t248387328\t0\t1000\t1000\t1000\t60";
    let aln = parse_paf_line(line, "chr12", None, 0).unwrap();
    assert_eq!(aln.hap_id, "HG00097#1");
    assert!(aln.mismatch_positions.is_empty()); // no CIGAR
    assert_eq!(aln.gap_identity, 0.0); // no gi:f tag
}

#[test]
fn parse_paf_line_invalid_target_start() {
    let line = "HG00097#1#contig.1\t1000\t0\t1000\t+\tCHM13#0#chr12\t248387328\tNOTANUM\t1000\t1000\t1000\t60";
    assert!(parse_paf_line(line, "chr12", None, 0).is_none());
}

#[test]
fn parse_paf_line_subset_matches_sample_name() {
    let line = "HG00097#1#contig.1\t1000\t0\t1000\t+\tCHM13#0#chr12\t248387328\t0\t1000\t1000\t1000\t60\tcg:Z:1000=";
    let mut subset = HashSet::new();
    subset.insert("HG00097".to_string()); // sample name, not hap ID
    assert!(parse_paf_line(line, "chr12", Some(&subset), 0).is_some());
}

#[test]
fn parse_paf_line_zero_length_block() {
    let line = "HG00097#1#contig.1\t0\t0\t0\t+\tCHM13#0#chr12\t248387328\t1000\t1000\t0\t0\t60\tcg:Z:";
    // Block length = 0, should pass min_aligned_length=0
    let result = parse_paf_line(line, "chr12", None, 0);
    // Might be Some or None depending on implementation — just verify no panic
    let _ = result;
}

// =======================================================================
// compute_window_pairwise — edge cases
// =======================================================================

#[test]
fn compute_window_pairwise_empty_alignments() {
    let results = compute_window_pairwise(&[], 0, 1000, "CHM13", None, None, 0.0);
    assert!(results.is_empty());
}

#[test]
fn compute_window_pairwise_single_haplotype() {
    let alns = vec![make_alignment("A#1", 0, 100, vec![])];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert!(results.is_empty()); // Need ≥2 haplotypes for pairs
}

#[test]
fn compute_window_pairwise_zero_length_window() {
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![]),
        make_alignment("B#1", 0, 100, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 50, 50, "CHM13", None, None, 0.0);
    assert!(results.is_empty()); // Zero-length window has no overlap
}

#[test]
fn compute_window_pairwise_window_before_alignments() {
    let alns = vec![
        make_alignment("A#1", 1000, 2000, vec![]),
        make_alignment("B#1", 1000, 2000, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 0, 500, "CHM13", None, None, 0.0);
    assert!(results.is_empty());
}

#[test]
fn compute_window_pairwise_window_after_alignments() {
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![]),
        make_alignment("B#1", 0, 100, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 200, 300, "CHM13", None, None, 0.0);
    assert!(results.is_empty());
}

#[test]
fn compute_window_pairwise_all_mismatches_zero_identity() {
    // Every position is discordant for one of the two haps
    let mut mm_a = Vec::new();
    let mut mm_b = Vec::new();
    for i in 0..100u64 {
        if i % 2 == 0 {
            mm_a.push(i);
        } else {
            mm_b.push(i);
        }
    }
    let alns = vec![
        make_alignment("A#1", 0, 100, mm_a),
        make_alignment("B#1", 0, 100, mm_b),
    ];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 1);
    // 100 discordant positions / 100 overlap = identity 0.0
    assert!(results[0].identity.abs() < 1e-10);
}

#[test]
fn compute_window_pairwise_three_haplotypes_three_pairs() {
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![]),
        make_alignment("B#1", 0, 100, vec![]),
        make_alignment("C#1", 0, 100, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 3); // C(3,2) = 3 pairs
}

#[test]
fn compute_window_pairwise_window_partially_covers_alignment() {
    // Alignment covers [0, 1000), window is [400, 600)
    let alns = vec![
        make_alignment("A#1", 0, 1000, vec![450, 500, 550, 900]),
        make_alignment("B#1", 0, 1000, vec![500, 700]),
    ];
    let results = compute_window_pairwise(&alns, 400, 600, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 1);
    // In window [400, 600): A has {450, 500, 550}, B has {500}
    // Symmetric diff: {450, 550} → 2 discordant out of 200 overlap
    assert!((results[0].identity - (1.0 - 2.0 / 200.0)).abs() < 1e-10);
    assert_eq!(results[0].overlap_bp, 200);
}

#[test]
fn compute_window_pairwise_query_ref_both_none() {
    // When both filters are None, all pairs emitted
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![]),
        make_alignment("B#1", 0, 100, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 1);
}

#[test]
fn compute_window_pairwise_query_ref_one_none() {
    // When only one of query/ref is Some, it's treated as both being None (no filter)
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![]),
        make_alignment("B#1", 0, 100, vec![]),
    ];
    let qf = HashSet::from(["A".to_string()]);
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", Some(&qf), None, 0.0);
    // With only one filter set, the cross-check requires both to be Some
    assert_eq!(results.len(), 1);
}

#[test]
fn compute_window_pairwise_cutoff_exact_boundary() {
    // Identity exactly equals cutoff → should be included (>=)
    let alns = vec![
        make_alignment("A#1", 0, 100, vec![5]),
        make_alignment("B#1", 0, 100, vec![]),
    ];
    // 1 discordant / 100 = identity 0.99
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.99);
    assert_eq!(results.len(), 1);
}

#[test]
fn compute_window_pairwise_a_length_b_length_reported() {
    let alns = vec![
        make_alignment("A#1", 10, 90, vec![]),
        make_alignment("B#1", 20, 80, vec![]),
    ];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].a_length, 80); // 90 - 10
    assert_eq!(results[0].b_length, 60); // 80 - 20
    assert_eq!(results[0].overlap_bp, 60); // [20, 80)
}

#[test]
fn compute_window_pairwise_dedup_across_alignment_blocks() {
    // Same haplotype with overlapping alignment blocks → mismatch positions should be deduped
    let alns = vec![
        make_alignment("A#1", 0, 60, vec![30, 40, 50]),
        make_alignment("A#1", 40, 100, vec![40, 50, 60]),
        make_alignment("B#1", 0, 100, vec![30, 40, 50, 60]),
    ];
    let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
    assert_eq!(results.len(), 1);
    // A deduplicated: {30, 40, 50, 60}, B: {30, 40, 50, 60}
    // Symmetric diff = empty → identity 1.0
    assert!((results[0].identity - 1.0).abs() < 1e-10);
}
