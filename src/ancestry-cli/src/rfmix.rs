//! Parser for RFMix v2 `.msp.tsv` output files.
//!
//! RFMix v2 outputs local ancestry calls in the `.msp.tsv` format (most likely
//! sub-population). Each row represents a genomic segment with ancestry calls
//! for each query haplotype.
//!
//! ## Format
//!
//! ```text
//! #Subpopulation order/codes: AFR=0\tEUR=1
//! #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
//! chr20\t82590\t728330\t0.00\t2.67\t339\t1\t1
//! ```
//!
//! - Line 1: Maps population names to integer codes
//! - Line 2: Column header with query haplotype IDs (sample.0, sample.1)
//! - Data lines: chromosome, positions, genetic positions, n_snps, per-haplotype ancestry codes

/// A single RFMix segment from `.msp.tsv`
#[derive(Debug, Clone)]
pub struct RfmixSegment {
    /// Chromosome name (e.g., "chr20")
    pub chrom: String,
    /// Segment start position (bp)
    pub start: u64,
    /// Segment end position (bp)
    pub end: u64,
    /// Segment start in genetic map coordinates (cM)
    pub start_cm: f64,
    /// Segment end in genetic map coordinates (cM)
    pub end_cm: f64,
    /// Number of SNPs in this segment
    pub n_snps: u32,
    /// Per-haplotype ancestry indices (index into population_names)
    pub hap_ancestries: Vec<usize>,
}

/// Parsed RFMix `.msp.tsv` result
#[derive(Debug, Clone)]
pub struct RfmixResult {
    /// Population names in order (index 0, 1, 2, ...)
    pub population_names: Vec<String>,
    /// Haplotype column names (e.g., ["HG00733.0", "HG00733.1"])
    pub haplotype_names: Vec<String>,
    /// Segments with per-haplotype ancestry calls
    pub segments: Vec<RfmixSegment>,
}

/// Parse an RFMix `.msp.tsv` file from a path.
pub fn parse_rfmix_msp(path: &std::path::Path) -> Result<RfmixResult, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read RFMix file {:?}: {}", path, e))?;
    parse_rfmix_msp_content(&content)
}

/// Parse RFMix `.msp.tsv` content from a string.
pub fn parse_rfmix_msp_content(content: &str) -> Result<RfmixResult, String> {
    let mut lines = content.lines();

    // Line 1: population codes
    // Format: "#Subpopulation order/codes: AFR=0\tEUR=1" or "#reference_panel_population:  AFR  EUR"
    let pop_line = lines.next().ok_or("Empty RFMix file: missing population header")?;
    let population_names = parse_population_header(pop_line)?;

    // Line 2: column header with haplotype names
    let header_line = lines.next().ok_or("Missing column header line")?;
    let haplotype_names = parse_column_header(header_line)?;

    // Data lines
    let mut segments = Vec::new();
    for (line_num, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let seg = parse_data_line(line, haplotype_names.len(), line_num + 3)?;
        segments.push(seg);
    }

    Ok(RfmixResult {
        population_names,
        haplotype_names,
        segments,
    })
}

/// Parse the population header line.
///
/// Supports two formats:
/// 1. `#Subpopulation order/codes: AFR=0\tEUR=1` (tab-separated key=value pairs)
/// 2. `#reference_panel_population:  AFR  EUR` (whitespace-separated names, order = index)
fn parse_population_header(line: &str) -> Result<Vec<String>, String> {
    let line = line.trim();
    if !line.starts_with('#') {
        return Err(format!("Population header should start with '#', got: {}", line));
    }

    // Try format 1: "AFR=0\tEUR=1" style
    if line.contains('=') {
        // Split on colon to get the assignments part
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() < 2 {
            return Err(format!("Cannot parse population header: {}", line));
        }

        let assignments = parts[1].trim();
        let mut pop_map: Vec<(usize, String)> = Vec::new();

        for token in assignments.split(['\t', ' ']) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if let Some((name, idx_str)) = token.split_once('=') {
                let idx: usize = idx_str.parse()
                    .map_err(|_| format!("Invalid population index '{}' in: {}", idx_str, line))?;
                pop_map.push((idx, name.to_string()));
            }
        }

        pop_map.sort_by_key(|(idx, _)| *idx);

        // Verify indices are 0..N-1
        for (i, (idx, _)) in pop_map.iter().enumerate() {
            if *idx != i {
                return Err(format!("Population indices not contiguous: expected {}, got {}", i, idx));
            }
        }

        Ok(pop_map.into_iter().map(|(_, name)| name).collect())
    } else {
        // Format 2: whitespace-separated names after colon
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() < 2 {
            return Err(format!("Cannot parse population header: {}", line));
        }

        let names: Vec<String> = parts[1].split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if names.is_empty() {
            return Err(format!("No population names found in header: {}", line));
        }

        Ok(names)
    }
}

/// Parse the column header line to extract haplotype names.
///
/// Format: `#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1`
fn parse_column_header(line: &str) -> Result<Vec<String>, String> {
    let line = line.trim();
    if !line.starts_with('#') {
        return Err(format!("Column header should start with '#', got: {}", line));
    }

    let fields: Vec<&str> = line.split('\t').collect();

    // First 6 columns are fixed: #chm, spos, epos, sgpos, egpos, n snps
    if fields.len() < 7 {
        return Err(format!(
            "Column header has too few fields ({}, need >= 7): {}",
            fields.len(), line
        ));
    }

    // Haplotype names start at column 6
    let hap_names: Vec<String> = fields[6..].iter().map(|s| s.to_string()).collect();

    if hap_names.is_empty() {
        return Err("No haplotype columns found in header".to_string());
    }

    Ok(hap_names)
}

/// Parse a data line from the `.msp.tsv` file.
fn parse_data_line(line: &str, n_haplotypes: usize, line_num: usize) -> Result<RfmixSegment, String> {
    let fields: Vec<&str> = line.split('\t').collect();

    let expected_fields = 6 + n_haplotypes;
    if fields.len() < expected_fields {
        return Err(format!(
            "Line {}: expected {} fields, got {}: {}",
            line_num, expected_fields, fields.len(), line
        ));
    }

    let chrom = fields[0].to_string();
    let start: u64 = fields[1].parse()
        .map_err(|_| format!("Line {}: invalid start position '{}'", line_num, fields[1]))?;
    let end: u64 = fields[2].parse()
        .map_err(|_| format!("Line {}: invalid end position '{}'", line_num, fields[2]))?;
    let start_cm: f64 = fields[3].parse()
        .map_err(|_| format!("Line {}: invalid start_cm '{}'", line_num, fields[3]))?;
    let end_cm: f64 = fields[4].parse()
        .map_err(|_| format!("Line {}: invalid end_cm '{}'", line_num, fields[4]))?;
    let n_snps: u32 = fields[5].parse()
        .map_err(|_| format!("Line {}: invalid n_snps '{}'", line_num, fields[5]))?;

    let mut hap_ancestries = Vec::with_capacity(n_haplotypes);
    for i in 0..n_haplotypes {
        let idx: usize = fields[6 + i].parse()
            .map_err(|_| format!("Line {}: invalid ancestry index '{}' for haplotype {}", line_num, fields[6 + i], i))?;
        hap_ancestries.push(idx);
    }

    Ok(RfmixSegment {
        chrom,
        start,
        end,
        start_cm,
        end_cm,
        n_snps,
        hap_ancestries,
    })
}

/// Convert RFMix segments to per-window ancestry calls.
///
/// Given a window size, creates a vector of per-window ancestry indices for each haplotype.
/// Windows that don't overlap any segment get `None`.
///
/// # Arguments
/// * `result` - Parsed RFMix result
/// * `window_size` - Window size in base pairs (e.g., 10000 for 10kb)
///
/// # Returns
/// A vector of length `n_haplotypes`, where each element is a `Vec<Option<usize>>`
/// giving the ancestry index for each window along the chromosome.
pub fn rfmix_to_windows(
    result: &RfmixResult,
    window_size: u64,
) -> Vec<Vec<Option<usize>>> {
    if result.segments.is_empty() || window_size == 0 {
        return vec![Vec::new(); result.haplotype_names.len()];
    }

    let n_haps = result.haplotype_names.len();

    // Determine genomic range
    let min_pos = result.segments.iter().map(|s| s.start).min().unwrap_or(0);
    let max_pos = result.segments.iter().map(|s| s.end).max().unwrap_or(0);

    if max_pos <= min_pos {
        return vec![Vec::new(); n_haps];
    }

    let n_windows = (max_pos - min_pos).div_ceil(window_size);

    let mut windows: Vec<Vec<Option<usize>>> = vec![vec![None; n_windows as usize]; n_haps];

    for seg in &result.segments {
        // Find which windows this segment overlaps
        let w_start = if seg.start >= min_pos {
            (seg.start - min_pos) / window_size
        } else {
            0
        };
        let w_end = if seg.end > min_pos {
            ((seg.end - min_pos).saturating_sub(1)) / window_size
        } else {
            continue;
        };

        for w in w_start..=w_end.min(n_windows - 1) {
            for (hap_idx, &ancestry) in seg.hap_ancestries.iter().enumerate() {
                if hap_idx < n_haps {
                    windows[hap_idx][w as usize] = Some(ancestry);
                }
            }
        }
    }

    windows
}

/// Get the window boundaries (start positions) for a given RFMix result and window size.
///
/// Returns the start position of each window.
pub fn rfmix_window_starts(result: &RfmixResult, window_size: u64) -> Vec<u64> {
    if result.segments.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let min_pos = result.segments.iter().map(|s| s.start).min().unwrap_or(0);
    let max_pos = result.segments.iter().map(|s| s.end).max().unwrap_or(0);

    if max_pos <= min_pos {
        return Vec::new();
    }

    let n_windows = (max_pos - min_pos).div_ceil(window_size);
    (0..n_windows).map(|w| min_pos + w * window_size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_msp() -> String {
        "#Subpopulation order/codes: AFR=0\tEUR=1\n\
         #chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1\n\
         chr20\t100000\t500000\t0.00\t1.50\t100\t1\t1\n\
         chr20\t500000\t800000\t1.50\t3.00\t80\t0\t1\n\
         chr20\t800000\t1200000\t3.00\t5.00\t120\t1\t0\n"
            .to_string()
    }

    #[test]
    fn test_parse_rfmix_msp_basic() {
        let content = make_test_msp();
        let result = parse_rfmix_msp_content(&content).unwrap();

        assert_eq!(result.population_names, vec!["AFR", "EUR"]);
        assert_eq!(result.haplotype_names, vec!["HG00733.0", "HG00733.1"]);
        assert_eq!(result.segments.len(), 3);

        let seg0 = &result.segments[0];
        assert_eq!(seg0.chrom, "chr20");
        assert_eq!(seg0.start, 100000);
        assert_eq!(seg0.end, 500000);
        assert!((seg0.start_cm - 0.0).abs() < 1e-6);
        assert!((seg0.end_cm - 1.5).abs() < 1e-6);
        assert_eq!(seg0.n_snps, 100);
        assert_eq!(seg0.hap_ancestries, vec![1, 1]); // EUR, EUR

        let seg1 = &result.segments[1];
        assert_eq!(seg1.hap_ancestries, vec![0, 1]); // AFR, EUR

        let seg2 = &result.segments[2];
        assert_eq!(seg2.hap_ancestries, vec![1, 0]); // EUR, AFR
    }

    #[test]
    fn test_parse_population_header_format1() {
        let line = "#Subpopulation order/codes: AFR=0\tEUR=1";
        let names = parse_population_header(line).unwrap();
        assert_eq!(names, vec!["AFR", "EUR"]);
    }

    #[test]
    fn test_parse_population_header_format2() {
        let line = "#reference_panel_population:  AFR  EUR";
        let names = parse_population_header(line).unwrap();
        assert_eq!(names, vec!["AFR", "EUR"]);
    }

    #[test]
    fn test_parse_population_header_three_way() {
        let line = "#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2";
        let names = parse_population_header(line).unwrap();
        assert_eq!(names, vec!["AFR", "EUR", "NAT"]);
    }

    #[test]
    fn test_parse_population_header_errors() {
        assert!(parse_population_header("not a header").is_err());
        assert!(parse_population_header("#:").is_err()); // no population names
    }

    #[test]
    fn test_parse_column_header() {
        let line = "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1";
        let names = parse_column_header(line).unwrap();
        assert_eq!(names, vec!["HG00733.0", "HG00733.1"]);
    }

    #[test]
    fn test_parse_column_header_errors() {
        // Too few fields
        assert!(parse_column_header("#chm\tspos\tepos").is_err());
        // Not a header
        assert!(parse_column_header("chm\tspos\tepos\tsgpos\tegpos\tn snps\thap").is_err());
    }

    #[test]
    fn test_rfmix_to_windows() {
        let content = make_test_msp();
        let result = parse_rfmix_msp_content(&content).unwrap();

        // Use 100kb windows
        let windows = rfmix_to_windows(&result, 100_000);

        assert_eq!(windows.len(), 2); // 2 haplotypes

        // Hap 0: EUR(1) from 100k-500k, AFR(0) from 500k-800k, EUR(1) from 800k-1200k
        // Window 0: 100k-200k → EUR(1)
        assert_eq!(windows[0][0], Some(1));
        // Window 4: 500k-600k → AFR(0)
        assert_eq!(windows[0][4], Some(0));
        // Window 7: 800k-900k → EUR(1)
        assert_eq!(windows[0][7], Some(1));

        // Hap 1: EUR(1) from 100k-500k, EUR(1) from 500k-800k, AFR(0) from 800k-1200k
        assert_eq!(windows[1][0], Some(1));
        assert_eq!(windows[1][4], Some(1));
        assert_eq!(windows[1][7], Some(0));
    }

    #[test]
    fn test_rfmix_to_windows_empty() {
        let result = RfmixResult {
            population_names: vec!["AFR".to_string(), "EUR".to_string()],
            haplotype_names: vec!["HG00733.0".to_string()],
            segments: vec![],
        };

        let windows = rfmix_to_windows(&result, 10000);
        assert_eq!(windows.len(), 1);
        assert!(windows[0].is_empty());
    }

    #[test]
    fn test_rfmix_window_starts() {
        let content = make_test_msp();
        let result = parse_rfmix_msp_content(&content).unwrap();

        let starts = rfmix_window_starts(&result, 100_000);

        // Range: 100k to 1200k, so 11 windows of 100k
        assert_eq!(starts.len(), 11);
        assert_eq!(starts[0], 100_000);
        assert_eq!(starts[1], 200_000);
        assert_eq!(starts[10], 1_100_000);
    }

    #[test]
    fn test_parse_real_rfmix_format() {
        // Test with the exact format from HG00733 chr20
        let content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t82590\t728330\t0.00\t2.67\t339\t1\t1
chr20\t728330\t775876\t2.67\t2.89\t44\t0\t1
chr20\t775876\t826094\t2.89\t3.20\t45\t1\t0";

        let result = parse_rfmix_msp_content(content).unwrap();

        assert_eq!(result.population_names, vec!["AFR", "EUR"]);
        assert_eq!(result.segments.len(), 3);

        // First segment: both haplotypes EUR
        assert_eq!(result.segments[0].hap_ancestries, vec![1, 1]);
        assert_eq!(result.segments[0].start, 82590);
        assert_eq!(result.segments[0].end, 728330);

        // Second segment: hap0=AFR, hap1=EUR
        assert_eq!(result.segments[1].hap_ancestries, vec![0, 1]);

        // Third segment: hap0=EUR, hap1=AFR
        assert_eq!(result.segments[2].hap_ancestries, vec![1, 0]);
    }

    #[test]
    fn test_parse_rfmix_file() {
        // Test file-based parsing if a real RFMix file exists in the data directory
        let path = std::path::Path::new("data/rfmix_test/rfmix_HG00733_chr20_v4.msp.tsv");
        if path.exists() {
            let result = parse_rfmix_msp(path).unwrap();

            assert_eq!(result.population_names.len(), 2);
            assert!(result.population_names.contains(&"AFR".to_string()));
            assert!(result.population_names.contains(&"EUR".to_string()));

            assert_eq!(result.haplotype_names.len(), 2);
            assert!(result.haplotype_names[0].contains("HG00733"));

            // Should have ~63 segments
            assert!(result.segments.len() > 50,
                "Expected ~63 segments, got {}", result.segments.len());
            assert!(result.segments.len() < 80,
                "Expected ~63 segments, got {}", result.segments.len());

            // All ancestry indices should be 0 or 1 (2-way)
            for seg in &result.segments {
                for &anc in &seg.hap_ancestries {
                    assert!(anc <= 1, "Ancestry index should be 0 or 1, got {}", anc);
                }
            }

            // Chromosome should be chr20
            for seg in &result.segments {
                assert_eq!(seg.chrom, "chr20");
            }
        }
    }
}
