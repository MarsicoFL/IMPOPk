#![allow(clippy::manual_is_multiple_of)]
//! PAF (Pairwise mApping Format) parser for direct identity computation.
//!
//! Extracts pairwise identity between haplotypes from pangenome alignments
//! to a shared reference (CHM13), without calling impg. This is >100x faster
//! than impg because the PAF already contains all alignment information.
//!
//! ## Method
//!
//! For each alignment in the PAF, we parse the extended CIGAR to determine
//! which target (CHM13) positions are matches (=) vs mismatches (X) or
//! deletions (D). For a given window on the reference:
//!
//! 1. Find all haplotypes with alignments overlapping the window
//! 2. For each haplotype, collect its mismatch positions within the window
//! 3. For each pair (A, B), compute concordance:
//!    - discordant = positions where exactly one haplotype mismatches the ref
//!    - identity = 1.0 - discordant / overlap_length
//!
//! This is mathematically equivalent to the reference-projection of pairwise
//! identity: two IBD haplotypes share the same mutations (mismatches to ref),
//! so their symmetric difference is near-zero, giving identity ≈ 1.0.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use flate2::bufread::MultiGzDecoder;

/// A parsed PAF alignment record with pre-computed mismatch positions.
#[derive(Debug, Clone)]
pub struct PafAlignment {
    /// Haplotype ID, e.g., "HG00097#1" (contig suffix stripped)
    pub hap_id: String,
    /// Target start position (0-based, inclusive)
    pub target_start: u64,
    /// Target end position (0-based, exclusive)
    pub target_end: u64,
    /// Gap-compressed identity from gi:f tag
    pub gap_identity: f64,
    /// Mismatch positions on target coordinates (sorted, absolute positions).
    /// Includes both substitutions (X) and deletions (D).
    pub mismatch_positions: Vec<u64>,
    /// Total aligned bases on target (= + X + D operations)
    pub aligned_bases: u64,
}

/// Result of pairwise identity computation for one window.
#[derive(Debug, Clone)]
pub struct PairwiseIdentity {
    pub group_a: String,
    pub group_b: String,
    pub identity: f64,
    /// Number of overlapping bases used for computation
    pub overlap_bp: u64,
    /// Coverage length of group A within the window (bp)
    pub a_length: u64,
    /// Coverage length of group B within the window (bp)
    pub b_length: u64,
}

/// Extract haplotype ID from a PAF query name.
///
/// PAF query names look like "HG00097#1#JBIRDD010000019.1".
/// We want "HG00097#1" (sample + haplotype number).
///
/// Format: `{sample}#{hap_num}#{contig}` → return `{sample}#{hap_num}`
pub fn extract_hap_id(query_name: &str) -> &str {
    let mut hash_positions = Vec::new();
    for (i, c) in query_name.char_indices() {
        if c == '#' {
            hash_positions.push(i);
        }
    }
    if hash_positions.len() >= 2 {
        // Return everything up to the second '#'
        &query_name[..hash_positions[1]]
    } else {
        // Fallback: return the whole name
        query_name
    }
}

/// Extract sample name from haplotype ID: "HG00097#1" → "HG00097"
pub fn extract_sample_from_hap(hap_id: &str) -> &str {
    hap_id.split('#').next().unwrap_or(hap_id)
}

/// Parse an extended CIGAR string and extract mismatch positions on target.
///
/// The CIGAR uses:
/// - `=` : sequence match (consumes query + target)
/// - `X` : sequence mismatch (consumes query + target)
/// - `I` : insertion in query (consumes query only — target position unchanged)
/// - `D` : deletion in query (consumes target only — gap in query)
/// - `M` : alignment match (ambiguous, treated as match for safety)
///
/// Returns (mismatch_positions_on_target, total_aligned_bases_on_target).
/// Mismatch positions include both X (substitutions) and D (deletions).
pub fn parse_cigar_mismatches(cigar: &str, target_start: u64) -> (Vec<u64>, u64) {
    let mut mismatches = Vec::new();
    let mut target_pos = target_start;
    let mut aligned_bases: u64 = 0;
    let mut num_buf = String::new();

    for ch in cigar.chars() {
        if ch.is_ascii_digit() {
            num_buf.push(ch);
            continue;
        }

        let count: u64 = if num_buf.is_empty() {
            1 // CIGAR ops without a count default to 1
        } else {
            num_buf.parse().unwrap_or(1)
        };
        num_buf.clear();

        match ch {
            '=' | 'M' => {
                // Match: advances target, no mismatches
                target_pos += count;
                aligned_bases += count;
            }
            'X' => {
                // Mismatch: record each position
                for i in 0..count {
                    mismatches.push(target_pos + i);
                }
                target_pos += count;
                aligned_bases += count;
            }
            'D' => {
                // Deletion in query: target positions exist but query doesn't cover them.
                // Treat as mismatches for pairwise comparison.
                for i in 0..count {
                    mismatches.push(target_pos + i);
                }
                target_pos += count;
                aligned_bases += count;
            }
            'I' => {
                // Insertion in query: no target positions consumed
                // These don't affect target-coordinate comparison
            }
            _ => {
                // Unknown op — skip (S, H, N, P, etc.)
                // S and H don't consume target
            }
        }
    }

    (mismatches, aligned_bases)
}

/// Parse the target chromosome name from PAF target field.
///
/// PAF target names look like "CHM13#0#chr12".
/// Returns "chr12".
pub fn extract_target_chrom(target_name: &str) -> &str {
    // Find the last '#' and return everything after it
    target_name.rsplit('#').next().unwrap_or(target_name)
}

/// Parse a single PAF line, returning an alignment if it matches the target chromosome.
///
/// Returns None if:
/// - The line doesn't target the specified chromosome
/// - The line is malformed
/// - The query haplotype is not in the subset (if subset is provided)
/// - The alignment is too short or low quality
pub fn parse_paf_line(
    line: &str,
    target_chrom: &str,
    subset: Option<&HashSet<String>>,
    min_aligned_length: u64,
) -> Option<PafAlignment> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return None;
    }

    // Field 6: target name (e.g., "CHM13#0#chr12")
    let target_name = fields[5];
    let chrom = extract_target_chrom(target_name);
    if chrom != target_chrom {
        return None;
    }

    // Field 1: query name (e.g., "HG00097#1#JBIRDD010000019.1")
    let query_name = fields[0];
    let hap_id = extract_hap_id(query_name);

    // Skip reference alignments (CHM13, GRCh38)
    let sample = extract_sample_from_hap(hap_id);
    if sample == "CHM13" || sample == "GRCh38" {
        return None;
    }

    // Check subset filter
    if let Some(subset_set) = subset {
        if !subset_set.contains(hap_id) && !subset_set.contains(sample) {
            return None;
        }
    }

    // Fields 8-9: target start/end (0-based)
    let target_start: u64 = fields[7].parse().ok()?;
    let target_end: u64 = fields[8].parse().ok()?;

    // Field 11: alignment block length
    let block_len: u64 = fields[10].parse().ok()?;
    if block_len < min_aligned_length {
        return None;
    }

    // Parse gi:f tag for gap-compressed identity
    let mut gap_identity = 0.0;
    let mut cigar_str = None;
    for field in &fields[12..] {
        if let Some(val) = field.strip_prefix("gi:f:") {
            gap_identity = val.parse().unwrap_or(0.0);
        } else if let Some(val) = field.strip_prefix("cg:Z:") {
            cigar_str = Some(val);
        }
    }

    // Parse CIGAR for mismatch positions
    let (mismatch_positions, aligned_bases) = if let Some(cigar) = cigar_str {
        parse_cigar_mismatches(cigar, target_start)
    } else {
        // No CIGAR available — use gi:f identity to estimate mismatches
        // This is a fallback; real data should always have CIGAR
        (Vec::new(), target_end - target_start)
    };

    Some(PafAlignment {
        hap_id: hap_id.to_string(),
        target_start,
        target_end,
        gap_identity,
        mismatch_positions,
        aligned_bases,
    })
}

/// Read a PAF file (possibly gzipped) and collect all alignments for a target chromosome.
///
/// # Arguments
/// * `paf_path` - Path to .paf or .paf.gz file
/// * `target_chrom` - Chromosome name to filter for (e.g., "chr12")
/// * `subset` - Optional set of sample/haplotype IDs to include
/// * `min_aligned_length` - Minimum alignment block length to include
///
/// Returns alignments sorted by target_start.
pub fn read_paf_alignments(
    paf_path: &str,
    target_chrom: &str,
    subset: Option<&HashSet<String>>,
    min_aligned_length: u64,
) -> Result<Vec<PafAlignment>, String> {
    let path = Path::new(paf_path);
    if !path.exists() {
        return Err(format!("PAF file not found: {}", paf_path));
    }

    let file = File::open(path).map_err(|e| format!("Failed to open PAF: {}", e))?;

    let reader: Box<dyn BufRead> = if paf_path.ends_with(".gz") {
        // Use MultiGzDecoder for BGZF (blocked gzip) compatibility.
        // Standard GzDecoder reads only the first gzip member; BGZF files
        // (produced by samtools/htslib) consist of many concatenated gzip blocks.
        let buf_file = BufReader::with_capacity(1024 * 1024, file);
        Box::new(BufReader::with_capacity(
            1024 * 1024,
            MultiGzDecoder::new(buf_file),
        ))
    } else {
        Box::new(BufReader::with_capacity(1024 * 1024, file))
    };

    let mut alignments = Vec::new();
    let mut total_lines = 0u64;
    let mut matched_lines = 0u64;

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Failed to read PAF line: {}", e))?;
        total_lines += 1;

        if total_lines % 1_000_000 == 0 {
            eprintln!(
                "  Read {} M PAF lines, {} alignments for {}...",
                total_lines / 1_000_000,
                matched_lines,
                target_chrom
            );
        }

        if let Some(aln) = parse_paf_line(&line, target_chrom, subset, min_aligned_length) {
            matched_lines += 1;
            alignments.push(aln);
        }
    }

    eprintln!(
        "Read {} total PAF lines, {} alignments for {}",
        total_lines, matched_lines, target_chrom
    );

    // Sort by target_start for efficient window processing
    alignments.sort_by_key(|a| a.target_start);

    Ok(alignments)
}

/// Read a PAF file once and collect alignments for multiple target chromosomes.
///
/// This is significantly faster than calling `read_paf_alignments` per chromosome
/// when processing multiple chromosomes, because the PAF file is only decompressed
/// and parsed once (for a 5.3GB compressed file, this saves minutes per chromosome).
///
/// # Arguments
/// * `paf_path` - Path to .paf or .paf.gz file
/// * `target_chroms` - Set of chromosome names to collect (e.g., {"chr1", "chr12", "chr20"})
/// * `subset` - Optional set of sample/haplotype IDs to include
/// * `min_aligned_length` - Minimum alignment block length to include
///
/// Returns a HashMap from chromosome name to sorted alignments.
pub fn read_paf_alignments_multi(
    paf_path: &str,
    target_chroms: &HashSet<String>,
    subset: Option<&HashSet<String>>,
    min_aligned_length: u64,
) -> Result<HashMap<String, Vec<PafAlignment>>, String> {
    let path = Path::new(paf_path);
    if !path.exists() {
        return Err(format!("PAF file not found: {}", paf_path));
    }

    let file = File::open(path).map_err(|e| format!("Failed to open PAF: {}", e))?;

    let reader: Box<dyn BufRead> = if paf_path.ends_with(".gz") {
        let buf_file = BufReader::with_capacity(1024 * 1024, file);
        Box::new(BufReader::with_capacity(
            1024 * 1024,
            MultiGzDecoder::new(buf_file),
        ))
    } else {
        Box::new(BufReader::with_capacity(1024 * 1024, file))
    };

    let mut per_chrom: HashMap<String, Vec<PafAlignment>> = HashMap::new();
    let mut total_lines = 0u64;
    let mut matched_lines = 0u64;

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Failed to read PAF line: {}", e))?;
        total_lines += 1;

        if total_lines % 1_000_000 == 0 {
            eprintln!(
                "  Read {} M PAF lines, {} alignments matched...",
                total_lines / 1_000_000,
                matched_lines,
            );
        }

        // Quick chromosome check: extract target chrom from PAF target field
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            continue;
        }
        let target_name = fields[5];
        let chrom = extract_target_chrom(target_name);
        if !target_chroms.contains(chrom) {
            continue;
        }

        let chrom_str = chrom.to_string();
        if let Some(aln) = parse_paf_line(&line, &chrom_str, subset, min_aligned_length) {
            matched_lines += 1;
            per_chrom.entry(chrom_str).or_default().push(aln);
        }
    }

    eprintln!(
        "Read {} total PAF lines, {} alignments across {} chromosomes",
        total_lines,
        matched_lines,
        per_chrom.len()
    );

    // Sort each chromosome's alignments by target_start
    for alns in per_chrom.values_mut() {
        alns.sort_by_key(|a| a.target_start);
    }

    Ok(per_chrom)
}

/// Compute the symmetric difference size of two sorted slices.
/// Both slices must be sorted in ascending order.
fn sorted_symmetric_difference_count(a: &[u64], b: &[u64]) -> u64 {
    let mut count = 0u64;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                count += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                count += 1;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    count += (a.len() - i) as u64;
    count += (b.len() - j) as u64;
    count
}

/// For a given window, collect per-haplotype mismatch sets and compute pairwise identity.
///
/// Uses the concordance method: two haplotypes are concordant at a position if
/// they both match the reference OR both mismatch. Discordant if exactly one mismatches.
///
/// identity(A, B) = 1.0 - |symmetric_diff(mismatches_A, mismatches_B)| / overlap_length
///
/// Alignments must be sorted by target_start (as returned by `read_paf_alignments`).
/// Uses binary search to skip non-overlapping alignments.
pub fn compute_window_pairwise(
    alignments: &[PafAlignment],
    window_start: u64,
    window_end: u64,
    _ref_name: &str,
    query_filter: Option<&HashSet<String>>,
    ref_filter: Option<&HashSet<String>>,
    cutoff: f64,
) -> Vec<PairwiseIdentity> {
    let window_start_u64 = window_start;
    let window_end_u64 = window_end;

    // Binary search: find the upper bound (first alignment starting at or after window_end)
    // All possible overlapping alignments are before this index.
    let upper = alignments.partition_point(|a| a.target_start < window_end);

    // Collect per-haplotype mismatch positions within the window (sorted Vecs, not HashSets)
    let mut hap_mismatches: HashMap<&str, Vec<u64>> = HashMap::new();
    let mut hap_coverage: HashMap<&str, (u64, u64)> = HashMap::new();

    for aln in &alignments[..upper] {
        // Skip alignments that end before window starts
        if aln.target_end <= window_start {
            continue;
        }

        // Extract mismatch positions within window bounds
        let start_idx = aln
            .mismatch_positions
            .partition_point(|&pos| pos < window_start_u64);
        let end_idx = aln
            .mismatch_positions
            .partition_point(|&pos| pos < window_end_u64);

        let window_mismatches = &aln.mismatch_positions[start_idx..end_idx];

        let entry = hap_mismatches
            .entry(aln.hap_id.as_str())
            .or_default();
        entry.extend_from_slice(window_mismatches);

        // Track coverage bounds
        let cov = hap_coverage.entry(aln.hap_id.as_str()).or_insert((
            aln.target_start.max(window_start),
            aln.target_end.min(window_end),
        ));
        cov.0 = cov.0.min(aln.target_start.max(window_start));
        cov.1 = cov.1.max(aln.target_end.min(window_end));
    }

    // Sort and deduplicate mismatch positions per haplotype
    // (needed when multiple alignment blocks contribute to the same window)
    for mm in hap_mismatches.values_mut() {
        mm.sort_unstable();
        mm.dedup();
    }

    // Collect haplotypes into sorted Vec for deterministic output
    let mut haps: Vec<&str> = hap_mismatches.keys().copied().collect();
    haps.sort_unstable();

    // Pre-compute sample names for filter checks
    let hap_samples: Vec<&str> = haps.iter().map(|h| extract_sample_from_hap(h)).collect();

    // Compute pairwise identity
    let mut results = Vec::new();

    for i in 0..haps.len() {
        for j in (i + 1)..haps.len() {
            // Apply query/ref filter early to skip unnecessary computation
            if let (Some(qf), Some(rf)) = (query_filter, ref_filter) {
                let sa = hap_samples[i];
                let sb = hap_samples[j];
                let cross = (qf.contains(sa) && rf.contains(sb))
                    || (qf.contains(sb) && rf.contains(sa));
                if !cross {
                    continue;
                }
            }

            // Canonical ordering (haps are already sorted, so i < j → haps[i] <= haps[j])
            let ga = haps[i];
            let gb = haps[j];

            // Compute overlap length from coverage bounds
            let cov_a = hap_coverage[ga];
            let cov_b = hap_coverage[gb];
            let overlap_start = cov_a.0.max(cov_b.0);
            let overlap_end = cov_a.1.min(cov_b.1);
            if overlap_start >= overlap_end {
                continue;
            }
            let overlap_len = overlap_end - overlap_start;

            // Get mismatch positions restricted to the overlap region
            let mm_a = &hap_mismatches[ga];
            let mm_b = &hap_mismatches[gb];

            let overlap_start_u64 = overlap_start;
            let overlap_end_u64 = overlap_end;

            // Restrict to overlap region using binary search
            let a_start = mm_a.partition_point(|&p| p < overlap_start_u64);
            let a_end = mm_a.partition_point(|&p| p < overlap_end_u64);
            let b_start = mm_b.partition_point(|&p| p < overlap_start_u64);
            let b_end = mm_b.partition_point(|&p| p < overlap_end_u64);

            let discordant = sorted_symmetric_difference_count(
                &mm_a[a_start..a_end],
                &mm_b[b_start..b_end],
            );

            let identity = if overlap_len > 0 {
                1.0 - (discordant as f64 / overlap_len as f64)
            } else {
                0.0
            };

            if identity >= cutoff {
                results.push(PairwiseIdentity {
                    group_a: ga.to_string(),
                    group_b: gb.to_string(),
                    identity,
                    overlap_bp: overlap_len,
                    a_length: cov_a.1 - cov_a.0,
                    b_length: cov_b.1 - cov_b.0,
                });
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // extract_hap_id
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_hap_id_with_contig() {
        assert_eq!(extract_hap_id("HG00097#1#JBIRDD010000019.1"), "HG00097#1");
    }

    #[test]
    fn test_extract_hap_id_with_chrom() {
        assert_eq!(extract_hap_id("GRCh38#0#chr12"), "GRCh38#0");
    }

    #[test]
    fn test_extract_hap_id_no_contig() {
        assert_eq!(extract_hap_id("HG00097#1"), "HG00097#1");
    }

    #[test]
    fn test_extract_hap_id_no_hash() {
        assert_eq!(extract_hap_id("HG00097"), "HG00097");
    }

    #[test]
    fn test_extract_hap_id_empty() {
        assert_eq!(extract_hap_id(""), "");
    }

    #[test]
    fn test_extract_hap_id_multiple_hashes() {
        assert_eq!(
            extract_hap_id("HG00097#1#contig#extra"),
            "HG00097#1"
        );
    }

    // -----------------------------------------------------------------------
    // extract_sample_from_hap
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_sample_basic() {
        assert_eq!(extract_sample_from_hap("HG00097#1"), "HG00097");
    }

    #[test]
    fn test_extract_sample_no_hash() {
        assert_eq!(extract_sample_from_hap("HG00097"), "HG00097");
    }

    // -----------------------------------------------------------------------
    // extract_target_chrom
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_chrom_chm13() {
        assert_eq!(extract_target_chrom("CHM13#0#chr12"), "chr12");
    }

    #[test]
    fn test_target_chrom_no_hash() {
        assert_eq!(extract_target_chrom("chr12"), "chr12");
    }

    // -----------------------------------------------------------------------
    // parse_cigar_mismatches
    // -----------------------------------------------------------------------

    #[test]
    fn test_cigar_all_match() {
        let (mm, aligned) = parse_cigar_mismatches("100=", 1000);
        assert!(mm.is_empty());
        assert_eq!(aligned, 100);
    }

    #[test]
    fn test_cigar_simple_mismatch() {
        let (mm, aligned) = parse_cigar_mismatches("10=1X10=", 1000);
        assert_eq!(mm, vec![1010]);
        assert_eq!(aligned, 21);
    }

    #[test]
    fn test_cigar_multiple_mismatches() {
        let (mm, aligned) = parse_cigar_mismatches("5=2X3=1X5=", 0);
        assert_eq!(mm, vec![5, 6, 10]);
        assert_eq!(aligned, 16);
    }

    #[test]
    fn test_cigar_with_deletion() {
        // Deletion consumes target positions
        let (mm, aligned) = parse_cigar_mismatches("10=3D10=", 100);
        // Positions 110, 111, 112 are deletions (mismatches)
        assert_eq!(mm, vec![110, 111, 112]);
        assert_eq!(aligned, 23);
    }

    #[test]
    fn test_cigar_with_insertion() {
        // Insertion doesn't consume target
        let (mm, aligned) = parse_cigar_mismatches("10=5I10=", 100);
        assert!(mm.is_empty());
        assert_eq!(aligned, 20); // Only = operations count on target
    }

    #[test]
    fn test_cigar_complex() {
        // From real data: 77=1X30=2X116=
        let (mm, aligned) = parse_cigar_mismatches("77=1X30=2X116=", 0);
        assert_eq!(mm, vec![77, 108, 109]);
        assert_eq!(aligned, 226);
    }

    #[test]
    fn test_cigar_mixed_indels() {
        let (mm, aligned) = parse_cigar_mismatches("5=1X2I3=1D4=", 50);
        // X at position 55, D at position 59 (after 5= + 1X + 2I + 3= = 9 target positions; I doesn't advance target)
        assert_eq!(mm, vec![55, 59]);
        assert_eq!(aligned, 14); // 5 + 1 + 3 + 1 + 4 = 14 target positions
    }

    #[test]
    fn test_cigar_empty() {
        let (mm, aligned) = parse_cigar_mismatches("", 0);
        assert!(mm.is_empty());
        assert_eq!(aligned, 0);
    }

    // -----------------------------------------------------------------------
    // parse_paf_line
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_paf_line_basic() {
        let line = "HG00097#1#contig.1\t100000\t0\t100000\t+\tCHM13#0#chr12\t133324548\t5000000\t5100000\t99000\t100000\t60\tgi:f:0.99\tbi:f:0.98\tcg:Z:99000=1000X";
        let aln = parse_paf_line(line, "chr12", None, 0).unwrap();
        assert_eq!(aln.hap_id, "HG00097#1");
        assert_eq!(aln.target_start, 5000000);
        assert_eq!(aln.target_end, 5100000);
        assert!((aln.gap_identity - 0.99).abs() < 1e-6);
        assert_eq!(aln.mismatch_positions.len(), 1000);
    }

    #[test]
    fn test_parse_paf_line_wrong_chrom() {
        let line = "HG00097#1#contig.1\t100000\t0\t100000\t+\tCHM13#0#chr1\t248387328\t1000\t2000\t900\t1000\t60\tgi:f:0.99\tcg:Z:1000=";
        assert!(parse_paf_line(line, "chr12", None, 0).is_none());
    }

    #[test]
    fn test_parse_paf_line_skips_chm13() {
        let line = "CHM13#0#chrM\t16569\t0\t16569\t+\tCHM13#0#chrM\t16569\t0\t16569\t16569\t16569\t255\tgi:f:1\tcg:Z:16569=";
        assert!(parse_paf_line(line, "chrM", None, 0).is_none());
    }

    #[test]
    fn test_parse_paf_line_skips_grch38() {
        let line = "GRCh38#0#chr12\t133275309\t0\t100000\t+\tCHM13#0#chr12\t133324548\t0\t100000\t99000\t100000\t60\tgi:f:0.99\tcg:Z:100000=";
        assert!(parse_paf_line(line, "chr12", None, 0).is_none());
    }

    #[test]
    fn test_parse_paf_line_with_subset_filter() {
        let line = "HG00097#1#contig.1\t100000\t0\t100000\t+\tCHM13#0#chr12\t133324548\t0\t100000\t99000\t100000\t60\tgi:f:0.99\tcg:Z:100000=";
        let mut subset = HashSet::new();
        subset.insert("HG00097#1".to_string());
        assert!(parse_paf_line(line, "chr12", Some(&subset), 0).is_some());

        let mut other_subset = HashSet::new();
        other_subset.insert("HG00099#1".to_string());
        assert!(parse_paf_line(line, "chr12", Some(&other_subset), 0).is_none());
    }

    #[test]
    fn test_parse_paf_line_min_length_filter() {
        let line = "HG00097#1#contig.1\t500\t0\t500\t+\tCHM13#0#chr12\t133324548\t0\t500\t490\t500\t20\tgi:f:0.98\tcg:Z:500=";
        assert!(parse_paf_line(line, "chr12", None, 0).is_some());
        assert!(parse_paf_line(line, "chr12", None, 1000).is_none());
    }

    #[test]
    fn test_parse_paf_line_no_cigar() {
        let line = "HG00097#1#contig.1\t100000\t0\t100000\t+\tCHM13#0#chr12\t133324548\t5000\t10000\t4800\t5000\t30\tgi:f:0.96";
        let aln = parse_paf_line(line, "chr12", None, 0).unwrap();
        assert!(aln.mismatch_positions.is_empty()); // fallback: no mismatches without CIGAR
        assert_eq!(aln.aligned_bases, 5000); // target_end - target_start
    }

    // -----------------------------------------------------------------------
    // compute_window_pairwise
    // -----------------------------------------------------------------------

    fn make_alignment(hap: &str, start: u64, end: u64, mismatches: Vec<u64>) -> PafAlignment {
        PafAlignment {
            hap_id: hap.to_string(),
            target_start: start,
            target_end: end,
            gap_identity: 1.0 - (mismatches.len() as f64 / (end - start) as f64),
            mismatch_positions: mismatches,
            aligned_bases: end - start,
        }
    }

    #[test]
    fn test_pairwise_perfect_match() {
        // Two haplotypes, same mismatch positions → identity = 1.0
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![10, 20, 30]),
            make_alignment("B#1", 0, 100, vec![10, 20, 30]),
        ];
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        assert!((results[0].identity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_no_shared_mismatches() {
        // Two haplotypes, different mismatch positions → lower identity
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![10, 20]),
            make_alignment("B#1", 0, 100, vec![50, 60]),
        ];
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        // 4 discordant positions out of 100 → identity = 0.96
        assert!((results[0].identity - 0.96).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_partial_overlap() {
        // A covers [0, 100), B covers [50, 150)
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![60, 70]),
            make_alignment("B#1", 50, 150, vec![60, 80]),
        ];
        // Window [0, 150): overlap is [50, 100)
        let results = compute_window_pairwise(&alns, 0, 150, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        // Overlap [50, 100) = 50 bp
        // A mismatches at 60, 70 (both in overlap)
        // B mismatches at 60, 80 (both in overlap)
        // Symmetric diff: {70, 80} → 2 discordant
        // identity = 1 - 2/50 = 0.96
        assert!((results[0].identity - 0.96).abs() < 1e-10);
        assert_eq!(results[0].overlap_bp, 50);
    }

    #[test]
    fn test_pairwise_no_overlap() {
        // Non-overlapping alignments → no pairs
        let alns = vec![
            make_alignment("A#1", 0, 50, vec![]),
            make_alignment("B#1", 100, 200, vec![]),
        ];
        let results = compute_window_pairwise(&alns, 0, 200, "CHM13", None, None, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_pairwise_cutoff_filter() {
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![10, 20, 30, 40, 50]),
            make_alignment("B#1", 0, 100, vec![]),
        ];
        // 5 discordant out of 100 → identity = 0.95
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.96);
        assert!(results.is_empty()); // Filtered by cutoff

        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.95);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_pairwise_query_ref_filter() {
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![]),
            make_alignment("B#1", 0, 100, vec![]),
            make_alignment("C#1", 0, 100, vec![]),
        ];
        let mut qf = HashSet::new();
        qf.insert("A".to_string());
        let mut rf = HashSet::new();
        rf.insert("B".to_string());

        let results = compute_window_pairwise(
            &alns, 0, 100, "CHM13",
            Some(&qf), Some(&rf), 0.0,
        );
        // Only A-B cross pair should be emitted (not A-C or B-C)
        assert_eq!(results.len(), 1);
        assert!(results[0].group_a == "A#1" || results[0].group_b == "A#1");
        assert!(results[0].group_a == "B#1" || results[0].group_b == "B#1");
    }

    #[test]
    fn test_pairwise_canonical_order() {
        let alns = vec![
            make_alignment("Z#1", 0, 100, vec![]),
            make_alignment("A#1", 0, 100, vec![]),
        ];
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].group_a, "A#1"); // A < Z
        assert_eq!(results[0].group_b, "Z#1");
    }

    #[test]
    fn test_pairwise_multiple_alignments_same_hap() {
        // Two alignment blocks from the same haplotype (different contigs)
        let alns = vec![
            make_alignment("A#1", 0, 50, vec![10]),
            make_alignment("A#1", 50, 100, vec![70]),
            make_alignment("B#1", 0, 100, vec![10, 70]),
        ];
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        // A: mismatches at {10, 70}, B: mismatches at {10, 70}
        // Symmetric diff = empty → identity = 1.0
        assert!((results[0].identity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_window_subset() {
        // Alignment covers [0, 200) but window is [100, 200)
        let alns = vec![
            make_alignment("A#1", 0, 200, vec![50, 150]),
            make_alignment("B#1", 0, 200, vec![50, 160]),
        ];
        let results = compute_window_pairwise(&alns, 100, 200, "CHM13", None, None, 0.0);
        assert_eq!(results.len(), 1);
        // In window [100, 200):
        // A mismatches: {150}
        // B mismatches: {160}
        // Symmetric diff: {150, 160} = 2 discordant
        // identity = 1 - 2/100 = 0.98
        assert!((results[0].identity - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_three_way_pairwise() {
        let alns = vec![
            make_alignment("A#1", 0, 100, vec![10]),
            make_alignment("B#1", 0, 100, vec![10]),
            make_alignment("C#1", 0, 100, vec![20]),
        ];
        let results = compute_window_pairwise(&alns, 0, 100, "CHM13", None, None, 0.0);
        // 3 pairs: A-B, A-C, B-C
        assert_eq!(results.len(), 3);
    }

    // -----------------------------------------------------------------------
    // sorted_symmetric_difference_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_sym_diff_empty_both() {
        assert_eq!(sorted_symmetric_difference_count(&[], &[]), 0);
    }

    #[test]
    fn test_sym_diff_one_empty() {
        assert_eq!(sorted_symmetric_difference_count(&[1, 2, 3], &[]), 3);
        assert_eq!(sorted_symmetric_difference_count(&[], &[4, 5]), 2);
    }

    #[test]
    fn test_sym_diff_identical() {
        assert_eq!(sorted_symmetric_difference_count(&[1, 5, 10], &[1, 5, 10]), 0);
    }

    #[test]
    fn test_sym_diff_disjoint() {
        assert_eq!(sorted_symmetric_difference_count(&[1, 2], &[3, 4]), 4);
    }

    #[test]
    fn test_sym_diff_partial_overlap() {
        assert_eq!(sorted_symmetric_difference_count(&[1, 3, 5], &[2, 3, 4]), 4);
    }

    // -----------------------------------------------------------------------
    // read_paf_alignments (file I/O)
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_paf_basic_file() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "HG00097#1#contig.1\t100000\t0\t100000\t+\tCHM13#0#chr12\t133324548\t5000000\t5100000\t99000\t100000\t60\tgi:f:0.99\tcg:Z:99000=1000X\n\
                       HG00099#1#contig.2\t80000\t0\t80000\t+\tCHM13#0#chr12\t133324548\t5020000\t5100000\t79000\t80000\t50\tgi:f:0.98\tcg:Z:79000=1000X\n\
                       HG00097#1#contig.3\t50000\t0\t50000\t+\tCHM13#0#chr1\t248387328\t1000\t51000\t49000\t50000\t40\tgi:f:0.97\tcg:Z:50000=\n";
        std::fs::write(&paf, content).unwrap();

        let alns = read_paf_alignments(paf.to_str().unwrap(), "chr12", None, 0).unwrap();
        assert_eq!(alns.len(), 2); // Only chr12 alignments
        assert_eq!(alns[0].hap_id, "HG00097#1");
        assert_eq!(alns[1].hap_id, "HG00099#1");
        // Sorted by target_start
        assert!(alns[0].target_start <= alns[1].target_start);
    }

    #[test]
    fn test_read_paf_with_subset() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n\
                       HG00099#1#c2\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t200\t300\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n";
        std::fs::write(&paf, content).unwrap();

        let mut subset = HashSet::new();
        subset.insert("HG00097#1".to_string());
        let alns = read_paf_alignments(paf.to_str().unwrap(), "chr12", Some(&subset), 0).unwrap();
        assert_eq!(alns.len(), 1);
        assert_eq!(alns[0].hap_id, "HG00097#1");
    }

    #[test]
    fn test_read_paf_nonexistent() {
        let result = read_paf_alignments("/nonexistent.paf", "chr12", None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_paf_skips_reference_haps() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "CHM13#0#chrM\t16569\t0\t16569\t+\tCHM13#0#chrM\t16569\t0\t16569\t16569\t16569\t255\tgi:f:1\tcg:Z:16569=\n\
                       GRCh38#0#chr12\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t100\t100\t60\tgi:f:1.0\tcg:Z:100=\n\
                       HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n";
        std::fs::write(&paf, content).unwrap();

        let alns = read_paf_alignments(paf.to_str().unwrap(), "chr12", None, 0).unwrap();
        assert_eq!(alns.len(), 1); // Only HG00097, not CHM13 or GRCh38
        assert_eq!(alns[0].hap_id, "HG00097#1");
    }

    // -----------------------------------------------------------------------
    // read_paf_alignments_multi
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_paf_multi_basic() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t133324548\t5000\t5100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n\
                       HG00097#1#c2\t100\t0\t100\t+\tCHM13#0#chr1\t248387328\t1000\t1100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n\
                       HG00099#1#c3\t100\t0\t100\t+\tCHM13#0#chr12\t133324548\t6000\t6100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n\
                       HG00099#1#c4\t100\t0\t100\t+\tCHM13#0#chr20\t64444167\t2000\t2100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n";
        std::fs::write(&paf, content).unwrap();

        let mut chroms = HashSet::new();
        chroms.insert("chr12".to_string());
        chroms.insert("chr1".to_string());

        let result = read_paf_alignments_multi(paf.to_str().unwrap(), &chroms, None, 0).unwrap();

        // Should have chr12 and chr1, not chr20
        assert!(result.contains_key("chr12"));
        assert!(result.contains_key("chr1"));
        assert!(!result.contains_key("chr20"));

        assert_eq!(result["chr12"].len(), 2);
        assert_eq!(result["chr1"].len(), 1);

        // chr12 alignments should be sorted by target_start
        assert!(result["chr12"][0].target_start <= result["chr12"][1].target_start);
    }

    #[test]
    fn test_read_paf_multi_matches_single() {
        // Verify multi-read produces identical results to single-read
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "HG00097#1#c1\t100000\t0\t100000\t+\tCHM13#0#chr12\t133324548\t5000000\t5100000\t99000\t100000\t60\tgi:f:0.99\tcg:Z:99000=1000X\n\
                       HG00099#1#c2\t80000\t0\t80000\t+\tCHM13#0#chr12\t133324548\t5020000\t5100000\t79000\t80000\t50\tgi:f:0.98\tcg:Z:79000=1000X\n\
                       HG00097#1#c3\t50000\t0\t50000\t+\tCHM13#0#chr1\t248387328\t1000\t51000\t49000\t50000\t40\tgi:f:0.97\tcg:Z:50000=\n";
        std::fs::write(&paf, content).unwrap();

        // Single-read
        let single_chr12 = read_paf_alignments(paf.to_str().unwrap(), "chr12", None, 0).unwrap();
        let single_chr1 = read_paf_alignments(paf.to_str().unwrap(), "chr1", None, 0).unwrap();

        // Multi-read
        let mut chroms = HashSet::new();
        chroms.insert("chr12".to_string());
        chroms.insert("chr1".to_string());
        let multi = read_paf_alignments_multi(paf.to_str().unwrap(), &chroms, None, 0).unwrap();

        // Compare chr12
        assert_eq!(single_chr12.len(), multi["chr12"].len());
        for (s, m) in single_chr12.iter().zip(multi["chr12"].iter()) {
            assert_eq!(s.hap_id, m.hap_id);
            assert_eq!(s.target_start, m.target_start);
            assert_eq!(s.target_end, m.target_end);
            assert_eq!(s.mismatch_positions, m.mismatch_positions);
        }

        // Compare chr1
        assert_eq!(single_chr1.len(), multi["chr1"].len());
        for (s, m) in single_chr1.iter().zip(multi["chr1"].iter()) {
            assert_eq!(s.hap_id, m.hap_id);
            assert_eq!(s.target_start, m.target_start);
        }
    }

    #[test]
    fn test_read_paf_multi_with_subset() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n\
                       HG00099#1#c2\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t200\t300\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n";
        std::fs::write(&paf, content).unwrap();

        let mut chroms = HashSet::new();
        chroms.insert("chr12".to_string());

        let mut subset = HashSet::new();
        subset.insert("HG00097#1".to_string());

        let result = read_paf_alignments_multi(
            paf.to_str().unwrap(), &chroms, Some(&subset), 0
        ).unwrap();

        assert_eq!(result["chr12"].len(), 1);
        assert_eq!(result["chr12"][0].hap_id, "HG00097#1");
    }

    #[test]
    fn test_read_paf_multi_nonexistent() {
        let result = read_paf_alignments_multi(
            "/nonexistent.paf",
            &HashSet::new(),
            None,
            0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_read_paf_multi_empty_chroms() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        std::fs::write(&paf, "HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n").unwrap();

        let result = read_paf_alignments_multi(
            paf.to_str().unwrap(),
            &HashSet::new(),
            None,
            0,
        ).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_read_paf_multi_skips_references() {
        let dir = tempfile::tempdir().unwrap();
        let paf = dir.path().join("test.paf");
        let content = "CHM13#0#chrM\t16569\t0\t16569\t+\tCHM13#0#chrM\t16569\t0\t16569\t16569\t16569\t255\tgi:f:1\tcg:Z:16569=\n\
                       GRCh38#0#chr12\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t100\t100\t60\tgi:f:1.0\tcg:Z:100=\n\
                       HG00097#1#c1\t100\t0\t100\t+\tCHM13#0#chr12\t1000\t0\t100\t99\t100\t60\tgi:f:0.99\tcg:Z:100=\n";
        std::fs::write(&paf, content).unwrap();

        let mut chroms = HashSet::new();
        chroms.insert("chr12".to_string());
        chroms.insert("chrM".to_string());

        let result = read_paf_alignments_multi(paf.to_str().unwrap(), &chroms, None, 0).unwrap();

        // chr12: only HG00097 (GRCh38 skipped)
        assert_eq!(result.get("chr12").map(|v| v.len()).unwrap_or(0), 1);
        // chrM: CHM13 skipped, so no alignments
        assert_eq!(result.get("chrM").map(|v| v.len()).unwrap_or(0), 0);
    }
}
