//! Parser for hap-ibd output files.
//!
//! hap-ibd (Browning & Browning) produces `.ibd` files with tab-separated fields:
//! ```text
//! sample1  hap1  sample2  hap2  chr  start_bp  end_bp  LOD_score
//! ```
//!
//! This module provides:
//! - [`HapIbdSegment`]: a parsed IBD segment from hap-ibd output
//! - [`parse_hapibd_file`]: read and parse a `.ibd` file
//! - [`parse_hapibd_content`]: parse hap-ibd output from a string
//! - [`hapibd_segments_for_pair`]: filter segments by sample pair
//! - [`hapibd_segments_for_chr`]: filter segments by chromosome
//! - [`hapibd_segments_above_lod`]: filter segments by minimum LOD score
//! - [`unique_pairs`]: extract unique sample pairs from segments
//!
//! ## Example
//!
//! ```rust,ignore
//! use impopk_ibd::hapibd::{parse_hapibd_file, hapibd_segments_for_pair};
//!
//! let segments = parse_hapibd_file("results.ibd").unwrap();
//! let pair_segs = hapibd_segments_for_pair(&segments, "HG00733", "NA12878");
//! for seg in &pair_segs {
//!     println!("{}\t{}\t{}\tLOD={:.1}", seg.chr, seg.start, seg.end, seg.lod);
//! }
//! ```

use std::path::Path;

/// A single IBD segment from hap-ibd output.
#[derive(Debug, Clone, PartialEq)]
pub struct HapIbdSegment {
    /// First sample identifier
    pub sample1: String,
    /// Haplotype index for sample1 (1 or 2)
    pub hap1: u8,
    /// Second sample identifier
    pub sample2: String,
    /// Haplotype index for sample2 (1 or 2)
    pub hap2: u8,
    /// Chromosome name
    pub chr: String,
    /// Start position in base pairs
    pub start: u64,
    /// End position in base pairs
    pub end: u64,
    /// LOD score (log of odds)
    pub lod: f64,
}

impl HapIbdSegment {
    /// Length of this segment in base pairs.
    pub fn length_bp(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Returns the segment as a (start, end) tuple for concordance analysis.
    pub fn as_interval(&self) -> (u64, u64) {
        (self.start, self.end)
    }

    /// Check if this segment involves the given sample (as either sample1 or sample2).
    pub fn involves_sample(&self, sample: &str) -> bool {
        self.sample1 == sample || self.sample2 == sample
    }

    /// Check if this segment involves both given samples (order-independent).
    pub fn involves_pair(&self, s1: &str, s2: &str) -> bool {
        (self.sample1 == s1 && self.sample2 == s2)
            || (self.sample1 == s2 && self.sample2 == s1)
    }
}

/// Parse a hap-ibd `.ibd` output file.
///
/// Handles:
/// - Comment lines (starting with `#`)
/// - Empty/whitespace-only lines
/// - Tab-separated fields: sample1, hap1, sample2, hap2, chr, start, end, LOD
///
/// Returns an error if the file cannot be read. Lines with parse errors are
/// skipped with a warning to stderr.
pub fn parse_hapibd_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<HapIbdSegment>> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to read hap-ibd file {:?}: {}", path.as_ref(), e))?;
    Ok(parse_hapibd_content(&content))
}

/// Parse hap-ibd content from a string.
///
/// This is the core parser, useful for testing without files.
pub fn parse_hapibd_content(content: &str) -> Vec<HapIbdSegment> {
    let mut segments = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_hapibd_line(line) {
            Some(seg) => segments.push(seg),
            None => {
                eprintln!(
                    "Warning: skipping malformed hap-ibd line {}: {}",
                    line_num + 1,
                    line
                );
            }
        }
    }
    segments
}

/// Parse a single tab-separated hap-ibd line.
fn parse_hapibd_line(line: &str) -> Option<HapIbdSegment> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 8 {
        return None;
    }
    let hap1 = fields[1].parse::<u8>().ok()?;
    let hap2 = fields[3].parse::<u8>().ok()?;
    let start = fields[5].parse::<u64>().ok()?;
    let end = fields[6].parse::<u64>().ok()?;
    let lod = fields[7].parse::<f64>().ok()?;

    Some(HapIbdSegment {
        sample1: fields[0].to_string(),
        hap1,
        sample2: fields[2].to_string(),
        hap2,
        chr: fields[4].to_string(),
        start,
        end,
        lod,
    })
}

/// Filter hap-ibd segments for a specific sample pair.
///
/// Returns all segments involving both `s1` and `s2` (order-independent).
pub fn hapibd_segments_for_pair<'a>(
    segments: &'a [HapIbdSegment],
    s1: &str,
    s2: &str,
) -> Vec<&'a HapIbdSegment> {
    segments
        .iter()
        .filter(|seg| seg.involves_pair(s1, s2))
        .collect()
}

/// Filter hap-ibd segments for a specific chromosome.
pub fn hapibd_segments_for_chr<'a>(
    segments: &'a [HapIbdSegment],
    chr: &str,
) -> Vec<&'a HapIbdSegment> {
    segments.iter().filter(|seg| seg.chr == chr).collect()
}

/// Filter segments by minimum LOD score.
pub fn hapibd_segments_above_lod(
    segments: &[HapIbdSegment],
    min_lod: f64,
) -> Vec<&HapIbdSegment> {
    segments.iter().filter(|seg| seg.lod >= min_lod).collect()
}

/// Get all unique sample pairs from hap-ibd segments.
pub fn unique_pairs(segments: &[HapIbdSegment]) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = segments
        .iter()
        .map(|seg| {
            let (a, b) = if seg.sample1 <= seg.sample2 {
                (seg.sample1.clone(), seg.sample2.clone())
            } else {
                (seg.sample2.clone(), seg.sample1.clone())
            };
            (a, b)
        })
        .collect();
    pairs.sort();
    pairs.dedup();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ibd_content() -> &'static str {
        "# hap-ibd output\n\
         # sample1\thap1\tsample2\thap2\tchr\tstart\tend\tLOD\n\
         HG00733\t1\tNA12878\t1\tchr20\t1000000\t5000000\t12.5\n\
         HG00733\t1\tNA12878\t2\tchr20\t8000000\t12000000\t8.3\n\
         HG00733\t2\tHG00514\t1\tchr20\t3000000\t7000000\t15.7\n\
         NA12878\t1\tHG00514\t2\tchr20\t500000\t2000000\t4.1\n\
         HG00733\t1\tNA12878\t1\tchr15\t20000000\t25000000\t20.0\n"
    }

    #[test]
    fn test_parse_hapibd_content_basic() {
        let segments = parse_hapibd_content(sample_ibd_content());
        assert_eq!(segments.len(), 5);
    }

    #[test]
    fn test_parse_hapibd_fields() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let seg = &segments[0];
        assert_eq!(seg.sample1, "HG00733");
        assert_eq!(seg.hap1, 1);
        assert_eq!(seg.sample2, "NA12878");
        assert_eq!(seg.hap2, 1);
        assert_eq!(seg.chr, "chr20");
        assert_eq!(seg.start, 1_000_000);
        assert_eq!(seg.end, 5_000_000);
        assert!((seg.lod - 12.5).abs() < 1e-9);
    }

    #[test]
    fn test_parse_hapibd_skips_comments_and_empty() {
        let content = "# comment\n\n  \n\
                        HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n\
                        # another comment\n";
        let segments = parse_hapibd_content(content);
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_parse_hapibd_skips_malformed_lines() {
        let content = "HG001\t1\tHG002\t2\tchr1\t100\t200\t5.0\n\
                        bad_line_too_few_fields\n\
                        HG001\t1\tHG002\tX\tchr1\t100\t200\t5.0\n\
                        HG003\t1\tHG004\t2\tchr1\t300\t400\t6.0\n";
        let segments = parse_hapibd_content(content);
        assert_eq!(segments.len(), 2); // first and last lines only
    }

    #[test]
    fn test_hapibd_segment_length() {
        let seg = HapIbdSegment {
            sample1: "A".into(),
            hap1: 1,
            sample2: "B".into(),
            hap2: 1,
            chr: "chr1".into(),
            start: 1_000_000,
            end: 5_000_000,
            lod: 10.0,
        };
        assert_eq!(seg.length_bp(), 4_000_000);
    }

    #[test]
    fn test_hapibd_segments_for_pair() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let pair = hapibd_segments_for_pair(&segments, "HG00733", "NA12878");
        assert_eq!(pair.len(), 3); // 2 on chr20 + 1 on chr15

        // Order-independent
        let pair_rev = hapibd_segments_for_pair(&segments, "NA12878", "HG00733");
        assert_eq!(pair_rev.len(), 3);
    }

    #[test]
    fn test_hapibd_segments_for_pair_no_match() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let pair = hapibd_segments_for_pair(&segments, "HG00733", "NONEXISTENT");
        assert!(pair.is_empty());
    }

    #[test]
    fn test_hapibd_segments_for_chr() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let chr20 = hapibd_segments_for_chr(&segments, "chr20");
        assert_eq!(chr20.len(), 4);
        let chr15 = hapibd_segments_for_chr(&segments, "chr15");
        assert_eq!(chr15.len(), 1);
    }

    #[test]
    fn test_hapibd_segments_above_lod() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let high_lod = hapibd_segments_above_lod(&segments, 10.0);
        assert_eq!(high_lod.len(), 3); // 12.5, 15.7, 20.0
    }

    #[test]
    fn test_unique_pairs() {
        let segments = parse_hapibd_content(sample_ibd_content());
        let pairs = unique_pairs(&segments);
        assert_eq!(pairs.len(), 3);
        // All pairs should be in canonical (sorted) order
        for (a, b) in &pairs {
            assert!(a <= b);
        }
    }

    #[test]
    fn test_parse_hapibd_empty_content() {
        let segments = parse_hapibd_content("");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_parse_hapibd_only_comments() {
        let content = "# header line 1\n# header line 2\n";
        let segments = parse_hapibd_content(content);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_hapibd_involves_sample() {
        let seg = HapIbdSegment {
            sample1: "HG00733".into(),
            hap1: 1,
            sample2: "NA12878".into(),
            hap2: 1,
            chr: "chr20".into(),
            start: 100,
            end: 200,
            lod: 5.0,
        };
        assert!(seg.involves_sample("HG00733"));
        assert!(seg.involves_sample("NA12878"));
        assert!(!seg.involves_sample("HG00514"));
    }

    #[test]
    fn test_hapibd_as_interval() {
        let seg = HapIbdSegment {
            sample1: "A".into(),
            hap1: 1,
            sample2: "B".into(),
            hap2: 1,
            chr: "chr1".into(),
            start: 1000,
            end: 5000,
            lod: 3.0,
        };
        assert_eq!(seg.as_interval(), (1000, 5000));
    }
}
