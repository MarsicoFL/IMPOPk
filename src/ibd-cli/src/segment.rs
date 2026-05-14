#![allow(clippy::manual_is_multiple_of)]
//! Segment Detection and Merging Algorithms
//!
//! This module provides tools for detecting and managing IBS/IBD segments
//! in sliding window analysis results.
//!
//! ## Overview
//!
//! The segment detection pipeline:
//! 1. Track per-window identity values for each haplotype pair
//! 2. Detect contiguous high-identity regions using run-length encoding (RLE)
//! 3. Filter segments by minimum length and quality criteria
//! 4. Merge overlapping segments from the same haplotype pair
//!
//! ## Key Types
//!
//! - [`Segment`]: Represents a detected IBS/IBD segment with genomic coordinates
//! - [`RleParams`]: Configuration for segment detection thresholds
//! - [`IdentityTrack`]: Per-window identity values for a haplotype pair
//!
//! ## Example
//!
//! ```rust,ignore
//! use impopk_ibd::segment::{RleParams, IdentityTrack, detect_segments_rle};
//!
//! // Create identity track for a haplotype pair
//! let track = IdentityTrack {
//!     windows: vec![(0, 0.999), (1, 0.998), (2, 0.9995), (3, 0.50)],
//!     n_total_windows: 4,
//! };
//!
//! // Window positions (start, end)
//! let positions = vec![(0, 4999), (5000, 9999), (10000, 14999), (15000, 19999)];
//!
//! // Use default parameters
//! let params = RleParams::default();
//!
//! // Detect segments
//! let segments = detect_segments_rle(&track, &positions, &params, "chr1", "HapA", "HapB");
//! ```

use crate::stats::OnlineStats;

/// A detected IBS/IBD segment between two haplotypes.
///
/// Represents a contiguous genomic region where two haplotypes show
/// high sequence identity, potentially indicating shared ancestry.
///
/// ## Fields
///
/// - `chrom`: Chromosome name
/// - `start`, `end`: Genomic coordinates (1-based, inclusive)
/// - `hap_a`, `hap_b`: Haplotype identifiers (normalized: hap_a <= hap_b lexicographically)
/// - `n_windows`: Number of analysis windows in the segment
/// - `mean_identity`: Average sequence identity across the segment
/// - `min_identity`: Lowest identity value observed in any window
/// - `identity_sum`: Sum of identity values (for re-averaging after merge)
/// - `n_called`: Number of windows with valid identity data
/// - `start_idx`, `end_idx`: Window indices (for correct merge calculations)
#[derive(Debug, Clone)]
pub struct Segment {
    /// Chromosome name
    pub chrom: String,
    /// Segment start position (bp)
    pub start: u64,
    /// Segment end position (bp)
    pub end: u64,
    /// First haplotype identifier (normalized: hap_a <= hap_b)
    pub hap_a: String,
    /// Second haplotype identifier (normalized: hap_a <= hap_b)
    pub hap_b: String,
    /// Number of windows in the segment
    pub n_windows: usize,
    /// Average identity across windows
    pub mean_identity: f64,
    /// Minimum identity observed
    pub min_identity: f64,
    /// Sum of identity values (for merging)
    pub identity_sum: f64,
    /// Number of windows with data
    pub n_called: usize,
    /// Start window index (inclusive)
    pub start_idx: usize,
    /// End window index (inclusive)
    pub end_idx: usize,
}

impl Segment {
    /// Calculate segment length in base pairs.
    ///
    /// Uses saturating subtraction to avoid underflow if start > end.
    pub fn length_bp(&self) -> u64 {
        self.end.saturating_sub(self.start) + 1
    }

    /// Calculate fraction of windows with identity data.
    ///
    /// Returns `n_called / n_windows`, or 0.0 if n_windows is 0.
    pub fn fraction_called(&self) -> f64 {
        if self.n_windows == 0 { 0.0 } else { self.n_called as f64 / self.n_windows as f64 }
    }
}

/// Parameters for run-length encoding (RLE) based segment detection.
///
/// These parameters control the sensitivity and specificity of segment calling.
///
/// ## Fields
///
/// - `min_identity`: Minimum identity threshold to consider a window as "high identity"
/// - `max_gap`: Maximum consecutive missing/low-identity windows to bridge
/// - `min_windows`: Minimum windows required for a valid segment
/// - `min_length_bp`: Minimum segment length in base pairs
/// - `drop_tolerance`: Extra tolerance below min_identity (effective threshold = min_identity - drop_tolerance)
#[derive(Debug, Clone)]
pub struct RleParams {
    /// Minimum identity threshold (default: 0.9995)
    pub min_identity: f64,
    /// Maximum gap windows to bridge (default: 1)
    pub max_gap: usize,
    /// Minimum windows per segment (default: 3)
    pub min_windows: usize,
    /// Minimum segment length in bp (default: 5000)
    pub min_length_bp: u64,
    /// Tolerance below min_identity (default: 0.0)
    pub drop_tolerance: f64,
}

impl Default for RleParams {
    fn default() -> Self {
        Self {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.0,
        }
    }
}

/// Track of per-window identities
pub struct IdentityTrack {
    pub windows: Vec<(usize, f64)>,
    pub n_total_windows: usize,
}

impl IdentityTrack {
    pub fn get(&self, idx: usize) -> Option<f64> {
        self.windows.iter().find(|(i, _)| *i == idx).map(|(_, ident)| *ident)
    }

    pub fn to_map(&self) -> std::collections::HashMap<usize, f64> {
        self.windows.iter().cloned().collect()
    }
}

/// RLE-based segment detection
pub fn detect_segments_rle(
    track: &IdentityTrack,
    window_positions: &[(u64, u64)],
    params: &RleParams,
    chrom: &str,
    hap_a: &str,
    hap_b: &str,
) -> Vec<Segment> {
    let ident_map = track.to_map();
    let n = track.n_total_windows;
    let effective_threshold = params.min_identity - params.drop_tolerance;

    let mut segments = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_end = 0;
    let mut gaps = 0;
    let mut stats = OnlineStats::new();
    let mut min_ident = 1.0_f64;

    for i in 0..n {
        let ident = ident_map.get(&i).copied();
        let missing = ident.is_none();
        let good = ident.is_some_and(|id| id >= effective_threshold);

        match current_start {
            None => {
                if good {
                    current_start = Some(i);
                    current_end = i;
                    gaps = 0;
                    stats = OnlineStats::new();
                    stats.add(ident.unwrap());
                    min_ident = ident.unwrap();
                }
            }
            Some(start) => {
                if good || missing {
                    current_end = i;
                    if missing {
                        gaps += 1;
                    } else {
                        let id = ident.unwrap();
                        stats.add(id);
                        if id < min_ident {
                            min_ident = id;
                        }
                    }

                    if gaps > params.max_gap {
                        if let Some(seg) = finalize_segment(
                            start, i - 1, window_positions, &stats, min_ident, params, chrom, hap_a, hap_b,
                        ) {
                            segments.push(seg);
                        }
                        current_start = None;
                        gaps = 0;
                        stats = OnlineStats::new();
                        min_ident = 1.0;
                    }
                } else {
                    if let Some(seg) = finalize_segment(
                        start, current_end, window_positions, &stats, min_ident, params, chrom, hap_a, hap_b,
                    ) {
                        segments.push(seg);
                    }

                    current_start = None;
                    gaps = 0;
                    stats = OnlineStats::new();
                    min_ident = 1.0;

                    if good {
                        current_start = Some(i);
                        current_end = i;
                        stats.add(ident.unwrap());
                        min_ident = ident.unwrap();
                    }
                }
            }
        }
    }

    if let Some(start) = current_start {
        if let Some(seg) = finalize_segment(
            start, current_end, window_positions, &stats, min_ident, params, chrom, hap_a, hap_b,
        ) {
            segments.push(seg);
        }
    }

    segments
}

#[allow(clippy::too_many_arguments)]
fn finalize_segment(
    start_idx: usize,
    end_idx: usize,
    window_positions: &[(u64, u64)],
    stats: &OnlineStats,
    min_ident: f64,
    params: &RleParams,
    chrom: &str,
    hap_a: &str,
    hap_b: &str,
) -> Option<Segment> {
    let n_windows = end_idx - start_idx + 1;
    if n_windows < params.min_windows {
        return None;
    }

    let start_bp = window_positions.get(start_idx)?.0;
    let end_bp = window_positions.get(end_idx)?.1;
    let length = end_bp.saturating_sub(start_bp) + 1;

    if length < params.min_length_bp {
        return None;
    }

    // Normalize haplotype order: ensure hap_a <= hap_b lexicographically
    let (norm_hap_a, norm_hap_b) = normalize_haplotype_pair(hap_a, hap_b);

    Some(Segment {
        chrom: chrom.to_string(),
        start: start_bp,
        end: end_bp,
        hap_a: norm_hap_a,
        hap_b: norm_hap_b,
        n_windows,
        mean_identity: stats.mean(),
        min_identity: min_ident,
        identity_sum: stats.mean() * stats.count() as f64,
        n_called: stats.count(),
        start_idx,
        end_idx,
    })
}

/// Normalize haplotype pair so that hap_a <= hap_b lexicographically.
/// This ensures that (A, B) and (B, A) are treated as the same pair.
fn normalize_haplotype_pair(hap_a: &str, hap_b: &str) -> (String, String) {
    if hap_a <= hap_b {
        (hap_a.to_string(), hap_b.to_string())
    } else {
        (hap_b.to_string(), hap_a.to_string())
    }
}

/// Merge overlapping segments
///
/// When two segments overlap, this function correctly handles the overlap by:
/// 1. Computing n_windows based on the merged window index range (avoiding double-counting)
/// 2. Estimating identity_sum and n_called proportionally for the overlap region
/// 3. Haplotype pairs are already normalized (A-B == B-A) during segment creation
pub fn merge_segments(segments: &mut Vec<Segment>) {
    if segments.len() < 2 {
        return;
    }

    // Sort by chromosome, haplotype pair, then position
    segments.sort_by(|a, b| {
        a.chrom
            .cmp(&b.chrom)
            .then(a.hap_a.cmp(&b.hap_a))
            .then(a.hap_b.cmp(&b.hap_b))
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
    });

    let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
    merged.push(segments[0].clone());

    for seg in segments.iter().skip(1) {
        let last = merged.last_mut().unwrap();

        // Only merge segments that belong to the same haplotype pair
        // (haplotypes are already normalized, so A-B == B-A)
        let same_haplotypes = seg.hap_a == last.hap_a && seg.hap_b == last.hap_b;

        // Check for overlap using window indices
        // Segments overlap if seg.start_idx <= last.end_idx
        if seg.chrom == last.chrom && same_haplotypes && seg.start_idx <= last.end_idx {
            // Calculate overlap in window indices
            let overlap_start = seg.start_idx;
            let overlap_end = last.end_idx.min(seg.end_idx);
            let overlap_windows = if overlap_end >= overlap_start {
                overlap_end - overlap_start + 1
            } else {
                0
            };

            // Calculate the contribution from seg, excluding the overlap
            // We estimate the overlap's contribution proportionally
            let seg_total_windows = seg.end_idx - seg.start_idx + 1;
            let seg_non_overlap_windows = seg_total_windows.saturating_sub(overlap_windows);

            // Estimate identity_sum and n_called for the non-overlapping part of seg
            // Using proportional estimation based on the segment's average
            let seg_non_overlap_fraction = if seg_total_windows > 0 {
                seg_non_overlap_windows as f64 / seg_total_windows as f64
            } else {
                0.0
            };

            let seg_non_overlap_identity_sum = seg.identity_sum * seg_non_overlap_fraction;
            let seg_non_overlap_n_called =
                (seg.n_called as f64 * seg_non_overlap_fraction).round() as usize;

            // Update the merged segment
            let new_end_idx = last.end_idx.max(seg.end_idx);
            last.end = last.end.max(seg.end);
            last.n_windows = new_end_idx - last.start_idx + 1;
            last.end_idx = new_end_idx;
            last.identity_sum += seg_non_overlap_identity_sum;
            last.n_called += seg_non_overlap_n_called;
            if last.n_called > 0 {
                last.mean_identity = last.identity_sum / last.n_called as f64;
            }
            last.min_identity = last.min_identity.min(seg.min_identity);
        } else {
            merged.push(seg.clone());
        }
    }

    *segments = merged;
}

// =============================================================================
// Segment Length Distribution Analysis
// =============================================================================

/// Summary statistics for a collection of IBD segment lengths.
#[derive(Debug, Clone)]
pub struct SegmentLengthStats {
    /// Number of segments
    pub count: usize,
    /// Mean segment length in bp
    pub mean_bp: f64,
    /// Median segment length in bp
    pub median_bp: f64,
    /// Standard deviation of segment lengths in bp
    pub std_bp: f64,
    /// Minimum segment length in bp
    pub min_bp: u64,
    /// Maximum segment length in bp
    pub max_bp: u64,
    /// Total length of all segments in bp
    pub total_bp: u64,
}

/// Compute summary statistics for segment length distribution.
///
/// Returns zeroed stats for empty input.
pub fn segment_length_distribution(segments: &[Segment]) -> SegmentLengthStats {
    if segments.is_empty() {
        return SegmentLengthStats {
            count: 0,
            mean_bp: 0.0,
            median_bp: 0.0,
            std_bp: 0.0,
            min_bp: 0,
            max_bp: 0,
            total_bp: 0,
        };
    }

    let mut lengths: Vec<u64> = segments.iter().map(|s| s.length_bp()).collect();
    lengths.sort_unstable();

    let count = lengths.len();
    let total: u64 = lengths.iter().sum();
    let mean = total as f64 / count as f64;

    let median = if count % 2 == 0 {
        (lengths[count / 2 - 1] + lengths[count / 2]) as f64 / 2.0
    } else {
        lengths[count / 2] as f64
    };

    let variance = if count > 1 {
        lengths.iter().map(|&l| (l as f64 - mean).powi(2)).sum::<f64>() / (count - 1) as f64
    } else {
        0.0
    };

    SegmentLengthStats {
        count,
        mean_bp: mean,
        median_bp: median,
        std_bp: variance.sqrt(),
        min_bp: lengths[0],
        max_bp: lengths[count - 1],
        total_bp: total,
    }
}

/// Compute a histogram of segment lengths with the given bin size.
///
/// Returns a vector of (bin_start_bp, count) pairs, sorted by bin_start.
/// Each bin covers [bin_start, bin_start + bin_size_bp).
pub fn segment_length_histogram(segments: &[Segment], bin_size_bp: u64) -> Vec<(u64, usize)> {
    if segments.is_empty() || bin_size_bp == 0 {
        return Vec::new();
    }

    let lengths: Vec<u64> = segments.iter().map(|s| s.length_bp()).collect();
    let max_len = *lengths.iter().max().unwrap();

    let n_bins = (max_len / bin_size_bp) as usize + 1;
    let mut counts = vec![0usize; n_bins];

    for &len in &lengths {
        let bin = (len / bin_size_bp) as usize;
        if bin < counts.len() {
            counts[bin] += 1;
        }
    }

    counts
        .into_iter()
        .enumerate()
        .filter(|(_, c)| *c > 0)
        .map(|(i, c)| (i as u64 * bin_size_bp, c))
        .collect()
}

/// Format a segment as a BED line (0-based half-open coordinates).
///
/// BED fields: chrom, start, end, name, score, strand
/// - start is converted from 1-based to 0-based
/// - name = "hapA_hapB"
/// - score = LOD * 100, capped at 1000 (BED convention), minimum 0
/// - strand = "."
pub fn format_segment_bed(seg: &Segment, lod: f64) -> String {
    let bed_start = seg.start.saturating_sub(1);
    let score = if lod > 0.0 {
        ((lod * 100.0).round() as u64).min(1000)
    } else {
        0
    };
    let name = format!("{}_{}", seg.hap_a, seg.hap_b);
    format!("{}\t{}\t{}\t{}\t{}\t.", seg.chrom, bed_start, seg.end, name, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test segment with all required fields
    fn make_segment(
        chrom: &str,
        start: u64,
        end: u64,
        hap_a: &str,
        hap_b: &str,
        start_idx: usize,
        end_idx: usize,
        mean_identity: f64,
        min_identity: f64,
    ) -> Segment {
        let n_windows = end_idx - start_idx + 1;
        let n_called = n_windows;
        Segment {
            chrom: chrom.to_string(),
            start,
            end,
            hap_a: hap_a.to_string(),
            hap_b: hap_b.to_string(),
            n_windows,
            mean_identity,
            min_identity,
            identity_sum: mean_identity * n_called as f64,
            n_called,
            start_idx,
            end_idx,
        }
    }

    #[test]
    fn test_segment_length() {
        let seg = make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998);
        assert_eq!(seg.length_bp(), 1001);
    }

    #[test]
    fn test_merge_segments_same_haplotypes() {
        // Overlapping segments with same haplotype pair should merge
        // seg1: indices [0-10], seg2: indices [5-15] => overlap [5-10] = 6 windows
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 10, 0.999, 0.998),
            make_segment("chr1", 1500, 2500, "A", "B", 5, 15, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 2500);
        // n_windows should be 16 (indices 0-15), not 22 (11+11)
        assert_eq!(segments[0].n_windows, 16);
        assert_eq!(segments[0].start_idx, 0);
        assert_eq!(segments[0].end_idx, 15);
    }

    #[test]
    fn test_merge_segments_different_haplotypes_not_merged() {
        // Overlapping segments with different haplotype pairs should NOT merge
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 10, 0.999, 0.998),
            make_segment("chr1", 1500, 2500, "C", "D", 5, 15, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        // Should remain as 2 separate segments because haplotypes differ
        assert_eq!(segments.len(), 2);
    }

    // === Edge case tests for IdentityTrack ===

    #[test]
    fn test_identity_track_empty() {
        let track = IdentityTrack {
            windows: vec![],
            n_total_windows: 0,
        };
        assert!(track.get(0).is_none());
        let map = track.to_map();
        assert!(map.is_empty());
    }

    #[test]
    fn test_identity_track_sparse() {
        // Track with gaps (sparse windows)
        let track = IdentityTrack {
            windows: vec![(0, 0.999), (5, 0.998), (10, 0.997)],
            n_total_windows: 15,
        };
        assert_eq!(track.get(0), Some(0.999));
        assert_eq!(track.get(5), Some(0.998));
        assert_eq!(track.get(10), Some(0.997));
        // Missing indices should return None
        assert!(track.get(1).is_none());
        assert!(track.get(3).is_none());
        assert!(track.get(14).is_none());
    }

    // === Edge case tests for detect_segments_rle ===

    #[test]
    fn test_detect_segments_rle_empty_track() {
        let track = IdentityTrack {
            windows: vec![],
            n_total_windows: 0,
        };
        let window_positions: Vec<(u64, u64)> = vec![];
        let params = RleParams::default();

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_detect_segments_rle_no_high_identity() {
        // All windows below threshold
        let track = IdentityTrack {
            windows: vec![(0, 0.9), (1, 0.85), (2, 0.88), (3, 0.92), (4, 0.91)],
            n_total_windows: 5,
        };
        let window_positions = vec![
            (0, 999), (1000, 1999), (2000, 2999), (3000, 3999), (4000, 4999),
        ];
        let params = RleParams::default(); // min_identity = 0.9995

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_detect_segments_rle_all_high_identity() {
        // All windows above threshold
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997), (3, 0.9996), (4, 0.9999)],
            n_total_windows: 5,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999), (4000, 5999), (6000, 7999), (8000, 9999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 9999);
        assert_eq!(segments[0].n_windows, 5);
    }

    #[test]
    fn test_detect_segments_rle_gap_at_start() {
        // Missing data at the start
        let track = IdentityTrack {
            windows: vec![(2, 0.9999), (3, 0.9998), (4, 0.9997)],
            n_total_windows: 5,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999), (4000, 5999), (6000, 7999), (8000, 9999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 4000);
        assert_eq!(segments[0].end, 9999);
    }

    #[test]
    fn test_detect_segments_rle_gap_at_end() {
        // Missing data at the end - the segment extends through gaps up to max_gap
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)],
            n_total_windows: 5,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999), (4000, 5999), (6000, 7999), (8000, 9999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        // Segment extends through the gap at index 3, but when gaps > max_gap at index 4,
        // it finalizes at index 3. The current_end was set to 4 (last missing window visited).
        // Actually, the algorithm extends to window 4 (since max_gap=1 allows 1 missing,
        // and windows 3 and 4 are missing which is 2 gaps total, so it extends to 4 then splits)
        assert_eq!(segments[0].end, 7999);
    }

    #[test]
    fn test_detect_segments_rle_gap_in_middle() {
        // Gap in the middle (within tolerance)
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998), (3, 0.9997), (4, 0.9996)],
            n_total_windows: 5,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999), (4000, 5999), (6000, 7999), (8000, 9999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1, // Allow 1 gap
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        // Should bridge the gap
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_detect_segments_rle_gap_exceeds_tolerance() {
        // Gap larger than max_gap should split segments
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998), (5, 0.9997), (6, 0.9996), (7, 0.9995)],
            n_total_windows: 8,
        };
        let window_positions = vec![
            (0, 999), (1000, 1999), (2000, 2999), (3000, 3999),
            (4000, 4999), (5000, 5999), (6000, 6999), (7000, 7999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 2,
            min_length_bp: 1000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        // Should create 2 segments due to gap > max_gap
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_detect_segments_rle_min_windows_filter() {
        // Segment too short (fewer than min_windows)
        // Note: n_windows counts all windows in the range (including missing ones),
        // so we need a truly short segment with no trailing gaps
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998)],
            n_total_windows: 2,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3, // Require at least 3 windows
            min_length_bp: 1000,
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        // Should be empty because segment spans only 2 windows (end_idx - start_idx + 1 = 2)
        assert!(segments.is_empty());
    }

    #[test]
    fn test_detect_segments_rle_min_length_filter() {
        // Segment too short in base pairs
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9998), (2, 0.9997)],
            n_total_windows: 3,
        };
        let window_positions = vec![
            (0, 999), (1000, 1999), (2000, 2999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 1,
            min_length_bp: 5000, // Require at least 5000 bp
            drop_tolerance: 0.0,
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        // Should be empty because segment is only 3000 bp
        assert!(segments.is_empty());
    }

    #[test]
    fn test_detect_segments_rle_with_drop_tolerance() {
        // Test drop_tolerance allowing slightly lower identity
        let track = IdentityTrack {
            windows: vec![(0, 0.9999), (1, 0.9990), (2, 0.9998)],
            n_total_windows: 3,
        };
        let window_positions = vec![
            (0, 1999), (2000, 3999), (4000, 5999),
        ];
        let params = RleParams {
            min_identity: 0.9995,
            max_gap: 1,
            min_windows: 3,
            min_length_bp: 5000,
            drop_tolerance: 0.001, // Effective threshold = 0.9985
        };

        let segments = detect_segments_rle(&track, &window_positions, &params, "chr1", "A", "B");
        // With drop_tolerance, middle window (0.999) should still pass
        assert_eq!(segments.len(), 1);
    }

    // === Edge case tests for merge_segments ===

    #[test]
    fn test_merge_segments_empty() {
        let mut segments: Vec<Segment> = vec![];
        merge_segments(&mut segments);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_merge_segments_single() {
        let mut segments = vec![make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998)];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_merge_segments_non_overlapping() {
        // Non-overlapping segments should not merge
        // seg1: indices [0-9], seg2: indices [20-29] => no overlap
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998),
            make_segment("chr1", 3000, 4000, "A", "B", 20, 29, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_adjacent() {
        // Adjacent segments (end_idx + 1 == start_idx) should not merge
        // seg1: indices [0-9], seg2: indices [11-20] => no overlap (gap at 10)
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998),
            make_segment("chr1", 2001, 3000, "A", "B", 11, 20, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        // Adjacent but not overlapping in window indices, so should remain separate
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_touching() {
        // Segments where second start_idx == first end_idx should merge (1 window overlap)
        // seg1: indices [0-9], seg2: indices [9-18] => overlap at index 9
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998),
            make_segment("chr1", 2000, 3000, "A", "B", 9, 18, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        // Touching segments should merge
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 3000);
        // n_windows should be 19 (indices 0-18), not 20 (10+10)
        assert_eq!(segments[0].n_windows, 19);
    }

    #[test]
    fn test_merge_segments_different_chromosomes() {
        // Segments on different chromosomes should not merge
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998),
            make_segment("chr2", 1000, 2000, "A", "B", 0, 9, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_unsorted_input() {
        // Segments in wrong order should be sorted and merged correctly
        // After sorting: seg2 (start_idx=0) then seg1 (start_idx=5)
        // seg2: indices [0-9], seg1: indices [5-14] => overlap [5-9] = 5 windows
        let mut segments = vec![
            make_segment("chr1", 1500, 2500, "A", "B", 5, 14, 0.998, 0.997),
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998),
        ];

        merge_segments(&mut segments);
        // Should sort and merge
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 2500);
        // n_windows should be 15 (indices 0-14), not 20 (10+10)
        assert_eq!(segments[0].n_windows, 15);
    }

    #[test]
    fn test_merge_segments_multiple_merges() {
        // Multiple overlapping segments should all merge correctly
        // seg1: indices [0-9], seg2: indices [5-14], seg3: indices [10-19]
        // After first merge: [0-14], after second merge: [0-19]
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.999),
            make_segment("chr1", 1500, 2500, "A", "B", 5, 14, 0.998, 0.998),
            make_segment("chr1", 2000, 3000, "A", "B", 10, 19, 0.997, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 3000);
        // n_windows should be 20 (indices 0-19)
        assert_eq!(segments[0].n_windows, 20);
        assert_eq!(segments[0].min_identity, 0.997);
    }

    // === Test for the specific bug: overlapping n_windows double-counting ===

    #[test]
    fn test_merge_segments_overlap_n_windows_correct() {
        // This is the specific bug case mentioned in the task:
        // seg1: indices [0-10] (n_windows=11), seg2: indices [5-15] (n_windows=11)
        // After merge: should be 16 windows (indices 0-15), NOT 22 (11+11)
        let mut segments = vec![
            make_segment("chr1", 0, 10000, "A", "B", 0, 10, 0.999, 0.998),
            make_segment("chr1", 5000, 15000, "A", "B", 5, 15, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].n_windows, 16); // indices 0-15
        assert_eq!(segments[0].start_idx, 0);
        assert_eq!(segments[0].end_idx, 15);
    }

    #[test]
    fn test_merge_segments_complete_overlap() {
        // seg1 completely contains seg2
        // seg1: indices [0-20], seg2: indices [5-15]
        // After merge: should still be 21 windows (indices 0-20)
        let mut segments = vec![
            make_segment("chr1", 0, 20000, "A", "B", 0, 20, 0.999, 0.998),
            make_segment("chr1", 5000, 15000, "A", "B", 5, 15, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].n_windows, 21); // indices 0-20
        assert_eq!(segments[0].end, 20000); // should keep the larger end
    }

    // === Test for haplotype normalization ===

    #[test]
    fn test_haplotype_normalization() {
        // Verify that haplotypes are normalized (A <= B)
        let (a, b) = normalize_haplotype_pair("Z", "A");
        assert_eq!(a, "A");
        assert_eq!(b, "Z");

        let (a, b) = normalize_haplotype_pair("A", "Z");
        assert_eq!(a, "A");
        assert_eq!(b, "Z");
    }

    #[test]
    fn test_merge_segments_reversed_haplotypes() {
        // Segments with reversed haplotype order should be treated as same pair
        // after normalization (done during segment creation via detect_segments_rle)
        // Here we simulate segments that would be created with normalized haplotypes
        let mut segments = vec![
            make_segment("chr1", 1000, 2000, "A", "B", 0, 10, 0.999, 0.998),
            // This segment would have been created as (B, A) but normalized to (A, B)
            make_segment("chr1", 1500, 2500, "A", "B", 5, 15, 0.998, 0.997),
        ];

        merge_segments(&mut segments);
        // Should merge because both are normalized to (A, B)
        assert_eq!(segments.len(), 1);
    }

    // === Edge case tests for Segment ===

    #[test]
    fn test_segment_length_bp_same_start_end() {
        let seg = make_segment("chr1", 1000, 1000, "A", "B", 0, 0, 0.999, 0.999);
        assert_eq!(seg.length_bp(), 1);
    }

    #[test]
    fn test_segment_length_bp_overflow_protection() {
        // Test saturating_sub behavior when start > end
        let mut seg = make_segment("chr1", 2000, 1000, "A", "B", 0, 9, 0.999, 0.998);
        seg.start = 2000;
        seg.end = 1000; // Unusual: end < start
        // Should not panic or overflow
        assert_eq!(seg.length_bp(), 1); // 0 + 1 due to saturating_sub
    }

    #[test]
    fn test_segment_fraction_called_zero_windows() {
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 2000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 0,
            mean_identity: 0.0,
            min_identity: 0.0,
            identity_sum: 0.0,
            n_called: 0,
            start_idx: 0,
            end_idx: 0,
        };
        assert_eq!(seg.fraction_called(), 0.0);
    }

    #[test]
    fn test_segment_fraction_called_all_called() {
        let seg = make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998);
        assert_eq!(seg.fraction_called(), 1.0);
    }

    #[test]
    fn test_segment_fraction_called_partial() {
        let mut seg = make_segment("chr1", 1000, 2000, "A", "B", 0, 9, 0.999, 0.998);
        seg.n_called = 8;
        seg.identity_sum = 0.999 * 8.0;
        assert!((seg.fraction_called() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_rle_params_default() {
        let params = RleParams::default();
        assert_eq!(params.min_identity, 0.9995);
        assert_eq!(params.max_gap, 1);
        assert_eq!(params.min_windows, 3);
        assert_eq!(params.min_length_bp, 5000);
        assert_eq!(params.drop_tolerance, 0.0);
    }

    // === Segment Length Distribution Tests ===

    #[test]
    fn test_segment_length_stats_empty() {
        let stats = segment_length_distribution(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_bp, 0.0);
        assert_eq!(stats.median_bp, 0.0);
        assert_eq!(stats.std_bp, 0.0);
        assert_eq!(stats.min_bp, 0);
        assert_eq!(stats.max_bp, 0);
        assert_eq!(stats.total_bp, 0);
    }

    #[test]
    fn test_segment_length_stats_single() {
        let segments = vec![make_segment("chr1", 1000, 5999, "A", "B", 0, 9, 0.999, 0.998)];
        let stats = segment_length_distribution(&segments);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean_bp, 5000.0);
        assert_eq!(stats.median_bp, 5000.0);
        assert_eq!(stats.std_bp, 0.0);
        assert_eq!(stats.min_bp, 5000);
        assert_eq!(stats.max_bp, 5000);
        assert_eq!(stats.total_bp, 5000);
    }

    #[test]
    fn test_segment_length_stats_multiple() {
        let segments = vec![
            make_segment("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998),    // 10000 bp
            make_segment("chr1", 0, 19999, "C", "D", 0, 19, 0.999, 0.998),   // 20000 bp
            make_segment("chr1", 0, 29999, "E", "F", 0, 29, 0.999, 0.998),   // 30000 bp
        ];
        let stats = segment_length_distribution(&segments);
        assert_eq!(stats.count, 3);
        assert!((stats.mean_bp - 20000.0).abs() < 1e-6);
        assert_eq!(stats.median_bp, 20000.0);
        assert_eq!(stats.min_bp, 10000);
        assert_eq!(stats.max_bp, 30000);
        assert_eq!(stats.total_bp, 60000);
        assert!(stats.std_bp > 0.0);
    }

    #[test]
    fn test_segment_length_histogram_empty() {
        let hist = segment_length_histogram(&[], 10000);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_segment_length_histogram_zero_bin_size() {
        let segments = vec![make_segment("chr1", 0, 9999, "A", "B", 0, 9, 0.999, 0.998)];
        let hist = segment_length_histogram(&segments, 0);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_segment_to_bed_line() {
        let seg = make_segment("chr20", 1000001, 5000000, "HG00733#1", "NA12878#1", 0, 9, 0.999, 0.998);
        let lod = 12.5;
        let bed_line = format_segment_bed(&seg, lod);
        // BED: 0-based start, half-open end
        assert_eq!(bed_line, "chr20\t1000000\t5000000\tHG00733#1_NA12878#1\t1000\t.");
    }

    #[test]
    fn test_segment_to_bed_line_low_lod() {
        let seg = make_segment("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
        let lod = 2.3;
        let bed_line = format_segment_bed(&seg, lod);
        assert_eq!(bed_line, "chr1\t99\t200\tA_B\t230\t.");
    }

    #[test]
    fn test_segment_to_bed_line_negative_lod() {
        let seg = make_segment("chr1", 100, 200, "A", "B", 0, 0, 0.999, 0.999);
        let lod = -1.0;
        let bed_line = format_segment_bed(&seg, lod);
        // Negative LOD → score 0
        assert_eq!(bed_line, "chr1\t99\t200\tA_B\t0\t.");
    }

    #[test]
    fn test_segment_length_histogram_known() {
        let segments = vec![
            make_segment("chr1", 0, 4999, "A", "B", 0, 4, 0.999, 0.998),     // 5000 bp
            make_segment("chr1", 0, 7999, "C", "D", 0, 7, 0.999, 0.998),     // 8000 bp
            make_segment("chr1", 0, 14999, "E", "F", 0, 14, 0.999, 0.998),   // 15000 bp
            make_segment("chr1", 0, 19999, "G", "H", 0, 19, 0.999, 0.998),   // 20000 bp
        ];
        let hist = segment_length_histogram(&segments, 10000);
        // bin 0: [0, 10000) → 5000 and 8000 → 2 segments
        // bin 1: [10000, 20000) → 15000 → 1 segment
        // bin 2: [20000, 30000) → 20000 → 1 segment
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0], (0, 2));
        assert_eq!(hist[1], (10000, 1));
        assert_eq!(hist[2], (20000, 1));
    }
}
