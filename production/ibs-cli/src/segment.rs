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
//! use hprc_ibd::segment::{RleParams, IdentityTrack, detect_segments_rle};
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
/// - `hap_a`, `hap_b`: Haplotype identifiers
/// - `n_windows`: Number of analysis windows in the segment
/// - `mean_identity`: Average sequence identity across the segment
/// - `min_identity`: Lowest identity value observed in any window
/// - `identity_sum`: Sum of identity values (for re-averaging after merge)
/// - `n_called`: Number of windows with valid identity data
#[derive(Debug, Clone)]
pub struct Segment {
    /// Chromosome name
    pub chrom: String,
    /// Segment start position (bp)
    pub start: u64,
    /// Segment end position (bp)
    pub end: u64,
    /// First haplotype identifier
    pub hap_a: String,
    /// Second haplotype identifier
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
        let good = ident.map_or(false, |id| id >= effective_threshold);

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

    Some(Segment {
        chrom: chrom.to_string(),
        start: start_bp,
        end: end_bp,
        hap_a: hap_a.to_string(),
        hap_b: hap_b.to_string(),
        n_windows,
        mean_identity: stats.mean(),
        min_identity: min_ident,
        identity_sum: stats.mean() * stats.count() as f64,
        n_called: stats.count(),
    })
}

/// Merge overlapping segments
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
        let same_haplotypes = seg.hap_a == last.hap_a && seg.hap_b == last.hap_b;

        if seg.chrom == last.chrom && same_haplotypes && seg.start <= last.end {
            last.end = last.end.max(seg.end);
            last.n_windows += seg.n_windows;
            last.identity_sum += seg.identity_sum;
            last.n_called += seg.n_called;
            last.mean_identity = last.identity_sum / last.n_called as f64;
            last.min_identity = last.min_identity.min(seg.min_identity);
        } else {
            merged.push(seg.clone());
        }
    }

    *segments = merged;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_length() {
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 2000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 10,
            mean_identity: 0.999,
            min_identity: 0.998,
            identity_sum: 9.99,
            n_called: 10,
        };
        assert_eq!(seg.length_bp(), 1001);
    }

    #[test]
    fn test_merge_segments_same_haplotypes() {
        // Overlapping segments with same haplotype pair should merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 1500,
                end: 2500,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 2500);
    }

    #[test]
    fn test_merge_segments_different_haplotypes_not_merged() {
        // Overlapping segments with different haplotype pairs should NOT merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 1500,
                end: 2500,
                hap_a: "C".to_string(),  // Different haplotype pair
                hap_b: "D".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
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
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_merge_segments_non_overlapping() {
        // Non-overlapping segments should not merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 3000,
                end: 4000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_adjacent() {
        // Adjacent segments (end == start - 1) should not merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 2001,
                end: 3000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        // Adjacent but not overlapping, so should remain separate
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_touching() {
        // Segments where second starts at first's end should merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 2000, // Starts at previous end
                end: 3000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        // Touching segments should merge
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 3000);
    }

    #[test]
    fn test_merge_segments_different_chromosomes() {
        // Segments on different chromosomes should not merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr2".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_merge_segments_unsorted_input() {
        // Segments in wrong order should be sorted and merged correctly
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1500,
                end: 2500,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.997,
                identity_sum: 9.98,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.998,
                identity_sum: 9.99,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        // Should sort and merge
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 2500);
    }

    #[test]
    fn test_merge_segments_multiple_merges() {
        // Multiple overlapping segments should all merge
        let mut segments = vec![
            Segment {
                chrom: "chr1".to_string(),
                start: 1000,
                end: 2000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.999,
                min_identity: 0.999,
                identity_sum: 9.99,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 1500,
                end: 2500,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.998,
                min_identity: 0.998,
                identity_sum: 9.98,
                n_called: 10,
            },
            Segment {
                chrom: "chr1".to_string(),
                start: 2000,
                end: 3000,
                hap_a: "A".to_string(),
                hap_b: "B".to_string(),
                n_windows: 10,
                mean_identity: 0.997,
                min_identity: 0.997,
                identity_sum: 9.97,
                n_called: 10,
            },
        ];

        merge_segments(&mut segments);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 1000);
        assert_eq!(segments[0].end, 3000);
        assert_eq!(segments[0].n_windows, 30);
        assert_eq!(segments[0].min_identity, 0.997);
    }

    // === Edge case tests for Segment ===

    #[test]
    fn test_segment_length_bp_same_start_end() {
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 1000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 1,
            mean_identity: 0.999,
            min_identity: 0.999,
            identity_sum: 0.999,
            n_called: 1,
        };
        assert_eq!(seg.length_bp(), 1);
    }

    #[test]
    fn test_segment_length_bp_overflow_protection() {
        // Test saturating_sub behavior when start > end
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 2000,
            end: 1000, // Unusual: end < start
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 10,
            mean_identity: 0.999,
            min_identity: 0.998,
            identity_sum: 9.99,
            n_called: 10,
        };
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
        };
        assert_eq!(seg.fraction_called(), 0.0);
    }

    #[test]
    fn test_segment_fraction_called_all_called() {
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 2000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 10,
            mean_identity: 0.999,
            min_identity: 0.998,
            identity_sum: 9.99,
            n_called: 10,
        };
        assert_eq!(seg.fraction_called(), 1.0);
    }

    #[test]
    fn test_segment_fraction_called_partial() {
        let seg = Segment {
            chrom: "chr1".to_string(),
            start: 1000,
            end: 2000,
            hap_a: "A".to_string(),
            hap_b: "B".to_string(),
            n_windows: 10,
            mean_identity: 0.999,
            min_identity: 0.998,
            identity_sum: 7.992,
            n_called: 8,
        };
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
}
