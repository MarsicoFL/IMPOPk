//! Segment detection and merging algorithms

use crate::stats::OnlineStats;

/// Represents an IBS/IBD segment
#[derive(Debug, Clone)]
pub struct Segment {
    pub chrom: String,
    pub start: u64,
    pub end: u64,
    pub hap_a: String,
    pub hap_b: String,
    pub n_windows: usize,
    pub mean_identity: f64,
    pub min_identity: f64,
    pub identity_sum: f64,
    pub n_called: usize,
}

impl Segment {
    pub fn length_bp(&self) -> u64 {
        self.end.saturating_sub(self.start) + 1
    }

    pub fn fraction_called(&self) -> f64 {
        if self.n_windows == 0 { 0.0 } else { self.n_called as f64 / self.n_windows as f64 }
    }
}

/// Parameters for RLE-based segment calling
#[derive(Debug, Clone)]
pub struct RleParams {
    pub min_identity: f64,
    pub max_gap: usize,
    pub min_windows: usize,
    pub min_length_bp: u64,
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

    segments.sort_by(|a, b| {
        a.chrom.cmp(&b.chrom).then(a.start.cmp(&b.start)).then(a.end.cmp(&b.end))
    });

    let mut merged: Vec<Segment> = Vec::with_capacity(segments.len());
    merged.push(segments[0].clone());

    for seg in segments.iter().skip(1) {
        let last = merged.last_mut().unwrap();

        if seg.chrom == last.chrom && seg.start <= last.end {
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
}
