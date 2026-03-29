#![allow(clippy::manual_is_multiple_of)]
//! Segment concordance metrics for IBD validation.
//!
//! Provides quantitative comparison between IBD segment sets (e.g., our tool vs hap-ibd):
//!
//! - [`segment_overlap_bp`]: base-pair overlap between two intervals
//! - [`segments_jaccard`]: Jaccard index over a genomic region
//! - [`segments_precision_recall`]: precision/recall using one set as ground truth
//! - [`per_window_concordance`]: window-level agreement fraction
//! - [`matched_segments`]: match segments by reciprocal overlap
//! - [`length_correlation`]: Pearson r of matched segment lengths
//! - [`f1_score`]: harmonic mean of precision and recall
//! - [`boundary_accuracy`]: start/end distance statistics for matched segments
//! - [`haplotype_level_concordance`]: per-haplotype-combo concordance metrics
//! - [`extract_haplotype_index`], [`extract_sample_id`]: haplotype ID parsing helpers

/// Compute base-pair overlap between two intervals.
///
/// Both intervals are [start, end) half-open. Returns 0 if no overlap.
pub fn segment_overlap_bp(a: (u64, u64), b: (u64, u64)) -> u64 {
    let start = a.0.max(b.0);
    let end = a.1.min(b.1);
    end.saturating_sub(start)
}

/// Compute total base pairs covered by a set of segments within a region.
///
/// Handles overlapping segments by merging them first.
fn covered_bp(segments: &[(u64, u64)], region: (u64, u64)) -> u64 {
    if segments.is_empty() {
        return 0;
    }
    // Clip segments to region and sort
    let mut clipped: Vec<(u64, u64)> = segments
        .iter()
        .filter_map(|&(s, e)| {
            let cs = s.max(region.0);
            let ce = e.min(region.1);
            if ce > cs {
                Some((cs, ce))
            } else {
                None
            }
        })
        .collect();
    if clipped.is_empty() {
        return 0;
    }
    clipped.sort_by_key(|&(s, _)| s);

    // Merge overlapping intervals
    let mut total = 0u64;
    let mut cur_start = clipped[0].0;
    let mut cur_end = clipped[0].1;
    for &(s, e) in &clipped[1..] {
        if s <= cur_end {
            cur_end = cur_end.max(e);
        } else {
            total += cur_end - cur_start;
            cur_start = s;
            cur_end = e;
        }
    }
    total += cur_end - cur_start;
    total
}

/// Compute the Jaccard index of two segment sets over a genomic region.
///
/// Jaccard = |intersection| / |union| where the sets are the base pairs covered.
/// Returns 0.0 if both sets are empty within the region.
pub fn segments_jaccard(
    ours: &[(u64, u64)],
    theirs: &[(u64, u64)],
    region: (u64, u64),
) -> f64 {
    // Compute intersection: bp that are in BOTH sets
    // Compute union: bp that are in EITHER set
    let region_len = region.1.saturating_sub(region.0);
    if region_len == 0 {
        return 0.0;
    }

    let covered_ours = covered_bp(ours, region);
    let covered_theirs = covered_bp(theirs, region);

    if covered_ours == 0 && covered_theirs == 0 {
        return 0.0; // Both empty → undefined, return 0
    }

    // To compute intersection, we need bp covered by both
    // Merge all segments, then count intersection via inclusion-exclusion
    let intersection = intersection_bp(ours, theirs, region);
    let union = covered_ours + covered_theirs - intersection;

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Compute base pairs in the intersection of two segment sets within a region.
fn intersection_bp(a: &[(u64, u64)], b: &[(u64, u64)], region: (u64, u64)) -> u64 {
    // Clip both to region
    let clip = |segs: &[(u64, u64)]| -> Vec<(u64, u64)> {
        segs.iter()
            .filter_map(|&(s, e)| {
                let cs = s.max(region.0);
                let ce = e.min(region.1);
                if ce > cs {
                    Some((cs, ce))
                } else {
                    None
                }
            })
            .collect()
    };

    let a_clipped = clip(a);
    let b_clipped = clip(b);

    if a_clipped.is_empty() || b_clipped.is_empty() {
        return 0;
    }

    // For each segment in a, compute overlap with each segment in b
    let mut total = 0u64;

    // Merge a segments first to avoid double-counting
    let a_merged = merge_intervals(&a_clipped);
    let b_merged = merge_intervals(&b_clipped);

    for &(as_, ae) in &a_merged {
        for &(bs, be) in &b_merged {
            total += segment_overlap_bp((as_, ae), (bs, be));
        }
    }
    total
}

/// Merge overlapping intervals into non-overlapping set.
fn merge_intervals(intervals: &[(u64, u64)]) -> Vec<(u64, u64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|&(s, _)| s);

    let mut merged = vec![sorted[0]];
    for &(s, e) in &sorted[1..] {
        let last = merged.last_mut().unwrap();
        if s <= last.1 {
            last.1 = last.1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    merged
}

/// Compute precision and recall of our segments against a ground truth.
///
/// - **Precision**: fraction of our called IBD bp that are also in ground truth
/// - **Recall**: fraction of ground truth IBD bp that we also call
///
/// Uses `theirs` as ground truth. Returns (precision, recall).
/// Returns (0.0, 0.0) if both sets are empty.
pub fn segments_precision_recall(
    ours: &[(u64, u64)],
    theirs: &[(u64, u64)],
    region: (u64, u64),
) -> (f64, f64) {
    let covered_ours = covered_bp(ours, region);
    let covered_theirs = covered_bp(theirs, region);
    let intersection = intersection_bp(ours, theirs, region);

    let precision = if covered_ours > 0 {
        intersection as f64 / covered_ours as f64
    } else {
        0.0
    };

    let recall = if covered_theirs > 0 {
        intersection as f64 / covered_theirs as f64
    } else {
        0.0
    };

    (precision, recall)
}

/// Compute per-window concordance between two segment sets.
///
/// Divides the region into non-overlapping windows of `window_size` bp.
/// For each window, checks if it is covered (>50% overlap) by segments in
/// both sets, or neither. Returns the fraction of concordant windows.
pub fn per_window_concordance(
    ours: &[(u64, u64)],
    theirs: &[(u64, u64)],
    region: (u64, u64),
    window_size: u64,
) -> f64 {
    if window_size == 0 || region.1 <= region.0 {
        return 0.0;
    }

    let ours_merged = merge_intervals(
        &ours
            .iter()
            .filter_map(|&(s, e)| {
                let cs = s.max(region.0);
                let ce = e.min(region.1);
                if ce > cs { Some((cs, ce)) } else { None }
            })
            .collect::<Vec<_>>(),
    );

    let theirs_merged = merge_intervals(
        &theirs
            .iter()
            .filter_map(|&(s, e)| {
                let cs = s.max(region.0);
                let ce = e.min(region.1);
                if ce > cs { Some((cs, ce)) } else { None }
            })
            .collect::<Vec<_>>(),
    );

    let mut concordant = 0u64;
    let mut total = 0u64;

    let mut pos = region.0;
    while pos < region.1 {
        let win_end = (pos + window_size).min(region.1);
        let win = (pos, win_end);
        let half = (win_end - pos) / 2;

        let ours_cov = window_coverage(&ours_merged, win);
        let theirs_cov = window_coverage(&theirs_merged, win);

        let ours_call = ours_cov > half;
        let theirs_call = theirs_cov > half;

        if ours_call == theirs_call {
            concordant += 1;
        }
        total += 1;
        pos = win_end;
    }

    if total == 0 {
        return 0.0;
    }
    concordant as f64 / total as f64
}

/// Compute bp coverage of merged segments within a window.
fn window_coverage(merged: &[(u64, u64)], window: (u64, u64)) -> u64 {
    let mut cov = 0u64;
    for &(s, e) in merged {
        cov += segment_overlap_bp((s, e), window);
    }
    cov
}

/// Match segments between two sets by reciprocal overlap.
///
/// Two segments are matched if their overlap fraction (relative to the shorter
/// segment) exceeds `min_overlap_frac`. Returns pairs of indices (ours_idx, theirs_idx).
pub fn matched_segments(
    ours: &[(u64, u64)],
    theirs: &[(u64, u64)],
    min_overlap_frac: f64,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for (i, &a) in ours.iter().enumerate() {
        for (j, &b) in theirs.iter().enumerate() {
            let overlap = segment_overlap_bp(a, b);
            if overlap == 0 {
                continue;
            }
            let len_a = a.1.saturating_sub(a.0);
            let len_b = b.1.saturating_sub(b.0);
            let shorter = len_a.min(len_b);
            if shorter == 0 {
                continue;
            }
            let frac = overlap as f64 / shorter as f64;
            if frac >= min_overlap_frac {
                matches.push((i, j));
            }
        }
    }
    matches
}

/// A pair of matched intervals: (our segment, their segment).
pub type MatchedInterval = ((u64, u64), (u64, u64));

/// Compute Pearson correlation coefficient between matched segment lengths.
///
/// Takes pairs of matched segments as `(our_interval, their_interval)`.
/// Returns 0.0 if fewer than 2 matches.
pub fn length_correlation(matches: &[MatchedInterval]) -> f64 {
    if matches.len() < 2 {
        return 0.0;
    }

    let n = matches.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;

    for &((s1, e1), (s2, e2)) in matches {
        let x = (e1 - s1) as f64;
        let y = (e2 - s2) as f64;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    let numerator = n * sum_xy - sum_x * sum_y;
    let denom_x = (n * sum_xx - sum_x * sum_x).max(0.0).sqrt();
    let denom_y = (n * sum_yy - sum_y * sum_y).max(0.0).sqrt();

    if denom_x < 1e-15 || denom_y < 1e-15 {
        return 0.0;
    }

    (numerator / (denom_x * denom_y)).clamp(-1.0, 1.0)
}

/// Compute F1 score from precision and recall.
pub fn f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall < 1e-15 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Segment boundary accuracy: measures how close detected segment boundaries
/// are to the ground truth boundaries.
///
/// For each matched pair of segments, computes the absolute distance between
/// start positions and end positions. Returns summary statistics.
#[derive(Debug, Clone)]
pub struct BoundaryAccuracy {
    /// Number of matched segment pairs analyzed
    pub n_matched: usize,
    /// Mean absolute distance between start positions (bp)
    pub mean_start_distance_bp: f64,
    /// Mean absolute distance between end positions (bp)
    pub mean_end_distance_bp: f64,
    /// Median absolute distance between start positions (bp)
    pub median_start_distance_bp: f64,
    /// Median absolute distance between end positions (bp)
    pub median_end_distance_bp: f64,
    /// Max absolute distance between start positions (bp)
    pub max_start_distance_bp: u64,
    /// Max absolute distance between end positions (bp)
    pub max_end_distance_bp: u64,
    /// Fraction of starts within `threshold_bp` of ground truth
    pub frac_start_within_threshold: f64,
    /// Fraction of ends within `threshold_bp` of ground truth
    pub frac_end_within_threshold: f64,
    /// The threshold used (bp)
    pub threshold_bp: u64,
}

/// Compute segment boundary accuracy between matched segments.
///
/// Takes matched segment pairs (ours, theirs) and a distance threshold.
/// Returns boundary accuracy statistics. Returns None if no matches.
pub fn boundary_accuracy(
    matches: &[MatchedInterval],
    threshold_bp: u64,
) -> Option<BoundaryAccuracy> {
    if matches.is_empty() {
        return None;
    }

    let n = matches.len();
    let mut start_distances: Vec<u64> = Vec::with_capacity(n);
    let mut end_distances: Vec<u64> = Vec::with_capacity(n);

    for &((our_start, our_end), (their_start, their_end)) in matches {
        start_distances.push(our_start.abs_diff(their_start));
        end_distances.push(our_end.abs_diff(their_end));
    }

    let mean_start = start_distances.iter().sum::<u64>() as f64 / n as f64;
    let mean_end = end_distances.iter().sum::<u64>() as f64 / n as f64;

    start_distances.sort_unstable();
    end_distances.sort_unstable();

    let median_start = if n % 2 == 0 {
        (start_distances[n / 2 - 1] + start_distances[n / 2]) as f64 / 2.0
    } else {
        start_distances[n / 2] as f64
    };
    let median_end = if n % 2 == 0 {
        (end_distances[n / 2 - 1] + end_distances[n / 2]) as f64 / 2.0
    } else {
        end_distances[n / 2] as f64
    };

    let max_start = *start_distances.last().unwrap();
    let max_end = *end_distances.last().unwrap();

    let frac_start = start_distances.iter().filter(|&&d| d <= threshold_bp).count() as f64 / n as f64;
    let frac_end = end_distances.iter().filter(|&&d| d <= threshold_bp).count() as f64 / n as f64;

    Some(BoundaryAccuracy {
        n_matched: n,
        mean_start_distance_bp: mean_start,
        mean_end_distance_bp: mean_end,
        median_start_distance_bp: median_start,
        median_end_distance_bp: median_end,
        max_start_distance_bp: max_start,
        max_end_distance_bp: max_end,
        frac_start_within_threshold: frac_start,
        frac_end_within_threshold: frac_end,
        threshold_bp,
    })
}

/// Extract haplotype index from a pangenome haplotype ID.
///
/// e.g., "HG00280#2#JBHDWB010000002.1" → 2
/// e.g., "HG00280#1#JBHDWB010000002.1:130787850-130792849" → 1
/// Returns None if no haplotype index can be extracted.
pub fn extract_haplotype_index(hap_id: &str) -> Option<u8> {
    let parts: Vec<&str> = hap_id.split('#').collect();
    if parts.len() >= 2 {
        parts[1].parse::<u8>().ok()
    } else {
        None
    }
}

/// Extract sample ID from a pangenome haplotype ID.
///
/// e.g., "HG00280#2#JBHDWB010000002.1" → "HG00280"
/// e.g., "HG00280" → "HG00280"
pub fn extract_sample_id(hap_id: &str) -> &str {
    hap_id.split('#').next().unwrap_or(hap_id)
}

/// Result of haplotype-level concordance analysis for a sample pair.
#[derive(Debug, Clone)]
pub struct HaplotypeConcordance {
    /// Sample1 ID
    pub sample1: String,
    /// Sample2 ID
    pub sample2: String,
    /// Number of haplotype combinations with segments in our data
    pub n_our_hap_combos: usize,
    /// Number of haplotype combinations with segments in hap-ibd
    pub n_hapibd_hap_combos: usize,
    /// Per haplotype-combination results: (hap1, hap2, jaccard, precision, recall, f1)
    pub per_hap_combo: Vec<HapComboResult>,
    /// Best haplotype-level Jaccard across all combinations
    pub best_jaccard: f64,
    /// Best haplotype-level F1 across all combinations
    pub best_f1: f64,
    /// Sample-level Jaccard (merging all haplotypes)
    pub sample_level_jaccard: f64,
    /// Sample-level F1
    pub sample_level_f1: f64,
}

/// Concordance result for a single haplotype combination.
#[derive(Debug, Clone)]
pub struct HapComboResult {
    /// Haplotype index for sample1
    pub hap1: u8,
    /// Haplotype index for sample2
    pub hap2: u8,
    /// Jaccard index for this haplotype combination
    pub jaccard: f64,
    /// Precision for this combination
    pub precision: f64,
    /// Recall for this combination
    pub recall: f64,
    /// F1 score for this combination
    pub f1: f64,
    /// Number of our segments for this combination
    pub n_ours: usize,
    /// Number of hap-ibd segments for this combination
    pub n_theirs: usize,
}

/// Compute haplotype-level concordance between our segments and hap-ibd segments.
///
/// Unlike sample-level concordance which merges all haplotype combinations,
/// this matches haplotype-specific segments: our HG00280#1 vs HG00323#2 is
/// compared separately against hap-ibd's HG00280 hap1 vs HG00323 hap2.
///
/// Parameters:
/// - `our_segments`: our IBD segments with full haplotype IDs
/// - `hapibd_segments`: hap-ibd segments with sample + haplotype index
/// - `sample1`, `sample2`: the sample pair to analyze
/// - `region`: genomic region for Jaccard computation
///
/// Returns None if no data exists for this pair in either dataset.
pub fn haplotype_level_concordance(
    our_segments: &[(String, String, u64, u64)], // (hap_a_id, hap_b_id, start, end)
    hapibd_segments: &[(String, u8, String, u8, u64, u64)], // (sample1, hap1, sample2, hap2, start, end)
    sample1: &str,
    sample2: &str,
    region: (u64, u64),
) -> Option<HaplotypeConcordance> {
    // Group our segments by haplotype combination
    // Our hap IDs: "HG00280#1#CONTIG" and "HG00323#2#CONTIG"
    // Extract sample and hap index for matching
    let mut our_by_hap: std::collections::HashMap<(u8, u8), Vec<(u64, u64)>> = std::collections::HashMap::new();
    let mut sample_level_ours: Vec<(u64, u64)> = Vec::new();

    for (hap_a, hap_b, start, end) in our_segments {
        let sa = extract_sample_id(hap_a);
        let sb = extract_sample_id(hap_b);
        let ha = extract_haplotype_index(hap_a).unwrap_or(0);
        let hb = extract_haplotype_index(hap_b).unwrap_or(0);

        // Check this segment involves the pair (order-independent)
        if (sa == sample1 && sb == sample2) || (sa == sample2 && sb == sample1) {
            // Normalize order: ensure sample1's hap comes first
            let (h1, h2) = if sa == sample1 { (ha, hb) } else { (hb, ha) };
            our_by_hap.entry((h1, h2)).or_default().push((*start, *end));
            sample_level_ours.push((*start, *end));
        }
    }

    // Group hap-ibd segments by haplotype combination
    let mut hapibd_by_hap: std::collections::HashMap<(u8, u8), Vec<(u64, u64)>> = std::collections::HashMap::new();
    let mut sample_level_hapibd: Vec<(u64, u64)> = Vec::new();

    for (s1, h1, s2, h2, start, end) in hapibd_segments {
        let (involves, normalized_h1, normalized_h2) = if s1 == sample1 && s2 == sample2 {
            (true, *h1, *h2)
        } else if s1 == sample2 && s2 == sample1 {
            (true, *h2, *h1)
        } else {
            (false, 0, 0)
        };

        if involves {
            hapibd_by_hap.entry((normalized_h1, normalized_h2)).or_default().push((*start, *end));
            sample_level_hapibd.push((*start, *end));
        }
    }

    if our_by_hap.is_empty() && hapibd_by_hap.is_empty() {
        return None;
    }

    // Compute per-haplotype-combination metrics
    let mut per_hap_combo = Vec::new();
    let mut all_combos: std::collections::HashSet<(u8, u8)> = std::collections::HashSet::new();
    for &key in our_by_hap.keys() {
        all_combos.insert(key);
    }
    for &key in hapibd_by_hap.keys() {
        all_combos.insert(key);
    }

    let empty: Vec<(u64, u64)> = Vec::new();
    let mut best_jaccard = 0.0_f64;
    let mut best_f1 = 0.0_f64;

    for &(h1, h2) in &all_combos {
        let ours = our_by_hap.get(&(h1, h2)).unwrap_or(&empty);
        let theirs = hapibd_by_hap.get(&(h1, h2)).unwrap_or(&empty);

        let jaccard = segments_jaccard(ours, theirs, region);
        let (precision, recall) = segments_precision_recall(ours, theirs, region);
        let f1 = f1_score(precision, recall);

        per_hap_combo.push(HapComboResult {
            hap1: h1,
            hap2: h2,
            jaccard,
            precision,
            recall,
            f1,
            n_ours: ours.len(),
            n_theirs: theirs.len(),
        });

        best_jaccard = best_jaccard.max(jaccard);
        best_f1 = best_f1.max(f1);
    }

    // Sample-level metrics for comparison
    let sample_jaccard = segments_jaccard(&sample_level_ours, &sample_level_hapibd, region);
    let (sp, sr) = segments_precision_recall(&sample_level_ours, &sample_level_hapibd, region);
    let sample_f1 = f1_score(sp, sr);

    Some(HaplotypeConcordance {
        sample1: sample1.to_string(),
        sample2: sample2.to_string(),
        n_our_hap_combos: our_by_hap.len(),
        n_hapibd_hap_combos: hapibd_by_hap.len(),
        per_hap_combo,
        best_jaccard,
        best_f1,
        sample_level_jaccard: sample_jaccard,
        sample_level_f1: sample_f1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_overlap_bp_full() {
        assert_eq!(segment_overlap_bp((100, 200), (100, 200)), 100);
    }

    #[test]
    fn test_segment_overlap_bp_partial() {
        assert_eq!(segment_overlap_bp((100, 200), (150, 250)), 50);
    }

    #[test]
    fn test_segment_overlap_bp_no_overlap() {
        assert_eq!(segment_overlap_bp((100, 200), (300, 400)), 0);
    }

    #[test]
    fn test_segment_overlap_bp_adjacent() {
        assert_eq!(segment_overlap_bp((100, 200), (200, 300)), 0);
    }

    #[test]
    fn test_segment_overlap_bp_contained() {
        assert_eq!(segment_overlap_bp((100, 400), (200, 300)), 100);
    }

    #[test]
    fn test_jaccard_perfect_overlap() {
        let segs = vec![(100, 200)];
        let j = segments_jaccard(&segs, &segs, (0, 1000));
        assert!((j - 1.0).abs() < 1e-9, "Expected Jaccard=1.0, got {}", j);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let ours = vec![(100, 200)];
        let theirs = vec![(300, 400)];
        let j = segments_jaccard(&ours, &theirs, (0, 1000));
        assert!((j - 0.0).abs() < 1e-9, "Expected Jaccard=0.0, got {}", j);
    }

    #[test]
    fn test_jaccard_half_overlap() {
        let ours = vec![(100, 300)];
        let theirs = vec![(200, 400)];
        // Overlap: 200-300 = 100bp, Union: 100-400 = 300bp
        let j = segments_jaccard(&ours, &theirs, (0, 1000));
        let expected = 100.0 / 300.0;
        assert!(
            (j - expected).abs() < 1e-9,
            "Expected Jaccard={}, got {}",
            expected,
            j
        );
    }

    #[test]
    fn test_jaccard_both_empty() {
        let empty: Vec<(u64, u64)> = vec![];
        let j = segments_jaccard(&empty, &empty, (0, 1000));
        assert!((j - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let ours = vec![(100, 200)];
        let empty: Vec<(u64, u64)> = vec![];
        let j = segments_jaccard(&ours, &empty, (0, 1000));
        assert!((j - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_recall_perfect() {
        let segs = vec![(100, 200)];
        let (p, r) = segments_precision_recall(&segs, &segs, (0, 1000));
        assert!((p - 1.0).abs() < 1e-9);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_recall_no_overlap() {
        let ours = vec![(100, 200)];
        let theirs = vec![(300, 400)];
        let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
        assert!((p - 0.0).abs() < 1e-9);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_recall_subset() {
        // Ours is a subset of theirs
        let ours = vec![(150, 200)];
        let theirs = vec![(100, 300)];
        let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
        assert!((p - 1.0).abs() < 1e-9); // All our bp are in truth
        assert!(
            (r - 0.25).abs() < 1e-9,
            "Expected recall=0.25, got {}",
            r
        ); // 50/200
    }

    #[test]
    fn test_per_window_concordance_perfect() {
        let segs = vec![(0, 100)];
        let c = per_window_concordance(&segs, &segs, (0, 100), 10);
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_window_concordance_no_segments() {
        let empty: Vec<(u64, u64)> = vec![];
        let c = per_window_concordance(&empty, &empty, (0, 100), 10);
        // Both empty → all windows agree (neither has IBD)
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_window_concordance_half_agree() {
        // Ours covers first half, theirs covers second half
        let ours = vec![(0, 50)];
        let theirs = vec![(50, 100)];
        // 10 windows of 10bp each. Windows 0-4: ours=yes theirs=no → disagree
        // Windows 5-9: ours=no theirs=yes → disagree
        let c = per_window_concordance(&ours, &theirs, (0, 100), 10);
        assert!((c - 0.0).abs() < 1e-9, "Expected 0.0, got {}", c);
    }

    #[test]
    fn test_per_window_concordance_mixed() {
        // ours: 0-60, theirs: 0-40
        // 10 windows of 10bp. 0-3: both yes. 4: ours=yes (10bp > 5), theirs=no → disagree
        // Actually: window [40,50): ours covers [40,50)=10bp > 5 → yes; theirs covers [40,40)=0 → no.
        // windows 5-9: neither → agree
        // windows 0-3: both → agree. window 4: disagree. windows 5-9: agree (ours covers [50,60) in win5)
        // window [50,60): ours covers [50,60)=10bp > 5 → yes, theirs covers 0 → no → disagree
        // So: windows 0-3 agree, 4 disagree, 5 disagree, 6-9 agree → 8/10 = 0.8
        let ours = vec![(0, 60)];
        let theirs = vec![(0, 40)];
        let c = per_window_concordance(&ours, &theirs, (0, 100), 10);
        assert!(
            (c - 0.8).abs() < 1e-9,
            "Expected concordance=0.8, got {}",
            c
        );
    }

    #[test]
    fn test_matched_segments_exact() {
        let ours = vec![(100, 200), (300, 400)];
        let theirs = vec![(100, 200), (300, 400)];
        let m = matched_segments(&ours, &theirs, 0.5);
        assert_eq!(m.len(), 2);
        assert!(m.contains(&(0, 0)));
        assert!(m.contains(&(1, 1)));
    }

    #[test]
    fn test_matched_segments_no_match() {
        let ours = vec![(100, 200)];
        let theirs = vec![(300, 400)];
        let m = matched_segments(&ours, &theirs, 0.5);
        assert!(m.is_empty());
    }

    #[test]
    fn test_matched_segments_partial() {
        // 50bp overlap out of 100bp shorter segment → 50% → matches at 0.5
        let ours = vec![(100, 200)];
        let theirs = vec![(150, 250)];
        let m = matched_segments(&ours, &theirs, 0.5);
        assert_eq!(m.len(), 1);

        // But not at 0.6
        let m2 = matched_segments(&ours, &theirs, 0.6);
        assert!(m2.is_empty());
    }

    #[test]
    fn test_length_correlation_perfect() {
        let matches = vec![
            ((0u64, 100u64), (0u64, 100u64)),
            ((0, 200), (0, 200)),
            ((0, 50), (0, 50)),
        ];
        let r = length_correlation(&matches);
        assert!((r - 1.0).abs() < 1e-9, "Expected r=1.0, got {}", r);
    }

    #[test]
    fn test_length_correlation_negative() {
        let matches = vec![
            ((0u64, 100u64), (0u64, 200u64)),
            ((0, 200), (0, 100)),
        ];
        let r = length_correlation(&matches);
        assert!(
            (r - (-1.0)).abs() < 1e-9,
            "Expected r=-1.0, got {}",
            r
        );
    }

    #[test]
    fn test_length_correlation_insufficient() {
        let matches = vec![((0u64, 100u64), (0u64, 100u64))];
        let r = length_correlation(&matches);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_length_correlation_known() {
        // Lengths: ours = [100, 200, 300], theirs = [110, 210, 310]
        // Almost perfect positive correlation
        let matches = vec![
            ((0u64, 100u64), (0u64, 110u64)),
            ((0, 200), (0, 210)),
            ((0, 300), (0, 310)),
        ];
        let r = length_correlation(&matches);
        assert!(r > 0.999, "Expected r≈1.0, got {}", r);
    }

    #[test]
    fn test_f1_score() {
        assert!((f1_score(1.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((f1_score(0.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((f1_score(0.5, 0.5) - 0.5).abs() < 1e-9);
        // precision=0.8, recall=0.6 → F1 = 2*0.48/1.4 ≈ 0.6857
        let f = f1_score(0.8, 0.6);
        assert!((f - 0.6857).abs() < 0.001);
    }

    #[test]
    fn test_jaccard_multiple_segments() {
        // ours: two segments covering 100bp each
        // theirs: one segment covering the same 200bp
        let ours = vec![(100, 200), (300, 400)];
        let theirs = vec![(100, 400)];
        // ours covers: 100-200 + 300-400 = 200bp
        // theirs covers: 100-400 = 300bp
        // intersection: 100-200 + 300-400 = 200bp
        // union: 100-400 = 300bp
        let j = segments_jaccard(&ours, &theirs, (0, 1000));
        let expected = 200.0 / 300.0;
        assert!(
            (j - expected).abs() < 1e-9,
            "Expected Jaccard={}, got {}",
            expected,
            j
        );
    }

    #[test]
    fn test_covered_bp_with_overlapping_segments() {
        // Two overlapping segments
        let segs = vec![(100, 300), (200, 400)];
        let bp = covered_bp(&segs, (0, 1000));
        assert_eq!(bp, 300); // 100-400
    }

    #[test]
    fn test_merge_intervals_empty() {
        let merged = merge_intervals(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_intervals_no_overlap() {
        let merged = merge_intervals(&[(100, 200), (300, 400)]);
        assert_eq!(merged, vec![(100, 200), (300, 400)]);
    }

    #[test]
    fn test_merge_intervals_overlap() {
        let merged = merge_intervals(&[(100, 300), (200, 400), (350, 500)]);
        assert_eq!(merged, vec![(100, 500)]);
    }

    // === Boundary accuracy tests ===

    #[test]
    fn test_boundary_accuracy_perfect_match() {
        let matches = vec![
            ((1000u64, 5000u64), (1000u64, 5000u64)),
            ((10000, 20000), (10000, 20000)),
        ];
        let acc = boundary_accuracy(&matches, 10000).unwrap();
        assert_eq!(acc.n_matched, 2);
        assert!((acc.mean_start_distance_bp - 0.0).abs() < 1e-9);
        assert!((acc.mean_end_distance_bp - 0.0).abs() < 1e-9);
        assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
        assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_accuracy_known_distances() {
        // Start distances: 1000, 3000 → mean 2000, median 2000
        // End distances: 2000, 4000 → mean 3000, median 3000
        let matches = vec![
            ((2000u64, 8000u64), (1000u64, 6000u64)),  // start diff=1000, end diff=2000
            ((15000, 25000), (12000, 21000)),            // start diff=3000, end diff=4000
        ];
        let acc = boundary_accuracy(&matches, 2500).unwrap();
        assert_eq!(acc.n_matched, 2);
        assert!((acc.mean_start_distance_bp - 2000.0).abs() < 1e-9);
        assert!((acc.mean_end_distance_bp - 3000.0).abs() < 1e-9);
        assert!((acc.median_start_distance_bp - 2000.0).abs() < 1e-9);
        assert!((acc.median_end_distance_bp - 3000.0).abs() < 1e-9);
        assert_eq!(acc.max_start_distance_bp, 3000);
        assert_eq!(acc.max_end_distance_bp, 4000);
        // Threshold 2500: starts within: 1000 yes, 3000 no → 0.5
        // Ends within: 2000 yes, 4000 no → 0.5
        assert!((acc.frac_start_within_threshold - 0.5).abs() < 1e-9);
        assert!((acc.frac_end_within_threshold - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_accuracy_empty() {
        let matches: Vec<MatchedInterval> = vec![];
        assert!(boundary_accuracy(&matches, 10000).is_none());
    }

    #[test]
    fn test_boundary_accuracy_single_match() {
        let matches = vec![
            ((5000u64, 15000u64), (6000u64, 14000u64)),
        ];
        let acc = boundary_accuracy(&matches, 5000).unwrap();
        assert_eq!(acc.n_matched, 1);
        assert!((acc.mean_start_distance_bp - 1000.0).abs() < 1e-9);
        assert!((acc.mean_end_distance_bp - 1000.0).abs() < 1e-9);
        // Both within 5000bp threshold
        assert!((acc.frac_start_within_threshold - 1.0).abs() < 1e-9);
        assert!((acc.frac_end_within_threshold - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_accuracy_odd_count_median() {
        let matches = vec![
            ((1000u64, 10000u64), (1000u64, 10000u64)), // start diff=0, end diff=0
            ((2000, 20000), (3000, 22000)),               // start diff=1000, end diff=2000
            ((5000, 30000), (8000, 35000)),               // start diff=3000, end diff=5000
        ];
        let acc = boundary_accuracy(&matches, 10000).unwrap();
        assert_eq!(acc.n_matched, 3);
        // Sorted start distances: [0, 1000, 3000] → median=1000
        assert!((acc.median_start_distance_bp - 1000.0).abs() < 1e-9);
        // Sorted end distances: [0, 2000, 5000] → median=2000
        assert!((acc.median_end_distance_bp - 2000.0).abs() < 1e-9);
    }

    // === Extract haplotype index tests ===

    #[test]
    fn test_extract_haplotype_index() {
        assert_eq!(extract_haplotype_index("HG00280#2#JBHDWB010000002.1"), Some(2));
        assert_eq!(extract_haplotype_index("HG00280#1#CONTIG"), Some(1));
        assert_eq!(extract_haplotype_index("HG00280#1#CONTIG:100-200"), Some(1));
        assert_eq!(extract_haplotype_index("HG00280"), None);
        assert_eq!(extract_haplotype_index(""), None);
    }

    #[test]
    fn test_extract_sample_id() {
        assert_eq!(extract_sample_id("HG00280#2#JBHDWB010000002.1"), "HG00280");
        assert_eq!(extract_sample_id("HG00280#1#CONTIG:100-200"), "HG00280");
        assert_eq!(extract_sample_id("HG00280"), "HG00280");
    }

    // === Haplotype-level concordance tests ===

    #[test]
    fn test_haplotype_concordance_perfect_match() {
        // Our segments: HG001#1 vs HG002#2, same as hap-ibd
        let our_segs = vec![
            ("HG001#1#CONTIG".to_string(), "HG002#2#CONTIG".to_string(), 1000u64, 5000u64),
        ];
        let hapibd_segs = vec![
            ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 1000u64, 5000u64),
        ];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
        ).unwrap();

        assert_eq!(result.n_our_hap_combos, 1);
        assert_eq!(result.n_hapibd_hap_combos, 1);
        assert!((result.best_jaccard - 1.0).abs() < 1e-9);
        assert!((result.best_f1 - 1.0).abs() < 1e-9);
        assert!((result.sample_level_jaccard - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_haplotype_concordance_wrong_haplotype() {
        // Our segments: HG001#1 vs HG002#1
        // hap-ibd: HG001 hap1 vs HG002 hap2 (different haplotype of HG002)
        let our_segs = vec![
            ("HG001#1#CONTIG".to_string(), "HG002#1#CONTIG".to_string(), 1000u64, 5000u64),
        ];
        let hapibd_segs = vec![
            ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 1000u64, 5000u64),
        ];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
        ).unwrap();

        // Haplotype-level: combo (1,1) has our seg but no hap-ibd → J=0
        //                   combo (1,2) has hap-ibd but no ours → J=0
        assert_eq!(result.per_hap_combo.len(), 2);
        assert!((result.best_jaccard - 0.0).abs() < 1e-9);

        // But sample-level should show perfect match (same genomic region)
        assert!((result.sample_level_jaccard - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_haplotype_concordance_multiple_combos() {
        // Multiple haplotype combos for same sample pair
        let our_segs = vec![
            ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 1000u64, 5000u64),
            ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 8000u64, 12000u64),
            ("HG001#2#C".to_string(), "HG002#1#C".to_string(), 20000u64, 25000u64),
        ];
        let hapibd_segs = vec![
            ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 1000u64, 5000u64),
            ("HG001".to_string(), 1u8, "HG002".to_string(), 2u8, 8000u64, 12000u64),
        ];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 30000),
        ).unwrap();

        assert_eq!(result.n_our_hap_combos, 3);
        assert_eq!(result.n_hapibd_hap_combos, 2);
        // (1,1) and (1,2) should have perfect Jaccard
        assert!((result.best_jaccard - 1.0).abs() < 1e-9);
        assert!((result.best_f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_haplotype_concordance_reversed_pair_order() {
        // hap-ibd has HG002 as sample1 and HG001 as sample2
        let our_segs = vec![
            ("HG001#1#C".to_string(), "HG002#2#C".to_string(), 1000u64, 5000u64),
        ];
        let hapibd_segs = vec![
            // Reversed order in hap-ibd
            ("HG002".to_string(), 2u8, "HG001".to_string(), 1u8, 1000u64, 5000u64),
        ];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
        ).unwrap();

        // Should still match correctly because normalization puts sample1's hap first
        assert!((result.best_jaccard - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_haplotype_concordance_no_data() {
        let our_segs: Vec<(String, String, u64, u64)> = vec![];
        let hapibd_segs: Vec<(String, u8, String, u8, u64, u64)> = vec![];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
        );
        assert!(result.is_none());
    }

    // === Additional edge case tests ===

    #[test]
    fn test_segment_overlap_bp_zero_length() {
        // Zero-length segment
        assert_eq!(segment_overlap_bp((100, 100), (100, 200)), 0);
        assert_eq!(segment_overlap_bp((100, 200), (150, 150)), 0);
    }

    #[test]
    fn test_jaccard_zero_region() {
        let segs = vec![(100, 200)];
        let j = segments_jaccard(&segs, &segs, (0, 0));
        assert!((j - 0.0).abs() < 1e-9, "Zero-size region should give 0.0");
    }

    #[test]
    fn test_jaccard_region_outside_segments() {
        let ours = vec![(100, 200)];
        let theirs = vec![(100, 200)];
        let j = segments_jaccard(&ours, &theirs, (500, 1000));
        assert!((j - 0.0).abs() < 1e-9, "Region outside all segments should give 0.0");
    }

    #[test]
    fn test_precision_recall_empty_ours() {
        let ours: Vec<(u64, u64)> = vec![];
        let theirs = vec![(100, 200)];
        let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
        assert!((p - 0.0).abs() < 1e-9);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_precision_recall_empty_theirs() {
        let ours = vec![(100, 200)];
        let theirs: Vec<(u64, u64)> = vec![];
        let (p, r) = segments_precision_recall(&ours, &theirs, (0, 1000));
        assert!((p - 0.0).abs() < 1e-9); // nothing in truth → no true positives
        assert!((r - 0.0).abs() < 1e-9); // no truth bp
    }

    #[test]
    fn test_per_window_concordance_zero_window() {
        let segs = vec![(0, 100)];
        let c = per_window_concordance(&segs, &segs, (0, 100), 0);
        assert!((c - 0.0).abs() < 1e-9, "Zero window size should return 0.0");
    }

    #[test]
    fn test_per_window_concordance_empty_region() {
        let segs = vec![(0, 100)];
        let c = per_window_concordance(&segs, &segs, (50, 50), 10);
        assert!((c - 0.0).abs() < 1e-9, "Empty region should return 0.0");
    }

    #[test]
    fn test_per_window_concordance_window_larger_than_region() {
        // One big window covers the whole region
        let ours = vec![(0, 50)];
        let theirs = vec![(0, 50)];
        let c = per_window_concordance(&ours, &theirs, (0, 50), 1000);
        // Single window [0,50), both cover it → agree
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_matched_segments_zero_overlap_threshold() {
        // With 0.0 threshold, any overlap should match
        let ours = vec![(100, 200)];
        let theirs = vec![(199, 300)]; // 1bp overlap
        let m = matched_segments(&ours, &theirs, 0.0);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_matched_segments_threshold_1_0() {
        // Threshold 1.0: requires overlap >= shorter segment length
        let ours = vec![(100, 200)]; // 100bp
        let theirs = vec![(100, 250)]; // 150bp, overlap=100, shorter=100, frac=1.0
        let m = matched_segments(&ours, &theirs, 1.0);
        assert_eq!(m.len(), 1); // 100/100 = 1.0, meets threshold
    }

    #[test]
    fn test_matched_segments_zero_length_segment() {
        let ours = vec![(100, 100)]; // zero length
        let theirs = vec![(100, 200)];
        let m = matched_segments(&ours, &theirs, 0.5);
        assert!(m.is_empty(), "Zero-length segment should never match");
    }

    #[test]
    fn test_length_correlation_empty() {
        let matches: Vec<MatchedInterval> = vec![];
        assert!((length_correlation(&matches) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_length_correlation_identical_lengths() {
        // All same length → std dev = 0 → r = 0
        let matches = vec![
            ((0u64, 100u64), (0u64, 100u64)),
            ((0, 100), (0, 100)),
            ((0, 100), (0, 100)),
        ];
        let r = length_correlation(&matches);
        assert!((r - 0.0).abs() < 1e-9, "Constant lengths → r=0, got {}", r);
    }

    #[test]
    fn test_f1_score_extreme_precision() {
        // Very high precision, very low recall
        let f = f1_score(1.0, 0.001);
        assert!(f < 0.003, "F1 with recall~0 should be very low");
    }

    #[test]
    fn test_f1_score_both_near_zero() {
        let f = f1_score(1e-20, 1e-20);
        assert!((f - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_accuracy_threshold_zero() {
        // Only exact matches count with threshold=0
        let matches = vec![
            ((1000u64, 5000u64), (1001u64, 5001u64)), // off by 1
        ];
        let acc = boundary_accuracy(&matches, 0).unwrap();
        assert!((acc.frac_start_within_threshold - 0.0).abs() < 1e-9);
        assert!((acc.frac_end_within_threshold - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract_haplotype_index_zero() {
        assert_eq!(extract_haplotype_index("HG00280#0#CONTIG"), Some(0));
    }

    #[test]
    fn test_extract_haplotype_index_non_numeric() {
        // Haplotype index that's not a number
        assert_eq!(extract_haplotype_index("sample#HAP1#CONTIG"), None);
    }

    #[test]
    fn test_extract_sample_id_empty() {
        assert_eq!(extract_sample_id(""), "");
    }

    #[test]
    fn test_covered_bp_fully_outside_region() {
        let segs = vec![(100, 200), (300, 400)];
        assert_eq!(covered_bp(&segs, (500, 600)), 0);
    }

    #[test]
    fn test_merge_intervals_single() {
        let merged = merge_intervals(&[(100, 200)]);
        assert_eq!(merged, vec![(100, 200)]);
    }

    #[test]
    fn test_merge_intervals_adjacent() {
        let merged = merge_intervals(&[(100, 200), (200, 300)]);
        // 200 <= 200 → merge
        assert_eq!(merged, vec![(100, 300)]);
    }

    #[test]
    fn test_haplotype_concordance_partial_overlap() {
        let our_segs = vec![
            ("HG001#1#C".to_string(), "HG002#1#C".to_string(), 1000u64, 6000u64),
        ];
        let hapibd_segs = vec![
            ("HG001".to_string(), 1u8, "HG002".to_string(), 1u8, 3000u64, 8000u64),
        ];
        let result = haplotype_level_concordance(
            &our_segs, &hapibd_segs, "HG001", "HG002", (0, 10000),
        ).unwrap();

        // Overlap: 3000-6000 = 3000bp, Union: 1000-8000 = 7000bp
        let expected_j = 3000.0 / 7000.0;
        let combo = &result.per_hap_combo[0];
        assert!((combo.jaccard - expected_j).abs() < 1e-9,
            "Expected Jaccard={:.4}, got {:.4}", expected_j, combo.jaccard);
        assert!((result.sample_level_jaccard - expected_j).abs() < 1e-9);
    }
}
