//! Concordance metrics for comparing local ancestry calls.
//!
//! Provides functions to compute per-window concordance, per-population
//! precision/recall/F1, switch point accuracy, and confusion matrices
//! between our ancestry calls and a ground truth (e.g., RFMix).

use std::collections::HashMap;

/// Per-window ancestry concordance: fraction of windows where both tools
/// agree on the ancestry assignment.
///
/// # Arguments
/// * `ours` - Our ancestry indices per window
/// * `truth` - Ground truth ancestry indices per window (same length)
///
/// # Returns
/// Fraction of matching windows (0.0 to 1.0), or 0.0 if empty.
pub fn per_window_ancestry_concordance(ours: &[usize], truth: &[usize]) -> f64 {
    let len = ours.len().min(truth.len());
    if len == 0 {
        return 0.0;
    }

    let matching = ours.iter().zip(truth.iter()).filter(|(a, b)| a == b).count();
    matching as f64 / len as f64
}

/// Per-population precision, recall, and F1 score.
///
/// For each population, computes:
/// - Precision: among windows we called as pop X, what fraction truly is pop X?
/// - Recall: among windows truly pop X, what fraction did we call pop X?
/// - F1: harmonic mean of precision and recall
///
/// # Arguments
/// * `ours` - Our ancestry indices per window
/// * `truth` - Ground truth ancestry indices per window
/// * `pop_names` - Names of populations (indexed by ancestry index)
///
/// # Returns
/// Map from population name to (precision, recall, f1).
pub fn per_population_concordance(
    ours: &[usize],
    truth: &[usize],
    pop_names: &[String],
) -> HashMap<String, (f64, f64, f64)> {
    let len = ours.len().min(truth.len());
    let mut result = HashMap::new();

    for (pop_idx, name) in pop_names.iter().enumerate() {
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_ = 0u64;

        for i in 0..len {
            let predicted = ours[i] == pop_idx;
            let actual = truth[i] == pop_idx;

            match (predicted, actual) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        result.insert(name.clone(), (precision, recall, f1));
    }

    result
}

/// N×N confusion matrix for ancestry calls.
///
/// `matrix[i][j]` = number of windows where truth is population i and we called population j.
///
/// # Arguments
/// * `ours` - Our ancestry indices per window
/// * `truth` - Ground truth ancestry indices per window
/// * `n_pops` - Number of populations
pub fn ancestry_confusion_matrix(ours: &[usize], truth: &[usize], n_pops: usize) -> Vec<Vec<u64>> {
    let mut matrix = vec![vec![0u64; n_pops]; n_pops];
    let len = ours.len().min(truth.len());

    for i in 0..len {
        if truth[i] < n_pops && ours[i] < n_pops {
            matrix[truth[i]][ours[i]] += 1;
        }
    }

    matrix
}

/// A switch point in an ancestry call sequence.
#[derive(Debug, Clone)]
pub struct SwitchPoint {
    /// Index in the window array where the switch occurs (between index-1 and index)
    pub window_index: usize,
    /// Ancestry before the switch
    pub from_ancestry: usize,
    /// Ancestry after the switch
    pub to_ancestry: usize,
}

/// Extract switch points from an ancestry sequence.
pub fn extract_switch_points(ancestries: &[usize]) -> Vec<SwitchPoint> {
    let mut switches = Vec::new();
    for i in 1..ancestries.len() {
        if ancestries[i] != ancestries[i - 1] {
            switches.push(SwitchPoint {
                window_index: i,
                from_ancestry: ancestries[i - 1],
                to_ancestry: ancestries[i],
            });
        }
    }
    switches
}

/// Detailed switch point accuracy report.
#[derive(Debug, Clone)]
pub struct SwitchAccuracyReport {
    /// Fraction of true switch points detected within tolerance
    pub detection_rate: f64,
    /// Mean distance (windows) from each true switch to nearest predicted switch
    pub mean_distance: f64,
    /// Number of true switch points matched by a prediction
    pub n_detected: usize,
    /// Number of true switch points not matched
    pub n_missed: usize,
    /// Number of predicted switches not matching any true switch
    pub n_spurious: usize,
    /// Precision: fraction of predicted switches that match a true switch
    pub switch_precision: f64,
}

/// Switch point accuracy between our calls and ground truth.
///
/// For each true switch point, finds the nearest predicted switch point.
/// Returns:
/// - Fraction of true switch points that have a predicted switch within the tolerance
/// - Mean distance (in windows) from each true switch to its nearest predicted switch
///
/// # Arguments
/// * `our_switches` - Switch point window indices from our calls
/// * `true_switches` - Switch point window indices from ground truth
/// * `tolerance_windows` - Maximum distance (in windows) to count as a match
///
/// # Returns
/// `(fraction_detected, mean_distance)` where fraction_detected is in [0, 1]
/// and mean_distance is the average window distance to nearest true switch.
pub fn switch_point_accuracy(
    our_switches: &[usize],
    true_switches: &[usize],
    tolerance_windows: usize,
) -> (f64, f64) {
    if true_switches.is_empty() {
        return (1.0, 0.0); // No switches to detect
    }

    if our_switches.is_empty() {
        return (0.0, f64::INFINITY); // Missed all switches
    }

    let mut detected = 0usize;
    let mut total_distance = 0.0f64;

    for &true_sw in true_switches {
        let nearest_dist = our_switches.iter()
            .map(|&our_sw| (our_sw as i64 - true_sw as i64).unsigned_abs() as usize)
            .min()
            .unwrap_or(usize::MAX);

        if nearest_dist <= tolerance_windows {
            detected += 1;
        }
        total_distance += nearest_dist as f64;
    }

    let fraction_detected = detected as f64 / true_switches.len() as f64;
    let mean_distance = total_distance / true_switches.len() as f64;

    (fraction_detected, mean_distance)
}

/// Compute detailed switch point accuracy report with matched/spurious classification.
///
/// Uses greedy matching: each true switch is matched to the nearest unmatched prediction.
/// Unmatched predictions are classified as spurious.
pub fn switch_point_accuracy_detailed(
    our_switches: &[usize],
    true_switches: &[usize],
    tolerance_windows: usize,
) -> SwitchAccuracyReport {
    if true_switches.is_empty() && our_switches.is_empty() {
        return SwitchAccuracyReport {
            detection_rate: 1.0,
            mean_distance: 0.0,
            n_detected: 0,
            n_missed: 0,
            n_spurious: 0,
            switch_precision: 1.0,
        };
    }

    if true_switches.is_empty() {
        return SwitchAccuracyReport {
            detection_rate: 1.0,
            mean_distance: 0.0,
            n_detected: 0,
            n_missed: 0,
            n_spurious: our_switches.len(),
            switch_precision: 0.0,
        };
    }

    if our_switches.is_empty() {
        return SwitchAccuracyReport {
            detection_rate: 0.0,
            mean_distance: f64::INFINITY,
            n_detected: 0,
            n_missed: true_switches.len(),
            n_spurious: 0,
            switch_precision: 1.0, // vacuously true
        };
    }

    // Greedy matching: for each true switch, find nearest unmatched prediction
    let mut used_pred = vec![false; our_switches.len()];
    let mut detected = 0usize;
    let mut total_distance = 0.0f64;

    for &true_sw in true_switches {
        let mut best_dist = usize::MAX;
        let mut best_idx = None;

        for (idx, &pred_sw) in our_switches.iter().enumerate() {
            if used_pred[idx] {
                continue;
            }
            let dist = (pred_sw as i64 - true_sw as i64).unsigned_abs() as usize;
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            if best_dist <= tolerance_windows {
                detected += 1;
                used_pred[idx] = true;
            }
        }
        total_distance += best_dist as f64;
    }

    let n_spurious = used_pred.iter().filter(|&&u| !u).count();
    let n_missed = true_switches.len() - detected;

    SwitchAccuracyReport {
        detection_rate: detected as f64 / true_switches.len() as f64,
        mean_distance: total_distance / true_switches.len() as f64,
        n_detected: detected,
        n_missed,
        n_spurious,
        switch_precision: if our_switches.is_empty() {
            1.0
        } else {
            detected as f64 / our_switches.len() as f64
        },
    }
}

/// Switch point accuracy using base-pair positions instead of window indices.
///
/// # Arguments
/// * `our_positions` - Switch point positions (bp) from our calls
/// * `true_positions` - Switch point positions (bp) from ground truth
/// * `tolerance_bp` - Maximum distance (bp) to count as a match
///
/// # Returns
/// `(fraction_detected, mean_distance_bp)`
pub fn switch_point_accuracy_bp(
    our_positions: &[u64],
    true_positions: &[u64],
    tolerance_bp: u64,
) -> (f64, f64) {
    if true_positions.is_empty() {
        return (1.0, 0.0);
    }

    if our_positions.is_empty() {
        return (0.0, f64::INFINITY);
    }

    let mut detected = 0usize;
    let mut total_distance = 0.0f64;

    for &true_pos in true_positions {
        let nearest_dist = our_positions.iter()
            .map(|&our_pos| (our_pos as i64 - true_pos as i64).unsigned_abs())
            .min()
            .unwrap_or(u64::MAX);

        if nearest_dist <= tolerance_bp {
            detected += 1;
        }
        total_distance += nearest_dist as f64;
    }

    let fraction_detected = detected as f64 / true_positions.len() as f64;
    let mean_distance = total_distance / true_positions.len() as f64;

    (fraction_detected, mean_distance)
}

/// Comprehensive concordance report comparing two sets of ancestry calls.
#[derive(Debug, Clone)]
pub struct ConcordanceReport {
    /// Overall per-window concordance
    pub overall_concordance: f64,
    /// Per-population precision, recall, F1
    pub per_population: HashMap<String, (f64, f64, f64)>,
    /// Confusion matrix (truth rows × predicted cols)
    pub confusion_matrix: Vec<Vec<u64>>,
    /// Fraction of true switch points detected within tolerance
    pub switch_detection_rate: f64,
    /// Mean distance to nearest true switch point (in windows)
    pub mean_switch_distance: f64,
    /// Number of windows compared
    pub n_windows: usize,
    /// Number of true switch points
    pub n_true_switches: usize,
    /// Number of predicted switch points
    pub n_predicted_switches: usize,
    /// Number of predicted switches that matched a true switch
    pub n_switches_detected: usize,
    /// Number of predicted switches with no matching true switch
    pub n_spurious_switches: usize,
    /// Switch precision: fraction of predictions that match truth
    pub switch_precision: f64,
}

/// Compute a comprehensive concordance report.
///
/// # Arguments
/// * `ours` - Our ancestry indices per window
/// * `truth` - Ground truth ancestry indices per window
/// * `pop_names` - Population names
/// * `switch_tolerance` - Tolerance in windows for switch point matching
pub fn compute_concordance_report(
    ours: &[usize],
    truth: &[usize],
    pop_names: &[String],
    switch_tolerance: usize,
) -> ConcordanceReport {
    let n_windows = ours.len().min(truth.len());
    let ours = &ours[..n_windows];
    let truth = &truth[..n_windows];

    let overall_concordance = per_window_ancestry_concordance(ours, truth);
    let per_population = per_population_concordance(ours, truth, pop_names);
    let confusion_matrix = ancestry_confusion_matrix(ours, truth, pop_names.len());

    let our_switches: Vec<usize> = extract_switch_points(ours).iter()
        .map(|s| s.window_index).collect();
    let true_switches: Vec<usize> = extract_switch_points(truth).iter()
        .map(|s| s.window_index).collect();

    let switch_report = switch_point_accuracy_detailed(
        &our_switches, &true_switches, switch_tolerance);

    ConcordanceReport {
        overall_concordance,
        per_population,
        confusion_matrix,
        switch_detection_rate: switch_report.detection_rate,
        mean_switch_distance: switch_report.mean_distance,
        n_windows,
        n_true_switches: true_switches.len(),
        n_predicted_switches: our_switches.len(),
        n_switches_detected: switch_report.n_detected,
        n_spurious_switches: switch_report.n_spurious,
        switch_precision: switch_report.switch_precision,
    }
}

/// Format a concordance report as a human-readable string.
pub fn format_concordance_report(report: &ConcordanceReport, pop_names: &[String]) -> String {
    let mut out = String::new();

    out.push_str(&format!("Overall concordance: {:.2}% ({} windows)\n",
        report.overall_concordance * 100.0, report.n_windows));
    out.push_str(&format!("Switch points: {} true, {} predicted ({} detected, {} spurious)\n",
        report.n_true_switches, report.n_predicted_switches,
        report.n_switches_detected, report.n_spurious_switches));
    out.push_str(&format!("Switch detection rate: {:.2}% (recall), precision: {:.2}%\n",
        report.switch_detection_rate * 100.0,
        report.switch_precision * 100.0));
    out.push_str(&format!("Mean switch distance: {:.1} windows\n\n",
        report.mean_switch_distance));

    out.push_str("Per-population metrics:\n");
    for name in pop_names {
        if let Some(&(prec, rec, f1)) = report.per_population.get(name) {
            out.push_str(&format!("  {}: precision={:.3}, recall={:.3}, F1={:.3}\n",
                name, prec, rec, f1));
        }
    }

    out.push_str("\nConfusion matrix (rows=truth, cols=predicted):\n");
    out.push_str(&format!("          {}\n", pop_names.iter()
        .map(|n| format!("{:>8}", n)).collect::<Vec<_>>().join("")));
    for (i, row) in report.confusion_matrix.iter().enumerate() {
        if i < pop_names.len() {
            out.push_str(&format!("{:>8}  {}\n", pop_names[i],
                row.iter().map(|v| format!("{:>8}", v)).collect::<Vec<_>>().join("")));
        }
    }

    out
}

/// An ancestry segment with genomic coordinates and population assignment.
#[derive(Debug, Clone)]
pub struct AncestryInterval {
    /// Start position (bp, inclusive)
    pub start: u64,
    /// End position (bp, exclusive)
    pub end: u64,
    /// Population index
    pub ancestry: usize,
}

/// Convert per-window ancestry calls to segments with base-pair coordinates.
///
/// Each window maps to `[window_start + i * window_size, window_start + (i+1) * window_size)`.
/// Adjacent windows with the same ancestry are merged into a single segment.
///
/// # Arguments
/// * `ancestries` - Per-window ancestry indices
/// * `window_start` - Genomic start position of the first window (bp)
/// * `window_size` - Size of each window (bp)
pub fn ancestries_to_segments(
    ancestries: &[usize],
    window_start: u64,
    window_size: u64,
) -> Vec<AncestryInterval> {
    if ancestries.is_empty() || window_size == 0 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut seg_start = window_start;
    let mut seg_ancestry = ancestries[0];

    for (i, &anc) in ancestries.iter().enumerate().skip(1) {
        if anc != seg_ancestry {
            segments.push(AncestryInterval {
                start: seg_start,
                end: window_start + i as u64 * window_size,
                ancestry: seg_ancestry,
            });
            seg_start = window_start + i as u64 * window_size;
            seg_ancestry = anc;
        }
    }
    // Final segment
    segments.push(AncestryInterval {
        start: seg_start,
        end: window_start + ancestries.len() as u64 * window_size,
        ancestry: seg_ancestry,
    });

    segments
}

/// Per-population segment-level Jaccard index.
///
/// For each population, extracts the intervals assigned to that population
/// by both tools, then computes Jaccard = |intersection_bp| / |union_bp|.
///
/// # Arguments
/// * `our_segments` - Our ancestry segments
/// * `truth_segments` - Ground truth ancestry segments
/// * `n_pops` - Number of populations
/// * `region` - Genomic region for clipping (start, end) in bp
///
/// # Returns
/// Vector of per-population Jaccard indices (indexed by population).
pub fn per_population_segment_jaccard(
    our_segments: &[AncestryInterval],
    truth_segments: &[AncestryInterval],
    n_pops: usize,
    region: (u64, u64),
) -> Vec<f64> {
    let mut jaccards = vec![0.0; n_pops];

    for (pop_idx, jaccard) in jaccards.iter_mut().enumerate() {
        let ours: Vec<(u64, u64)> = our_segments.iter()
            .filter(|s| s.ancestry == pop_idx)
            .map(|s| (s.start, s.end))
            .collect();
        let theirs: Vec<(u64, u64)> = truth_segments.iter()
            .filter(|s| s.ancestry == pop_idx)
            .map(|s| (s.start, s.end))
            .collect();

        let intersection = intersection_bp(&ours, &theirs, region);
        let covered_ours = covered_bp(&ours, region);
        let covered_theirs = covered_bp(&theirs, region);
        let union = covered_ours + covered_theirs - intersection;

        *jaccard = if union > 0 { intersection as f64 / union as f64 } else { 0.0 };
    }

    jaccards
}

/// Per-population segment-level precision and recall.
///
/// For each population:
/// - Precision = bp we call as pop X that truth also calls pop X / total bp we call as pop X
/// - Recall = bp truth calls as pop X that we also call pop X / total bp truth calls as pop X
///
/// # Returns
/// Vector of per-population (precision, recall, f1) tuples.
pub fn per_population_segment_precision_recall(
    our_segments: &[AncestryInterval],
    truth_segments: &[AncestryInterval],
    n_pops: usize,
    region: (u64, u64),
) -> Vec<(f64, f64, f64)> {
    let mut results = vec![(0.0, 0.0, 0.0); n_pops];

    for (pop_idx, result) in results.iter_mut().enumerate() {
        let ours: Vec<(u64, u64)> = our_segments.iter()
            .filter(|s| s.ancestry == pop_idx)
            .map(|s| (s.start, s.end))
            .collect();
        let theirs: Vec<(u64, u64)> = truth_segments.iter()
            .filter(|s| s.ancestry == pop_idx)
            .map(|s| (s.start, s.end))
            .collect();

        let intersection = intersection_bp(&ours, &theirs, region);
        let covered_ours = covered_bp(&ours, region);
        let covered_theirs = covered_bp(&theirs, region);

        let precision = if covered_ours > 0 { intersection as f64 / covered_ours as f64 } else { 0.0 };
        let recall = if covered_theirs > 0 { intersection as f64 / covered_theirs as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        *result = (precision, recall, f1);
    }

    results
}

/// Total base pairs covered by a set of segments within a region, handling overlaps.
fn covered_bp(segments: &[(u64, u64)], region: (u64, u64)) -> u64 {
    if segments.is_empty() {
        return 0;
    }
    let mut clipped: Vec<(u64, u64)> = segments.iter()
        .filter_map(|&(s, e)| {
            let cs = s.max(region.0);
            let ce = e.min(region.1);
            if ce > cs { Some((cs, ce)) } else { None }
        })
        .collect();
    if clipped.is_empty() {
        return 0;
    }
    clipped.sort_by_key(|&(s, _)| s);

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

/// Base pairs in the intersection of two segment sets within a region.
fn intersection_bp(a: &[(u64, u64)], b: &[(u64, u64)], region: (u64, u64)) -> u64 {
    let clip = |segs: &[(u64, u64)]| -> Vec<(u64, u64)> {
        segs.iter()
            .filter_map(|&(s, e)| {
                let cs = s.max(region.0);
                let ce = e.min(region.1);
                if ce > cs { Some((cs, ce)) } else { None }
            })
            .collect()
    };

    let a_clipped = clip(a);
    let b_clipped = clip(b);

    if a_clipped.is_empty() || b_clipped.is_empty() {
        return 0;
    }

    let mut total = 0u64;
    for &(a_start, a_end) in &a_clipped {
        for &(b_start, b_end) in &b_clipped {
            let overlap_start = a_start.max(b_start);
            let overlap_end = a_end.min(b_end);
            if overlap_end > overlap_start {
                total += overlap_end - overlap_start;
            }
        }
    }
    total
}

/// Comprehensive segment-level concordance report.
#[derive(Debug, Clone)]
pub struct SegmentConcordanceReport {
    /// Per-population Jaccard index (bp-level)
    pub jaccard_per_pop: Vec<f64>,
    /// Per-population (precision, recall, F1) at bp level
    pub precision_recall_per_pop: Vec<(f64, f64, f64)>,
    /// Population names
    pub pop_names: Vec<String>,
    /// Region analyzed
    pub region: (u64, u64),
}

/// Compute segment-level concordance from per-window ancestry calls.
///
/// Converts window-level ancestry assignments to segments, then computes
/// per-population Jaccard, precision, recall, and F1 at the base-pair level.
///
/// # Arguments
/// * `ours` - Our per-window ancestry indices
/// * `truth` - Ground truth per-window ancestry indices
/// * `pop_names` - Population names
/// * `window_start` - Start position of the first window (bp)
/// * `window_size` - Window size (bp)
pub fn compute_segment_concordance(
    ours: &[usize],
    truth: &[usize],
    pop_names: &[String],
    window_start: u64,
    window_size: u64,
) -> SegmentConcordanceReport {
    let n_pops = pop_names.len();
    let len = ours.len().min(truth.len());
    let ours = &ours[..len];
    let truth = &truth[..len];

    let our_segs = ancestries_to_segments(ours, window_start, window_size);
    let truth_segs = ancestries_to_segments(truth, window_start, window_size);

    let region = (window_start, window_start + len as u64 * window_size);

    let jaccard_per_pop = per_population_segment_jaccard(&our_segs, &truth_segs, n_pops, region);
    let precision_recall_per_pop = per_population_segment_precision_recall(&our_segs, &truth_segs, n_pops, region);

    SegmentConcordanceReport {
        jaccard_per_pop,
        precision_recall_per_pop,
        pop_names: pop_names.to_vec(),
        region,
    }
}

/// Format a segment concordance report as a human-readable string.
pub fn format_segment_concordance(report: &SegmentConcordanceReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Segment-level concordance (region: {}-{}, {:.2} Mb)\n",
        report.region.0, report.region.1,
        (report.region.1 - report.region.0) as f64 / 1_000_000.0
    ));
    for (i, name) in report.pop_names.iter().enumerate() {
        let (prec, rec, f1) = report.precision_recall_per_pop[i];
        out.push_str(&format!(
            "  {}: Jaccard={:.4}, precision={:.4}, recall={:.4}, F1={:.4}\n",
            name, report.jaccard_per_pop[i], prec, rec, f1
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_concordance() {
        let calls = vec![0, 0, 0, 1, 1, 1, 0, 0];
        let truth = vec![0, 0, 0, 1, 1, 1, 0, 0];

        assert!((per_window_ancestry_concordance(&calls, &truth) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_concordance() {
        let calls = vec![0, 0, 0, 0];
        let truth = vec![1, 1, 1, 1];

        assert!((per_window_ancestry_concordance(&calls, &truth) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_concordance() {
        let calls = vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 0, 0]; // 4/5 match

        let conc = per_window_ancestry_concordance(&calls, &truth);
        assert!((conc - 0.8).abs() < 1e-10, "Expected 0.8, got {}", conc);
    }

    #[test]
    fn test_concordance_empty() {
        assert!((per_window_ancestry_concordance(&[], &[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_concordance_different_lengths() {
        let calls = vec![0, 0, 1];
        let truth = vec![0, 0, 1, 1, 1]; // only first 3 compared

        let conc = per_window_ancestry_concordance(&calls, &truth);
        assert!((conc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_population_perfect() {
        let calls = vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 1, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let metrics = per_population_concordance(&calls, &truth, &names);

        assert!((metrics["AFR"].0 - 1.0).abs() < 1e-10); // precision
        assert!((metrics["AFR"].1 - 1.0).abs() < 1e-10); // recall
        assert!((metrics["AFR"].2 - 1.0).abs() < 1e-10); // F1

        assert!((metrics["EUR"].0 - 1.0).abs() < 1e-10);
        assert!((metrics["EUR"].1 - 1.0).abs() < 1e-10);
        assert!((metrics["EUR"].2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_population_all_wrong() {
        let calls = vec![1, 1, 1, 1]; // predict all EUR
        let truth = vec![0, 0, 0, 0]; // all AFR
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let metrics = per_population_concordance(&calls, &truth, &names);

        // AFR: precision=NaN→0 (never predicted), recall=0 (never detected)
        assert!((metrics["AFR"].0 - 0.0).abs() < 1e-10, "AFR precision should be 0");
        assert!((metrics["AFR"].1 - 0.0).abs() < 1e-10, "AFR recall should be 0");

        // EUR: precision=0 (all FP), recall=NaN→0 (no true EUR)
        assert!((metrics["EUR"].0 - 0.0).abs() < 1e-10, "EUR precision should be 0");
    }

    #[test]
    fn test_confusion_matrix() {
        let calls = vec![0, 0, 1, 1, 0];
        let truth = vec![0, 1, 1, 0, 0];

        let matrix = ancestry_confusion_matrix(&calls, &truth, 2);

        // truth=0, predicted=0: windows 0, 4 → 2
        assert_eq!(matrix[0][0], 2);
        // truth=0, predicted=1: window 3 → 1
        assert_eq!(matrix[0][1], 1);
        // truth=1, predicted=0: window 1 → 1
        assert_eq!(matrix[1][0], 1);
        // truth=1, predicted=1: window 2 → 1
        assert_eq!(matrix[1][1], 1);
    }

    #[test]
    fn test_extract_switch_points() {
        let ancestries = vec![0, 0, 0, 1, 1, 0, 0, 1];
        let switches = extract_switch_points(&ancestries);

        assert_eq!(switches.len(), 3);
        assert_eq!(switches[0].window_index, 3);
        assert_eq!(switches[0].from_ancestry, 0);
        assert_eq!(switches[0].to_ancestry, 1);
        assert_eq!(switches[1].window_index, 5);
        assert_eq!(switches[2].window_index, 7);
    }

    #[test]
    fn test_switch_point_accuracy_exact() {
        let our_switches = vec![3, 7, 12];
        let true_switches = vec![3, 7, 12];

        let (frac, mean_dist) = switch_point_accuracy(&our_switches, &true_switches, 2);
        assert!((frac - 1.0).abs() < 1e-10);
        assert!((mean_dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_switch_point_accuracy_within_tolerance() {
        let our_switches = vec![4, 8]; // off by 1
        let true_switches = vec![3, 7];

        let (frac, mean_dist) = switch_point_accuracy(&our_switches, &true_switches, 2);
        assert!((frac - 1.0).abs() < 1e-10, "All within tolerance, should be 1.0");
        assert!((mean_dist - 1.0).abs() < 1e-10, "Mean distance should be 1.0");
    }

    #[test]
    fn test_switch_point_accuracy_missed() {
        let our_switches = vec![10, 20]; // far from truth
        let true_switches = vec![3, 7];

        let (frac, _) = switch_point_accuracy(&our_switches, &true_switches, 2);
        assert!((frac - 0.0).abs() < 1e-10, "All outside tolerance, should be 0.0");
    }

    #[test]
    fn test_switch_point_accuracy_no_true_switches() {
        let our_switches = vec![3, 7];
        let true_switches: Vec<usize> = vec![];

        let (frac, mean_dist) = switch_point_accuracy(&our_switches, &true_switches, 2);
        assert!((frac - 1.0).abs() < 1e-10);
        assert!((mean_dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_switch_point_accuracy_no_predicted() {
        let our_switches: Vec<usize> = vec![];
        let true_switches = vec![3, 7];

        let (frac, mean_dist) = switch_point_accuracy(&our_switches, &true_switches, 2);
        assert!((frac - 0.0).abs() < 1e-10);
        assert!(mean_dist.is_infinite());
    }

    #[test]
    fn test_switch_point_accuracy_bp() {
        let our_pos = vec![500_000u64, 1_200_000];
        let true_pos = vec![480_000u64, 1_150_000];

        // Tolerance 100kb: both within tolerance
        let (frac, mean_dist) = switch_point_accuracy_bp(&our_pos, &true_pos, 100_000);
        assert!((frac - 1.0).abs() < 1e-10);
        assert!((mean_dist - 35_000.0).abs() < 1e-6); // (20k + 50k) / 2

        // Tolerance 10kb: only the first is within
        let (frac2, _) = switch_point_accuracy_bp(&our_pos, &true_pos, 25_000);
        assert!((frac2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_concordance_report() {
        let ours =  vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0];
        let truth = vec![0, 0, 0, 1, 1, 0, 0, 0, 0, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let report = compute_concordance_report(&ours, &truth, &names, 2);

        // 9/10 match
        assert!((report.overall_concordance - 0.9).abs() < 1e-10);
        assert_eq!(report.n_windows, 10);

        // Truth switches at index 3 (0→1) and 5 (1→0) = 2 switches
        assert_eq!(report.n_true_switches, 2);
        // Predicted switches at 3 (0→1) and 6 (1→0) = 2 switches
        assert_eq!(report.n_predicted_switches, 2);

        // Both switch points should be detected within tolerance 2
        assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_format_concordance_report() {
        let ours =  vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 1, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let report = compute_concordance_report(&ours, &truth, &names, 2);
        let formatted = format_concordance_report(&report, &names);

        assert!(formatted.contains("100.00%"));
        assert!(formatted.contains("AFR"));
        assert!(formatted.contains("EUR"));
    }

    // ========================================================================
    // Segment-level overlap metrics tests
    // ========================================================================

    #[test]
    fn test_ancestries_to_segments_basic() {
        let ancestries = vec![0, 0, 1, 1, 1, 0];
        let segs = ancestries_to_segments(&ancestries, 1000, 100);

        assert_eq!(segs.len(), 3);
        // First: pop 0, [1000, 1200)
        assert_eq!(segs[0].start, 1000);
        assert_eq!(segs[0].end, 1200);
        assert_eq!(segs[0].ancestry, 0);
        // Second: pop 1, [1200, 1500)
        assert_eq!(segs[1].start, 1200);
        assert_eq!(segs[1].end, 1500);
        assert_eq!(segs[1].ancestry, 1);
        // Third: pop 0, [1500, 1600)
        assert_eq!(segs[2].start, 1500);
        assert_eq!(segs[2].end, 1600);
        assert_eq!(segs[2].ancestry, 0);
    }

    #[test]
    fn test_ancestries_to_segments_homogeneous() {
        let ancestries = vec![1, 1, 1, 1];
        let segs = ancestries_to_segments(&ancestries, 0, 10_000);

        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 40_000);
        assert_eq!(segs[0].ancestry, 1);
    }

    #[test]
    fn test_ancestries_to_segments_empty() {
        let segs = ancestries_to_segments(&[], 0, 100);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_ancestries_to_segments_single() {
        let segs = ancestries_to_segments(&[2], 5000, 500);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start, 5000);
        assert_eq!(segs[0].end, 5500);
        assert_eq!(segs[0].ancestry, 2);
    }

    #[test]
    fn test_ancestries_to_segments_zero_window_size() {
        let segs = ancestries_to_segments(&[0, 1], 0, 0);
        assert!(segs.is_empty());
    }

    #[test]
    fn test_per_population_segment_jaccard_perfect() {
        // Same calls → Jaccard = 1.0 for all populations
        let ancestries = vec![0, 0, 1, 1, 0];
        let our_segs = ancestries_to_segments(&ancestries, 0, 100);
        let truth_segs = ancestries_to_segments(&ancestries, 0, 100);
        let region = (0, 500);

        let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);
        assert!((jaccards[0] - 1.0).abs() < 1e-10, "Pop 0 should be 1.0, got {}", jaccards[0]);
        assert!((jaccards[1] - 1.0).abs() < 1e-10, "Pop 1 should be 1.0, got {}", jaccards[1]);
    }

    #[test]
    fn test_per_population_segment_jaccard_no_overlap() {
        // Completely opposite calls
        let ours = vec![0, 0, 0, 0];
        let truth = vec![1, 1, 1, 1];
        let our_segs = ancestries_to_segments(&ours, 0, 100);
        let truth_segs = ancestries_to_segments(&truth, 0, 100);
        let region = (0, 400);

        let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);
        assert!((jaccards[0] - 0.0).abs() < 1e-10, "Pop 0 Jaccard should be 0");
        assert!((jaccards[1] - 0.0).abs() < 1e-10, "Pop 1 Jaccard should be 0");
    }

    #[test]
    fn test_per_population_segment_jaccard_partial() {
        // Ours: [0, 0, 1, 1] → pop0 covers [0, 200), pop1 covers [200, 400)
        // Truth: [0, 1, 1, 1] → pop0 covers [0, 100), pop1 covers [100, 400)
        let ours = vec![0, 0, 1, 1];
        let truth = vec![0, 1, 1, 1];
        let our_segs = ancestries_to_segments(&ours, 0, 100);
        let truth_segs = ancestries_to_segments(&truth, 0, 100);
        let region = (0, 400);

        let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);

        // Pop 0: ours = [0,200), truth = [0,100). Intersection = 100, union = 200. J = 0.5
        assert!((jaccards[0] - 0.5).abs() < 1e-10, "Pop 0 Jaccard: expected 0.5, got {}", jaccards[0]);

        // Pop 1: ours = [200,400), truth = [100,400). Intersection = 200, union = 300. J = 2/3
        assert!((jaccards[1] - 2.0/3.0).abs() < 1e-10, "Pop 1 Jaccard: expected 0.667, got {}", jaccards[1]);
    }

    #[test]
    fn test_per_population_segment_precision_recall_perfect() {
        let ancestries = vec![0, 0, 1, 1, 0];
        let our_segs = ancestries_to_segments(&ancestries, 0, 100);
        let truth_segs = ancestries_to_segments(&ancestries, 0, 100);
        let region = (0, 500);

        let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);
        // All perfect
        assert!((pr[0].0 - 1.0).abs() < 1e-10); // pop0 precision
        assert!((pr[0].1 - 1.0).abs() < 1e-10); // pop0 recall
        assert!((pr[0].2 - 1.0).abs() < 1e-10); // pop0 F1
        assert!((pr[1].0 - 1.0).abs() < 1e-10); // pop1 precision
        assert!((pr[1].1 - 1.0).abs() < 1e-10); // pop1 recall
        assert!((pr[1].2 - 1.0).abs() < 1e-10); // pop1 F1
    }

    #[test]
    fn test_per_population_segment_precision_recall_partial() {
        // Ours: [0, 0, 1, 1], Truth: [0, 1, 1, 1]
        let ours = vec![0, 0, 1, 1];
        let truth = vec![0, 1, 1, 1];
        let our_segs = ancestries_to_segments(&ours, 0, 100);
        let truth_segs = ancestries_to_segments(&truth, 0, 100);
        let region = (0, 400);

        let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);

        // Pop 0: ours=[0,200) truth=[0,100). intersection=100.
        //   Precision = 100/200 = 0.5, Recall = 100/100 = 1.0
        assert!((pr[0].0 - 0.5).abs() < 1e-10, "Pop 0 precision: expected 0.5, got {}", pr[0].0);
        assert!((pr[0].1 - 1.0).abs() < 1e-10, "Pop 0 recall: expected 1.0, got {}", pr[0].1);

        // Pop 1: ours=[200,400) truth=[100,400). intersection=200.
        //   Precision = 200/200 = 1.0, Recall = 200/300 = 2/3
        assert!((pr[1].0 - 1.0).abs() < 1e-10, "Pop 1 precision: expected 1.0, got {}", pr[1].0);
        assert!((pr[1].1 - 2.0/3.0).abs() < 1e-10, "Pop 1 recall: expected 0.667, got {}", pr[1].1);
    }

    #[test]
    fn test_per_population_segment_precision_recall_all_wrong() {
        // All predicted as 0, truth is all 1
        let ours = vec![0, 0, 0, 0];
        let truth = vec![1, 1, 1, 1];
        let our_segs = ancestries_to_segments(&ours, 0, 100);
        let truth_segs = ancestries_to_segments(&truth, 0, 100);
        let region = (0, 400);

        let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);
        // Pop 0: ours=400bp, truth=0bp, intersection=0. Precision=0, Recall=0 (no truth bp)
        assert!((pr[0].0 - 0.0).abs() < 1e-10);
        assert!((pr[0].1 - 0.0).abs() < 1e-10);
        // Pop 1: ours=0bp, truth=400bp, intersection=0. Precision=0 (no ours bp), Recall=0
        assert!((pr[1].0 - 0.0).abs() < 1e-10);
        assert!((pr[1].1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_covered_bp_basic() {
        let segments = vec![(100, 300), (500, 800)];
        let region = (0, 1000);
        assert_eq!(covered_bp(&segments, region), 500); // 200 + 300
    }

    #[test]
    fn test_covered_bp_overlapping() {
        let segments = vec![(100, 300), (200, 500)];
        let region = (0, 1000);
        assert_eq!(covered_bp(&segments, region), 400); // merged: [100, 500)
    }

    #[test]
    fn test_covered_bp_clipped() {
        let segments = vec![(100, 600)];
        let region = (200, 400);
        assert_eq!(covered_bp(&segments, region), 200); // clipped to [200, 400)
    }

    #[test]
    fn test_covered_bp_empty() {
        assert_eq!(covered_bp(&[], (0, 1000)), 0);
    }

    #[test]
    fn test_intersection_bp_basic() {
        let a = vec![(100, 300)];
        let b = vec![(200, 400)];
        let region = (0, 500);
        assert_eq!(intersection_bp(&a, &b, region), 100); // [200, 300)
    }

    #[test]
    fn test_intersection_bp_no_overlap() {
        let a = vec![(100, 200)];
        let b = vec![(300, 400)];
        let region = (0, 500);
        assert_eq!(intersection_bp(&a, &b, region), 0);
    }

    #[test]
    fn test_compute_segment_concordance_basic() {
        // Perfect agreement
        let ours =  vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 1, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);

        assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
        assert!((report.jaccard_per_pop[1] - 1.0).abs() < 1e-10);
        assert_eq!(report.region, (0, 50_000));
    }

    #[test]
    fn test_compute_segment_concordance_mixed() {
        // Ours: [EUR, EUR, AFR, AFR], Truth: [EUR, AFR, AFR, AFR]
        // EUR: ours=[0,20k), truth=[0,10k). J=10k/20k = 0.5
        // AFR: ours=[20k,40k), truth=[10k,40k). J=20k/30k = 0.667
        let ours =  vec![1, 1, 0, 0]; // 1=EUR, 0=AFR
        let truth = vec![1, 0, 0, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);

        // AFR: ours=[20k,40k), truth=[10k,40k). intersection=20k, union=30k. J=2/3
        assert!((report.jaccard_per_pop[0] - 2.0/3.0).abs() < 1e-10,
            "AFR Jaccard: expected 0.667, got {}", report.jaccard_per_pop[0]);
        // EUR: ours=[0,20k), truth=[0,10k). intersection=10k, union=20k. J=0.5
        assert!((report.jaccard_per_pop[1] - 0.5).abs() < 1e-10,
            "EUR Jaccard: expected 0.5, got {}", report.jaccard_per_pop[1]);
    }

    #[test]
    fn test_compute_segment_concordance_three_pops() {
        // 3-way ancestry: perfect agreement
        let ancestries = vec![0, 0, 1, 1, 2, 2, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string(), "NAT".to_string()];

        let report = compute_segment_concordance(&ancestries, &ancestries, &names, 0, 1000);

        assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
        assert!((report.jaccard_per_pop[1] - 1.0).abs() < 1e-10);
        assert!((report.jaccard_per_pop[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_format_segment_concordance() {
        let ours =  vec![0, 0, 1, 1, 0];
        let truth = vec![0, 0, 1, 1, 0];
        let names = vec!["AFR".to_string(), "EUR".to_string()];

        let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);
        let formatted = format_segment_concordance(&report);

        assert!(formatted.contains("AFR"));
        assert!(formatted.contains("EUR"));
        assert!(formatted.contains("Jaccard"));
        assert!(formatted.contains("1.0000"));
    }

    #[test]
    fn test_segment_jaccard_empty_segments() {
        let jaccards = per_population_segment_jaccard(&[], &[], 2, (0, 1000));
        assert!((jaccards[0] - 0.0).abs() < 1e-10);
        assert!((jaccards[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_segment_precision_recall_empty() {
        let pr = per_population_segment_precision_recall(&[], &[], 2, (0, 1000));
        assert!((pr[0].0 - 0.0).abs() < 1e-10);
        assert!((pr[0].2 - 0.0).abs() < 1e-10);
    }

    // ========================================================================
    // switch_point_accuracy_detailed tests
    // ========================================================================

    #[test]
    fn test_detailed_accuracy_perfect_match() {
        let ours = vec![3, 7, 12];
        let truth = vec![3, 7, 12];
        let report = switch_point_accuracy_detailed(&ours, &truth, 2);
        assert_eq!(report.n_detected, 3);
        assert_eq!(report.n_missed, 0);
        assert_eq!(report.n_spurious, 0);
        assert!((report.detection_rate - 1.0).abs() < 1e-10);
        assert!((report.switch_precision - 1.0).abs() < 1e-10);
        assert!((report.mean_distance - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_all_spurious() {
        let ours = vec![3, 7, 12];
        let truth: Vec<usize> = vec![];
        let report = switch_point_accuracy_detailed(&ours, &truth, 2);
        assert_eq!(report.n_detected, 0);
        assert_eq!(report.n_missed, 0);
        assert_eq!(report.n_spurious, 3);
        assert!((report.detection_rate - 1.0).abs() < 1e-10); // vacuously
        assert!((report.switch_precision - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_all_missed() {
        let ours: Vec<usize> = vec![];
        let truth = vec![3, 7, 12];
        let report = switch_point_accuracy_detailed(&ours, &truth, 2);
        assert_eq!(report.n_detected, 0);
        assert_eq!(report.n_missed, 3);
        assert_eq!(report.n_spurious, 0);
        assert!((report.detection_rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_mixed() {
        // True: [5, 15], Pred: [4, 10, 20]
        // 5 matched by 4 (dist=1), 15 matched by 10 (dist=5, > tol=2) → missed
        // Result: 1 detected (5→4), 1 missed (15), 2 spurious (10, 20)
        let ours = vec![4, 10, 20];
        let truth = vec![5, 15];
        let report = switch_point_accuracy_detailed(&ours, &truth, 2);
        assert_eq!(report.n_detected, 1);
        assert_eq!(report.n_missed, 1);
        assert_eq!(report.n_spurious, 2);
        assert!((report.detection_rate - 0.5).abs() < 1e-10);
        assert!((report.switch_precision - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_both_empty() {
        let report = switch_point_accuracy_detailed(&[], &[], 5);
        assert_eq!(report.n_detected, 0);
        assert_eq!(report.n_missed, 0);
        assert_eq!(report.n_spurious, 0);
        assert!((report.detection_rate - 1.0).abs() < 1e-10);
        assert!((report.switch_precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_greedy_matching() {
        // True: [5, 10], Pred: [6]
        // 6 matches 5 (dist=1), 10 is missed
        let report = switch_point_accuracy_detailed(&[6], &[5, 10], 2);
        assert_eq!(report.n_detected, 1);
        assert_eq!(report.n_missed, 1);
        assert_eq!(report.n_spurious, 0);
        assert!((report.switch_precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_concordance_report_integration() {
        // Test that ConcordanceReport now includes spurious switch info
        let ours =  vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0];
        let truth = vec![0, 0, 0, 1, 1, 0, 0, 0, 0, 0];
        let names = vec!["A".to_string(), "B".to_string()];
        let report = compute_concordance_report(&ours, &truth, &names, 2);

        // Both switches detected within tolerance 2
        assert_eq!(report.n_switches_detected, 2);
        assert_eq!(report.n_spurious_switches, 0);
        assert!((report.switch_precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_detailed_accuracy_concordance_report_with_spurious() {
        // ours has extra switches not in truth
        // Switches at indices: 1 (0→1), 2 (1→0), 3 (0→1), 4 (1→0) = 4 switches
        let ours =  vec![0, 1, 0, 1, 0, 0, 0, 0];
        let truth = vec![0, 0, 0, 0, 0, 0, 0, 0]; // 0 switches
        let names = vec!["A".to_string(), "B".to_string()];
        let report = compute_concordance_report(&ours, &truth, &names, 2);

        assert_eq!(report.n_true_switches, 0);
        assert_eq!(report.n_predicted_switches, 4);
        assert_eq!(report.n_switches_detected, 0);
        assert_eq!(report.n_spurious_switches, 4);
        assert!((report.switch_precision - 0.0).abs() < 1e-10);
    }
}
