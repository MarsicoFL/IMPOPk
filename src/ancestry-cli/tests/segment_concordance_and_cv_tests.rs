//! Tests for ancestry segment concordance, confusion matrix, switch points,
//! cross-validation edge cases, and ancestries_to_segments.
//!
//! Cycle 23: targets uncovered edge cases in per_population_segment_jaccard,
//! per_population_segment_precision_recall, ancestry_confusion_matrix,
//! ancestries_to_segments, and CrossValidationResult formatting.

use std::collections::HashMap;

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    CrossValidationResult,
    cross_validate, cross_validate_kfold,
};

use hprc_ancestry_cli::concordance::{
    ancestry_confusion_matrix, ancestries_to_segments, extract_switch_points,
    per_population_concordance, per_population_segment_jaccard,
    per_population_segment_precision_recall, per_window_ancestry_concordance,
    switch_point_accuracy, switch_point_accuracy_detailed, AncestryInterval,
};

// ============================================================================
// per_window_ancestry_concordance
// ============================================================================

#[test]
fn test_per_window_concordance_empty() {
    assert_eq!(per_window_ancestry_concordance(&[], &[]), 0.0);
}

#[test]
fn test_per_window_concordance_perfect() {
    let ours = vec![0, 1, 2, 0, 1];
    let truth = vec![0, 1, 2, 0, 1];
    assert!((per_window_ancestry_concordance(&ours, &truth) - 1.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_none_matching() {
    let ours = vec![0, 0, 0];
    let truth = vec![1, 1, 1];
    assert!((per_window_ancestry_concordance(&ours, &truth) - 0.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_different_lengths() {
    // Should use the shorter length
    let ours = vec![0, 1, 2, 0, 1];
    let truth = vec![0, 1, 2]; // shorter
    // First 3 match → 3/3 = 1.0
    assert!((per_window_ancestry_concordance(&ours, &truth) - 1.0).abs() < 1e-10);
}

#[test]
fn test_per_window_concordance_half() {
    let ours = vec![0, 1, 0, 1];
    let truth = vec![0, 0, 1, 1];
    // Matches: [0]==, [3]== → 2/4 = 0.5
    assert!((per_window_ancestry_concordance(&ours, &truth) - 0.5).abs() < 1e-10);
}

// ============================================================================
// per_population_concordance
// ============================================================================

#[test]
fn test_per_population_concordance_perfect_3pop() {
    let ours = vec![0, 0, 1, 1, 2, 2];
    let truth = vec![0, 0, 1, 1, 2, 2];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];

    let result = per_population_concordance(&ours, &truth, &pop_names);
    for (_, (p, r, f1)) in &result {
        assert!((*p - 1.0).abs() < 1e-10);
        assert!((*r - 1.0).abs() < 1e-10);
        assert!((*f1 - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_per_population_concordance_all_wrong() {
    let ours = vec![0, 0, 0, 0]; // Predict all EUR
    let truth = vec![1, 1, 1, 1]; // All AFR
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let result = per_population_concordance(&ours, &truth, &pop_names);
    // EUR: TP=0, FP=4, FN=0 → precision=0, recall=0 (no true EUR windows)
    // AFR: TP=0, FP=0, FN=4 → precision=0 (no predictions), recall=0
    assert!((result["EUR"].0 - 0.0).abs() < 1e-10); // precision
    assert!((result["AFR"].1 - 0.0).abs() < 1e-10); // recall
}

#[test]
fn test_per_population_concordance_one_pop_never_predicted() {
    let ours = vec![0, 0, 0]; // Only predict pop 0
    let truth = vec![0, 1, 0]; // Pop 1 exists in truth
    let pop_names = vec!["A".to_string(), "B".to_string()];

    let result = per_population_concordance(&ours, &truth, &pop_names);
    // Pop B: TP=0, FP=0, FN=1 → precision=0 (no predictions), recall=0
    assert!((result["B"].0 - 0.0).abs() < 1e-10);
    assert!((result["B"].1 - 0.0).abs() < 1e-10);
}

// ============================================================================
// ancestry_confusion_matrix
// ============================================================================

#[test]
fn test_confusion_matrix_perfect() {
    let ours = vec![0, 0, 1, 1, 2, 2];
    let truth = vec![0, 0, 1, 1, 2, 2];
    let matrix = ancestry_confusion_matrix(&ours, &truth, 3);
    // Diagonal should have all counts
    assert_eq!(matrix[0][0], 2);
    assert_eq!(matrix[1][1], 2);
    assert_eq!(matrix[2][2], 2);
    // Off-diagonal should be 0
    assert_eq!(matrix[0][1], 0);
    assert_eq!(matrix[1][0], 0);
}

#[test]
fn test_confusion_matrix_all_confused() {
    let ours = vec![1, 1, 1]; // Predict all pop 1
    let truth = vec![0, 0, 0]; // All pop 0
    let matrix = ancestry_confusion_matrix(&ours, &truth, 2);
    assert_eq!(matrix[0][1], 3); // truth=0, pred=1
    assert_eq!(matrix[0][0], 0);
    assert_eq!(matrix[1][0], 0);
    assert_eq!(matrix[1][1], 0);
}

#[test]
fn test_confusion_matrix_out_of_range_indices() {
    // Indices >= n_pops should be silently ignored
    let ours = vec![0, 5, 1]; // 5 is out of range for 3 pops
    let truth = vec![0, 1, 1];
    let matrix = ancestry_confusion_matrix(&ours, &truth, 3);
    assert_eq!(matrix[0][0], 1); // truth=0, pred=0
    assert_eq!(matrix[1][1], 1); // truth=1, pred=1
    // The 5 index should be ignored
    let total: u64 = matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 2); // Only 2 valid entries
}

#[test]
fn test_confusion_matrix_empty() {
    let matrix = ancestry_confusion_matrix(&[], &[], 3);
    let total: u64 = matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 0);
}

// ============================================================================
// extract_switch_points
// ============================================================================

#[test]
fn test_extract_switch_points_no_switches() {
    let ancestries = vec![0, 0, 0, 0];
    let switches = extract_switch_points(&ancestries);
    assert!(switches.is_empty());
}

#[test]
fn test_extract_switch_points_alternating() {
    let ancestries = vec![0, 1, 0, 1, 0];
    let switches = extract_switch_points(&ancestries);
    assert_eq!(switches.len(), 4);
    assert_eq!(switches[0].window_index, 1);
    assert_eq!(switches[0].from_ancestry, 0);
    assert_eq!(switches[0].to_ancestry, 1);
}

#[test]
fn test_extract_switch_points_single_switch() {
    let ancestries = vec![0, 0, 1, 1];
    let switches = extract_switch_points(&ancestries);
    assert_eq!(switches.len(), 1);
    assert_eq!(switches[0].window_index, 2);
    assert_eq!(switches[0].from_ancestry, 0);
    assert_eq!(switches[0].to_ancestry, 1);
}

#[test]
fn test_extract_switch_points_empty() {
    let switches = extract_switch_points(&[]);
    assert!(switches.is_empty());
}

#[test]
fn test_extract_switch_points_single_element() {
    let switches = extract_switch_points(&[0]);
    assert!(switches.is_empty());
}

// ============================================================================
// switch_point_accuracy
// ============================================================================

#[test]
fn test_switch_point_accuracy_no_true_switches() {
    let (frac, dist) = switch_point_accuracy(&[5, 10], &[], 3);
    // No true switches → perfect detection
    assert!((frac - 1.0).abs() < 1e-10);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_switch_point_accuracy_no_predicted_switches() {
    let (frac, dist) = switch_point_accuracy(&[], &[5, 10], 3);
    assert!((frac - 0.0).abs() < 1e-10);
    assert!(dist.is_infinite());
}

#[test]
fn test_switch_point_accuracy_perfect_detection() {
    let (frac, dist) = switch_point_accuracy(&[5, 10, 15], &[5, 10, 15], 0);
    assert!((frac - 1.0).abs() < 1e-10);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_switch_point_accuracy_within_tolerance() {
    // Predicted at 6,11 but true at 5,10 — within tolerance of 2
    let (frac, _dist) = switch_point_accuracy(&[6, 11], &[5, 10], 2);
    assert!((frac - 1.0).abs() < 1e-10);
}

#[test]
fn test_switch_point_accuracy_outside_tolerance() {
    // Predicted at 100 but true at 5 — outside tolerance of 2
    let (frac, dist) = switch_point_accuracy(&[100], &[5], 2);
    assert!((frac - 0.0).abs() < 1e-10);
    assert!((dist - 95.0).abs() < 1e-10);
}

// ============================================================================
// switch_point_accuracy_detailed
// ============================================================================

#[test]
fn test_switch_point_accuracy_detailed_both_empty() {
    let report = switch_point_accuracy_detailed(&[], &[], 5);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert_eq!(report.n_detected, 0);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 0);
    assert!((report.switch_precision - 1.0).abs() < 1e-10);
}

#[test]
fn test_switch_point_accuracy_detailed_no_true() {
    let report = switch_point_accuracy_detailed(&[5, 10], &[], 3);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert_eq!(report.n_spurious, 2);
    assert!((report.switch_precision - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_point_accuracy_detailed_no_predicted() {
    let report = switch_point_accuracy_detailed(&[], &[5, 10], 3);
    assert!((report.detection_rate - 0.0).abs() < 1e-10);
    assert_eq!(report.n_missed, 2);
    assert_eq!(report.n_spurious, 0);
    assert!((report.switch_precision - 1.0).abs() < 1e-10); // vacuously true
}

#[test]
fn test_switch_point_accuracy_detailed_greedy_matching() {
    // True switches at 5 and 10; predicted at 6 (near 5) and 11 (near 10)
    let report = switch_point_accuracy_detailed(&[6, 11], &[5, 10], 3);
    assert_eq!(report.n_detected, 2);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 0);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert!((report.switch_precision - 1.0).abs() < 1e-10);
}

#[test]
fn test_switch_point_accuracy_detailed_mixed() {
    // True at 5,10,20; predicted at 6,15
    // 6 matches 5 (dist=1, within 3), 15 doesn't match 10 (dist=5, outside 3), matches 20? dist=5, outside 3
    // So: detected=1 (just 5→6), missed=2 (10,20), spurious=1 (15)
    let report = switch_point_accuracy_detailed(&[6, 15], &[5, 10, 20], 3);
    assert_eq!(report.n_detected, 1);
    assert_eq!(report.n_missed, 2);
    // After matching 6→5, 15 is unmatched → spurious
    assert_eq!(report.n_spurious, 1);
}

// ============================================================================
// ancestries_to_segments
// ============================================================================

#[test]
fn test_ancestries_to_segments_empty() {
    let segments = ancestries_to_segments(&[], 0, 1000);
    assert!(segments.is_empty());
}

#[test]
fn test_ancestries_to_segments_zero_window_size() {
    let segments = ancestries_to_segments(&[0, 1, 2], 0, 0);
    assert!(segments.is_empty());
}

#[test]
fn test_ancestries_to_segments_single_window() {
    let segments = ancestries_to_segments(&[2], 1000, 5000);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 1000);
    assert_eq!(segments[0].end, 6000);
    assert_eq!(segments[0].ancestry, 2);
}

#[test]
fn test_ancestries_to_segments_all_same() {
    let segments = ancestries_to_segments(&[1, 1, 1, 1], 0, 1000);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 0);
    assert_eq!(segments[0].end, 4000);
    assert_eq!(segments[0].ancestry, 1);
}

#[test]
fn test_ancestries_to_segments_two_segments() {
    let segments = ancestries_to_segments(&[0, 0, 1, 1], 0, 1000);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start, 0);
    assert_eq!(segments[0].end, 2000);
    assert_eq!(segments[0].ancestry, 0);
    assert_eq!(segments[1].start, 2000);
    assert_eq!(segments[1].end, 4000);
    assert_eq!(segments[1].ancestry, 1);
}

#[test]
fn test_ancestries_to_segments_alternating() {
    let segments = ancestries_to_segments(&[0, 1, 0, 1, 0], 0, 1000);
    assert_eq!(segments.len(), 5);
    for (i, seg) in segments.iter().enumerate() {
        assert_eq!(seg.start, i as u64 * 1000);
        assert_eq!(seg.end, (i as u64 + 1) * 1000);
        assert_eq!(seg.ancestry, i % 2);
    }
}

#[test]
fn test_ancestries_to_segments_with_offset() {
    let segments = ancestries_to_segments(&[0, 1], 5_000_000, 10000);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start, 5_000_000);
    assert_eq!(segments[0].end, 5_010_000);
    assert_eq!(segments[1].start, 5_010_000);
    assert_eq!(segments[1].end, 5_020_000);
}

// ============================================================================
// per_population_segment_jaccard
// ============================================================================

#[test]
fn test_per_population_segment_jaccard_perfect_match() {
    let ours = vec![
        AncestryInterval { start: 0, end: 5000, ancestry: 0 },
        AncestryInterval { start: 5000, end: 10000, ancestry: 1 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 5000, ancestry: 0 },
        AncestryInterval { start: 5000, end: 10000, ancestry: 1 },
    ];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 2, (0, 10000));
    assert!((jaccards[0] - 1.0).abs() < 1e-9);
    assert!((jaccards[1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_per_population_segment_jaccard_no_overlap() {
    // We assign everything to pop 0, truth assigns to pop 1
    let ours = vec![
        AncestryInterval { start: 0, end: 10000, ancestry: 0 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 10000, ancestry: 1 },
    ];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 2, (0, 10000));
    assert!((jaccards[0] - 0.0).abs() < 1e-9); // pop 0: ours covers 10k, truth 0 → J=0
    assert!((jaccards[1] - 0.0).abs() < 1e-9); // pop 1: ours covers 0, truth 10k → J=0
}

#[test]
fn test_per_population_segment_jaccard_partial() {
    // Pop 0: ours [0,6000), truth [3000,10000) → intersection [3000,6000)=3000
    //   covered_ours=6000, covered_truth=7000, union=10000, J=3000/10000=0.3
    let ours = vec![
        AncestryInterval { start: 0, end: 6000, ancestry: 0 },
        AncestryInterval { start: 6000, end: 10000, ancestry: 1 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 3000, ancestry: 1 },
        AncestryInterval { start: 3000, end: 10000, ancestry: 0 },
    ];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 2, (0, 10000));
    // Pop 0: ours [0,6000), truth [3000,10000)
    //   intersection = [3000,6000) = 3000bp
    //   covered_ours = 6000, covered_truth = 7000
    //   union = 6000+7000-3000 = 10000
    //   J = 3000/10000 = 0.3
    assert!((jaccards[0] - 0.3).abs() < 1e-9);
}

#[test]
fn test_per_population_segment_jaccard_empty() {
    let ours: Vec<AncestryInterval> = vec![];
    let truth: Vec<AncestryInterval> = vec![];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 3, (0, 10000));
    assert_eq!(jaccards.len(), 3);
    for j in &jaccards {
        assert!((j - 0.0).abs() < 1e-9);
    }
}

// ============================================================================
// per_population_segment_precision_recall
// ============================================================================

#[test]
fn test_per_population_precision_recall_perfect() {
    let ours = vec![
        AncestryInterval { start: 0, end: 5000, ancestry: 0 },
        AncestryInterval { start: 5000, end: 10000, ancestry: 1 },
    ];
    let truth = ours.clone();
    let results = per_population_segment_precision_recall(&ours, &truth, 2, (0, 10000));
    for (p, r, f1) in &results {
        assert!((*p - 1.0).abs() < 1e-9);
        assert!((*r - 1.0).abs() < 1e-9);
        assert!((*f1 - 1.0).abs() < 1e-9);
    }
}

#[test]
fn test_per_population_precision_recall_all_wrong() {
    let ours = vec![AncestryInterval { start: 0, end: 10000, ancestry: 0 }];
    let truth = vec![AncestryInterval { start: 0, end: 10000, ancestry: 1 }];
    let results = per_population_segment_precision_recall(&ours, &truth, 2, (0, 10000));
    // Pop 0: precision=0 (no truth), recall=0 (no truth)
    // Pop 1: precision=0 (no ours), recall=0 (no ours)
    for (p, r, _f1) in &results {
        assert!((*p - 0.0).abs() < 1e-9);
        assert!((*r - 0.0).abs() < 1e-9);
    }
}

#[test]
fn test_per_population_precision_recall_empty() {
    let results = per_population_segment_precision_recall(&[], &[], 3, (0, 10000));
    assert_eq!(results.len(), 3);
    for (p, r, f1) in &results {
        assert_eq!(*p, 0.0);
        assert_eq!(*r, 0.0);
        assert_eq!(*f1, 0.0);
    }
}

// ============================================================================
// CrossValidationResult formatting
// ============================================================================

/// Helper to build a CrossValidationResult with manually-specified metrics.
fn make_cv_result(
    accuracy_per_pop: HashMap<String, f64>,
    overall_accuracy: f64,
    n_windows_per_pop: HashMap<String, usize>,
    confusion: HashMap<(String, String), usize>,
    precision_per_pop: HashMap<String, f64>,
    recall_per_pop: HashMap<String, f64>,
    f1_per_pop: HashMap<String, f64>,
) -> CrossValidationResult {
    CrossValidationResult {
        accuracy_per_pop,
        overall_accuracy,
        n_windows_per_pop,
        confusion,
        precision_per_pop,
        recall_per_pop,
        f1_per_pop,
    }
}

#[test]
fn test_cross_validation_result_confusion_matrix_tsv_format() {
    let mut confusion = HashMap::new();
    confusion.insert(("A".to_string(), "A".to_string()), 10);
    confusion.insert(("A".to_string(), "B".to_string()), 3);
    confusion.insert(("B".to_string(), "A".to_string()), 2);
    confusion.insert(("B".to_string(), "B".to_string()), 8);

    // Manually compute metrics: A precision=10/(10+2)=0.833, recall=10/(10+3)=0.769
    //                            B precision=8/(8+3)=0.727, recall=8/(8+2)=0.8
    let result = make_cv_result(
        [("A".to_string(), 0.769), ("B".to_string(), 0.8)].into(),
        0.783,
        [("A".to_string(), 13), ("B".to_string(), 10)].into(),
        confusion,
        [("A".to_string(), 0.833), ("B".to_string(), 0.727)].into(),
        [("A".to_string(), 0.769), ("B".to_string(), 0.8)].into(),
        [("A".to_string(), 0.8), ("B".to_string(), 0.762)].into(),
    );

    let tsv = result.confusion_matrix_tsv();
    assert!(tsv.starts_with("true_pop\tpred_pop\tcount\n"));
    assert!(tsv.contains("A\tA\t10"));
    assert!(tsv.contains("A\tB\t3"));
    assert!(tsv.contains("B\tA\t2"));
    assert!(tsv.contains("B\tB\t8"));

    let lines: Vec<&str> = tsv.trim().lines().collect();
    // header + 4 data lines (2 pops × 2 pops)
    assert_eq!(lines.len(), 5);
}

#[test]
fn test_cross_validation_result_has_bias_boundary() {
    // Exactly 50% — not considered bias
    let result = make_cv_result(
        [("A".to_string(), 0.5)].into(),
        0.5,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );
    assert!(!result.has_bias());
}

#[test]
fn test_cross_validation_result_has_bias_below_boundary() {
    // 49.9% — considered bias
    let result = make_cv_result(
        [("A".to_string(), 0.499)].into(),
        0.499,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );
    assert!(result.has_bias());
}

#[test]
fn test_cross_validation_result_confusion_matrix_tsv_three_pops() {
    let mut confusion = HashMap::new();
    confusion.insert(("A".to_string(), "A".to_string()), 10);
    confusion.insert(("B".to_string(), "B".to_string()), 10);
    confusion.insert(("C".to_string(), "C".to_string()), 10);

    let result = make_cv_result(
        [("A".to_string(), 1.0), ("B".to_string(), 1.0), ("C".to_string(), 1.0)].into(),
        1.0,
        HashMap::new(),
        confusion,
        [("A".to_string(), 1.0), ("B".to_string(), 1.0), ("C".to_string(), 1.0)].into(),
        [("A".to_string(), 1.0), ("B".to_string(), 1.0), ("C".to_string(), 1.0)].into(),
        [("A".to_string(), 1.0), ("B".to_string(), 1.0), ("C".to_string(), 1.0)].into(),
    );

    let tsv = result.confusion_matrix_tsv();
    assert!(tsv.contains("A\tA\t10"));
    assert!(tsv.contains("B\tB\t10"));
    assert!(tsv.contains("C\tC\t10"));

    let lines: Vec<&str> = tsv.trim().lines().collect();
    // header + 9 data lines (3 pops × 3 pops)
    assert_eq!(lines.len(), 10);
}

#[test]
fn test_cross_validation_result_confusion_matrix_tsv_empty() {
    let result = make_cv_result(
        HashMap::new(),
        0.0,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );
    let tsv = result.confusion_matrix_tsv();
    assert!(tsv.starts_with("true_pop\tpred_pop\tcount\n"));
    // No pops → only header
    let lines: Vec<&str> = tsv.trim().lines().collect();
    assert_eq!(lines.len(), 1);
}

// ============================================================================
// cross_validate: edge cases
// ============================================================================

#[test]
fn test_cross_validate_empty_haplotype_observations() {
    // Observations exist but are empty vectors
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string(), "pop_a#2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let mut observations = HashMap::new();
    observations.insert("pop_a#1".to_string(), vec![]); // Empty!
    observations.insert("pop_a#2".to_string(), vec![]); // Empty!

    let result = cross_validate(&observations, &pops, &params);
    assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
}

#[test]
fn test_cross_validate_three_pops_one_single_haplotype() {
    // Pop C has only 1 haplotype → should be skipped in LOO
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string(), "pop_a#2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["pop_b#1".to_string(), "pop_b#2".to_string()],
        },
        AncestralPopulation {
            name: "pop_c".to_string(),
            haplotypes: vec!["pop_c#1".to_string()], // Only 1 haplotype
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let mut observations = HashMap::new();
    for pop in &pops {
        for hap in &pop.haplotypes {
            let mut sims = HashMap::new();
            for p in &pops {
                for h in &p.haplotypes {
                    let sim = if p.name == pop.name { 0.95 } else { 0.80 };
                    sims.insert(h.clone(), sim);
                }
            }
            observations.insert(hap.clone(), vec![
                AncestryObservation {
                    chrom: "chr1".to_string(), start: 0, end: 5000,
                    sample: hap.clone(),
                    similarities: sims,
                    coverage_ratios: None,
            haplotype_consistency_bonus: None,
                },
            ]);
        }
    }

    let result = cross_validate(&observations, &pops, &params);
    // pop_c should have 0 windows tested (only 1 haplotype, can't do LOO)
    assert_eq!(*result.n_windows_per_pop.get("pop_c").unwrap_or(&0), 0);
    // pop_a and pop_b should have windows tested
    assert!(*result.n_windows_per_pop.get("pop_a").unwrap_or(&0) > 0);
    assert!(*result.n_windows_per_pop.get("pop_b").unwrap_or(&0) > 0);
}

// ============================================================================
// cross_validate_kfold: more scenarios
// ============================================================================

#[test]
fn test_kfold_k_larger_than_haplotypes() {
    // k=10 but only 2 haplotypes per pop → each fold gets at most 1 test hap
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string(), "pop_a#2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["pop_b#1".to_string(), "pop_b#2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);

    let mut observations = HashMap::new();
    for pop in &pops {
        for hap in &pop.haplotypes {
            let mut sims = HashMap::new();
            for p in &pops {
                for h in &p.haplotypes {
                    let sim = if p.name == pop.name { 0.95 } else { 0.80 };
                    sims.insert(h.clone(), sim);
                }
            }
            observations.insert(hap.clone(), vec![
                AncestryObservation {
                    chrom: "chr1".to_string(), start: 0, end: 5000,
                    sample: hap.clone(),
                    similarities: sims,
                    coverage_ratios: None,
            haplotype_consistency_bonus: None,
                },
            ]);
        }
    }

    // Should not panic with k=10 and only 2 haplotypes
    let result = cross_validate_kfold(&observations, &pops, &params, 10);
    // Should still produce some results
    assert!(result.overall_accuracy >= 0.0);
}

#[test]
fn test_kfold_k_zero_clamped_to_two() {
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string(), "pop_a#2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["pop_b#1".to_string(), "pop_b#2".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    // k=0 should be clamped to k=2 (minimum)
    let result = cross_validate_kfold(&observations, &pops, &params, 0);
    assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
}

#[test]
fn test_kfold_single_haplotype_pops_all_in_reference() {
    // All populations have only 1 haplotype → no test haplotypes
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["pop_a#1".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["pop_b#1".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops.clone(), 0.001);
    let observations: HashMap<String, Vec<AncestryObservation>> = HashMap::new();

    let result = cross_validate_kfold(&observations, &pops, &params, 3);
    // No test haplotypes possible → 0 accuracy
    assert!((result.overall_accuracy - 0.0).abs() < 1e-10);
}

// ============================================================================
// print_summary (just verify it doesn't panic)
// ============================================================================

#[test]
fn test_cross_validation_result_print_summary_no_panic() {
    let mut confusion = HashMap::new();
    confusion.insert(("A".to_string(), "A".to_string()), 5);
    confusion.insert(("A".to_string(), "B".to_string()), 1);
    confusion.insert(("B".to_string(), "A".to_string()), 2);
    confusion.insert(("B".to_string(), "B".to_string()), 4);

    let result = make_cv_result(
        [("A".to_string(), 0.833), ("B".to_string(), 0.667)].into(),
        0.75,
        [("A".to_string(), 6), ("B".to_string(), 6)].into(),
        confusion,
        [("A".to_string(), 0.714), ("B".to_string(), 0.8)].into(),
        [("A".to_string(), 0.833), ("B".to_string(), 0.667)].into(),
        [("A".to_string(), 0.769), ("B".to_string(), 0.727)].into(),
    );

    // Should not panic
    result.print_summary();
}

#[test]
fn test_cross_validation_result_print_summary_empty() {
    let result = make_cv_result(
        HashMap::new(),
        0.0,
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
    );

    // Should not panic even with empty data
    result.print_summary();
}
