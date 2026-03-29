//! Edge case tests for the ancestry concordance metrics module.
//!
//! Tests cover:
//! - per_window_ancestry_concordance: known patterns, different lengths, all-agree/disagree
//! - switch_point_accuracy: exact match, no switches, missed all, tolerance boundary
//! - per_population_concordance: partial precision/recall, absent population
//! - confusion_matrix: 2-pop/3-pop, out-of-range indices
//! - extract_switch_points: no switches, alternating, single element
//! - compute_concordance_report: comprehensive report validation
//! - format_concordance_report: output contains expected content

use hprc_ancestry_cli::concordance::*;

// =============================================
// per_window_ancestry_concordance edge cases
// =============================================

#[test]
fn test_concordance_all_agree() {
    let calls = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let truth = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let c = per_window_ancestry_concordance(&calls, &truth);
    assert!((c - 1.0).abs() < 1e-10);
}

#[test]
fn test_concordance_all_disagree() {
    let calls = vec![0, 0, 0, 0, 0];
    let truth = vec![1, 1, 1, 1, 1];
    let c = per_window_ancestry_concordance(&calls, &truth);
    assert!((c - 0.0).abs() < 1e-10);
}

#[test]
fn test_concordance_50_percent() {
    let calls = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0];
    let truth = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // First 4 match + last 2 match = 6/10 = 0.6
    let c = per_window_ancestry_concordance(&calls, &truth);
    assert!((c - 0.6).abs() < 1e-10, "Expected 0.6, got {}", c);
}

#[test]
fn test_concordance_empty() {
    assert!((per_window_ancestry_concordance(&[], &[]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_concordance_single_element_match() {
    assert!((per_window_ancestry_concordance(&[0], &[0]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_concordance_single_element_mismatch() {
    assert!((per_window_ancestry_concordance(&[0], &[1]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_concordance_truncates_to_shorter() {
    // Only first 3 compared
    let calls = vec![0, 1, 0];
    let truth = vec![0, 1, 0, 1, 1, 1, 1];
    let c = per_window_ancestry_concordance(&calls, &truth);
    assert!((c - 1.0).abs() < 1e-10, "Should only compare first 3, all match");
}

#[test]
fn test_concordance_three_populations() {
    let calls = vec![0, 1, 2, 0, 1, 2];
    let truth = vec![0, 1, 2, 2, 1, 0]; // 4/6 match
    let c = per_window_ancestry_concordance(&calls, &truth);
    let expected = 4.0 / 6.0;
    assert!((c - expected).abs() < 1e-10, "Expected {}, got {}", expected, c);
}

// =============================================
// switch_point_accuracy edge cases
// =============================================

#[test]
fn test_switch_accuracy_exact_match() {
    let our = vec![5, 10, 15];
    let truth = vec![5, 10, 15];
    let (frac, dist) = switch_point_accuracy(&our, &truth, 2);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_no_true_switches() {
    let our = vec![5, 10];
    let truth: Vec<usize> = vec![];
    let (frac, dist) = switch_point_accuracy(&our, &truth, 2);
    assert!((frac - 1.0).abs() < 1e-10, "No true switches → trivially detected");
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_no_predicted_switches() {
    let our: Vec<usize> = vec![];
    let truth = vec![5, 10];
    let (frac, dist) = switch_point_accuracy(&our, &truth, 2);
    assert!((frac - 0.0).abs() < 1e-10, "No predictions → 0 detection");
    assert!(dist.is_infinite());
}

#[test]
fn test_switch_accuracy_both_empty() {
    let our: Vec<usize> = vec![];
    let truth: Vec<usize> = vec![];
    let (frac, dist) = switch_point_accuracy(&our, &truth, 2);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_tolerance_boundary() {
    let our = vec![5]; // off by exactly 2
    let truth = vec![3];
    // tolerance=2: |5-3|=2 ≤ 2 → detected
    let (frac1, _) = switch_point_accuracy(&our, &truth, 2);
    assert!((frac1 - 1.0).abs() < 1e-10);

    // tolerance=1: |5-3|=2 > 1 → not detected
    let (frac2, _) = switch_point_accuracy(&our, &truth, 1);
    assert!((frac2 - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_mean_distance() {
    // Two true switches: predicted off by 1 and 3
    let our = vec![6, 13];
    let truth = vec![5, 10];
    let (_, dist) = switch_point_accuracy(&our, &truth, 5);
    // |6-5|=1, |13-10|=3 → mean = 2.0
    assert!((dist - 2.0).abs() < 1e-10, "Expected mean distance=2.0, got {}", dist);
}

#[test]
fn test_switch_accuracy_zero_tolerance() {
    let our = vec![5, 10];
    let truth = vec![5, 11]; // one exact, one off by 1
    let (frac, _) = switch_point_accuracy(&our, &truth, 0);
    assert!((frac - 0.5).abs() < 1e-10, "Only exact match counts at tolerance=0");
}

// =============================================
// switch_point_accuracy_bp edge cases
// =============================================

#[test]
fn test_switch_accuracy_bp_exact() {
    let our = vec![500_000u64, 1_000_000];
    let truth = vec![500_000u64, 1_000_000];
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 0);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_bp_empty_both() {
    let (frac, dist) = switch_point_accuracy_bp(&[], &[], 100);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_bp_no_predicted() {
    let (frac, dist) = switch_point_accuracy_bp(&[], &[100_000], 100_000);
    assert!((frac - 0.0).abs() < 1e-10);
    assert!(dist.is_infinite());
}

// =============================================
// per_population_concordance edge cases
// =============================================

#[test]
fn test_per_pop_partial() {
    // 2 pops, ours misclassifies some AFR as EUR
    let calls = vec![0, 0, 1, 0, 0]; // predict: 3 AFR, 2 EUR
    let truth = vec![0, 0, 0, 0, 1]; // truth: 4 AFR, 1 EUR
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let metrics = per_population_concordance(&calls, &truth, &names);

    // AFR: predicted at {0,1,3,4}, truth at {0,1,2,3}
    // TP(predict=AFR,truth=AFR)={0,1,3}=3, FP(predict=AFR,truth=EUR)={4}=1, FN(truth=AFR,predict=EUR)={2}=1
    // precision=3/(3+1)=0.75, recall=3/(3+1)=0.75
    assert!((metrics["AFR"].0 - 0.75).abs() < 1e-10, "AFR precision: got {}", metrics["AFR"].0);
    assert!((metrics["AFR"].1 - 0.75).abs() < 1e-10, "AFR recall: got {}", metrics["AFR"].1);

    // EUR: predicted at {2}, truth at {4}
    // TP=0, FP=1 (index 2), FN=1 (index 4) → precision=0, recall=0
    assert!((metrics["EUR"].0 - 0.0).abs() < 1e-10, "EUR precision: got {}", metrics["EUR"].0);
    assert!((metrics["EUR"].1 - 0.0).abs() < 1e-10, "EUR recall: got {}", metrics["EUR"].1);
}

#[test]
fn test_per_pop_absent_population() {
    // 3 populations, but NAT never appears in either calls or truth
    let calls = vec![0, 0, 1, 1, 0];
    let truth = vec![0, 0, 1, 1, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string(), "NAT".to_string()];

    let metrics = per_population_concordance(&calls, &truth, &names);

    // NAT: TP=0, FP=0, FN=0 → precision=0, recall=0, F1=0
    assert!((metrics["NAT"].0 - 0.0).abs() < 1e-10);
    assert!((metrics["NAT"].1 - 0.0).abs() < 1e-10);
    assert!((metrics["NAT"].2 - 0.0).abs() < 1e-10);
}

#[test]
fn test_per_pop_everything_one_class() {
    // Predict all as EUR, truth is all AFR
    let calls = vec![1, 1, 1, 1];
    let truth = vec![0, 0, 0, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let metrics = per_population_concordance(&calls, &truth, &names);

    // AFR: TP=0, FP=0, FN=4 → precision=0 (never predicted), recall=0
    assert!((metrics["AFR"].0 - 0.0).abs() < 1e-10);
    assert!((metrics["AFR"].1 - 0.0).abs() < 1e-10);

    // EUR: TP=0, FP=4, FN=0 → precision=0 (all FP), recall=0 (no true EUR)
    assert!((metrics["EUR"].0 - 0.0).abs() < 1e-10);
}

#[test]
fn test_per_pop_empty() {
    let metrics = per_population_concordance(&[], &[], &["AFR".to_string()]);
    assert!((metrics["AFR"].0 - 0.0).abs() < 1e-10);
}

// =============================================
// confusion_matrix edge cases
// =============================================

#[test]
fn test_confusion_matrix_perfect_2pop() {
    let calls = vec![0, 0, 0, 1, 1];
    let truth = vec![0, 0, 0, 1, 1];
    let matrix = ancestry_confusion_matrix(&calls, &truth, 2);

    // Diagonal only
    assert_eq!(matrix[0][0], 3); // AFR→AFR
    assert_eq!(matrix[0][1], 0); // AFR→EUR
    assert_eq!(matrix[1][0], 0); // EUR→AFR
    assert_eq!(matrix[1][1], 2); // EUR→EUR
}

#[test]
fn test_confusion_matrix_3pop() {
    let calls = vec![0, 1, 2, 0, 1, 2]; // alternating
    let truth = vec![0, 0, 0, 1, 1, 1]; // first 3 AFR, last 3 EUR
    let matrix = ancestry_confusion_matrix(&calls, &truth, 3);

    // truth=0 (AFR): called 0,1,2 → matrix[0][0]=1, matrix[0][1]=1, matrix[0][2]=1
    assert_eq!(matrix[0][0], 1);
    assert_eq!(matrix[0][1], 1);
    assert_eq!(matrix[0][2], 1);

    // truth=1 (EUR): called 0,1,2 → matrix[1][0]=1, matrix[1][1]=1, matrix[1][2]=1
    assert_eq!(matrix[1][0], 1);
    assert_eq!(matrix[1][1], 1);
    assert_eq!(matrix[1][2], 1);

    // truth=2 (NAT): never appears → all zeros
    assert_eq!(matrix[2][0], 0);
    assert_eq!(matrix[2][1], 0);
    assert_eq!(matrix[2][2], 0);
}

#[test]
fn test_confusion_matrix_out_of_range() {
    // Ancestry index >= n_pops → should be skipped
    let calls = vec![0, 1, 5]; // 5 is out of range for n_pops=2
    let truth = vec![0, 1, 0];
    let matrix = ancestry_confusion_matrix(&calls, &truth, 2);

    // Only first two windows counted
    assert_eq!(matrix[0][0], 1);
    assert_eq!(matrix[1][1], 1);
    // Total in matrix should be 2 (third window skipped)
    let total: u64 = matrix.iter().flat_map(|row| row.iter()).sum();
    assert_eq!(total, 2);
}

#[test]
fn test_confusion_matrix_empty() {
    let matrix = ancestry_confusion_matrix(&[], &[], 2);
    assert_eq!(matrix, vec![vec![0, 0], vec![0, 0]]);
}

// =============================================
// extract_switch_points edge cases
// =============================================

#[test]
fn test_switch_points_no_switches() {
    let ancestries = vec![0, 0, 0, 0, 0];
    let switches = extract_switch_points(&ancestries);
    assert!(switches.is_empty());
}

#[test]
fn test_switch_points_single_element() {
    let ancestries = vec![0];
    let switches = extract_switch_points(&ancestries);
    assert!(switches.is_empty());
}

#[test]
fn test_switch_points_empty() {
    let ancestries: Vec<usize> = vec![];
    let switches = extract_switch_points(&ancestries);
    assert!(switches.is_empty());
}

#[test]
fn test_switch_points_alternating() {
    let ancestries = vec![0, 1, 0, 1, 0];
    let switches = extract_switch_points(&ancestries);
    assert_eq!(switches.len(), 4);
    // Verify all switch indices
    assert_eq!(switches[0].window_index, 1);
    assert_eq!(switches[1].window_index, 2);
    assert_eq!(switches[2].window_index, 3);
    assert_eq!(switches[3].window_index, 4);
}

#[test]
fn test_switch_points_from_to() {
    let ancestries = vec![0, 1, 2];
    let switches = extract_switch_points(&ancestries);
    assert_eq!(switches.len(), 2);
    assert_eq!(switches[0].from_ancestry, 0);
    assert_eq!(switches[0].to_ancestry, 1);
    assert_eq!(switches[1].from_ancestry, 1);
    assert_eq!(switches[1].to_ancestry, 2);
}

#[test]
fn test_switch_points_two_elements() {
    let ancestries = vec![0, 1];
    let switches = extract_switch_points(&ancestries);
    assert_eq!(switches.len(), 1);
    assert_eq!(switches[0].window_index, 1);
}

// =============================================
// compute_concordance_report edge cases
// =============================================

#[test]
fn test_report_perfect() {
    let calls = vec![0, 0, 1, 1, 0, 0];
    let truth = vec![0, 0, 1, 1, 0, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&calls, &truth, &names, 2);

    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
    assert_eq!(report.n_windows, 6);
    // Switch points at indices 2 and 4
    assert_eq!(report.n_true_switches, 2);
    assert_eq!(report.n_predicted_switches, 2);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
    assert!((report.mean_switch_distance - 0.0).abs() < 1e-10);
}

#[test]
fn test_report_empty() {
    let report = compute_concordance_report(&[], &[], &["AFR".to_string()], 2);
    assert_eq!(report.n_windows, 0);
    assert!((report.overall_concordance - 0.0).abs() < 1e-10);
}

#[test]
fn test_report_no_switches() {
    let calls = vec![0, 0, 0, 0];
    let truth = vec![0, 0, 0, 0];
    let names = vec!["AFR".to_string()];

    let report = compute_concordance_report(&calls, &truth, &names, 2);
    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 0);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
}

#[test]
fn test_report_different_lengths() {
    // Should truncate to shorter
    let calls = vec![0, 1, 0];
    let truth = vec![0, 1, 0, 1, 1, 1];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&calls, &truth, &names, 2);
    assert_eq!(report.n_windows, 3);
    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
}

// =============================================
// format_concordance_report edge cases
// =============================================

#[test]
fn test_format_contains_key_info() {
    let calls = vec![0, 0, 1, 1, 0];
    let truth = vec![0, 0, 1, 0, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&calls, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    assert!(formatted.contains("concordance"), "Should mention concordance");
    assert!(formatted.contains("AFR"), "Should mention AFR");
    assert!(formatted.contains("EUR"), "Should mention EUR");
    assert!(formatted.contains("precision"), "Should mention precision");
    assert!(formatted.contains("recall"), "Should mention recall");
    assert!(formatted.contains("F1"), "Should mention F1");
    assert!(formatted.contains("Switch"), "Should mention switch points");
}

#[test]
fn test_format_perfect_shows_100() {
    let calls = vec![0, 0, 1, 1];
    let truth = vec![0, 0, 1, 1];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&calls, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    assert!(formatted.contains("100.00%"), "Perfect concordance should show 100.00%");
}

// =============================================
// Integration: full pipeline test
// =============================================

#[test]
fn test_concordance_pipeline_known_scenario() {
    // Realistic scenario: mostly EUR with AFR segment in the middle
    // Truth: EUR, EUR, EUR, AFR, AFR, AFR, EUR, EUR, EUR, EUR
    // Ours:  EUR, EUR, EUR, AFR, AFR, EUR, EUR, EUR, EUR, EUR
    // We miss one AFR window (5th position)
    let truth = vec![1, 1, 1, 0, 0, 0, 1, 1, 1, 1];
    let calls = vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    // Concordance: 9/10 = 0.9
    let c = per_window_ancestry_concordance(&calls, &truth);
    assert!((c - 0.9).abs() < 1e-10);

    // Per-population metrics
    let metrics = per_population_concordance(&calls, &truth, &names);
    // AFR: TP=2, FP=0, FN=1 → precision=1.0, recall=2/3=0.667
    assert!((metrics["AFR"].0 - 1.0).abs() < 1e-10);
    assert!((metrics["AFR"].1 - 2.0 / 3.0).abs() < 1e-10);

    // EUR: TP=7, FP=1, FN=0 → precision=7/8=0.875, recall=1.0
    assert!((metrics["EUR"].0 - 7.0 / 8.0).abs() < 1e-10);
    assert!((metrics["EUR"].1 - 1.0).abs() < 1e-10);

    // Confusion matrix
    let matrix = ancestry_confusion_matrix(&calls, &truth, 2);
    assert_eq!(matrix[0][0], 2); // AFR→AFR (TP)
    assert_eq!(matrix[0][1], 1); // AFR→EUR (FN)
    assert_eq!(matrix[1][0], 0); // EUR→AFR (FP)
    assert_eq!(matrix[1][1], 7); // EUR→EUR (TN for AFR / TP for EUR)

    // Switch points
    let true_switches = extract_switch_points(&truth);
    let our_switches = extract_switch_points(&calls);
    assert_eq!(true_switches.len(), 2); // at index 3 and 6
    assert_eq!(our_switches.len(), 2);  // at index 3 and 5

    let true_sw_idx: Vec<usize> = true_switches.iter().map(|s| s.window_index).collect();
    let our_sw_idx: Vec<usize> = our_switches.iter().map(|s| s.window_index).collect();

    let (frac, mean_dist) = switch_point_accuracy(&our_sw_idx, &true_sw_idx, 2);
    assert!((frac - 1.0).abs() < 1e-10, "Both switches within tolerance 2");
    // Distance: |3-3|=0, |5-6|=1 → mean=0.5
    assert!((mean_dist - 0.5).abs() < 1e-10, "Expected mean distance=0.5, got {}", mean_dist);
}

// =============================================================================
// Segment-level concordance edge case tests
// =============================================================================

// --- ancestries_to_segments edge cases ---

#[test]
fn test_segments_large_window_offset() {
    // Start at a large genomic offset (e.g., chr20 midpoint)
    let ancestries = vec![0, 0, 1];
    let segs = ancestries_to_segments(&ancestries, 30_000_000, 10_000);

    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].start, 30_000_000);
    assert_eq!(segs[0].end, 30_020_000);
    assert_eq!(segs[0].ancestry, 0);
    assert_eq!(segs[1].start, 30_020_000);
    assert_eq!(segs[1].end, 30_030_000);
    assert_eq!(segs[1].ancestry, 1);
}

#[test]
fn test_segments_many_alternating_populations() {
    // Rapid alternation: 0, 1, 2, 0, 1, 2, 0, 1, 2, 0
    let ancestries = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
    let segs = ancestries_to_segments(&ancestries, 0, 5000);

    // Every window is its own segment (10 segments)
    assert_eq!(segs.len(), 10);
    for (i, seg) in segs.iter().enumerate() {
        assert_eq!(seg.start, i as u64 * 5000);
        assert_eq!(seg.end, (i as u64 + 1) * 5000);
        assert_eq!(seg.ancestry, i % 3);
    }
}

#[test]
fn test_segments_consecutive_same_merged() {
    // Long run of same ancestry → single merged segment
    let ancestries = vec![2; 100];
    let segs = ancestries_to_segments(&ancestries, 1_000_000, 10_000);

    assert_eq!(segs.len(), 1);
    assert_eq!(segs[0].start, 1_000_000);
    assert_eq!(segs[0].end, 2_000_000); // 1M + 100*10k
    assert_eq!(segs[0].ancestry, 2);
}

#[test]
fn test_segments_two_elements_different() {
    let segs = ancestries_to_segments(&[0, 1], 0, 100);
    assert_eq!(segs.len(), 2);
    assert_eq!(segs[0].end, 100);
    assert_eq!(segs[1].start, 100);
    assert_eq!(segs[1].end, 200);
}

// --- per_population_segment_jaccard edge cases ---

#[test]
fn test_jaccard_region_clips_segments() {
    // Segments extend beyond region → should be clipped
    // Our segments: pop0 covers [0, 500), pop1 covers [500, 1000)
    // Region: [200, 800)
    let ours = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let truth = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // same
    let our_segs = ancestries_to_segments(&ours, 0, 100);
    let truth_segs = ancestries_to_segments(&truth, 0, 100);
    let region = (200, 800);

    let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);
    // Perfect agreement within the clipped region
    assert!((jaccards[0] - 1.0).abs() < 1e-10, "Pop 0 should be 1.0 (clipped), got {}", jaccards[0]);
    assert!((jaccards[1] - 1.0).abs() < 1e-10, "Pop 1 should be 1.0 (clipped), got {}", jaccards[1]);
}

#[test]
fn test_jaccard_three_pop_one_absent() {
    // 3 populations, but pop 2 never appears in either set
    let ancestries = vec![0, 0, 1, 1, 0, 0];
    let our_segs = ancestries_to_segments(&ancestries, 0, 100);
    let truth_segs = ancestries_to_segments(&ancestries, 0, 100);
    let region = (0, 600);

    let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 3, region);
    assert!((jaccards[0] - 1.0).abs() < 1e-10);
    assert!((jaccards[1] - 1.0).abs() < 1e-10);
    assert!((jaccards[2] - 0.0).abs() < 1e-10, "Absent pop should have Jaccard 0");
}

#[test]
fn test_jaccard_complete_swap() {
    // Ours: all pop 0. Truth: all pop 1.
    let ours = vec![0, 0, 0, 0];
    let truth = vec![1, 1, 1, 1];
    let our_segs = ancestries_to_segments(&ours, 0, 1000);
    let truth_segs = ancestries_to_segments(&truth, 0, 1000);
    let region = (0, 4000);

    let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);
    // Pop 0: ours covers 4000bp, truth covers 0bp. intersection=0, union=4000. J=0
    assert!((jaccards[0] - 0.0).abs() < 1e-10);
    // Pop 1: ours covers 0bp, truth covers 4000bp. intersection=0, union=4000. J=0
    assert!((jaccards[1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_jaccard_single_window_segments() {
    // Each window is a different population, single-window granularity
    let ours = vec![0, 1, 0, 1];
    let truth = vec![0, 0, 1, 1];
    let our_segs = ancestries_to_segments(&ours, 0, 100);
    let truth_segs = ancestries_to_segments(&truth, 0, 100);
    let region = (0, 400);

    let jaccards = per_population_segment_jaccard(&our_segs, &truth_segs, 2, region);
    // Pop 0: ours=[0,100)+[200,300), truth=[0,200). intersection=[0,100)=100. union=100+200-100=200. J=0.5
    // But note: intersection is computed pairwise, so need to be careful
    // ours pop0: [(0,100), (200,300)] — covered=200
    // truth pop0: [(0,200)] — covered=200
    // intersection: (0,100)∩(0,200)=100, (200,300)∩(0,200)=0 → total=100
    // union = 200+200-100 = 300
    // Actually wait — truth pop0 end is at 200, our (200,300) starts at 200 so no overlap
    // intersection = 100. union = 200+200-100 = 300. J = 100/300 = 1/3
    assert!((jaccards[0] - 1.0 / 3.0).abs() < 1e-10, "Pop 0 Jaccard: expected 1/3, got {}", jaccards[0]);
    // Pop 1: ours=[(100,200),(300,400)] covered=200. truth=[(200,400)] covered=200.
    // intersection: (100,200)∩(200,400)=0, (300,400)∩(200,400)=100 → total=100
    // union = 200+200-100 = 300. J = 1/3
    assert!((jaccards[1] - 1.0 / 3.0).abs() < 1e-10, "Pop 1 Jaccard: expected 1/3, got {}", jaccards[1]);
}

// --- per_population_segment_precision_recall edge cases ---

#[test]
fn test_precision_recall_asymmetric() {
    // Ours predicts a very broad pop0 region; truth has a narrow pop0 region
    // Ours: [0, 0, 0, 0, 0] = pop0 over [0, 500)
    // Truth: [1, 1, 0, 1, 1] = pop0 over [200, 300), pop1 over [0,200)+[300,500)
    let ours = vec![0, 0, 0, 0, 0];
    let truth = vec![1, 1, 0, 1, 1];
    let our_segs = ancestries_to_segments(&ours, 0, 100);
    let truth_segs = ancestries_to_segments(&truth, 0, 100);
    let region = (0, 500);

    let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);

    // Pop 0: ours=500bp, truth=100bp. intersection=100bp.
    //   Precision = 100/500 = 0.2, Recall = 100/100 = 1.0
    assert!((pr[0].0 - 0.2).abs() < 1e-10, "Pop 0 precision: expected 0.2, got {}", pr[0].0);
    assert!((pr[0].1 - 1.0).abs() < 1e-10, "Pop 0 recall: expected 1.0, got {}", pr[0].1);
    // F1 = 2 * 0.2 * 1.0 / (0.2 + 1.0) = 0.4/1.2 = 1/3
    assert!((pr[0].2 - 1.0 / 3.0).abs() < 1e-10, "Pop 0 F1: expected 1/3, got {}", pr[0].2);

    // Pop 1: ours=0bp, truth=400bp. intersection=0.
    //   Precision = 0 (no ours bp), Recall = 0
    assert!((pr[1].0 - 0.0).abs() < 1e-10);
    assert!((pr[1].1 - 0.0).abs() < 1e-10);
}

#[test]
fn test_precision_recall_with_region_clipping() {
    // Segments span [0, 1000) but region is [300, 700)
    let ours = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // pop0=[0,500), pop1=[500,1000)
    let truth = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1]; // pop0=[0,300), pop1=[300,1000)
    let our_segs = ancestries_to_segments(&ours, 0, 100);
    let truth_segs = ancestries_to_segments(&truth, 0, 100);
    let region = (300, 700);

    let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);

    // In region [300, 700):
    // Pop 0: ours=[300,500)=200bp, truth=[0,300) clipped=nothing → truth=0bp
    //   intersection=0, precision=0/200=0, recall=0/0=0
    assert!((pr[0].0 - 0.0).abs() < 1e-10, "Pop 0 precision: expected 0.0, got {}", pr[0].0);
    assert!((pr[0].1 - 0.0).abs() < 1e-10, "Pop 0 recall: expected 0.0, got {}", pr[0].1);

    // Pop 1: ours=[500,700)=200bp, truth=[300,700)=400bp
    //   intersection=200bp, precision=200/200=1.0, recall=200/400=0.5
    assert!((pr[1].0 - 1.0).abs() < 1e-10, "Pop 1 precision: expected 1.0, got {}", pr[1].0);
    assert!((pr[1].1 - 0.5).abs() < 1e-10, "Pop 1 recall: expected 0.5, got {}", pr[1].1);
}

#[test]
fn test_precision_recall_f1_harmonic_mean() {
    // Verify F1 = 2*P*R/(P+R) for a known case
    let ours = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1]; // pop0=[0,300), pop1=[300,1000)
    let truth = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // pop0=[0,500), pop1=[500,1000)
    let our_segs = ancestries_to_segments(&ours, 0, 100);
    let truth_segs = ancestries_to_segments(&truth, 0, 100);
    let region = (0, 1000);

    let pr = per_population_segment_precision_recall(&our_segs, &truth_segs, 2, region);

    // Pop 0: ours=300bp, truth=500bp. intersection=300. P=300/300=1.0, R=300/500=0.6
    let p = pr[0].0;
    let r = pr[0].1;
    let expected_f1 = 2.0 * p * r / (p + r);
    assert!((pr[0].2 - expected_f1).abs() < 1e-10, "F1 should be harmonic mean: expected {}, got {}", expected_f1, pr[0].2);
    assert!((p - 1.0).abs() < 1e-10);
    assert!((r - 0.6).abs() < 1e-10);
}

// --- compute_segment_concordance edge cases ---

#[test]
fn test_segment_concordance_mismatched_input_lengths() {
    // Ours has 10 windows, truth has 5 → should truncate to 5
    let ours = vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
    let truth = vec![0, 0, 1, 1, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 1000);

    // Should only consider first 5 windows
    assert_eq!(report.region, (0, 5000));
    // Perfect agreement on the first 5 windows
    assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
    assert!((report.jaccard_per_pop[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_segment_concordance_large_window_size() {
    // Very large windows (1 Mb each)
    let ours = vec![0, 1, 0];
    let truth = vec![0, 0, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 10_000_000, 1_000_000);

    // Region: [10M, 13M)
    assert_eq!(report.region, (10_000_000, 13_000_000));

    // Pop 0: ours=[10M,11M)+[12M,13M)=2Mb, truth=[10M,13M)=3Mb
    //   intersection=2Mb, union=3Mb. Jaccard = 2/3
    assert!((report.jaccard_per_pop[0] - 2.0 / 3.0).abs() < 1e-10,
        "Pop 0 Jaccard: expected 2/3, got {}", report.jaccard_per_pop[0]);

    // Pop 1: ours=[11M,12M)=1Mb, truth=0bp
    //   intersection=0, union=1Mb. Jaccard = 0
    assert!((report.jaccard_per_pop[1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_segment_concordance_empty_input() {
    let names = vec!["AFR".to_string(), "EUR".to_string()];
    let report = compute_segment_concordance(&[], &[], &names, 0, 10_000);

    assert_eq!(report.region, (0, 0));
    assert!((report.jaccard_per_pop[0] - 0.0).abs() < 1e-10);
    assert!((report.jaccard_per_pop[1] - 0.0).abs() < 1e-10);
}

#[test]
fn test_segment_concordance_single_window() {
    let names = vec!["EUR".to_string()];
    let report = compute_segment_concordance(&[0], &[0], &names, 5000, 100);

    assert_eq!(report.region, (5000, 5100));
    assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
    assert!((report.precision_recall_per_pop[0].0 - 1.0).abs() < 1e-10);
    assert!((report.precision_recall_per_pop[0].1 - 1.0).abs() < 1e-10);
}

// --- format_segment_concordance edge cases ---

#[test]
fn test_format_segment_concordance_three_pop() {
    let ours = vec![0, 1, 2, 0, 1, 2];
    let truth = vec![0, 1, 2, 0, 1, 2]; // perfect
    let names = vec!["AFR".to_string(), "EUR".to_string(), "NAT".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);
    let formatted = format_segment_concordance(&report);

    assert!(formatted.contains("AFR"), "Should contain AFR");
    assert!(formatted.contains("EUR"), "Should contain EUR");
    assert!(formatted.contains("NAT"), "Should contain NAT");
    assert!(formatted.contains("Jaccard"), "Should contain Jaccard");
    assert!(formatted.contains("1.0000"), "Perfect Jaccard should show 1.0000");
    assert!(formatted.contains("Segment-level"), "Should indicate segment-level");
}

#[test]
fn test_format_segment_concordance_zero_jaccard() {
    // Complete disagreement
    let ours = vec![0, 0, 0, 0];
    let truth = vec![1, 1, 1, 1];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 100);
    let formatted = format_segment_concordance(&report);

    assert!(formatted.contains("0.0000"), "Zero Jaccard should show 0.0000");
}

#[test]
fn test_format_segment_concordance_region_display() {
    let names = vec!["EUR".to_string()];
    let report = compute_segment_concordance(&[0, 0], &[0, 0], &names, 1_000_000, 500_000);
    let formatted = format_segment_concordance(&report);

    // Region should be 1M-2M = 1.00 Mb
    assert!(formatted.contains("1000000"), "Should show region start");
    assert!(formatted.contains("2000000"), "Should show region end");
    assert!(formatted.contains("1.00 Mb"), "Should show region size in Mb");
}

// --- Segment-level integration: realistic ancestry painting scenario ---

#[test]
fn test_segment_concordance_realistic_admixed() {
    // Simulate a realistic admixed chromosome:
    // Truth: long EUR tract, short AFR tract, long EUR tract
    // EUR(0): windows 0-19, AFR(1): windows 20-24, EUR(0): windows 25-49
    let mut truth = vec![0; 50];
    for i in 20..25 {
        truth[i] = 1;
    }

    // Our prediction: correctly identifies EUR, but shifts AFR tract by 2 windows
    // EUR(0): windows 0-21, AFR(1): windows 22-26, EUR(0): windows 27-49
    let mut ours = vec![0; 50];
    for i in 22..27 {
        ours[i] = 1;
    }

    let names = vec!["EUR".to_string(), "AFR".to_string()];
    let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);

    // Region: [0, 500_000)
    assert_eq!(report.region, (0, 500_000));

    // AFR (pop 1): ours=[220k,270k), truth=[200k,250k)
    //   intersection=[220k,250k)=30kb, union=[200k,270k)=70kb. Jaccard = 3/7
    let afr_jaccard = report.jaccard_per_pop[1];
    assert!((afr_jaccard - 3.0 / 7.0).abs() < 1e-10,
        "AFR Jaccard: expected 3/7={}, got {}", 3.0 / 7.0, afr_jaccard);

    // EUR (pop 0): ours=[0,220k)+[270k,500k)=450kb, truth=[0,200k)+[250k,500k)=450kb
    //   intersection=[0,200k)+[270k,500k)=200k+230k=430kb
    //   Wait—need to compute carefully:
    //   ours pop0: [(0, 220000), (270000, 500000)]
    //   truth pop0: [(0, 200000), (250000, 500000)]
    //   intersection: (0,220k)∩(0,200k)=200k, (0,220k)∩(250k,500k)=0, (270k,500k)∩(0,200k)=0, (270k,500k)∩(250k,500k)=230k
    //   total intersection = 200k + 230k = 430kb
    //   covered_ours = 220k + 230k = 450kb
    //   covered_truth = 200k + 250k = 450kb
    //   union = 450k + 450k - 430k = 470kb
    //   Jaccard = 430/470 = 43/47
    let eur_jaccard = report.jaccard_per_pop[0];
    assert!((eur_jaccard - 430.0 / 470.0).abs() < 1e-10,
        "EUR Jaccard: expected {}, got {}", 430.0 / 470.0, eur_jaccard);

    // Verify precision/recall for AFR
    let (afr_prec, afr_rec, _) = report.precision_recall_per_pop[1];
    // AFR precision: intersection(30k) / ours(50k) = 0.6
    assert!((afr_prec - 0.6).abs() < 1e-10, "AFR precision: expected 0.6, got {}", afr_prec);
    // AFR recall: intersection(30k) / truth(50k) = 0.6
    assert!((afr_rec - 0.6).abs() < 1e-10, "AFR recall: expected 0.6, got {}", afr_rec);
}

// =============================================
// switch_point_accuracy_detailed edge cases
// =============================================

#[test]
fn test_detailed_switch_accuracy_both_empty() {
    let report = switch_point_accuracy_detailed(&[], &[], 2);
    assert_eq!(report.n_detected, 0);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 0);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert!((report.switch_precision - 1.0).abs() < 1e-10);
    assert!((report.mean_distance - 0.0).abs() < 1e-10);
}

#[test]
fn test_detailed_switch_accuracy_no_true_with_predictions() {
    // No true switches, but we made 3 spurious predictions
    let report = switch_point_accuracy_detailed(&[5, 10, 15], &[], 2);
    assert_eq!(report.n_detected, 0);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 3);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert!((report.switch_precision - 0.0).abs() < 1e-10);
}

#[test]
fn test_detailed_switch_accuracy_no_predictions_with_true() {
    // 2 true switches but no predictions
    let report = switch_point_accuracy_detailed(&[], &[5, 10], 2);
    assert_eq!(report.n_detected, 0);
    assert_eq!(report.n_missed, 2);
    assert_eq!(report.n_spurious, 0);
    assert!((report.detection_rate - 0.0).abs() < 1e-10);
    assert!((report.switch_precision - 1.0).abs() < 1e-10); // vacuously true
    assert!(report.mean_distance.is_infinite());
}

#[test]
fn test_detailed_switch_accuracy_perfect_match() {
    let report = switch_point_accuracy_detailed(&[5, 10, 15], &[5, 10, 15], 0);
    assert_eq!(report.n_detected, 3);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 0);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    assert!((report.switch_precision - 1.0).abs() < 1e-10);
    assert!((report.mean_distance - 0.0).abs() < 1e-10);
}

#[test]
fn test_detailed_switch_accuracy_within_tolerance() {
    // True switches at 5, 10. Predicted at 6, 12 (off by 1 and 2).
    let report = switch_point_accuracy_detailed(&[6, 12], &[5, 10], 2);
    assert_eq!(report.n_detected, 2);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 0);
    assert!((report.detection_rate - 1.0).abs() < 1e-10);
    // mean_distance = (|6-5| + |12-10|) / 2 = (1 + 2) / 2 = 1.5
    assert!((report.mean_distance - 1.5).abs() < 1e-10);
}

#[test]
fn test_detailed_switch_accuracy_partial_with_spurious() {
    // True at 10, 20. Predicted at 10, 50, 70.
    // Match: 10↔10, 20→nearest is 50 (dist=30 > tolerance=5) → missed.
    // Spurious: 50 and 70 (unmatched predictions)
    let report = switch_point_accuracy_detailed(&[10, 50, 70], &[10, 20], 5);
    assert_eq!(report.n_detected, 1);
    assert_eq!(report.n_missed, 1);
    assert_eq!(report.n_spurious, 2);
    assert!((report.detection_rate - 0.5).abs() < 1e-10);
    // switch_precision = 1 detected / 3 predictions = 1/3
    assert!((report.switch_precision - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_detailed_switch_accuracy_greedy_matching() {
    // Greedy matching: true at 10, 12. Predicted at 11.
    // 11 matches true=10 (dist=1), leaving true=12 unmatched (nearest unused is none).
    let report = switch_point_accuracy_detailed(&[11], &[10, 12], 2);
    assert_eq!(report.n_detected, 1); // 11 matches 10 (dist=1)
    assert_eq!(report.n_missed, 1);   // 12 unmatched (11 already used)
    assert_eq!(report.n_spurious, 0); // 11 was matched
}

// =============================================
// switch_point_accuracy_bp additional edge cases
// =============================================

#[test]
fn test_switch_accuracy_bp_tolerance_boundary() {
    // Positions: truth at 1000000, predicted at 1000500
    // Tolerance = 500 → exactly at boundary → detected
    let (frac, _) = switch_point_accuracy_bp(&[1_000_500], &[1_000_000], 500);
    assert!((frac - 1.0).abs() < 1e-10);

    // Tolerance = 499 → just outside → not detected
    let (frac2, _) = switch_point_accuracy_bp(&[1_000_500], &[1_000_000], 499);
    assert!((frac2 - 0.0).abs() < 1e-10);
}

#[test]
fn test_switch_accuracy_bp_multiple_matches() {
    // Multiple true positions, each has a nearby prediction
    let our = vec![100_000, 500_000, 900_000];
    let truth = vec![100_100, 500_200, 900_050];
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 500);
    assert!((frac - 1.0).abs() < 1e-10);
    // Mean distance: (100 + 200 + 50) / 3 ≈ 116.67
    assert!((dist - 350.0 / 3.0).abs() < 1e-6,
        "Expected mean dist={}, got {}", 350.0 / 3.0, dist);
}
