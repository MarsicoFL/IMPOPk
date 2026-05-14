//! Tests for compute_concordance_report and Population/non_ibd_emission in
//! ancestry-cli context. Also tests for parse_similarity_data edge cases.

use impopk_ancestry_cli::concordance::{
    compute_concordance_report, format_concordance_report,
};

// ===========================================================================
// compute_concordance_report tests
// ===========================================================================

#[test]
fn test_concordance_report_perfect_agreement() {
    let ours = vec![0, 0, 1, 1, 0, 0, 2, 2, 0, 0];
    let truth = vec![0, 0, 1, 1, 0, 0, 2, 2, 0, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
    assert_eq!(report.n_windows, 10);
    // With perfect agreement, all per-population metrics should have F1 = 1.0
    for name in &pop_names {
        if let Some(&(prec, rec, f1)) = report.per_population.get(name) {
            assert!((prec - 1.0).abs() < 1e-10, "{} precision not 1.0", name);
            assert!((rec - 1.0).abs() < 1e-10, "{} recall not 1.0", name);
            assert!((f1 - 1.0).abs() < 1e-10, "{} F1 not 1.0", name);
        }
    }
    // Switches should be identical
    assert_eq!(report.n_true_switches, report.n_predicted_switches);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
}

#[test]
fn test_concordance_report_complete_disagreement() {
    let ours = vec![0, 0, 0, 0, 0];
    let truth = vec![1, 1, 1, 1, 1];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert!((report.overall_concordance - 0.0).abs() < 1e-10);
    assert_eq!(report.n_windows, 5);
    // Confusion matrix: all truth=1 predicted as 0
    assert_eq!(report.confusion_matrix[1][0], 5);
    assert_eq!(report.confusion_matrix[1][1], 0);
    assert_eq!(report.confusion_matrix[0][0], 0);
}

#[test]
fn test_concordance_report_empty_inputs() {
    let ours: Vec<usize> = vec![];
    let truth: Vec<usize> = vec![];
    let pop_names = vec!["EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert_eq!(report.n_windows, 0);
    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 0);
}

#[test]
fn test_concordance_report_mismatched_lengths() {
    // Report should truncate to the shorter length
    let ours = vec![0, 1, 0, 1, 0, 1, 0];
    let truth = vec![0, 1, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert_eq!(report.n_windows, 3);
    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
}

#[test]
fn test_concordance_report_switch_points() {
    // Truth: [0,0,1,1,0,0] — switches at index 2 and 4
    // Ours:  [0,0,0,1,1,0] — switches at index 3 and 5
    let ours = vec![0, 0, 0, 1, 1, 0];
    let truth = vec![0, 0, 1, 1, 0, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert_eq!(report.n_true_switches, 2);
    assert_eq!(report.n_predicted_switches, 2);
    // With tolerance=1, switch at index 3 detects truth switch at index 2 (dist=1)
    // and switch at index 5 detects truth switch at index 4 (dist=1)
    assert_eq!(report.n_switches_detected, 2);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
}

#[test]
fn test_concordance_report_no_switches() {
    // Homogeneous arrays — no switch points
    let ours = vec![0, 0, 0, 0, 0];
    let truth = vec![0, 0, 0, 0, 0];
    let pop_names = vec!["EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 0);
    assert_eq!(report.n_switches_detected, 0);
    assert_eq!(report.n_spurious_switches, 0);
}

#[test]
fn test_concordance_report_spurious_switches() {
    // Truth has no switches, but ours alternates
    let ours = vec![0, 1, 0, 1, 0];
    let truth = vec![0, 0, 0, 0, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 4);
    assert_eq!(report.n_spurious_switches, 4);
}

#[test]
fn test_concordance_report_three_populations() {
    // 3-way ancestry: EUR(0), AFR(1), AMR(2)
    let ours = vec![0, 0, 1, 1, 2, 2, 0, 0];
    let truth = vec![0, 0, 1, 1, 2, 2, 0, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
    // 3x3 confusion matrix
    assert_eq!(report.confusion_matrix.len(), 3);
    for row in &report.confusion_matrix {
        assert_eq!(row.len(), 3);
    }
    assert_eq!(report.confusion_matrix[0][0], 4); // EUR correct
    assert_eq!(report.confusion_matrix[1][1], 2); // AFR correct
    assert_eq!(report.confusion_matrix[2][2], 2); // AMR correct
}

// ===========================================================================
// format_concordance_report tests
// ===========================================================================

#[test]
fn test_format_concordance_report_contains_key_values() {
    let ours = vec![0, 0, 1, 1, 0];
    let truth = vec![0, 0, 1, 1, 0];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);
    let formatted = format_concordance_report(&report, &pop_names);

    assert!(formatted.contains("100.00%"), "Should show 100% concordance");
    assert!(formatted.contains("5 windows"), "Should show window count");
    assert!(formatted.contains("EUR"), "Should mention EUR");
    assert!(formatted.contains("AFR"), "Should mention AFR");
}

#[test]
fn test_format_concordance_report_zero_concordance() {
    let ours = vec![0, 0, 0];
    let truth = vec![1, 1, 1];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);
    let formatted = format_concordance_report(&report, &pop_names);

    assert!(formatted.contains("0.00%"), "Should show 0% concordance");
}

// ===========================================================================
// Confusion matrix consistency checks
// ===========================================================================

#[test]
fn test_concordance_report_confusion_matrix_sums_to_n_windows() {
    let ours = vec![0, 1, 2, 0, 1, 2, 0, 1];
    let truth = vec![0, 0, 1, 1, 2, 2, 0, 1];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string(), "AMR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    let total: u64 = report.confusion_matrix.iter()
        .flat_map(|row| row.iter())
        .sum();
    assert_eq!(total, report.n_windows as u64);
}

#[test]
fn test_concordance_report_switch_precision_and_recall_bounded() {
    let ours = vec![0, 1, 0, 1, 0, 1, 0];
    let truth = vec![0, 0, 1, 1, 0, 0, 1];
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 1);

    assert!(report.switch_detection_rate >= 0.0 && report.switch_detection_rate <= 1.0);
    assert!(report.switch_precision >= 0.0 && report.switch_precision <= 1.0);
}
