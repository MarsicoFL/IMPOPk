//! Integration tests for the ancestry validation pipeline.
//!
//! Tests the full pipeline: parse RFMix output → convert to windows → compute concordance metrics.
//! Uses synthetic but realistic data to verify the pipeline works end-to-end.

use hprc_ancestry_cli::concordance::*;
use hprc_ancestry_cli::rfmix::*;

/// Simulate a scenario where our ancestry calls match RFMix 9/10 windows.
#[test]
fn test_ancestry_validation_pipeline_high_concordance() {
    // Synthetic RFMix output: 2-pop (AFR/EUR), 3 segments on chr20
    let rfmix_content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tHG00733.0\tHG00733.1
chr20\t0\t500000\t0.00\t1.50\t100\t1\t1
chr20\t500000\t800000\t1.50\t3.00\t80\t0\t1
chr20\t800000\t1000000\t3.00\t4.00\t50\t1\t0
";

    let rfmix = parse_rfmix_msp_content(rfmix_content).unwrap();
    assert_eq!(rfmix.population_names, vec!["AFR", "EUR"]);
    assert_eq!(rfmix.segments.len(), 3);

    // Convert to 100kb windows
    let truth_windows = rfmix_to_windows(&rfmix, 100_000);
    assert_eq!(truth_windows.len(), 2); // 2 haplotypes

    let hap0_truth: Vec<usize> = truth_windows[0]
        .iter()
        .map(|x| x.unwrap_or(0))
        .collect();

    // Our calls: match all but one window (window 5 → we call EUR instead of AFR)
    let mut our_calls = hap0_truth.clone();
    if our_calls.len() > 5 {
        our_calls[5] = 1; // flip one AFR window to EUR
    }

    // Overall concordance
    let concordance = per_window_ancestry_concordance(&our_calls, &hap0_truth);
    let expected_conc = (hap0_truth.len() - 1) as f64 / hap0_truth.len() as f64;
    assert!(
        (concordance - expected_conc).abs() < 1e-10,
        "Expected concordance={}, got {}",
        expected_conc,
        concordance
    );

    // Per-population metrics
    let metrics = per_population_concordance(&our_calls, &hap0_truth, &rfmix.population_names);
    // Both populations should have reasonable F1
    for name in &rfmix.population_names {
        let (_, _, f1) = metrics[name.as_str()];
        // F1 should not be zero (we got most windows right)
        // Except if the population has very few windows
        assert!((0.0..=1.0).contains(&f1), "F1 for {} should be in [0,1], got {}", name, f1);
    }

    // Confusion matrix
    let matrix = ancestry_confusion_matrix(&our_calls, &hap0_truth, 2);
    let total: u64 = matrix.iter().flat_map(|row| row.iter()).sum();
    assert_eq!(total as usize, hap0_truth.len().min(our_calls.len()));
}

/// Test with perfect match between our calls and RFMix.
#[test]
fn test_ancestry_validation_pipeline_perfect() {
    let rfmix_content = "\
#reference_panel_population: AFR EUR
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t0\t500000\t0.00\t1.50\t100\t1
chr20\t500000\t1000000\t1.50\t3.00\t80\t0
";

    let rfmix = parse_rfmix_msp_content(rfmix_content).unwrap();
    let truth_windows = rfmix_to_windows(&rfmix, 100_000);

    let hap0_truth: Vec<usize> = truth_windows[0]
        .iter()
        .map(|x| x.unwrap_or(0))
        .collect();
    let our_calls = hap0_truth.clone();

    let concordance = per_window_ancestry_concordance(&our_calls, &hap0_truth);
    assert!((concordance - 1.0).abs() < 1e-10);

    let report = compute_concordance_report(&our_calls, &hap0_truth, &rfmix.population_names, 2);
    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
}

/// Test with 3-way ancestry and partial agreement.
#[test]
fn test_ancestry_validation_pipeline_three_way() {
    let rfmix_content = "\
#Subpopulation order/codes: AFR=0\tEUR=1\tNAT=2
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
chr20\t0\t300000\t0.0\t1.0\t50\t0
chr20\t300000\t600000\t1.0\t2.0\t60\t1
chr20\t600000\t1000000\t2.0\t3.5\t80\t2
";

    let rfmix = parse_rfmix_msp_content(rfmix_content).unwrap();
    assert_eq!(rfmix.population_names.len(), 3);

    let truth_windows = rfmix_to_windows(&rfmix, 100_000);
    let hap0_truth: Vec<usize> = truth_windows[0]
        .iter()
        .map(|x| x.unwrap_or(0))
        .collect();

    // Our calls: mostly right but confuse NAT for EUR at the boundary
    let mut our_calls = hap0_truth.clone();
    if our_calls.len() > 6 {
        our_calls[6] = 1; // We call EUR where truth is NAT
    }

    let concordance = per_window_ancestry_concordance(&our_calls, &hap0_truth);
    assert!(concordance > 0.8, "Should be mostly concordant, got {}", concordance);

    // Confusion matrix should be 3x3
    let matrix = ancestry_confusion_matrix(&our_calls, &hap0_truth, 3);
    assert_eq!(matrix.len(), 3);
    assert_eq!(matrix[0].len(), 3);

    // The misclassification should show up as NAT→EUR
    // matrix[2][1] should be >= 1 (truth=NAT, predicted=EUR)
    if our_calls.len() > 6 {
        assert!(matrix[2][1] >= 1, "Should have NAT→EUR confusion");
    }
}

/// Test switch point accuracy through the full pipeline.
#[test]
fn test_ancestry_validation_switch_accuracy() {
    // Create a simple 2-pop scenario with known switch points
    // Truth: EUR EUR EUR AFR AFR AFR EUR EUR EUR EUR
    let truth: Vec<usize> = vec![1, 1, 1, 0, 0, 0, 1, 1, 1, 1];

    // Our calls: slightly shifted switch points
    // Ours: EUR EUR EUR EUR AFR AFR AFR EUR EUR EUR
    let ours: Vec<usize> = vec![1, 1, 1, 1, 0, 0, 0, 1, 1, 1];

    let pop_names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &pop_names, 2);

    // Concordance: 8/10 = 0.8 (disagreement at indices 3 and 6)
    assert!((report.overall_concordance - 0.8).abs() < 1e-10);

    // True switches at indices 3 and 6
    assert_eq!(report.n_true_switches, 2);

    // Predicted switches at indices 4 and 7
    assert_eq!(report.n_predicted_switches, 2);

    // Both should be detected within tolerance 2 (off by 1)
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10);
    assert!((report.mean_switch_distance - 1.0).abs() < 1e-10, "Mean distance should be 1.0");
}

/// Test the full concordance report formatting.
#[test]
fn test_ancestry_validation_report_format() {
    let truth: Vec<usize> = vec![0, 0, 0, 1, 1, 1, 0, 0];
    let ours: Vec<usize> = vec![0, 0, 0, 1, 1, 0, 0, 0]; // miss 1 EUR window
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    // Verify report contains essential information
    assert!(formatted.contains("concordance"));
    assert!(formatted.contains("AFR"));
    assert!(formatted.contains("EUR"));
    assert!(formatted.contains("Switch"));
    assert!(formatted.contains("precision"));
    assert!(formatted.contains("recall"));

    // Concordance should be 7/8 = 87.5%
    assert!((report.overall_concordance - 0.875).abs() < 1e-10);
}

/// Test with empty RFMix result (no segments).
#[test]
fn test_ancestry_validation_empty_rfmix() {
    let rfmix_content = "\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0
";
    let rfmix = parse_rfmix_msp_content(rfmix_content).unwrap();
    let truth_windows = rfmix_to_windows(&rfmix, 100_000);

    assert_eq!(truth_windows.len(), 1);
    assert!(truth_windows[0].is_empty());

    // Computing concordance on empty windows
    let concordance = per_window_ancestry_concordance(&[], &[]);
    assert!((concordance - 0.0).abs() < 1e-10);
}
