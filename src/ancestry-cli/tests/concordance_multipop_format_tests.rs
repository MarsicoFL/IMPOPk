//! Tests for ancestry concordance with multi-population (5+) scenarios,
//! format validation for report outputs, segment conversion edge cases,
//! and comprehensive integration tests.

use hprc_ancestry_cli::concordance::{
    ancestries_to_segments, ancestry_confusion_matrix, compute_concordance_report,
    compute_segment_concordance, extract_switch_points, format_concordance_report,
    format_segment_concordance, per_population_concordance,
    per_population_segment_jaccard, per_population_segment_precision_recall,
    per_window_ancestry_concordance, switch_point_accuracy, switch_point_accuracy_bp,
    switch_point_accuracy_detailed, AncestryInterval,
};

// ============================================================================
// 5-population scenarios
// ============================================================================

fn five_pop_names() -> Vec<String> {
    vec![
        "AFR".to_string(),
        "EUR".to_string(),
        "EAS".to_string(),
        "CSA".to_string(),
        "AMR".to_string(),
    ]
}

#[test]
fn five_pop_perfect_concordance() {
    let calls = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let truth = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let names = five_pop_names();

    let conc = per_window_ancestry_concordance(&calls, &truth);
    assert!((conc - 1.0).abs() < 1e-10);

    let metrics = per_population_concordance(&calls, &truth, &names);
    for name in &names {
        let (prec, rec, f1) = metrics[name];
        assert!((prec - 1.0).abs() < 1e-10, "{} precision should be 1.0", name);
        assert!((rec - 1.0).abs() < 1e-10, "{} recall should be 1.0", name);
        assert!((f1 - 1.0).abs() < 1e-10, "{} F1 should be 1.0", name);
    }
}

#[test]
fn five_pop_confusion_matrix_dimensions() {
    let calls = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let truth = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];

    let matrix = ancestry_confusion_matrix(&calls, &truth, 5);
    assert_eq!(matrix.len(), 5);
    for row in &matrix {
        assert_eq!(row.len(), 5);
    }
    // Total counts should equal number of windows
    let total: u64 = matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 10);
}

#[test]
fn five_pop_confusion_matrix_diagonal() {
    let calls = vec![0, 1, 2, 3, 4];
    let truth = vec![0, 1, 2, 3, 4];

    let matrix = ancestry_confusion_matrix(&calls, &truth, 5);
    for i in 0..5 {
        assert_eq!(matrix[i][i], 1, "Diagonal [{i}][{i}] should be 1");
        for j in 0..5 {
            if i != j {
                assert_eq!(matrix[i][j], 0, "Off-diagonal [{i}][{j}] should be 0");
            }
        }
    }
}

#[test]
fn five_pop_mixed_concordance() {
    // Simulate realistic admixed individual with some misassignments
    let truth = vec![0, 0, 0, 1, 1, 2, 2, 2, 3, 4]; // 10 windows
    let calls = vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4]; // 7/10 correct
    let names = five_pop_names();

    let conc = per_window_ancestry_concordance(&calls, &truth);
    // Match: 0,1,3,4,5,6,8,9 = 8/10
    assert!((conc - 0.8).abs() < 1e-10, "Expected 0.8, got {}", conc);

    let metrics = per_population_concordance(&calls, &truth, &names);
    // AFR: truth=[0,1,2], predicted=[0,1] → tp=2, fp=0, fn=1
    assert!((metrics["AFR"].0 - 1.0).abs() < 1e-10); // precision=2/2=1.0
    assert!((metrics["AFR"].1 - 2.0 / 3.0).abs() < 1e-10); // recall=2/3
}

#[test]
fn five_pop_segment_concordance() {
    let ours = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let truth = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let names = five_pop_names();

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);
    assert_eq!(report.pop_names.len(), 5);
    assert_eq!(report.region, (0, 100_000));
    for i in 0..5 {
        assert!(
            (report.jaccard_per_pop[i] - 1.0).abs() < 1e-10,
            "Pop {} Jaccard should be 1.0, got {}",
            names[i],
            report.jaccard_per_pop[i]
        );
    }
}

#[test]
fn five_pop_segment_concordance_shifted() {
    // All predictions shifted by 1 window
    let ours = vec![4, 0, 0, 1, 1, 2, 2, 3, 3, 4];
    let truth = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
    let names = five_pop_names();

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 10_000);
    // No population should have perfect Jaccard except maybe by coincidence
    // Each pop has overlap less than 100%
    for i in 0..5 {
        assert!(
            report.jaccard_per_pop[i] < 1.0,
            "Pop {} should not have perfect Jaccard with shifted calls",
            names[i]
        );
    }
}

#[test]
fn five_pop_absent_population() {
    // Population 4 (AMR) never appears in data
    let ours = vec![0, 0, 1, 1, 2, 2, 3, 3, 0, 1];
    let truth = vec![0, 0, 1, 1, 2, 2, 3, 3, 0, 1];
    let names = five_pop_names();

    let metrics = per_population_concordance(&ours, &truth, &names);
    // AMR never appears → tp=0, fp=0, fn=0 → precision=0, recall=0
    assert!((metrics["AMR"].0 - 0.0).abs() < 1e-10);
    assert!((metrics["AMR"].1 - 0.0).abs() < 1e-10);
    assert!((metrics["AMR"].2 - 0.0).abs() < 1e-10);
}

// ============================================================================
// format_concordance_report validation
// ============================================================================

#[test]
fn format_concordance_report_contains_all_populations() {
    let ours = vec![0, 0, 1, 2, 2];
    let truth = vec![0, 0, 1, 2, 2];
    let names = vec!["AFR".to_string(), "EUR".to_string(), "EAS".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    for name in &names {
        assert!(formatted.contains(name), "Format should contain {}", name);
    }
    assert!(formatted.contains("Overall concordance"));
    assert!(formatted.contains("Switch points"));
    assert!(formatted.contains("Switch detection rate"));
    assert!(formatted.contains("Confusion matrix"));
}

#[test]
fn format_concordance_report_percentage_format() {
    let ours = vec![0, 0, 0, 0, 0];
    let truth = vec![0, 0, 0, 0, 0];
    let names = vec!["AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    // Should contain "100.00%" for perfect concordance
    assert!(
        formatted.contains("100.00%"),
        "Perfect concordance should show 100.00%, got:\n{}",
        formatted
    );
}

#[test]
fn format_concordance_report_switch_info() {
    let ours = vec![0, 0, 1, 1, 0, 0]; // 2 switches at 2, 4
    let truth = vec![0, 0, 0, 1, 1, 0]; // 2 switches at 3, 5
    let names = vec!["A".to_string(), "B".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    assert!(formatted.contains("2 true"));
    assert!(formatted.contains("2 predicted"));
}

#[test]
fn format_concordance_report_per_pop_metrics() {
    let ours = vec![0, 0, 1, 1, 0];
    let truth = vec![0, 1, 1, 1, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    let formatted = format_concordance_report(&report, &names);

    // Should have precision, recall, F1 for each population
    assert!(formatted.contains("precision="));
    assert!(formatted.contains("recall="));
    assert!(formatted.contains("F1="));
}

// ============================================================================
// format_segment_concordance validation
// ============================================================================

#[test]
fn format_segment_concordance_region_in_mb() {
    let ours = vec![0, 0, 1, 1, 0];
    let truth = vec![0, 0, 1, 1, 0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 1_000_000, 10_000);
    let formatted = format_segment_concordance(&report);

    // Region: 1_000_000 to 1_050_000 = 0.05 Mb
    assert!(formatted.contains("Mb"), "Should mention Mb");
    assert!(formatted.contains("1000000"), "Should contain region start");
    assert!(formatted.contains("1050000"), "Should contain region end");
}

#[test]
fn format_segment_concordance_all_pops_listed() {
    let ours = vec![0, 1, 2, 0, 1];
    let truth = vec![0, 1, 2, 0, 1];
    let names = vec!["AFR".to_string(), "EUR".to_string(), "NAT".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 1000);
    let formatted = format_segment_concordance(&report);

    for name in &names {
        assert!(formatted.contains(name), "Should list population {}", name);
    }
    assert!(formatted.contains("Jaccard="));
    assert!(formatted.contains("precision="));
    assert!(formatted.contains("recall="));
    assert!(formatted.contains("F1="));
}

#[test]
fn format_segment_concordance_four_decimal_places() {
    let ours = vec![0, 0, 1, 1];
    let truth = vec![0, 1, 1, 1];
    let names = vec!["A".to_string(), "B".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 100);
    let formatted = format_segment_concordance(&report);

    // Should have 4 decimal places in Jaccard values (format uses {:.4})
    // A: Jaccard=0.5000
    assert!(
        formatted.contains("0.5000"),
        "Should have 4 decimal places, got:\n{}",
        formatted
    );
}

// ============================================================================
// ancestries_to_segments: edge cases
// ============================================================================

#[test]
fn ancestries_to_segments_alternating_every_window() {
    // Every window changes — maximum fragmentation
    let ancestries = vec![0, 1, 0, 1, 0];
    let segs = ancestries_to_segments(&ancestries, 0, 100);
    assert_eq!(segs.len(), 5, "Alternating should create 5 segments");
    for (i, seg) in segs.iter().enumerate() {
        assert_eq!(seg.start, i as u64 * 100);
        assert_eq!(seg.end, (i + 1) as u64 * 100);
        assert_eq!(seg.ancestry, i % 2);
    }
}

#[test]
fn ancestries_to_segments_large_window_size() {
    // Realistic window size (10 kb)
    let ancestries = vec![0, 0, 1, 1, 1, 2, 2, 0];
    let segs = ancestries_to_segments(&ancestries, 50_000_000, 10_000);
    assert_eq!(segs.len(), 4);
    assert_eq!(segs[0].start, 50_000_000);
    assert_eq!(segs[0].end, 50_020_000); // 2 windows
    assert_eq!(segs[0].ancestry, 0);
    assert_eq!(segs[1].start, 50_020_000);
    assert_eq!(segs[1].end, 50_050_000); // 3 windows
    assert_eq!(segs[1].ancestry, 1);
    assert_eq!(segs[3].end, 50_080_000); // final end
}

#[test]
fn ancestries_to_segments_many_populations() {
    // Rapid switching between 5 populations
    let ancestries = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let segs = ancestries_to_segments(&ancestries, 0, 10_000);
    assert_eq!(segs.len(), 10); // each window is its own segment
    for (i, seg) in segs.iter().enumerate() {
        assert_eq!(seg.ancestry, i % 5);
    }
}

#[test]
fn ancestries_to_segments_contiguous_coverage() {
    // Verify no gaps between segments
    let ancestries = vec![0, 0, 1, 2, 2, 0, 1];
    let segs = ancestries_to_segments(&ancestries, 1000, 500);
    // Check each segment end == next segment start
    for i in 0..segs.len() - 1 {
        assert_eq!(
            segs[i].end, segs[i + 1].start,
            "Gap between segment {} and {}",
            i,
            i + 1
        );
    }
    // First starts at window_start, last ends at window_start + len*window_size
    assert_eq!(segs[0].start, 1000);
    assert_eq!(segs.last().unwrap().end, 1000 + 7 * 500);
}

// ============================================================================
// per_population_segment_jaccard: edge cases
// ============================================================================

#[test]
fn segment_jaccard_population_not_in_data() {
    // 3 populations declared, only 2 appear in data
    let ours = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 1 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 1 },
    ];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 3, (0, 1000));
    assert!((jaccards[0] - 1.0).abs() < 1e-10);
    assert!((jaccards[1] - 1.0).abs() < 1e-10);
    assert!((jaccards[2] - 0.0).abs() < 1e-10, "Absent pop should have J=0");
}

#[test]
fn segment_jaccard_one_side_empty_per_pop() {
    // We call everything as pop0, truth has pop0 and pop1
    let ours = vec![AncestryInterval { start: 0, end: 1000, ancestry: 0 }];
    let truth = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 1 },
    ];
    let jaccards = per_population_segment_jaccard(&ours, &truth, 2, (0, 1000));
    // Pop 0: ours=1000bp, truth=500bp, intersection=500bp, union=1000bp → J=0.5
    assert!((jaccards[0] - 0.5).abs() < 1e-10);
    // Pop 1: ours=0bp, truth=500bp, intersection=0bp, union=500bp → J=0
    assert!((jaccards[1] - 0.0).abs() < 1e-10);
}

// ============================================================================
// per_population_segment_precision_recall: edge cases
// ============================================================================

#[test]
fn segment_precision_recall_population_not_in_ours() {
    // We never call pop 2, truth has some pop 2
    let ours = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 1 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 300, ancestry: 0 },
        AncestryInterval { start: 300, end: 700, ancestry: 2 },
        AncestryInterval { start: 700, end: 1000, ancestry: 1 },
    ];
    let pr = per_population_segment_precision_recall(&ours, &truth, 3, (0, 1000));
    // Pop 2: ours_covered=0, truth_covered=400. precision=0, recall=0
    assert!((pr[2].0 - 0.0).abs() < 1e-10, "Pop 2 precision should be 0");
    assert!((pr[2].1 - 0.0).abs() < 1e-10, "Pop 2 recall should be 0");
    assert!((pr[2].2 - 0.0).abs() < 1e-10, "Pop 2 F1 should be 0");
}

#[test]
fn segment_precision_recall_population_not_in_truth() {
    // We call some pop 2, but truth never has pop 2
    let ours = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 2 },
    ];
    let truth = vec![
        AncestryInterval { start: 0, end: 500, ancestry: 0 },
        AncestryInterval { start: 500, end: 1000, ancestry: 1 },
    ];
    let pr = per_population_segment_precision_recall(&ours, &truth, 3, (0, 1000));
    // Pop 2: ours_covered=500, truth_covered=0. intersection=0
    // precision=0/500=0 (no intersection), recall=0 (no truth)
    assert!((pr[2].0 - 0.0).abs() < 1e-10, "Pop 2 precision should be 0");
    assert!((pr[2].1 - 0.0).abs() < 1e-10, "Pop 2 recall should be 0");
}

// ============================================================================
// switch_point_accuracy: additional edge cases
// ============================================================================

#[test]
fn switch_point_accuracy_single_true_single_pred_far() {
    let (frac, dist) = switch_point_accuracy(&[50], &[10], 5);
    assert!((frac - 0.0).abs() < 1e-10); // too far
    assert!((dist - 40.0).abs() < 1e-10);
}

#[test]
fn switch_point_accuracy_single_true_single_pred_close() {
    let (frac, dist) = switch_point_accuracy(&[12], &[10], 5);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 2.0).abs() < 1e-10);
}

#[test]
fn switch_point_accuracy_many_switches() {
    let true_sw: Vec<usize> = (0..100).map(|i| i * 10).collect();
    let our_sw: Vec<usize> = (0..100).map(|i| i * 10 + 1).collect(); // off by 1
    let (frac, dist) = switch_point_accuracy(&our_sw, &true_sw, 2);
    assert!((frac - 1.0).abs() < 1e-10, "All within tolerance");
    assert!((dist - 1.0).abs() < 1e-10, "Mean distance should be 1");
}

// ============================================================================
// switch_point_accuracy_bp: edge cases
// ============================================================================

#[test]
fn switch_point_accuracy_bp_empty_both() {
    let (frac, dist) = switch_point_accuracy_bp(&[], &[], 1000);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn switch_point_accuracy_bp_large_positions() {
    // Chromosome-scale positions
    let our_pos = vec![50_000_000u64, 120_000_000];
    let true_pos = vec![50_010_000u64, 120_050_000];
    let (frac, dist) = switch_point_accuracy_bp(&our_pos, &true_pos, 100_000);
    assert!((frac - 1.0).abs() < 1e-10); // both within 100kb
    assert!((dist - 30_000.0).abs() < 1e-6); // (10k + 50k) / 2 = 30k
}

// ============================================================================
// switch_point_accuracy_detailed: greedy matching edge cases
// ============================================================================

#[test]
fn detailed_accuracy_greedy_non_optimal() {
    // Greedy matching may not be optimal
    // True: [5, 10], Pred: [7]
    // Greedy: 5→7 (dist=2, within tol=3), 10 unmatched
    // Optimal: 10→7 (dist=3, within tol=3), 5 unmatched
    // Both are valid, but greedy picks the first true switch
    let report = switch_point_accuracy_detailed(&[7], &[5, 10], 3);
    assert_eq!(report.n_detected, 1);
    assert_eq!(report.n_missed, 1);
    assert_eq!(report.n_spurious, 0);
}

#[test]
fn detailed_accuracy_multiple_preds_near_one_true() {
    // Many predictions clustered near one true switch
    let report = switch_point_accuracy_detailed(&[9, 10, 11, 12], &[10], 2);
    // Greedy: 10→9 (dist=1, within tol) → detected=1, spurious=3
    assert_eq!(report.n_detected, 1);
    assert_eq!(report.n_missed, 0);
    assert_eq!(report.n_spurious, 3);
    assert!((report.switch_precision - 0.25).abs() < 1e-10); // 1/4
}

// ============================================================================
// compute_concordance_report: integration with 5 populations
// ============================================================================

#[test]
fn concordance_report_five_pops_integration() {
    let ours = vec![0, 0, 1, 1, 2, 3, 3, 4, 4, 4];
    let truth = vec![0, 0, 1, 2, 2, 3, 3, 4, 4, 4];
    let names = five_pop_names();

    let report = compute_concordance_report(&ours, &truth, &names, 2);

    // 9/10 correct (only window 3 differs: ours=1, truth=2)
    assert!((report.overall_concordance - 0.9).abs() < 1e-10);
    assert_eq!(report.n_windows, 10);

    // Check confusion matrix dimensions
    assert_eq!(report.confusion_matrix.len(), 5);
    for row in &report.confusion_matrix {
        assert_eq!(row.len(), 5);
    }
    // Total entries = n_windows
    let total: u64 = report.confusion_matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 10);

    // All populations should have entries
    assert_eq!(report.per_population.len(), 5);
    for name in &names {
        assert!(report.per_population.contains_key(name));
    }
}

#[test]
fn concordance_report_no_switches() {
    // Homogeneous ancestry — no switches
    let ours = vec![0, 0, 0, 0, 0];
    let truth = vec![0, 0, 0, 0, 0];
    let names = vec!["AFR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 0);
    assert_eq!(report.n_switches_detected, 0);
    assert_eq!(report.n_spurious_switches, 0);
    assert!((report.switch_detection_rate - 1.0).abs() < 1e-10); // vacuously true
    assert!((report.switch_precision - 1.0).abs() < 1e-10);
}

#[test]
fn concordance_report_single_window() {
    let ours = vec![0];
    let truth = vec![0];
    let names = vec!["AFR".to_string(), "EUR".to_string()];

    let report = compute_concordance_report(&ours, &truth, &names, 2);
    assert_eq!(report.n_windows, 1);
    assert!((report.overall_concordance - 1.0).abs() < 1e-10);
    assert_eq!(report.n_true_switches, 0);
    assert_eq!(report.n_predicted_switches, 0);
}

// ============================================================================
// extract_switch_points: edge cases
// ============================================================================

#[test]
fn extract_switch_points_empty() {
    assert!(extract_switch_points(&[]).is_empty());
}

#[test]
fn extract_switch_points_single() {
    assert!(extract_switch_points(&[0]).is_empty());
}

#[test]
fn extract_switch_points_no_switches() {
    assert!(extract_switch_points(&[1, 1, 1, 1, 1]).is_empty());
}

#[test]
fn extract_switch_points_every_window() {
    let switches = extract_switch_points(&[0, 1, 2, 3, 4]);
    assert_eq!(switches.len(), 4);
    for (i, sw) in switches.iter().enumerate() {
        assert_eq!(sw.window_index, i + 1);
        assert_eq!(sw.from_ancestry, i);
        assert_eq!(sw.to_ancestry, i + 1);
    }
}

#[test]
fn extract_switch_points_back_and_forth() {
    let switches = extract_switch_points(&[0, 1, 0, 1, 0]);
    assert_eq!(switches.len(), 4);
    assert_eq!(switches[0].from_ancestry, 0);
    assert_eq!(switches[0].to_ancestry, 1);
    assert_eq!(switches[1].from_ancestry, 1);
    assert_eq!(switches[1].to_ancestry, 0);
}

// ============================================================================
// compute_segment_concordance: edge cases
// ============================================================================

#[test]
fn segment_concordance_empty_input() {
    let names = vec!["A".to_string()];
    let report = compute_segment_concordance(&[], &[], &names, 0, 1000);
    assert_eq!(report.region, (0, 0));
    // With empty input, Jaccard should be 0
    assert!((report.jaccard_per_pop[0] - 0.0).abs() < 1e-10);
}

#[test]
fn segment_concordance_single_window() {
    let names = vec!["A".to_string(), "B".to_string()];
    let report = compute_segment_concordance(&[0], &[0], &names, 5000, 10_000);
    assert_eq!(report.region, (5000, 15000));
    assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
}

#[test]
fn segment_concordance_different_lengths_truncated() {
    // ours has 5 windows, truth has 3 → should use min(5, 3) = 3
    let ours = vec![0, 0, 1, 1, 1];
    let truth = vec![0, 0, 1];
    let names = vec!["A".to_string(), "B".to_string()];

    let report = compute_segment_concordance(&ours, &truth, &names, 0, 100);
    // Only 3 windows used
    assert_eq!(report.region, (0, 300));
    assert!((report.jaccard_per_pop[0] - 1.0).abs() < 1e-10);
    assert!((report.jaccard_per_pop[1] - 1.0).abs() < 1e-10);
}

// ============================================================================
// Property: Jaccard symmetry
// ============================================================================

#[test]
fn segment_jaccard_symmetry() {
    let a = vec![
        AncestryInterval { start: 0, end: 300, ancestry: 0 },
        AncestryInterval { start: 300, end: 500, ancestry: 1 },
    ];
    let b = vec![
        AncestryInterval { start: 0, end: 200, ancestry: 0 },
        AncestryInterval { start: 200, end: 500, ancestry: 1 },
    ];
    let j_ab = per_population_segment_jaccard(&a, &b, 2, (0, 500));
    let j_ba = per_population_segment_jaccard(&b, &a, 2, (0, 500));
    for i in 0..2 {
        assert!(
            (j_ab[i] - j_ba[i]).abs() < 1e-10,
            "Jaccard should be symmetric for pop {}: {} vs {}",
            i,
            j_ab[i],
            j_ba[i]
        );
    }
}

#[test]
fn segment_precision_recall_asymmetry() {
    // Precision and recall should swap when we swap ours/truth
    let a = vec![
        AncestryInterval { start: 0, end: 200, ancestry: 0 },
        AncestryInterval { start: 200, end: 500, ancestry: 1 },
    ];
    let b = vec![
        AncestryInterval { start: 0, end: 100, ancestry: 0 },
        AncestryInterval { start: 100, end: 500, ancestry: 1 },
    ];
    let pr_ab = per_population_segment_precision_recall(&a, &b, 2, (0, 500));
    let pr_ba = per_population_segment_precision_recall(&b, &a, 2, (0, 500));
    // For pop 0: a has 200bp, b has 100bp, intersection=100bp
    // pr_ab: precision=100/200=0.5, recall=100/100=1.0
    // pr_ba: precision=100/100=1.0, recall=100/200=0.5
    assert!((pr_ab[0].0 - pr_ba[0].1).abs() < 1e-10, "Precision/recall should swap");
    assert!((pr_ab[0].1 - pr_ba[0].0).abs() < 1e-10, "Recall/precision should swap");
}
