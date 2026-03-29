/// Edge case tests for algo_dev cycle 1 changes:
/// - PulseEstimate.cv field (CV = 1/√(n_k × π_m))
/// - DemographicResult.low_tract_warning field
/// - Lagrange projection replaces uniform scaling for BW constraint
///
/// These test edge cases not covered by algo_dev's inline tests.
use hprc_ancestry_cli::demography::*;

// ==================== CV field ====================

#[test]
fn cv_single_pulse_100_tracts() {
    // Single pulse with proportion=1.0, n_tracts=100
    // CV = 1/√(100 × 1.0) = 0.1
    let tracts = make_quantile_tracts(100, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert_eq!(result.n_pulses, 1);
    assert_eq!(result.pulses.len(), 1);
    let cv = result.pulses[0].cv;
    let expected = 1.0 / (100.0_f64).sqrt(); // 0.1
    assert!((cv - expected).abs() < 0.01,
        "CV should be ~{:.3}, got {:.3}", expected, cv);
}

#[test]
fn cv_decreases_with_more_tracts() {
    // More tracts should give smaller CV
    let params = DemographyParams::default();
    let tracts_50 = make_quantile_tracts(50, 1e-5, 20_000.0);
    let tracts_200 = make_quantile_tracts(200, 1e-5, 20_000.0);
    let r50 = infer_demography(&tracts_50, "POP", &params);
    let r200 = infer_demography(&tracts_200, "POP", &params);
    assert!(r50.pulses[0].cv > r200.pulses[0].cv,
        "CV with 50 tracts ({:.3}) should be > CV with 200 tracts ({:.3})",
        r50.pulses[0].cv, r200.pulses[0].cv);
}

#[test]
fn cv_finite_for_valid_input() {
    let tracts = make_quantile_tracts(30, 5e-6, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    for pulse in &result.pulses {
        assert!(pulse.cv.is_finite(), "CV should be finite, got {}", pulse.cv);
        assert!(pulse.cv > 0.0, "CV should be positive, got {}", pulse.cv);
    }
}

#[test]
fn cv_exactly_3_tracts() {
    // Minimum for inference — CV should be high but finite
    // CV = 1/√(3 × 1.0) ≈ 0.577
    let tracts = make_quantile_tracts(3, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    if result.n_pulses >= 1 {
        let cv = result.pulses[0].cv;
        assert!(cv.is_finite());
        assert!(cv > 0.3, "CV with 3 tracts should be high, got {:.3}", cv);
    }
}

#[test]
fn cv_empty_pulses_when_insufficient_data() {
    // 2 tracts = insufficient → no pulses → no CV to check
    let tracts = vec![30_000.0, 50_000.0];
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.pulses.is_empty());
}

// ==================== low_tract_warning field ====================

#[test]
fn low_tract_warning_none_for_sufficient_data() {
    // 100 tracts should not trigger warning
    let tracts = make_quantile_tracts(100, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.low_tract_warning.is_none(),
        "Should not warn with 100 tracts, got: {:?}", result.low_tract_warning);
}

#[test]
fn low_tract_warning_set_for_few_tracts() {
    // 10 tracts — should trigger warning (< 20)
    let tracts = make_quantile_tracts(10, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.low_tract_warning.is_some(),
        "Should warn with 10 tracts");
    let warning = result.low_tract_warning.unwrap();
    assert!(warning.contains("10"), "Warning should mention tract count: {}", warning);
}

#[test]
fn low_tract_warning_boundary_at_20() {
    // Exactly 20 tracts — should NOT trigger warning
    let tracts = make_quantile_tracts(20, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.low_tract_warning.is_none(),
        "Should not warn at exactly 20 tracts, got: {:?}", result.low_tract_warning);
}

#[test]
fn low_tract_warning_at_19() {
    // 19 tracts — should trigger warning
    let tracts = make_quantile_tracts(19, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.low_tract_warning.is_some(),
        "Should warn at 19 tracts");
}

#[test]
fn low_tract_warning_zero_tracts() {
    let tracts: Vec<f64> = vec![];
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    // With 0 tracts, the first early-return path is taken
    // No warning for n==0 (original code sets None for empty input)
    assert!(result.low_tract_warning.is_none());
}

#[test]
fn low_tract_warning_2_tracts() {
    let tracts = vec![30_000.0, 50_000.0];
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);
    // n=2 → first early return with low_tract_warning
    assert!(result.low_tract_warning.is_some(),
        "Should warn with 2 tracts");
}

#[test]
fn low_tract_warning_all_below_lmin() {
    // 50 tracts all below l_min → valid_tracts = 0 → second early return
    let tracts: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 100.0).collect();
    let params = DemographyParams {
        l_min_bp: 20_000.0,
        ..DemographyParams::default()
    };
    let result = infer_demography(&tracts, "POP", &params);
    assert!(result.low_tract_warning.is_some());
}

// ==================== Lagrange projection edge cases ====================

#[test]
fn lagrange_extreme_constraint_far_below() {
    // Constrain to 10% of natural mean — rates should still be positive
    let tracts = make_quantile_tracts(200, 1e-5, 20_000.0);
    let params_free = DemographyParams::default();
    let r_free = infer_demography(&tracts, "POP", &params_free);
    let free_rate = r_free.pulses[0].rate;

    let target = free_rate * 0.1;
    let params = DemographyParams {
        bw_constraint: Some(target),
        ..DemographyParams::default()
    };
    let r = infer_demography(&tracts, "POP", &params);
    assert!(r.pulses[0].rate > 0.0, "Rate must stay positive even with extreme constraint");
    assert!(r.pulses[0].rate.is_finite());
}

#[test]
fn lagrange_extreme_constraint_far_above() {
    // Constrain to 10x natural mean
    let tracts = make_quantile_tracts(200, 1e-5, 20_000.0);
    let params_free = DemographyParams::default();
    let r_free = infer_demography(&tracts, "POP", &params_free);
    let free_rate = r_free.pulses[0].rate;

    let target = free_rate * 10.0;
    let params = DemographyParams {
        bw_constraint: Some(target),
        ..DemographyParams::default()
    };
    let r = infer_demography(&tracts, "POP", &params);
    assert!(r.pulses[0].rate.is_finite());
    assert!(r.log_likelihood.is_finite());
}

#[test]
fn lagrange_constraint_zero_is_skipped() {
    // lambda_bw = 0 should be skipped (the if lambda_bw > 0.0 check)
    let tracts = make_quantile_tracts(100, 1e-5, 20_000.0);
    let params_free = DemographyParams::default();
    let r_free = infer_demography(&tracts, "POP", &params_free);

    let params_zero = DemographyParams {
        bw_constraint: Some(0.0),
        ..DemographyParams::default()
    };
    let r_zero = infer_demography(&tracts, "POP", &params_zero);
    // With bw_constraint = 0.0, the constraint is skipped so results should match free
    assert!((r_free.pulses[0].rate - r_zero.pulses[0].rate).abs() / r_free.pulses[0].rate < 0.01,
        "Zero constraint should be skipped: free={:.2e} vs zero={:.2e}",
        r_free.pulses[0].rate, r_zero.pulses[0].rate);
}

#[test]
fn lagrange_preserves_rate_positivity_with_small_constraint() {
    // Very small target — Lagrange with .max(1e-15) should keep rates positive
    let tracts = make_quantile_tracts(100, 1e-5, 20_000.0);
    let params = DemographyParams {
        bw_constraint: Some(1e-15),
        ..DemographyParams::default()
    };
    let r = infer_demography(&tracts, "POP", &params);
    for pulse in &r.pulses {
        assert!(pulse.rate > 0.0, "Rate must be positive, got {:.2e}", pulse.rate);
    }
}

#[test]
fn lagrange_negative_constraint_is_skipped() {
    // Negative lambda_bw should be skipped
    let tracts = make_quantile_tracts(100, 1e-5, 20_000.0);
    let params_free = DemographyParams::default();
    let r_free = infer_demography(&tracts, "POP", &params_free);

    let params_neg = DemographyParams {
        bw_constraint: Some(-1.0),
        ..DemographyParams::default()
    };
    let r_neg = infer_demography(&tracts, "POP", &params_neg);
    // Negative constraint should be skipped
    assert!((r_free.pulses[0].rate - r_neg.pulses[0].rate).abs() / r_free.pulses[0].rate < 0.01,
        "Negative constraint should be skipped");
}

#[test]
fn lagrange_at_natural_mean_is_noop() {
    // Constraining at the natural EM mean should change nothing
    let tracts = make_quantile_tracts(200, 1e-5, 20_000.0);
    let params_free = DemographyParams::default();
    let r_free = infer_demography(&tracts, "POP", &params_free);
    let natural_mean: f64 = r_free.pulses.iter()
        .map(|p| p.proportion * p.rate)
        .sum();

    let params_at_mean = DemographyParams {
        bw_constraint: Some(natural_mean),
        ..DemographyParams::default()
    };
    let r_at_mean = infer_demography(&tracts, "POP", &params_at_mean);
    let constrained_mean: f64 = r_at_mean.pulses.iter()
        .map(|p| p.proportion * p.rate)
        .sum();
    assert!((constrained_mean - natural_mean).abs() / natural_mean < 0.05,
        "Constraining at natural mean should be near-noop: {:.2e} vs {:.2e}",
        constrained_mean, natural_mean);
}

// ==================== write_demography_tsv with new fields ====================

#[test]
fn write_tsv_with_cv_field_no_panic() {
    // Verify TSV writer handles PulseEstimate with cv field without panic
    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 100,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 200.0, proportion: 1.0, rate: 2e-6, cv: 0.1,
        }],
        bic: -500.0,
        log_likelihood: -245.0,
        ks_statistic: Some(0.05),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let tmp = std::env::temp_dir().join("demog_cv_test.tsv");
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.lines().count() == 2); // header + 1 row
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_with_low_tract_warning_no_panic() {
    // Verify TSV writer handles DemographicResult with warning without panic
    let pooled = vec![DemographicResult {
        population: "AFR".to_string(),
        n_tracts: 5,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 100.0, proportion: 1.0, rate: 1e-6, cv: 0.447,
        }],
        bic: 100.0,
        log_likelihood: -45.0,
        ks_statistic: Some(0.15),
        single_pulse_rejected: false,
        low_tract_warning: Some("5 tracts may be insufficient".to_string()),
    }];
    let tmp = std::env::temp_dir().join("demog_warning_test.tsv");
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.lines().count() == 2);
    std::fs::remove_file(&tmp).ok();
}

// ==================== PulseEstimate struct ====================

#[test]
fn pulse_estimate_cv_field_accessible() {
    let pulse = PulseEstimate {
        generations: 100.0,
        proportion: 0.5,
        rate: 1e-6,
        cv: 0.141,
    };
    assert!((pulse.cv - 0.141).abs() < 1e-6);
}

#[test]
fn pulse_estimate_debug_includes_cv() {
    let pulse = PulseEstimate {
        generations: 100.0,
        proportion: 0.5,
        rate: 1e-6,
        cv: 0.141,
    };
    let debug = format!("{:?}", pulse);
    assert!(debug.contains("cv"), "Debug output should include cv field: {}", debug);
}

#[test]
fn pulse_estimate_clone_preserves_cv() {
    let pulse = PulseEstimate {
        generations: 100.0,
        proportion: 0.5,
        rate: 1e-6,
        cv: 0.333,
    };
    let cloned = pulse.clone();
    assert_eq!(cloned.cv, pulse.cv);
}

// ==================== DemographicResult struct ====================

#[test]
fn demographic_result_debug_includes_warning() {
    let result = DemographicResult {
        population: "POP".to_string(),
        n_tracts: 5,
        n_pulses: 0,
        pulses: vec![],
        bic: f64::INFINITY,
        log_likelihood: f64::NEG_INFINITY,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: Some("test warning".to_string()),
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("low_tract_warning"), "Debug should include warning field");
    assert!(debug.contains("test warning"));
}

#[test]
fn demographic_result_clone_preserves_warning() {
    let result = DemographicResult {
        population: "POP".to_string(),
        n_tracts: 5,
        n_pulses: 0,
        pulses: vec![],
        bic: f64::INFINITY,
        log_likelihood: f64::NEG_INFINITY,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: Some("warning text".to_string()),
    };
    let cloned = result.clone();
    assert_eq!(cloned.low_tract_warning, result.low_tract_warning);
}

// ==================== Integration: infer_demography returns both new fields ====================

#[test]
fn infer_demography_returns_both_new_fields() {
    let tracts = make_quantile_tracts(50, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);

    // Should have cv on each pulse
    for pulse in &result.pulses {
        assert!(pulse.cv.is_finite());
        assert!(pulse.cv > 0.0);
    }

    // 50 tracts >= 20 threshold → no warning
    assert!(result.low_tract_warning.is_none());
}

#[test]
fn infer_demography_both_fields_with_few_tracts() {
    let tracts = make_quantile_tracts(5, 1e-5, 20_000.0);
    let params = DemographyParams::default();
    let result = infer_demography(&tracts, "POP", &params);

    // Should have warning for < 20 tracts
    assert!(result.low_tract_warning.is_some());

    // CV should be high but finite
    for pulse in &result.pulses {
        assert!(pulse.cv.is_finite());
        assert!(pulse.cv > 0.3);
    }
}

// Helper to generate deterministic tract lengths
fn make_quantile_tracts(n: usize, lambda: f64, l_min: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            l_min + (-((1.0 - p).ln()) / lambda)
        })
        .collect()
}
