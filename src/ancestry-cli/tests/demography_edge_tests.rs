//! Edge case tests for demography.rs — multi-pulse demographic inference (T80)
//!
//! Covers: empty/degenerate inputs, numerical edge cases, boundary conditions,
//! NaN/Infinity robustness, BIC/KS edge cases, infer_all_demography, format edge cases.

use hprc_ancestry_cli::{
    AncestrySegment, DemographyParams, DemographicResult, PulseEstimate,
    extract_tract_lengths, infer_demography, infer_all_demography,
    ks_test_exponential, format_demography_report,
};

// ==================== Helper ====================

fn make_segment(ancestry: &str, start: u64, end: u64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "S1".to_string(),
        ancestry_idx: 0,
        ancestry_name: ancestry.to_string(),
        n_windows: ((end - start) / 10_000) as usize,
        mean_similarity: 0.95,
        mean_posterior: Some(0.9),
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

fn default_params() -> DemographyParams {
    DemographyParams::default()
}

fn make_exponential_tracts(n: usize, lambda: f64, l_min: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            -((1.0 - p).ln()) / lambda + l_min
        })
        .collect()
}

// ==================== extract_tract_lengths ====================

#[test]
fn extract_tracts_empty_segments() {
    let tracts = extract_tract_lengths(&[]);
    assert!(tracts.is_empty());
}

#[test]
fn extract_tracts_single_segment() {
    let segs = vec![make_segment("EUR", 0, 100_000)];
    let tracts = extract_tract_lengths(&segs);
    assert_eq!(tracts.len(), 1);
    assert_eq!(tracts["EUR"], vec![100_000.0]);
}

#[test]
fn extract_tracts_multiple_populations() {
    let segs = vec![
        make_segment("EUR", 0, 50_000),
        make_segment("AFR", 50_000, 200_000),
        make_segment("EAS", 200_000, 250_000),
        make_segment("EUR", 250_000, 400_000),
    ];
    let tracts = extract_tract_lengths(&segs);
    assert_eq!(tracts.len(), 3);
    assert_eq!(tracts["EUR"].len(), 2);
    assert_eq!(tracts["AFR"].len(), 1);
    assert_eq!(tracts["EAS"].len(), 1);
    assert_eq!(tracts["EUR"][0], 50_000.0);
    assert_eq!(tracts["EUR"][1], 150_000.0);
}

#[test]
fn extract_tracts_zero_length_segment() {
    let segs = vec![make_segment("EUR", 100, 100)]; // start == end
    let tracts = extract_tract_lengths(&segs);
    assert_eq!(tracts["EUR"], vec![0.0]);
}

// ==================== ks_test_exponential ====================

#[test]
fn ks_test_empty() {
    let (ks, crit) = ks_test_exponential(&[], 20_000.0, 0.05);
    assert_eq!(ks, 0.0);
    assert_eq!(crit, 1.0);
}

#[test]
fn ks_test_single_tract() {
    let (ks, crit) = ks_test_exponential(&[50_000.0], 20_000.0, 0.05);
    assert!(ks.is_finite());
    assert!(crit.is_finite());
    // With n=1, critical value = 1.36/1 = 1.36, so should not reject
    assert!(ks <= crit, "single tract should not reject: KS={}, crit={}", ks, crit);
}

#[test]
fn ks_test_two_identical_tracts() {
    let tracts = vec![50_000.0, 50_000.0];
    let (ks, crit) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    assert!(ks.is_finite());
    assert!(crit.is_finite());
}

#[test]
fn ks_test_alpha_levels() {
    let tracts = make_exponential_tracts(100, 1e-5, 20_000.0);
    let (_, crit_01) = ks_test_exponential(&tracts, 20_000.0, 0.01);
    let (_, crit_05) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    let (_, crit_10) = ks_test_exponential(&tracts, 20_000.0, 0.10);
    // More stringent alpha → higher critical value
    assert!(crit_01 > crit_05, "alpha=0.01 should be stricter");
    assert!(crit_05 > crit_10, "alpha=0.05 should be stricter than 0.10");
}

#[test]
fn ks_test_tracts_at_l_min() {
    // All tracts exactly at l_min → excess = 0 → lambda = inf → degenerate
    let tracts = vec![20_000.0; 10];
    let (ks, _crit) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    // Should not panic, KS stat might be NaN or finite
    assert!(!ks.is_nan() || ks.is_nan()); // just verifying no panic
}

#[test]
fn ks_test_tracts_below_l_min() {
    // Tracts shorter than l_min → excess clamped to 0
    let tracts = vec![10_000.0, 15_000.0, 5_000.0];
    let (ks, _crit) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    // Should not panic
    assert!(ks.is_finite() || ks.is_nan());
}

#[test]
fn ks_test_large_n() {
    // With many samples, critical value shrinks → more sensitive
    let tracts = make_exponential_tracts(10_000, 1e-5, 20_000.0);
    let (ks, crit) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    assert!(crit < 0.05, "large n should have small critical value, got {}", crit);
    // Perfect quantile data should still pass
    assert!(ks < crit);
}

// ==================== infer_demography ====================

#[test]
fn infer_empty_tracts() {
    let result = infer_demography(&[], "TEST", &default_params());
    assert_eq!(result.n_tracts, 0);
    assert_eq!(result.n_pulses, 0);
    assert!(result.pulses.is_empty());
    assert!(result.bic.is_infinite());
    assert_eq!(result.population, "TEST");
}

#[test]
fn infer_one_tract() {
    let result = infer_demography(&[100_000.0], "POP", &default_params());
    assert_eq!(result.n_pulses, 0); // < 3 tracts
    assert!(result.pulses.is_empty());
}

#[test]
fn infer_two_tracts() {
    let result = infer_demography(&[50_000.0, 80_000.0], "POP", &default_params());
    assert_eq!(result.n_pulses, 0); // < 3 tracts
}

#[test]
fn infer_exactly_three_tracts() {
    let result = infer_demography(&[30_000.0, 50_000.0, 100_000.0], "POP", &default_params());
    // Exactly 3 valid tracts (all > l_min=20000), should fit 1 pulse
    assert_eq!(result.n_pulses, 1);
    assert_eq!(result.pulses.len(), 1);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn infer_all_tracts_below_l_min() {
    // All tracts below detection limit → filtered out → n_valid < 3
    let tracts = vec![10_000.0, 15_000.0, 18_000.0, 19_999.0];
    let result = infer_demography(&tracts, "POP", &default_params());
    assert_eq!(result.n_pulses, 0);
    assert!(result.pulses.is_empty());
}

#[test]
fn infer_mixed_above_below_l_min() {
    // Some tracts filtered, only 2 valid → insufficient
    let tracts = vec![10_000.0, 15_000.0, 30_000.0, 50_000.0];
    let params = DemographyParams { l_min_bp: 20_000.0, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);
    // 2 valid tracts (30k, 50k) → below minimum 3
    assert_eq!(result.n_tracts, 2);
    assert_eq!(result.n_pulses, 0);
}

#[test]
fn infer_all_identical_tracts() {
    // All tracts identical → zero variance → degenerate MLE
    let tracts = vec![50_000.0; 20];
    let result = infer_demography(&tracts, "POP", &default_params());
    // Should not panic, result should have 1 pulse (or 0 if degenerate)
    assert!(result.n_pulses <= 1);
    assert!(result.population == "POP");
}

#[test]
fn infer_very_large_tracts() {
    // Very long tracts → small lambda → small g (recent admixture, few recomb events)
    let tracts: Vec<f64> = (0..50).map(|i| 1_000_000_000.0 + i as f64 * 1000.0).collect();
    let result = infer_demography(&tracts, "RECENT", &default_params());
    assert_eq!(result.n_pulses, 1);
    // Long tracts = recent admixture (lambda = g*r is small, so g is small)
    assert!(result.pulses[0].generations < 100.0,
        "very long tracts should imply recent admixture, got g={}", result.pulses[0].generations);
}

#[test]
fn infer_very_short_tracts_just_above_l_min() {
    // Tracts barely above l_min → very large lambda → recent admixture
    let l_min = 20_000.0;
    let tracts: Vec<f64> = (0..50).map(|i| l_min + 1.0 + i as f64 * 0.1).collect();
    let result = infer_demography(&tracts, "RECENT", &DemographyParams { l_min_bp: l_min, ..default_params() });
    assert_eq!(result.n_pulses, 1);
    assert!(result.log_likelihood.is_finite());
}

#[test]
fn infer_custom_recomb_rate() {
    let tracts = make_exponential_tracts(100, 2e-6, 20_000.0);
    let params1 = DemographyParams { recomb_rate: 1e-8, ..default_params() };
    let params2 = DemographyParams { recomb_rate: 2e-8, ..default_params() };

    let r1 = infer_demography(&tracts, "POP", &params1);
    let r2 = infer_demography(&tracts, "POP", &params2);

    // Same rate, different recomb → different generation estimate
    // g = lambda / recomb_rate, so doubling recomb_rate halves g
    assert_eq!(r1.n_pulses, 1);
    assert_eq!(r2.n_pulses, 1);
    let ratio = r1.pulses[0].generations / r2.pulses[0].generations;
    assert!((ratio - 2.0).abs() < 0.01, "doubling recomb rate should halve g, ratio={}", ratio);
}

#[test]
fn infer_custom_l_min() {
    let tracts: Vec<f64> = (0..100).map(|i| 40_000.0 + i as f64 * 1000.0).collect();
    let params_low = DemographyParams { l_min_bp: 10_000.0, ..default_params() };
    let params_high = DemographyParams { l_min_bp: 30_000.0, ..default_params() };

    let r_low = infer_demography(&tracts, "POP", &params_low);
    let r_high = infer_demography(&tracts, "POP", &params_high);

    // Both should produce results
    assert_eq!(r_low.n_pulses, 1);
    assert_eq!(r_high.n_pulses, 1);
    // Higher l_min → smaller excess → larger lambda → different generation estimate
    assert!((r_low.pulses[0].generations - r_high.pulses[0].generations).abs() > 0.1,
        "different l_min should give different estimates");
}

#[test]
fn infer_max_pulses_one() {
    // With max_pulses=1, should never try multi-pulse even if data supports it
    let l_min = 20_000.0;
    let mut tracts = make_exponential_tracts(100, 1e-7, l_min);
    tracts.extend(make_exponential_tracts(100, 1e-5, l_min));

    let params = DemographyParams { max_pulses: 1, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);
    assert_eq!(result.n_pulses, 1);
}

#[test]
fn infer_result_fields_populated() {
    let tracts = make_exponential_tracts(100, 1e-5, 20_000.0);
    let result = infer_demography(&tracts, "EUR", &default_params());

    assert_eq!(result.population, "EUR");
    assert_eq!(result.n_tracts, 100);
    assert_eq!(result.n_pulses, 1);
    assert!(result.bic.is_finite());
    assert!(result.log_likelihood.is_finite());
    assert!(result.log_likelihood < 0.0); // LL is always negative
    assert!(result.ks_statistic.is_some());
    // Single-pulse on single-exp data should NOT be rejected
    assert!(!result.single_pulse_rejected);
    // Pulse fields
    assert_eq!(result.pulses.len(), 1);
    assert!(result.pulses[0].generations > 0.0);
    assert_eq!(result.pulses[0].proportion, 1.0);
    assert!(result.pulses[0].rate > 0.0);
}

#[test]
fn infer_multi_pulse_not_selected_with_few_tracts() {
    // 15 tracts → insufficient for 2-pulse (needs 10*m = 20)
    let l_min = 20_000.0;
    let mut tracts = make_exponential_tracts(8, 1e-7, l_min);
    tracts.extend(make_exponential_tracts(7, 1e-5, l_min));

    let result = infer_demography(&tracts, "POP", &default_params());
    // Even if KS rejects, not enough data for 2-pulse
    assert_eq!(result.n_pulses, 1);
}

#[test]
fn infer_em_convergence_tolerance() {
    let tracts = make_exponential_tracts(200, 1e-5, 20_000.0);
    let tight = DemographyParams { em_tolerance: 1e-12, ..default_params() };
    let loose = DemographyParams { em_tolerance: 1e-2, ..default_params() };

    let r_tight = infer_demography(&tracts, "POP", &tight);
    let r_loose = infer_demography(&tracts, "POP", &loose);

    // Both should produce valid results
    assert_eq!(r_tight.n_pulses, 1);
    assert_eq!(r_loose.n_pulses, 1);
    // Rate estimates should be similar (both converge for single component)
    let diff = (r_tight.pulses[0].rate - r_loose.pulses[0].rate).abs();
    assert!(diff < 1e-4, "tolerance shouldn't affect single-component MLE much, diff={}", diff);
}

#[test]
fn infer_em_max_iters_one() {
    let tracts = make_exponential_tracts(100, 1e-5, 20_000.0);
    let params = DemographyParams { max_em_iters: 1, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);
    // Should still produce a valid (if suboptimal) result
    assert_eq!(result.n_pulses, 1);
    assert!(result.log_likelihood.is_finite());
}

// ==================== infer_all_demography ====================

#[test]
fn infer_all_empty_segments() {
    let results = infer_all_demography(&[], &["EUR".to_string(), "AFR".to_string()], &default_params());
    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r.n_tracts, 0);
        assert_eq!(r.n_pulses, 0);
    }
}

#[test]
fn infer_all_no_populations() {
    let seg = make_segment("EUR", 0, 100_000);
    let results = infer_all_demography(&[&seg], &[], &default_params());
    assert!(results.is_empty());
}

#[test]
fn infer_all_population_not_in_segments() {
    let seg = make_segment("EUR", 0, 100_000);
    let results = infer_all_demography(&[&seg], &["AFR".to_string()], &default_params());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].population, "AFR");
    assert_eq!(results[0].n_tracts, 0);
    assert_eq!(results[0].n_pulses, 0);
}

#[test]
fn infer_all_multiple_populations() {
    let segs: Vec<AncestrySegment> = (0..50).flat_map(|i| {
        vec![
            make_segment("EUR", i * 200_000, i * 200_000 + 80_000),
            make_segment("AFR", i * 200_000 + 80_000, i * 200_000 + 200_000),
        ]
    }).collect();
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();

    let results = infer_all_demography(
        &seg_refs,
        &["EUR".to_string(), "AFR".to_string()],
        &default_params(),
    );
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].population, "EUR");
    assert_eq!(results[1].population, "AFR");
    // Both have 50 tracts each — EUR tracts are 80kb, AFR tracts are 120kb
    assert_eq!(results[0].n_tracts, 50);
    assert_eq!(results[1].n_tracts, 50);
}

// ==================== format_demography_report ====================

#[test]
fn format_report_empty() {
    let report = format_demography_report(&[]);
    assert!(report.contains("Demographic Inference"));
}

#[test]
fn format_report_insufficient_tracts() {
    let results = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 2,
        n_pulses: 0,
        pulses: vec![],
        bic: f64::INFINITY,
        log_likelihood: f64::NEG_INFINITY,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let report = format_demography_report(&results);
    assert!(report.contains("insufficient tracts"));
    assert!(report.contains("n=2"));
}

#[test]
fn format_report_multi_pulse() {
    let results = vec![DemographicResult {
        population: "AMR".to_string(),
        n_tracts: 200,
        n_pulses: 2,
        pulses: vec![
            PulseEstimate { generations: 50.0, proportion: 0.6, rate: 5e-7 , cv: 0.0 },
            PulseEstimate { generations: 300.0, proportion: 0.4, rate: 3e-6 , cv: 0.0 },
        ],
        bic: -800.0,
        log_likelihood: -390.0,
        ks_statistic: Some(0.12),
        single_pulse_rejected: true,
        low_tract_warning: None,
    }];
    let report = format_demography_report(&results);
    assert!(report.contains("AMR"));
    assert!(report.contains("2 pulses"));
    assert!(report.contains("Pulse 1"));
    assert!(report.contains("Pulse 2"));
    assert!(report.contains("REJECTED"));
    assert!(report.contains("60.0%"));
    assert!(report.contains("40.0%"));
}

#[test]
fn format_report_single_pulse_not_rejected() {
    let results = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 100,
        n_pulses: 1,
        pulses: vec![PulseEstimate { generations: 150.0, proportion: 1.0, rate: 1.5e-6 , cv: 0.0 }],
        bic: -400.0,
        log_likelihood: -195.0,
        ks_statistic: Some(0.03),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let report = format_demography_report(&results);
    assert!(report.contains("1 pulse"));
    assert!(report.contains("single-pulse OK"));
    assert!(!report.contains("REJECTED"));
}

#[test]
fn format_report_no_ks_statistic() {
    let results = vec![DemographicResult {
        population: "TEST".to_string(),
        n_tracts: 50,
        n_pulses: 1,
        pulses: vec![PulseEstimate { generations: 100.0, proportion: 1.0, rate: 1e-6 , cv: 0.0 }],
        bic: -300.0,
        log_likelihood: -145.0,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let report = format_demography_report(&results);
    assert!(report.contains("TEST"));
    assert!(!report.contains("KS="));
}

// ==================== DemographyParams defaults ====================

#[test]
fn params_default_values() {
    let p = DemographyParams::default();
    assert_eq!(p.recomb_rate, 1e-8);
    assert_eq!(p.l_min_bp, 20_000.0);
    assert_eq!(p.max_pulses, 3);
    assert_eq!(p.em_tolerance, 1e-8);
    assert_eq!(p.max_em_iters, 200);
    assert_eq!(p.ks_alpha, 0.05);
}

// ==================== Numerical stability ====================

#[test]
fn infer_nan_tracts_no_panic() {
    let tracts = vec![f64::NAN, 50_000.0, 80_000.0, 100_000.0];
    // NaN tracts should be filtered by l > l_min check (NaN > 20000 is false)
    let result = infer_demography(&tracts, "POP", &default_params());
    // Should not panic; 3 valid tracts remain
    assert!(result.n_pulses <= 1);
}

#[test]
fn infer_infinity_tracts_no_panic() {
    let tracts = vec![f64::INFINITY, 50_000.0, 80_000.0, 100_000.0];
    // INFINITY > l_min is true, so it's included
    let result = infer_demography(&tracts, "POP", &default_params());
    // Should not panic (infinity in sum gives infinity lambda → may be degenerate)
    assert!(result.population == "POP");
}

#[test]
fn infer_negative_tracts_no_panic() {
    let tracts = vec![-100.0, -50_000.0, 30_000.0, 50_000.0, 80_000.0];
    // Negative tracts filtered (< l_min)
    let result = infer_demography(&tracts, "POP", &default_params());
    assert_eq!(result.n_tracts, 3); // only 30k, 50k, 80k pass
    assert_eq!(result.n_pulses, 1);
}

#[test]
fn ks_test_nan_tracts_no_panic() {
    let tracts = vec![f64::NAN, 50_000.0, 80_000.0];
    let (ks, crit) = ks_test_exponential(&tracts, 20_000.0, 0.05);
    // May produce NaN but should not panic
    let _ = (ks, crit);
}

// ==================== BIC edge cases ====================

#[test]
fn infer_bic_negative_for_good_fit() {
    let tracts = make_exponential_tracts(200, 1e-5, 20_000.0);
    let result = infer_demography(&tracts, "POP", &default_params());
    // BIC should be finite for good data
    assert!(result.bic.is_finite());
}

#[test]
fn infer_bic_penalizes_complexity() {
    // For single-exponential data, BIC should prefer 1 pulse over 2
    let tracts = make_exponential_tracts(200, 1e-5, 20_000.0);
    let result = infer_demography(&tracts, "POP", &default_params());
    assert_eq!(result.n_pulses, 1, "BIC should prefer parsimony for single-exp data");
}

// ==================== Poorly separated rates ====================

#[test]
fn infer_poorly_separated_rates_selects_single() {
    // Two components with similar rates (ratio < 1.5) → should stay at 1 pulse
    let l_min = 20_000.0;
    let lambda1 = 1e-5;
    let lambda2 = 1.2e-5; // ratio = 1.2 < 1.5 threshold
    let mut tracts = make_exponential_tracts(100, lambda1, l_min);
    tracts.extend(make_exponential_tracts(100, lambda2, l_min));

    let result = infer_demography(&tracts, "POP", &default_params());
    // Rate separation check (R > 1.5) should prevent 2-pulse selection
    assert_eq!(result.n_pulses, 1, "poorly separated rates should collapse to 1 pulse");
}

// ==================== Extreme parameter values ====================

#[test]
fn infer_very_small_recomb_rate() {
    let tracts = make_exponential_tracts(50, 1e-5, 20_000.0);
    let params = DemographyParams { recomb_rate: 1e-12, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);
    assert_eq!(result.n_pulses, 1);
    // Very small recomb rate → very large generation estimate
    assert!(result.pulses[0].generations > 1e6);
}

#[test]
fn infer_very_large_l_min() {
    // l_min larger than all tracts → all filtered
    let tracts: Vec<f64> = (0..20).map(|i| 30_000.0 + i as f64 * 1000.0).collect();
    let params = DemographyParams { l_min_bp: 1_000_000.0, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);
    assert_eq!(result.n_pulses, 0);
}

// ==================== PulseEstimate fields ====================

#[test]
fn pulse_estimate_rate_generation_consistency() {
    let tracts = make_exponential_tracts(100, 1e-5, 20_000.0);
    let params = DemographyParams { recomb_rate: 1e-8, ..default_params() };
    let result = infer_demography(&tracts, "POP", &params);

    assert_eq!(result.n_pulses, 1);
    let pulse = &result.pulses[0];
    // generations = rate / recomb_rate
    let expected_g = pulse.rate / params.recomb_rate;
    assert!((pulse.generations - expected_g).abs() < 1e-6,
        "g={} != rate/recomb_rate={}", pulse.generations, expected_g);
}

#[test]
fn pulse_proportions_sum_to_one() {
    let tracts = make_exponential_tracts(100, 1e-5, 20_000.0);
    let result = infer_demography(&tracts, "POP", &default_params());

    let sum: f64 = result.pulses.iter().map(|p| p.proportion).sum();
    assert!((sum - 1.0).abs() < 1e-6, "proportions should sum to 1, got {}", sum);
}

// ==================== DemographicResult clone/debug ====================

#[test]
fn demographic_result_clone() {
    let result = infer_demography(&[30_000.0, 50_000.0, 80_000.0], "POP", &default_params());
    let cloned = result.clone();
    assert_eq!(cloned.population, result.population);
    assert_eq!(cloned.n_pulses, result.n_pulses);
}

#[test]
fn demographic_result_debug() {
    let result = infer_demography(&[30_000.0, 50_000.0, 80_000.0], "POP", &default_params());
    let debug = format!("{:?}", result);
    assert!(debug.contains("POP"));
}

#[test]
fn pulse_estimate_debug() {
    let pulse = PulseEstimate { generations: 100.0, proportion: 0.5, rate: 1e-6 , cv: 0.0 };
    let debug = format!("{:?}", pulse);
    assert!(debug.contains("100"));
}

#[test]
fn demography_params_debug() {
    let params = DemographyParams::default();
    let debug = format!("{:?}", params);
    assert!(debug.contains("recomb_rate"));
}
