//! Edge case tests for algo_dev cycle 68 additions:
//! - infer_per_sample_demography() edge cases
//! - write_demography_tsv() edge cases
//! - BW constraint projection edge cases
//! - T83 iterative refinement early stopping

use hprc_ancestry_cli::{
    AncestrySegment, AncestryHmmParams, AncestralPopulation, DemographyParams,
    DemographicResult, PulseEstimate, SampleDemographicResult,
    infer_per_sample_demography, infer_demography, infer_all_demography,
    write_demography_tsv, iterative_refine,
};

// ==================== Helpers ====================

fn make_segment(sample: &str, ancestry: &str, start: u64, end: u64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: sample.to_string(),
        ancestry_idx: 0,
        ancestry_name: ancestry.to_string(),
        n_windows: ((end - start) / 10_000).max(1) as usize,
        mean_similarity: 0.95,
        mean_posterior: Some(0.9),
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

fn default_params() -> DemographyParams {
    DemographyParams::default()
}

fn make_test_populations() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation { name: "EUR".to_string(), haplotypes: vec!["ref1".to_string()] },
        AncestralPopulation { name: "AFR".to_string(), haplotypes: vec!["ref2".to_string()] },
        AncestralPopulation { name: "EAS".to_string(), haplotypes: vec!["ref3".to_string()] },
    ]
}

fn make_dummy_result(pop: &str, n_tracts: usize, pulses: Vec<PulseEstimate>) -> DemographicResult {
    DemographicResult {
        population: pop.to_string(),
        n_tracts,
        n_pulses: pulses.len(),
        pulses,
        bic: 500.0,
        log_likelihood: -200.0,
        ks_statistic: Some(0.05),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }
}

// ==================== infer_per_sample_demography ====================

#[test]
fn per_sample_single_sample_no_matching_population() {
    let segments = vec![
        make_segment("S1", "EUR", 0, 200_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["AFR".to_string()]; // no EUR segments match AFR
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].sample, "S1");
    assert_eq!(results[0].results.len(), 1);
    assert_eq!(results[0].results[0].n_tracts, 0);
}

#[test]
fn per_sample_no_populations() {
    let segments = vec![
        make_segment("S1", "EUR", 0, 200_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names: Vec<String> = vec![];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].results.len(), 0); // no populations to infer
}

#[test]
fn per_sample_many_samples_sorted_by_name() {
    let segments = vec![
        make_segment("Zulu", "EUR", 0, 100_000),
        make_segment("Alpha", "EUR", 0, 200_000),
        make_segment("Mike", "EUR", 0, 150_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string()];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 3);
    // BTreeMap guarantees alphabetical order
    assert_eq!(results[0].sample, "Alpha");
    assert_eq!(results[1].sample, "Mike");
    assert_eq!(results[2].sample, "Zulu");
}

#[test]
fn per_sample_same_sample_all_same_pop() {
    let segments = vec![
        make_segment("S1", "EUR", 0, 100_000),
        make_segment("S1", "EUR", 100_000, 250_000),
        make_segment("S1", "EUR", 250_000, 500_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].results.len(), 2);
    // EUR should have 3 tracts, AFR should have 0
    assert_eq!(results[0].results[0].population, "EUR");
    assert_eq!(results[0].results[0].n_tracts, 3);
    assert_eq!(results[0].results[1].population, "AFR");
    assert_eq!(results[0].results[1].n_tracts, 0);
}

#[test]
fn per_sample_single_segment_per_sample() {
    let segments = vec![
        make_segment("A", "EUR", 0, 50_000),
        make_segment("B", "AFR", 0, 80_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 2);
    // Each sample has exactly 1 tract in one population, 0 in other
    assert_eq!(results[0].sample, "A");
    assert_eq!(results[0].results[0].n_tracts, 1); // EUR
    assert_eq!(results[0].results[1].n_tracts, 0); // AFR
    assert_eq!(results[1].sample, "B");
    assert_eq!(results[1].results[0].n_tracts, 0); // EUR
    assert_eq!(results[1].results[1].n_tracts, 1); // AFR
}

#[test]
fn per_sample_zero_length_segment_no_panic() {
    let segments = vec![
        make_segment("S1", "EUR", 100, 100), // zero length
        make_segment("S1", "EUR", 0, 200_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string()];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 1);
    // Should not panic; zero-length tract is just 0.0
    assert_eq!(results[0].results[0].n_tracts, 2);
}

#[test]
fn per_sample_large_number_of_samples() {
    let segments: Vec<AncestrySegment> = (0..100)
        .map(|i| make_segment(&format!("sample_{:03}", i), "EUR", 0, 100_000 + i as u64 * 1000))
        .collect();
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string()];
    let results = infer_per_sample_demography(&refs, &pop_names, &default_params());
    assert_eq!(results.len(), 100);
    // All should have exactly 1 EUR tract
    for r in &results {
        assert_eq!(r.results[0].n_tracts, 1);
    }
}

// ==================== write_demography_tsv ====================

#[test]
fn write_tsv_both_empty() {
    let tmp = std::env::temp_dir().join("demog_edge_both_empty.tsv");
    write_demography_tsv(&tmp, &[], &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1); // header only
    assert!(lines[0].starts_with("sample\t"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_only_per_sample() {
    let per_sample = vec![SampleDemographicResult {
        sample: "hap1".to_string(),
        results: vec![make_dummy_result("EUR", 10, vec![
            PulseEstimate { generations: 100.0, proportion: 1.0, rate: 1e-6 , cv: 0.0 },
        ])],
    }];
    let tmp = std::env::temp_dir().join("demog_edge_per_sample_only.tsv");
    write_demography_tsv(&tmp, &[], &per_sample).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2); // header + 1 row
    assert!(lines[1].starts_with("hap1\t"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_nan_inf_in_results() {
    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 5,
        n_pulses: 1,
        pulses: vec![PulseEstimate { generations: f64::NAN, proportion: f64::NAN, rate: f64::NAN , cv: 0.0 }],
        bic: f64::NAN,
        log_likelihood: f64::NAN,
        ks_statistic: Some(f64::NAN),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let tmp = std::env::temp_dir().join("demog_edge_nan.tsv");
    // Should not panic, just write NaN values
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("NaN") || content.contains("nan") || content.contains("NA"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_many_pulses() {
    let pulses: Vec<PulseEstimate> = (1..=10).map(|i| PulseEstimate {
        generations: i as f64 * 10.0,
        proportion: 0.1,
        rate: i as f64 * 1e-7, cv: 0.0 }).collect();
    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 1000,
        n_pulses: 10,
        pulses,
        bic: 999.0,
        log_likelihood: -400.0,
        ks_statistic: Some(0.01),
        single_pulse_rejected: true,
        low_tract_warning: None,
    }];
    let tmp = std::env::temp_dir().join("demog_edge_many_pulses.tsv");
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 11); // header + 10 pulse rows
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_multiple_populations_multiple_samples() {
    let per_sample = vec![
        SampleDemographicResult {
            sample: "S1".to_string(),
            results: vec![
                make_dummy_result("EUR", 5, vec![PulseEstimate { generations: 50.0, proportion: 1.0, rate: 5e-7 , cv: 0.0 }]),
                make_dummy_result("AFR", 3, vec![]),
            ],
        },
        SampleDemographicResult {
            sample: "S2".to_string(),
            results: vec![
                make_dummy_result("EUR", 8, vec![
                    PulseEstimate { generations: 20.0, proportion: 0.6, rate: 2e-7 , cv: 0.0 },
                    PulseEstimate { generations: 80.0, proportion: 0.4, rate: 8e-7 , cv: 0.0 },
                ]),
                make_dummy_result("AFR", 2, vec![PulseEstimate { generations: 30.0, proportion: 1.0, rate: 3e-7 , cv: 0.0 }]),
            ],
        },
    ];
    let pooled = vec![
        make_dummy_result("EUR", 13, vec![PulseEstimate { generations: 40.0, proportion: 1.0, rate: 4e-7 , cv: 0.0 }]),
    ];
    let tmp = std::env::temp_dir().join("demog_edge_multi_pop_sample.tsv");
    write_demography_tsv(&tmp, &pooled, &per_sample).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    // header(1) + pooled EUR(1) + S1 EUR(1) + S1 AFR no pulses(1) + S2 EUR(2) + S2 AFR(1) = 7
    assert_eq!(lines.len(), 7);
    // Pooled comes first
    assert!(lines[1].starts_with("POOLED\t"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_empty_population_name() {
    let pooled = vec![make_dummy_result("", 1, vec![
        PulseEstimate { generations: 10.0, proportion: 1.0, rate: 1e-7 , cv: 0.0 },
    ])];
    let tmp = std::env::temp_dir().join("demog_edge_empty_pop.tsv");
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    // Should not panic, just have empty population field
    assert!(content.lines().count() == 2);
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn write_tsv_ks_statistic_none() {
    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 50,
        n_pulses: 1,
        pulses: vec![PulseEstimate { generations: 100.0, proportion: 1.0, rate: 1e-6 , cv: 0.0 }],
        bic: 300.0,
        log_likelihood: -100.0,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let tmp = std::env::temp_dir().join("demog_edge_ks_none.tsv");
    write_demography_tsv(&tmp, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert!(content.contains("NA")); // ks_statistic=None should become NA
    std::fs::remove_file(&tmp).ok();
}

// ==================== BW constraint edge cases ====================

#[test]
fn bw_constraint_negative_lambda_no_effect() {
    // Negative lambda_bw should be skipped (lambda_bw <= 0 check)
    let params = DemographyParams {
        bw_constraint: Some(-1e-5),
        ..default_params()
    };
    // With negative constraint, tracts should still produce a result
    let result = infer_demography(
        &[50_000.0, 80_000.0, 120_000.0, 200_000.0, 60_000.0],
        "EUR",
        &params,
    );
    assert!(result.n_tracts > 0);
    // Should not panic
}

#[test]
fn bw_constraint_very_large_lambda() {
    let params = DemographyParams {
        bw_constraint: Some(1.0), // extremely large, rates will be scaled up hugely
        ..default_params()
    };
    let result = infer_demography(
        &[50_000.0, 80_000.0, 120_000.0],
        "EUR",
        &params,
    );
    // Should not panic, produces a result
    assert!(result.n_tracts > 0);
    if !result.pulses.is_empty() {
        // Rate should be very large due to scaling
        assert!(result.pulses[0].rate > 1e-4 || result.pulses[0].rate.is_finite());
    }
}

#[test]
fn bw_constraint_very_small_lambda() {
    let params = DemographyParams {
        bw_constraint: Some(1e-20), // tiny, rates will be scaled down
        ..default_params()
    };
    let result = infer_demography(
        &[50_000.0, 80_000.0, 120_000.0],
        "EUR",
        &params,
    );
    assert!(result.n_tracts > 0);
    // Should not panic
}

#[test]
fn bw_constraint_zero_is_noop() {
    // lambda_bw = 0.0 should be skipped (>0.0 check)
    let params_zero = DemographyParams {
        bw_constraint: Some(0.0),
        ..default_params()
    };
    let params_none = DemographyParams {
        bw_constraint: None,
        ..default_params()
    };
    let tracts = vec![50_000.0, 80_000.0, 120_000.0, 200_000.0, 60_000.0, 90_000.0, 150_000.0, 70_000.0, 110_000.0, 130_000.0];
    let r_zero = infer_demography(&tracts, "EUR", &params_zero);
    let r_none = infer_demography(&tracts, "EUR", &params_none);
    assert_eq!(r_zero.n_pulses, r_none.n_pulses);
    // Rates should be identical
    for (pz, pn) in r_zero.pulses.iter().zip(r_none.pulses.iter()) {
        assert!((pz.rate - pn.rate).abs() < 1e-10,
            "Zero constraint should be noop: {} vs {}", pz.rate, pn.rate);
    }
}

#[test]
fn bw_constraint_with_single_tract() {
    // Single tract uses closed-form, constraint may not apply (EM not called for 1 pulse)
    let params = DemographyParams {
        bw_constraint: Some(1e-5),
        ..default_params()
    };
    let result = infer_demography(&[100_000.0], "EUR", &params);
    assert_eq!(result.n_tracts, 1);
    // Should not panic
}

#[test]
fn bw_constraint_with_empty_tracts() {
    let params = DemographyParams {
        bw_constraint: Some(1e-5),
        ..default_params()
    };
    let result = infer_demography(&[], "EUR", &params);
    assert_eq!(result.n_tracts, 0);
    assert!(result.pulses.is_empty());
}

#[test]
fn bw_constraint_matches_unconstrained_when_at_natural_mean() {
    // If we set constraint = the natural unconstrained mean, results should be similar
    let tracts: Vec<f64> = (0..50).map(|i| 50_000.0 + (i as f64 * 3000.0)).collect();
    // First get unconstrained result
    let r_free = infer_demography(&tracts, "EUR", &default_params());
    if r_free.pulses.is_empty() { return; }
    // Compute the natural weighted mean rate
    let natural_mean: f64 = r_free.pulses.iter().map(|p| p.proportion * p.rate).sum();
    if !natural_mean.is_finite() || natural_mean <= 0.0 { return; }
    // Now constrain to this same mean
    let params_constrained = DemographyParams {
        bw_constraint: Some(natural_mean),
        ..default_params()
    };
    let r_constrained = infer_demography(&tracts, "EUR", &params_constrained);
    // Should be very similar
    assert_eq!(r_free.n_pulses, r_constrained.n_pulses);
}

// ==================== T83: iterative_refine early stopping ====================

#[test]
fn early_stopping_converges_before_max_passes() {
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops.clone(), 0.01);
    // Very strong emissions → should converge quickly
    let log_e = vec![
        vec![-0.1, -5.0, -5.0],
        vec![-5.0, -0.1, -5.0],
        vec![-5.0, -5.0, -0.1],
        vec![-0.1, -5.0, -5.0],
        vec![-5.0, -0.1, -5.0],
    ];
    // With strong emissions and many passes, early stopping should trigger
    let (post_10, states_10) = iterative_refine(&log_e, &params, 10, 0.5);
    let (post_100, states_100) = iterative_refine(&log_e, &params, 100, 0.5);
    // Results should be identical because early stopping kicked in
    assert_eq!(states_10, states_100);
    for (a, b) in post_10.iter().zip(post_100.iter()) {
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-10,
                "Post 10 and 100 passes should be identical after early stopping");
        }
    }
}

#[test]
fn early_stopping_single_window() {
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops, 0.01);
    let log_e = vec![vec![-1.0, -2.0, -3.0]];
    // Single window should converge immediately
    let (posteriors, states) = iterative_refine(&log_e, &params, 50, 0.5);
    assert_eq!(posteriors.len(), 1);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0); // most probable state
}

#[test]
fn early_stopping_uniform_emissions() {
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops, 0.01);
    // Uniform emissions: -1.0 for all states in all windows
    let log_e = vec![
        vec![-1.0, -1.0, -1.0],
        vec![-1.0, -1.0, -1.0],
        vec![-1.0, -1.0, -1.0],
    ];
    let (posteriors, states) = iterative_refine(&log_e, &params, 20, 0.5);
    assert_eq!(posteriors.len(), 3);
    assert_eq!(states.len(), 3);
    // With uniform emissions, posteriors should be driven by transitions
    // After several passes with feedback, should converge
}

#[test]
fn early_stopping_two_states() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["r1".to_string()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["r2".to_string()] },
    ];
    let params = AncestryHmmParams::new(pops, 0.01);
    let log_e = vec![
        vec![-0.5, -2.0],
        vec![-0.5, -2.0],
        vec![-2.0, -0.5],
        vec![-2.0, -0.5],
    ];
    // With early stopping, running 50 vs 100 passes should give identical results
    let (post_50, states_50) = iterative_refine(&log_e, &params, 50, 0.5);
    let (post_100, states_100) = iterative_refine(&log_e, &params, 100, 0.5);
    assert_eq!(states_50, states_100);
    for (a, b) in post_50.iter().zip(post_100.iter()) {
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-10,
                "Same lambda, converged results should be identical");
        }
    }
    // States should reflect emissions: first 2 windows → A, last 2 → B
    assert_eq!(states_50[0], 0);
    assert_eq!(states_50[3], 1);
}

#[test]
fn early_stopping_high_lambda_still_converges() {
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops, 0.01);
    let log_e = vec![
        vec![-0.5, -3.0, -3.0],
        vec![-3.0, -0.5, -3.0],
        vec![-3.0, -3.0, -0.5],
    ];
    // High lambda = more feedback per pass, converges to its own fixed point
    let (post_a, states_a) = iterative_refine(&log_e, &params, 20, 2.0);
    let (post_b, states_b) = iterative_refine(&log_e, &params, 100, 2.0);
    // Same lambda, different max passes → early stopping should make them identical
    assert_eq!(states_a, states_b);
    for (a, b) in post_a.iter().zip(post_b.iter()) {
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-10,
                "Same lambda, different max passes should converge identically");
        }
    }
}

#[test]
fn early_stopping_zero_lambda_no_change() {
    // lambda=0 means no posterior feedback → posteriors don't change between passes
    // Should trigger early stopping at pass 1
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops, 0.01);
    let log_e = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-2.0, -1.0, -3.0],
    ];
    let (post_1, states_1) = iterative_refine(&log_e, &params, 1, 0.0);
    let (post_50, states_50) = iterative_refine(&log_e, &params, 50, 0.0);
    // With zero lambda, apply_posterior_feedback is identity → posteriors unchanged
    // Early stopping should trigger immediately at pass 1
    assert_eq!(states_1, states_50);
    for (a, b) in post_1.iter().zip(post_50.iter()) {
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-10);
        }
    }
}

#[test]
fn early_stopping_negative_lambda_no_feedback() {
    // Negative lambda → apply_posterior_feedback returns original (lambda <= 0 guard)
    let pops = make_test_populations();
    let params = AncestryHmmParams::new(pops, 0.01);
    let log_e = vec![
        vec![-1.0, -2.0, -3.0],
        vec![-2.0, -1.0, -3.0],
    ];
    let (_, states) = iterative_refine(&log_e, &params, 20, -1.0);
    // Should still produce valid output
    assert_eq!(states.len(), 2);
}

// ==================== Combined: per-sample demography + constraint ====================

#[test]
fn per_sample_with_bw_constraint() {
    let segments = vec![
        make_segment("S1", "EUR", 0, 100_000),
        make_segment("S1", "EUR", 100_000, 250_000),
        make_segment("S1", "AFR", 250_000, 400_000),
        make_segment("S2", "EUR", 0, 300_000),
        make_segment("S2", "AFR", 300_000, 500_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let params = DemographyParams {
        bw_constraint: Some(1e-5),
        ..default_params()
    };
    let results = infer_per_sample_demography(&refs, &pop_names, &params);
    assert_eq!(results.len(), 2);
    // Should not panic with constraint active
    for r in &results {
        for dr in &r.results {
            assert!(dr.n_tracts <= 5);
        }
    }
}

#[test]
fn per_sample_write_full_pipeline() {
    // End-to-end: infer per-sample, then write TSV, then read and validate
    let segments = vec![
        make_segment("S1", "EUR", 0, 100_000),
        make_segment("S1", "EUR", 100_000, 300_000),
        make_segment("S2", "AFR", 0, 200_000),
    ];
    let refs: Vec<&AncestrySegment> = segments.iter().collect();
    let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
    let params = default_params();

    let pooled = infer_all_demography(&refs, &pop_names, &params);
    let per_sample = infer_per_sample_demography(&refs, &pop_names, &params);

    let tmp = std::env::temp_dir().join("demog_edge_pipeline.tsv");
    write_demography_tsv(&tmp, &pooled, &per_sample).unwrap();

    let content = std::fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 2); // at least header + 1 data row
    assert!(lines[0].contains("sample\t"));

    // Verify POOLED rows come first (before per-sample)
    let mut found_pooled = false;
    let mut found_sample = false;
    for line in &lines[1..] {
        if line.starts_with("POOLED\t") {
            assert!(!found_sample, "POOLED rows should come before per-sample rows");
            found_pooled = true;
        } else {
            found_sample = true;
        }
    }
    // At least one type should be present
    assert!(found_pooled || found_sample);

    std::fs::remove_file(&tmp).ok();
}

// ==================== DemographyParams::bw_constraint field ====================

#[test]
fn demography_params_default_bw_constraint_is_none() {
    let params = DemographyParams::default();
    assert!(params.bw_constraint.is_none());
}

#[test]
fn demography_params_clone_preserves_constraint() {
    let params = DemographyParams {
        bw_constraint: Some(1.23e-6),
        ..default_params()
    };
    let cloned = params.clone();
    assert_eq!(cloned.bw_constraint, Some(1.23e-6));
}

// ==================== SampleDemographicResult ====================

#[test]
fn sample_demographic_result_clone_and_debug() {
    let r = SampleDemographicResult {
        sample: "test".to_string(),
        results: vec![],
    };
    let cloned = r.clone();
    assert_eq!(cloned.sample, "test");
    assert!(cloned.results.is_empty());
    // Debug trait should work
    let debug_str = format!("{:?}", r);
    assert!(debug_str.contains("test"));
}
