//! Edge case tests for coverage-ratio auxiliary emission features.
//!
//! Tests the coverage-ratio parsing, emission computation, and integration
//! with the full ancestry HMM pipeline (Viterbi, forward-backward, Baum-Welch).

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestryHmmParams, AncestryObservation, AncestralPopulation, EmissionModel,
    parse_similarity_data_with_coverage, coverage_ratio,
    extract_ancestry_segments, estimate_admixture_proportions,
    viterbi, forward_backward, posterior_decode,
};

// ----- Helpers -----

fn make_two_pop() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR_H1".to_string(), "EUR_H2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR_H1".to_string(), "AFR_H2".to_string()],
        },
    ]
}

fn make_three_pop() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["EUR_H1".to_string(), "EUR_H2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["AFR_H1".to_string(), "AFR_H2".to_string()],
        },
        AncestralPopulation {
            name: "EAS".to_string(),
            haplotypes: vec!["EAS_H1".to_string(), "EAS_H2".to_string()],
        },
    ]
}

fn make_obs(
    start: u64,
    sims: &[(&str, f64)],
    covs: Option<&[(&str, f64)]>,
) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 5000,
        sample: "QUERY".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: covs.map(|c| c.iter().map(|(k, v)| (k.to_string(), *v)).collect()),
            haplotype_consistency_bonus: None,
    }
}

fn make_header_with_coverage() -> String {
    "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity".to_string()
}

// =============================================================================
// SECTION 1: coverage_ratio() edge cases
// =============================================================================

#[test]
fn test_coverage_ratio_one_zero() {
    // One length is zero, the other isn't
    assert_eq!(coverage_ratio(0, 500), 0.0);
    assert_eq!(coverage_ratio(500, 0), 0.0);
}

#[test]
fn test_coverage_ratio_both_one() {
    assert!((coverage_ratio(1, 1) - 1.0).abs() < 1e-15);
}

#[test]
fn test_coverage_ratio_large_disparity() {
    // 1 vs 1_000_000
    let r = coverage_ratio(1, 1_000_000);
    assert!((r - 1e-6).abs() < 1e-10);
}

#[test]
fn test_coverage_ratio_u64_max_vs_one() {
    // Extreme: u64::MAX vs 1
    let r = coverage_ratio(1, u64::MAX);
    assert!(r > 0.0);
    assert!(r < 1e-15);
}

#[test]
fn test_coverage_ratio_near_equal() {
    // Very close lengths
    let r = coverage_ratio(999999, 1000000);
    assert!((r - 0.999999).abs() < 1e-6);
}

// =============================================================================
// SECTION 2: parse_similarity_data_with_coverage() edge cases
// =============================================================================

#[test]
fn test_parse_coverage_empty_data() {
    // Header only, no data lines
    let lines = vec![make_header_with_coverage()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q".to_string()],
        &["R".to_string()],
        "estimated.identity",
    )
    .unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_parse_coverage_no_matching_pairs() {
    // Data lines exist but no query/reference match
    let header = make_header_with_coverage();
    let line = "chr1\t1\t5000\tX#1#s:1-5000\tY#1#s:1-5000\t4000\t3000\t2000\t0.7\t0.8\t0.75\t0.90";
    let lines = vec![header, line.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q".to_string()],
        &["R".to_string()],
        "estimated.identity",
    )
    .unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_parse_coverage_reversed_query_ref_order() {
    // group.a is reference, group.b is query (reversed from typical)
    let header = make_header_with_coverage();
    let line = "chr1\t1\t5000\tREF#1#s:1-5000\tQRY#1#s:1-5000\t4000\t5000\t3000\t0.7\t0.8\t0.75\t0.92";
    let lines = vec![header, line.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["QRY#1".to_string()],
        &["REF#1".to_string()],
        "estimated.identity",
    )
    .unwrap();
    assert_eq!(result.len(), 1, "Should match even with reversed order");
    let obs = &result["QRY#1"][0];
    assert!((obs.similarities["REF#1"] - 0.92).abs() < 1e-6);
    let covs = obs.coverage_ratios.as_ref().unwrap();
    assert!((covs["REF#1"] - 4.0 / 5.0).abs() < 1e-6);
}

#[test]
fn test_parse_coverage_multiple_windows_sorted() {
    // Three windows out of order should be sorted by position
    let header = make_header_with_coverage();
    let line1 = "chr1\t20000\t25000\tQ#1#s:1-25000\tR#1#s:1-25000\t4000\t4000\t3000\t0.7\t0.8\t0.75\t0.93";
    let line2 = "chr1\t1\t5000\tQ#1#s:1-5000\tR#1#s:1-5000\t3000\t4000\t2500\t0.6\t0.7\t0.65\t0.88";
    let line3 = "chr1\t10000\t15000\tQ#1#s:1-15000\tR#1#s:1-15000\t5000\t5000\t4500\t0.8\t0.9\t0.85\t0.95";
    let lines = vec![header, line1.to_string(), line2.to_string(), line3.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q#1".to_string()],
        &["R#1".to_string()],
        "estimated.identity",
    )
    .unwrap();
    let obs = &result["Q#1"];
    assert_eq!(obs.len(), 3);
    assert_eq!(obs[0].start, 1);
    assert_eq!(obs[1].start, 10000);
    assert_eq!(obs[2].start, 20000);
}

#[test]
fn test_parse_coverage_zero_lengths() {
    // Both group.a.length and group.b.length are 0
    let header = make_header_with_coverage();
    let line = "chr1\t1\t5000\tQ#1#s:1-5000\tR#1#s:1-5000\t0\t0\t0\t0.0\t0.0\t0.0\t0.50";
    let lines = vec![header, line.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q#1".to_string()],
        &["R#1".to_string()],
        "estimated.identity",
    )
    .unwrap();
    let obs = &result["Q#1"][0];
    let covs = obs.coverage_ratios.as_ref().unwrap();
    assert_eq!(covs["R#1"], 0.0, "Zero lengths should give coverage ratio 0.0");
}

#[test]
fn test_parse_coverage_multiple_refs() {
    // Multiple reference haplotypes in same window
    let header = make_header_with_coverage();
    let line1 = "chr1\t1\t5000\tQ#1#s:1-5000\tEUR_H1#1#s:1-5000\t4000\t3500\t3000\t0.7\t0.8\t0.75\t0.95";
    let line2 = "chr1\t1\t5000\tQ#1#s:1-5000\tEUR_H2#1#s:1-5000\t4000\t4000\t3800\t0.8\t0.9\t0.85\t0.96";
    let line3 = "chr1\t1\t5000\tQ#1#s:1-5000\tAFR_H1#1#s:1-5000\t4000\t2000\t1800\t0.5\t0.6\t0.55\t0.80";
    let lines = vec![header, line1.to_string(), line2.to_string(), line3.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q#1".to_string()],
        &["EUR_H1#1".to_string(), "EUR_H2#1".to_string(), "AFR_H1#1".to_string()],
        "estimated.identity",
    )
    .unwrap();
    let obs = &result["Q#1"][0];
    assert_eq!(obs.similarities.len(), 3, "Should have 3 reference haplotypes");
    let covs = obs.coverage_ratios.as_ref().unwrap();
    assert_eq!(covs.len(), 3, "Should have 3 coverage ratios");
    // EUR_H1: min(4000,3500)/max = 0.875
    assert!((covs["EUR_H1#1"] - 3500.0 / 4000.0).abs() < 1e-6);
    // EUR_H2: 4000/4000 = 1.0
    assert!((covs["EUR_H2#1"] - 1.0).abs() < 1e-6);
    // AFR_H1: 2000/4000 = 0.5
    assert!((covs["AFR_H1#1"] - 0.5).abs() < 1e-6);
}

#[test]
fn test_parse_coverage_missing_sim_column_errors() {
    // Ask for a column that doesn't exist
    let header = make_header_with_coverage();
    let lines = vec![header];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q".to_string()],
        &["R".to_string()],
        "nonexistent.column",
    );
    assert!(result.is_err(), "Should error on missing similarity column");
}

#[test]
fn test_parse_coverage_jaccard_column() {
    // Use jaccard.similarity instead of estimated.identity
    let header = make_header_with_coverage();
    let line = "chr1\t1\t5000\tQ#1#s:1-5000\tR#1#s:1-5000\t4000\t3000\t2500\t0.70\t0.80\t0.75\t0.90";
    let lines = vec![header, line.to_string()];
    let result = parse_similarity_data_with_coverage(
        lines.into_iter(),
        &["Q#1".to_string()],
        &["R#1".to_string()],
        "jaccard.similarity",
    )
    .unwrap();
    let obs = &result["Q#1"][0];
    assert!((obs.similarities["R#1"] - 0.70).abs() < 1e-6, "Should parse jaccard column");
}

// =============================================================================
// SECTION 3: log_emission_with_coverage() edge cases
// =============================================================================

#[test]
fn test_coverage_emission_empty_coverage_map() {
    // coverage_ratios = Some(empty HashMap)
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 5000,
        sample: "Q".to_string(),
        similarities: [
            ("EUR_H1".to_string(), 0.95),
            ("EUR_H2".to_string(), 0.94),
            ("AFR_H1".to_string(), 0.85),
            ("AFR_H2".to_string(), 0.84),
        ]
        .into_iter()
        .collect(),
        coverage_ratios: Some(HashMap::new()), // Empty map
            haplotype_consistency_bonus: None,
    };

    // Should fall back to similarity-only because coverage map is empty
    let em_with = params.log_emission(&obs, 0);
    params.set_coverage_weight(0.0);
    let em_without = params.log_emission(&obs, 0);
    assert!(
        (em_with - em_without).abs() < 1e-10,
        "Empty coverage map should fall back: {} vs {}",
        em_with,
        em_without
    );
}

#[test]
fn test_coverage_emission_partial_coverage_data() {
    // Coverage data for only one population's haplotypes, not the other
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.95),
            ("EUR_H2", 0.94),
            ("AFR_H1", 0.85),
            ("AFR_H2", 0.84),
        ],
        Some(&[
            ("EUR_H1", 0.98),
            ("EUR_H2", 0.97),
            // No AFR coverage data
        ]),
    );

    // Should still produce valid emissions
    let em0 = params.log_emission(&obs, 0);
    let em1 = params.log_emission(&obs, 1);
    assert!(em0.is_finite(), "State 0 emission should be finite");
    assert!(em1.is_finite(), "State 1 emission should be finite");
    assert!(em0 <= 0.0);
    assert!(em1 <= 0.0);
}

#[test]
fn test_coverage_emission_all_zero_coverage() {
    // All coverage ratios are 0.0
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.95),
            ("EUR_H2", 0.94),
            ("AFR_H1", 0.85),
            ("AFR_H2", 0.84),
        ],
        Some(&[
            ("EUR_H1", 0.0),
            ("EUR_H2", 0.0),
            ("AFR_H1", 0.0),
            ("AFR_H2", 0.0),
        ]),
    );

    // Zero coverage should fall back (log_emission_with_coverage checks for > 0)
    let em = params.log_emission(&obs, 0);
    assert!(em.is_finite(), "Zero coverage should still produce finite emission");
}

#[test]
fn test_coverage_emission_identical_coverage_all_pops() {
    // All populations have the same coverage ratio
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops.clone(), 0.001);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.95),
            ("EUR_H2", 0.94),
            ("AFR_H1", 0.85),
            ("AFR_H2", 0.84),
        ],
        Some(&[
            ("EUR_H1", 0.90),
            ("EUR_H2", 0.90),
            ("AFR_H1", 0.90),
            ("AFR_H2", 0.90),
        ]),
    );

    // Identical coverage across populations shouldn't change relative rankings
    let mut params_no = AncestryHmmParams::new(pops, 0.001);
    params_no.set_coverage_weight(0.0);

    let diff_with = params.log_emission(&obs, 0) - params.log_emission(&obs, 1);
    let diff_without = params_no.log_emission(&obs, 0) - params_no.log_emission(&obs, 1);

    // Equal coverage means coverage softmax is equal → no relative change
    assert!(
        (diff_with - diff_without).abs() < 1e-6,
        "Equal coverage should not change relative advantage: {:.6} vs {:.6}",
        diff_with,
        diff_without
    );
}

#[test]
fn test_coverage_emission_very_large_weight() {
    // Very large coverage weight should amplify the coverage signal
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(100.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.90),
            ("EUR_H2", 0.89),
            ("AFR_H1", 0.91), // AFR has higher similarity
            ("AFR_H2", 0.90),
        ],
        Some(&[
            ("EUR_H1", 0.99), // EUR has much higher coverage
            ("EUR_H2", 0.98),
            ("AFR_H1", 0.50),
            ("AFR_H2", 0.50),
        ]),
    );

    // With huge coverage weight, coverage should dominate even though AFR has higher sim
    let em0 = params.log_emission(&obs, 0);
    let em1 = params.log_emission(&obs, 1);
    assert!(em0.is_finite());
    assert!(em1.is_finite());
    // EUR (state 0) should be favored by coverage despite lower similarity
    assert!(
        em0 > em1,
        "High coverage weight should make EUR (high cov) dominate AFR (high sim): EUR={:.4} AFR={:.4}",
        em0, em1
    );
}

#[test]
fn test_coverage_emission_negative_weight_not_panic() {
    // Negative weight (unusual but shouldn't crash)
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(-1.0);

    let obs = make_obs(
        0,
        &[("EUR_H1", 0.95), ("EUR_H2", 0.94), ("AFR_H1", 0.85), ("AFR_H2", 0.84)],
        Some(&[("EUR_H1", 0.90), ("EUR_H2", 0.90), ("AFR_H1", 0.80), ("AFR_H2", 0.80)]),
    );

    // Negative weight shouldn't activate coverage (coverage_weight > 0 check)
    let em = params.log_emission(&obs, 0);
    assert!(em.is_finite(), "Negative weight should fall back to sim-only");
}

#[test]
fn test_coverage_emission_three_populations() {
    // Three-population case: coverage advantages differ per population
    let pops = make_three_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.90), ("EUR_H2", 0.89),
            ("AFR_H1", 0.90), ("AFR_H2", 0.89),
            ("EAS_H1", 0.90), ("EAS_H2", 0.89),
        ],
        Some(&[
            ("EUR_H1", 0.98), ("EUR_H2", 0.97), // EUR has highest coverage
            ("AFR_H1", 0.70), ("AFR_H2", 0.69),
            ("EAS_H1", 0.50), ("EAS_H2", 0.49),
        ]),
    );

    // With equal similarities, coverage should determine ranking
    let em_eur = params.log_emission(&obs, 0);
    let em_afr = params.log_emission(&obs, 1);
    let em_eas = params.log_emission(&obs, 2);
    assert!(em_eur > em_afr, "EUR should rank above AFR (higher cov)");
    assert!(em_afr > em_eas, "AFR should rank above EAS (higher cov)");
}

// =============================================================================
// SECTION 4: HMM pipeline with coverage (Viterbi, FB, posterior decode)
// =============================================================================

#[test]
fn test_viterbi_with_coverage_empty_observations() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs: Vec<AncestryObservation> = vec![];
    let states = viterbi(&obs, &params);
    assert!(states.is_empty());
}

#[test]
fn test_viterbi_with_coverage_single_observation() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = vec![make_obs(
        0,
        &[("EUR_H1", 0.95), ("EUR_H2", 0.94), ("AFR_H1", 0.85), ("AFR_H2", 0.84)],
        Some(&[("EUR_H1", 0.98), ("EUR_H2", 0.97), ("AFR_H1", 0.70), ("AFR_H2", 0.69)]),
    )];

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 2, "State should be 0 or 1");
}

#[test]
fn test_viterbi_coverage_flips_assignment() {
    // Similarity favors AFR, but coverage strongly favors EUR
    let pops = make_two_pop();

    // Without coverage: AFR should win
    let mut params_no_cov = AncestryHmmParams::new(pops.clone(), 0.001);
    params_no_cov.set_coverage_weight(0.0);

    // With coverage: EUR should win due to strong coverage signal
    let mut params_cov = AncestryHmmParams::new(pops, 0.001);
    params_cov.set_coverage_weight(5.0);

    let obs: Vec<AncestryObservation> = (0..10)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.88),
                    ("EUR_H2", 0.87),
                    ("AFR_H1", 0.92), // AFR has higher similarity
                    ("AFR_H2", 0.91),
                ],
                Some(&[
                    ("EUR_H1", 0.99), // EUR has much higher coverage
                    ("EUR_H2", 0.98),
                    ("AFR_H1", 0.40),
                    ("AFR_H2", 0.39),
                ]),
            )
        })
        .collect();

    let states_no_cov = viterbi(&obs, &params_no_cov);
    let states_cov = viterbi(&obs, &params_cov);

    // Without coverage: mostly AFR (state 1)
    let afr_count_no_cov = states_no_cov.iter().filter(|&&s| s == 1).count();
    assert!(
        afr_count_no_cov >= 8,
        "Without coverage, AFR should dominate: {}/10",
        afr_count_no_cov
    );

    // With coverage: mostly EUR (state 0)
    let eur_count_cov = states_cov.iter().filter(|&&s| s == 0).count();
    assert!(
        eur_count_cov >= 8,
        "With strong coverage weight, EUR should dominate: {}/10",
        eur_count_cov
    );
}

#[test]
fn test_forward_backward_coverage_posteriors_valid() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.90),
                    ("EUR_H2", 0.88),
                    ("AFR_H1", 0.80),
                    ("AFR_H2", 0.78),
                ]),
            )
        })
        .collect();

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 20);

    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 2, "Window {}: should have 2 posteriors", t);
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Window {} posteriors should sum to 1: {:.10}",
            t,
            sum
        );
        for &p in post {
            assert!(p >= 0.0 && p <= 1.0, "Posterior out of range at window {}: {}", t, p);
        }
    }
}

#[test]
fn test_posterior_decode_with_coverage() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs: Vec<AncestryObservation> = (0..10)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.93),
                    ("AFR_H1", 0.70),
                    ("AFR_H2", 0.68),
                ]),
            )
        })
        .collect();

    let states = posterior_decode(&obs, &params);
    assert_eq!(states.len(), 10);
    for &s in &states {
        assert!(s < 2, "Posterior-decoded state should be valid: {}", s);
    }
    // EUR should dominate (higher sim + higher coverage)
    let eur_count = states.iter().filter(|&&s| s == 0).count();
    assert!(eur_count >= 8, "EUR should dominate: {}/10", eur_count);
}

#[test]
fn test_forward_backward_coverage_mixed_ancestry() {
    // Create a mixed-ancestry sequence: EUR for first half, AFR for second
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.01); // Higher switch prob
    params.set_coverage_weight(1.0);

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| {
            if w < 10 {
                // EUR ancestry: EUR has higher sim + higher cov
                make_obs(
                    w * 5000,
                    &[
                        ("EUR_H1", 0.96),
                        ("EUR_H2", 0.95),
                        ("AFR_H1", 0.80),
                        ("AFR_H2", 0.79),
                    ],
                    Some(&[
                        ("EUR_H1", 0.97),
                        ("EUR_H2", 0.96),
                        ("AFR_H1", 0.60),
                        ("AFR_H2", 0.58),
                    ]),
                )
            } else {
                // AFR ancestry: AFR has higher sim + higher cov
                make_obs(
                    w * 5000,
                    &[
                        ("EUR_H1", 0.80),
                        ("EUR_H2", 0.79),
                        ("AFR_H1", 0.96),
                        ("AFR_H2", 0.95),
                    ],
                    Some(&[
                        ("EUR_H1", 0.60),
                        ("EUR_H2", 0.58),
                        ("AFR_H1", 0.97),
                        ("AFR_H2", 0.96),
                    ]),
                )
            }
        })
        .collect();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 20);

    // First half should be EUR (state 0)
    let eur_first_half = states[..10].iter().filter(|&&s| s == 0).count();
    assert!(eur_first_half >= 8, "First half should be EUR: {}/10", eur_first_half);

    // Second half should be AFR (state 1)
    let afr_second_half = states[10..].iter().filter(|&&s| s == 1).count();
    assert!(afr_second_half >= 8, "Second half should be AFR: {}/10", afr_second_half);
}

// =============================================================================
// SECTION 5: Segment extraction & admixture with coverage
// =============================================================================

#[test]
fn test_extract_segments_with_coverage_observations() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    // 5 windows all EUR
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.98),
                    ("EUR_H2", 0.97),
                    ("AFR_H1", 0.70),
                    ("AFR_H2", 0.69),
                ]),
            )
        })
        .collect();

    let states = viterbi(&obs, &params);
    let posteriors = forward_backward(&obs, &params);
    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));

    assert!(!segments.is_empty(), "Should produce at least one segment");
    for seg in &segments {
        assert!(seg.lod_score.is_finite(), "LOD should be finite");
        assert!(seg.discriminability >= 0.0, "Discriminability should be non-negative");
        if let Some(mp) = seg.mean_posterior {
            assert!(mp >= 0.0 && mp <= 1.0, "Mean posterior should be in [0,1]");
        }
    }
}

#[test]
fn test_admixture_proportions_with_coverage() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_coverage_weight(1.0);

    // Mixed ancestry: 60% EUR, 40% AFR
    let obs: Vec<AncestryObservation> = (0..10)
        .map(|w| {
            if w < 6 {
                make_obs(
                    w * 5000,
                    &[("EUR_H1", 0.96), ("EUR_H2", 0.95), ("AFR_H1", 0.80), ("AFR_H2", 0.79)],
                    Some(&[("EUR_H1", 0.97), ("EUR_H2", 0.96), ("AFR_H1", 0.60), ("AFR_H2", 0.58)]),
                )
            } else {
                make_obs(
                    w * 5000,
                    &[("EUR_H1", 0.80), ("EUR_H2", 0.79), ("AFR_H1", 0.96), ("AFR_H2", 0.95)],
                    Some(&[("EUR_H1", 0.60), ("EUR_H2", 0.58), ("AFR_H1", 0.97), ("AFR_H2", 0.96)]),
                )
            }
        })
        .collect();

    let states = viterbi(&obs, &params);
    let posteriors = forward_backward(&obs, &params);
    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
    let pop_names: Vec<String> = vec!["EUR".to_string(), "AFR".to_string()];
    let admix = estimate_admixture_proportions(&segments, "QUERY", &pop_names);

    // Proportions should sum to 1
    let sum: f64 = admix.proportions.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Admixture proportions should sum to 1.0: {}",
        sum
    );
}

// =============================================================================
// SECTION 6: Coverage with other features (emission model, normalization)
// =============================================================================

#[test]
fn test_coverage_with_mean_emission_model() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_emission_model(EmissionModel::Mean);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.95),
            ("EUR_H2", 0.90), // large spread
            ("AFR_H1", 0.85),
            ("AFR_H2", 0.80),
        ],
        Some(&[
            ("EUR_H1", 0.98),
            ("EUR_H2", 0.97),
            ("AFR_H1", 0.70),
            ("AFR_H2", 0.69),
        ]),
    );

    let em0 = params.log_emission(&obs, 0);
    let em1 = params.log_emission(&obs, 1);
    assert!(em0.is_finite());
    assert!(em1.is_finite());
    // EUR should still be favored (both sim and coverage favor it)
    assert!(em0 > em1, "EUR should be favored: {} > {}", em0, em1);
}

#[test]
fn test_coverage_with_topk_emission_model() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_emission_model(EmissionModel::TopK(1));
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.95),
            ("EUR_H2", 0.90),
            ("AFR_H1", 0.85),
            ("AFR_H2", 0.80),
        ],
        Some(&[
            ("EUR_H1", 0.98),
            ("EUR_H2", 0.97),
            ("AFR_H1", 0.70),
            ("AFR_H2", 0.69),
        ]),
    );

    // TopK(1) should be equivalent to Max
    let em0 = params.log_emission(&obs, 0);
    assert!(em0.is_finite());
    assert!(em0 <= 0.0);
}

#[test]
fn test_coverage_with_normalization_active() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    // Build observations for normalization learning
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95 + (w as f64 * 0.001)),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.90),
                    ("EUR_H2", 0.89),
                    ("AFR_H1", 0.80),
                    ("AFR_H2", 0.79),
                ]),
            )
        })
        .collect();

    // Learn normalization
    params.learn_normalization(&obs);

    // Emissions should still be valid after normalization
    for state in 0..2 {
        let em = params.log_emission(&obs[0], state);
        assert!(em.is_finite(), "State {} emission should be finite after normalization", state);
    }

    // Forward-backward should work
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 20);
    for post in &posteriors {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Posteriors should sum to 1 with normalization + coverage"
        );
    }
}

// =============================================================================
// SECTION 7: Baum-Welch with coverage
// =============================================================================

#[test]
fn test_baum_welch_with_coverage_does_not_crash() {
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    let obs: Vec<AncestryObservation> = (0..30)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.90),
                    ("EUR_H2", 0.88),
                    ("AFR_H1", 0.75),
                    ("AFR_H2", 0.73),
                ]),
            )
        })
        .collect();

    let all_obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];
    let ll = params.baum_welch(&all_obs_slices, 3, 1e-4);
    assert!(ll.is_finite(), "Baum-Welch log-likelihood should be finite with coverage");
}

#[test]
fn test_baum_welch_with_coverage_produces_finite_ll() {
    // Verify BW with coverage produces finite, reasonable log-likelihood
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    let obs: Vec<AncestryObservation> = (0..50)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.90 + (w as f64 % 10.0) * 0.005),
                    ("EUR_H2", 0.89),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.90),
                    ("EUR_H2", 0.88),
                    ("AFR_H1", 0.75),
                    ("AFR_H2", 0.73),
                ]),
            )
        })
        .collect();

    let all_obs_slices: Vec<&[AncestryObservation]> = vec![obs.as_slice()];
    let ll = params.baum_welch(&all_obs_slices, 5, 1e-10);

    assert!(ll.is_finite(), "BW LL should be finite: {}", ll);
    assert!(ll < 0.0, "BW LL should be negative (log-likelihood): {}", ll);

    // After training, params should still be valid
    assert!(params.emission_std > 0.0, "Temperature should remain positive");
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Transition row should sum to 1: {}", sum);
    }
}

// =============================================================================
// SECTION 8: Numerical stability edge cases
// =============================================================================

#[test]
fn test_coverage_near_one_everywhere() {
    // All coverage ratios are very close to 1.0
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs = make_obs(
        0,
        &[
            ("EUR_H1", 0.999),
            ("EUR_H2", 0.998),
            ("AFR_H1", 0.997),
            ("AFR_H2", 0.996),
        ],
        Some(&[
            ("EUR_H1", 0.9999),
            ("EUR_H2", 0.9998),
            ("AFR_H1", 0.9997),
            ("AFR_H2", 0.9996),
        ]),
    );

    for state in 0..2 {
        let em = params.log_emission(&obs, state);
        assert!(em.is_finite(), "Near-1 coverage should not produce NaN/Inf: {}", em);
        assert!(em <= 0.0);
    }
}

#[test]
fn test_coverage_emission_long_sequence_stability() {
    // 1000 windows with coverage enabled
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(0.5);

    let obs: Vec<AncestryObservation> = (0..1000)
        .map(|w| {
            make_obs(
                w * 5000,
                &[
                    ("EUR_H1", 0.95),
                    ("EUR_H2", 0.94),
                    ("AFR_H1", 0.85),
                    ("AFR_H2", 0.84),
                ],
                Some(&[
                    ("EUR_H1", 0.90),
                    ("EUR_H2", 0.88),
                    ("AFR_H1", 0.75),
                    ("AFR_H2", 0.73),
                ]),
            )
        })
        .collect();

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 1000);

    // Check all posteriors valid (no NaN/Inf after 1000 windows)
    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Window {} posteriors diverged: sum={}",
            t,
            sum
        );
        for &p in post {
            assert!(
                p.is_finite() && p >= 0.0,
                "Window {} has invalid posterior: {}",
                t,
                p
            );
        }
    }
}

#[test]
fn test_coverage_mixed_none_and_some_observations() {
    // Mix of observations with and without coverage data in same sequence
    let pops = make_two_pop();
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_coverage_weight(1.0);

    let obs: Vec<AncestryObservation> = (0..10)
        .map(|w| {
            if w % 2 == 0 {
                // Even windows: with coverage
                make_obs(
                    w * 5000,
                    &[
                        ("EUR_H1", 0.95),
                        ("EUR_H2", 0.94),
                        ("AFR_H1", 0.85),
                        ("AFR_H2", 0.84),
                    ],
                    Some(&[
                        ("EUR_H1", 0.90),
                        ("EUR_H2", 0.88),
                        ("AFR_H1", 0.70),
                        ("AFR_H2", 0.68),
                    ]),
                )
            } else {
                // Odd windows: no coverage
                make_obs(
                    w * 5000,
                    &[
                        ("EUR_H1", 0.95),
                        ("EUR_H2", 0.94),
                        ("AFR_H1", 0.85),
                        ("AFR_H2", 0.84),
                    ],
                    None,
                )
            }
        })
        .collect();

    // Pipeline should handle mixed coverage/no-coverage gracefully
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 10);

    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 10);

    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Window {} posteriors should sum to 1 with mixed coverage: {}",
            t,
            sum
        );
    }
}
