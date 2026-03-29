//! Tier-2 edge case tests for untested ancestry transformation functions:
//! - blend_posteriors_with_emissions (posterior-emission interpolation)
//! - apply_changepoint_prior (state persistence bonus)
//! - local_rerank_emissions (neighborhood rank-based emissions)
//! - parse_population_groups (string spec parsing)
//! - bayesian_shrink_emissions additional edge cases
//!
//! Focus: shape mismatches, NaN/Inf blending, rank logic with ties, parsing failures.

use hprc_ancestry_cli::hmm::{
    blend_posteriors_with_emissions, apply_changepoint_prior,
    local_rerank_emissions, parse_population_groups,
    AncestralPopulation,
};

// ===========================================================================
// Helpers
// ===========================================================================

fn make_pops(names: &[&str]) -> Vec<AncestralPopulation> {
    names.iter().map(|&name| AncestralPopulation {
        name: name.to_string(),
        haplotypes: vec![format!("{}_h1", name)],
    }).collect()
}

// ===========================================================================
// blend_posteriors_with_emissions
// ===========================================================================

#[test]
fn blend_empty_emissions() {
    let result = blend_posteriors_with_emissions(&[], &[], 0.5);
    assert!(result.is_empty());
}

#[test]
fn blend_lambda_zero_returns_emissions() {
    let emissions = vec![vec![-1.0, -2.0], vec![-0.5, -1.5]];
    let posteriors = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.0);
    assert_eq!(result, emissions);
}

#[test]
fn blend_lambda_negative_returns_emissions() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.8, 0.2]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, -0.5);
    assert_eq!(result, emissions);
}

#[test]
fn blend_lambda_one_returns_log_posteriors() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.8, 0.2]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 1.0);
    assert!((result[0][0] - 0.8_f64.ln()).abs() < 1e-10);
    assert!((result[0][1] - 0.2_f64.ln()).abs() < 1e-10);
}

#[test]
fn blend_lambda_one_zero_posterior_gives_neg_inf() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![1.0, 0.0]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 1.0);
    assert!((result[0][0] - 0.0).abs() < 1e-10); // ln(1) = 0
    assert_eq!(result[0][1], f64::NEG_INFINITY); // ln(0) = -inf
}

#[test]
fn blend_interpolation_midpoint() {
    // lambda=0.5 → equal weight to emission and log-posterior
    let emissions = vec![vec![-2.0, -1.0]];
    let posteriors = vec![vec![0.3, 0.7]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.5);
    let expected_0 = 0.5 * (-2.0) + 0.5 * 0.3_f64.ln();
    let expected_1 = 0.5 * (-1.0) + 0.5 * 0.7_f64.ln();
    assert!((result[0][0] - expected_0).abs() < 1e-10);
    assert!((result[0][1] - expected_1).abs() < 1e-10);
}

#[test]
fn blend_emission_neg_inf_uses_log_posterior() {
    // When emission is NEG_INFINITY but posterior is valid → use log_posterior
    let ni = f64::NEG_INFINITY;
    let emissions = vec![vec![ni, -1.0]];
    let posteriors = vec![vec![0.6, 0.4]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.5);
    // e[0] = NEG_INFINITY, p[0] finite → falls to log_p branch
    assert!((result[0][0] - 0.6_f64.ln()).abs() < 1e-10);
}

#[test]
fn blend_both_neg_inf_stays_neg_inf() {
    // When both emission and posterior are non-finite → log_p = NEG_INFINITY
    let ni = f64::NEG_INFINITY;
    let emissions = vec![vec![ni, -1.0]];
    let posteriors = vec![vec![0.0, 0.5]]; // p=0 → log_p = -inf
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.5);
    assert_eq!(result[0][0], ni);
}

#[test]
fn blend_negative_posterior_gives_neg_inf() {
    // Negative posterior → not > 0 → log_p = NEG_INFINITY
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![-0.1, 0.5]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.5);
    // For pop 0: e=-1 finite, log_p=NEG_INF (p<0) → e is finite but log_p isn't → return e
    assert_eq!(result[0][0], -1.0);
}

#[test]
fn blend_large_lambda_near_one() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.9, 0.1]];
    let result = blend_posteriors_with_emissions(&emissions, &posteriors, 0.99);
    // Should be very close to log-posteriors
    assert!((result[0][0] - 0.9_f64.ln()).abs() < 0.1);
}

// ===========================================================================
// apply_changepoint_prior
// ===========================================================================

#[test]
fn changepoint_empty_emissions() {
    let result = apply_changepoint_prior(&[], &[0, 1], 0.5);
    assert!(result.is_empty());
}

#[test]
fn changepoint_empty_states() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_changepoint_prior(&emissions, &[], 0.5);
    assert_eq!(result, emissions);
}

#[test]
fn changepoint_zero_bonus() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_changepoint_prior(&emissions, &[0], 0.0);
    assert_eq!(result, emissions);
}

#[test]
fn changepoint_negative_bonus() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = apply_changepoint_prior(&emissions, &[0], -1.0);
    assert_eq!(result, emissions);
}

#[test]
fn changepoint_applies_bonus_to_decoded_state() {
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let states = vec![1]; // decoded state is 1
    let bonus = 0.5;
    let result = apply_changepoint_prior(&emissions, &states, bonus);
    // State 1 gets bonus
    assert!((result[0][1] - (-2.0 + 0.5)).abs() < 1e-10);
    // Others unchanged
    assert!((result[0][0] - (-1.0)).abs() < 1e-10);
    assert!((result[0][2] - (-3.0)).abs() < 1e-10);
}

#[test]
fn changepoint_multiple_windows() {
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -0.5],
        vec![-2.0, -1.0],
    ];
    let states = vec![0, 1, 0];
    let bonus = 1.0;
    let result = apply_changepoint_prior(&emissions, &states, bonus);
    // Window 0: state 0 gets bonus
    assert!((result[0][0] - (- 1.0 + 1.0)).abs() < 1e-10);
    assert!((result[0][1] - (-2.0)).abs() < 1e-10);
    // Window 1: state 1 gets bonus
    assert!((result[1][0] - (-1.5)).abs() < 1e-10);
    assert!((result[1][1] - (-0.5 + 1.0)).abs() < 1e-10);
}

// ===========================================================================
// local_rerank_emissions
// ===========================================================================

#[test]
fn rerank_empty() {
    let result = local_rerank_emissions(&[], 3);
    assert!(result.is_empty());
}

#[test]
fn rerank_zero_radius() {
    let emissions = vec![vec![-1.0, -2.0]];
    let result = local_rerank_emissions(&emissions, 0);
    assert_eq!(result, emissions);
}

#[test]
fn rerank_single_pop() {
    // k=1 → only one rank → rank_score[0] = ln(1) - ln(1) = 0
    let emissions = vec![vec![-5.0]; 5];
    let result = local_rerank_emissions(&emissions, 2);
    assert_eq!(result.len(), 5);
    for row in &result {
        assert_eq!(row.len(), 1);
        assert!(row[0].is_finite());
    }
}

#[test]
fn rerank_two_pops_clear_winner() {
    // Pop 0 always much higher → rank 0 (best) everywhere
    let emissions = vec![
        vec![-0.1, -5.0],
        vec![-0.2, -4.0],
        vec![-0.1, -6.0],
    ];
    let result = local_rerank_emissions(&emissions, 1);
    // Pop 0 should get best rank score, pop 1 worst rank score
    for row in &result {
        assert!(row[0] > row[1], "pop 0 should rank higher: {} vs {}", row[0], row[1]);
    }
}

#[test]
fn rerank_all_neg_infinity() {
    // All NEG_INFINITY → cumulative gets -1e6 penalty each → all tied
    let ni = f64::NEG_INFINITY;
    let emissions = vec![vec![ni, ni, ni]; 5];
    let result = local_rerank_emissions(&emissions, 2);
    assert_eq!(result.len(), 5);
    for row in &result {
        for &val in row {
            assert!(val.is_finite(), "rerank produced non-finite: {}", val);
        }
    }
}

#[test]
fn rerank_mixed_neg_inf_and_finite() {
    let ni = f64::NEG_INFINITY;
    let emissions = vec![
        vec![-1.0, ni, -0.5],
        vec![-1.0, -2.0, -0.5],
        vec![-1.0, ni, -0.5],
    ];
    let result = local_rerank_emissions(&emissions, 1);
    assert_eq!(result.len(), 3);
    for row in &result {
        for &val in row {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn rerank_radius_larger_than_data() {
    // radius > n → window spans entire data
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-0.5, -1.5],
    ];
    let result = local_rerank_emissions(&emissions, 100);
    assert_eq!(result.len(), 2);
    // Both windows see all data → same ranking
    assert_eq!(result[0], result[1]);
}

#[test]
fn rerank_output_sums_to_one_prob() {
    // Rank scores are designed as log-probabilities → exp should sum to 1
    let emissions: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![-1.0 + i as f64 * 0.1, -2.0, -0.5 - i as f64 * 0.05])
        .collect();
    let result = local_rerank_emissions(&emissions, 3);
    for (t, row) in result.iter().enumerate() {
        let sum_prob: f64 = row.iter().map(|&v| v.exp()).sum();
        assert!(
            (sum_prob - 1.0).abs() < 0.01,
            "rank probs don't sum to 1 at t={}: sum={}", t, sum_prob
        );
    }
}

// ===========================================================================
// parse_population_groups
// ===========================================================================

#[test]
fn parse_groups_simple() {
    let pops = make_pops(&["AFR", "EUR", "EAS"]);
    let result = parse_population_groups("AFR,EUR;EAS", &pops);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}

#[test]
fn parse_groups_single_group() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("AFR,EUR", &pops);
    assert_eq!(result, Some(vec![vec![0, 1]]));
}

#[test]
fn parse_groups_each_separate() {
    let pops = make_pops(&["AFR", "EUR", "EAS"]);
    let result = parse_population_groups("AFR;EUR;EAS", &pops);
    assert_eq!(result, Some(vec![vec![0], vec![1], vec![2]]));
}

#[test]
fn parse_groups_unknown_pop_returns_none() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("AFR,UNKNOWN", &pops);
    assert_eq!(result, None);
}

#[test]
fn parse_groups_empty_spec() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("", &pops);
    // Empty spec → empty group_str → empty name → skip → no groups → Some(empty)
    assert_eq!(result, Some(vec![]));
}

#[test]
fn parse_groups_whitespace_trimmed() {
    let pops = make_pops(&["AFR", "EUR", "EAS"]);
    let result = parse_population_groups("AFR , EUR ; EAS", &pops);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}

#[test]
fn parse_groups_trailing_semicolon() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("AFR;EUR;", &pops);
    // Trailing semicolon → last group_str is "" → empty → no group added
    assert_eq!(result, Some(vec![vec![0], vec![1]]));
}

#[test]
fn parse_groups_trailing_comma() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("AFR,EUR,", &pops);
    // Trailing comma → last name is "" → trimmed → empty → skipped
    assert_eq!(result, Some(vec![vec![0, 1]]));
}

#[test]
fn parse_groups_empty_pops_list() {
    let pops: Vec<AncestralPopulation> = vec![];
    let result = parse_population_groups("AFR", &pops);
    assert_eq!(result, None); // Can't find AFR in empty list
}

#[test]
fn parse_groups_duplicate_pop_in_group() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("AFR,AFR", &pops);
    // Allows duplicates (no dedup check) — indices [0, 0]
    assert_eq!(result, Some(vec![vec![0, 0]]));
}

#[test]
fn parse_groups_case_sensitive() {
    let pops = make_pops(&["AFR", "EUR"]);
    let result = parse_population_groups("afr", &pops);
    assert_eq!(result, None); // case-sensitive match fails
}
