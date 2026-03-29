//! Tests for T52: emission-aware BW transition dampening.
//!
//! Validates that `transition_dampening` properly blends MLE transitions
//! with prior transitions during Baum-Welch M-step. This prevents
//! double-correction when pairwise emissions already bias transitions.

use hprc_ancestry_cli::hmm::{
    AncestryHmmParams, AncestralPopulation, AncestryObservation,
};

fn transition_deviation(trans: &[Vec<f64>], prior: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for (i, row) in trans.iter().enumerate() {
        for (j, &t) in row.iter().enumerate() {
            total += (t - prior[i][j]).abs();
        }
    }
    total
}

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_obs(chrom: &str, start: u64, end: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: chrom.to_string(),
        start,
        end,
        sample: "query".to_string(),
        similarities: sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_3pop_params(switch_prob: f64, dampening: f64) -> AncestryHmmParams {
    let pops = vec![
        make_pop("EUR", &["eur1", "eur2"]),
        make_pop("AFR", &["afr1", "afr2"]),
        make_pop("EAS", &["eas1", "eas2"]),
    ];
    let mut params = AncestryHmmParams::new(pops, switch_prob);
    params.transition_dampening = dampening;
    params
}

fn make_obs_sequence(n: usize, pattern: &str) -> Vec<AncestryObservation> {
    // Generate observations where EUR is dominant for first half, AFR for second
    (0..n).map(|i| {
        let (eur_sim, afr_sim, eas_sim) = match pattern {
            "eur_afr_switch" => {
                if i < n / 2 {
                    (0.99, 0.95, 0.93)
                } else {
                    (0.93, 0.99, 0.93)
                }
            }
            "noisy" => {
                let noise = ((i as f64) * 0.7).sin() * 0.01;
                if i < n / 2 {
                    (0.99 + noise, 0.95 + noise * 0.5, 0.93 + noise * 0.3)
                } else {
                    (0.93 + noise, 0.99 + noise * 0.5, 0.93 + noise * 0.3)
                }
            }
            _ => (0.99, 0.95, 0.93),
        };
        make_obs("chr1", i as u64 * 10000, (i as u64 + 1) * 10000, &[
            ("eur1", eur_sim), ("eur2", eur_sim - 0.005),
            ("afr1", afr_sim), ("afr2", afr_sim - 0.005),
            ("eas1", eas_sim), ("eas2", eas_sim - 0.005),
        ])
    }).collect()
}

#[test]
fn dampening_zero_is_full_mle() {
    // With dampening=0, BW should fully update transitions from data
    let mut params = make_3pop_params(0.01, 0.0);
    let prior_trans = params.transitions.clone();

    let obs = make_obs_sequence(50, "eur_afr_switch");
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let _ll = params.baum_welch(&obs_refs, 5, 1e-6);

    // Transitions should have changed from prior (BW learned something)
    let changed = params.transitions.iter().enumerate().any(|(i, row)| {
        row.iter().enumerate().any(|(j, &t)| (t - prior_trans[i][j]).abs() > 1e-6)
    });
    assert!(changed, "With dampening=0, transitions should change during BW");
}

#[test]
fn dampening_one_freezes_transitions() {
    // With dampening=1.0, transitions should NOT change from prior
    let mut params = make_3pop_params(0.01, 1.0);
    let prior_trans = params.transitions.clone();

    let obs = make_obs_sequence(50, "eur_afr_switch");
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let _ll = params.baum_welch(&obs_refs, 5, 1e-6);

    // Transitions should be essentially unchanged (dampened to prior)
    for i in 0..3 {
        for j in 0..3 {
            let diff = (params.transitions[i][j] - prior_trans[i][j]).abs();
            assert!(diff < 0.01,
                "dampening=1.0 should freeze transitions, but T[{i}][{j}] changed by {diff:.6}");
        }
    }
}

#[test]
fn dampening_half_blends_mle_and_prior() {
    // With dampening=0.5, result should be between full MLE (d=0) and frozen (d=1)
    let obs = make_obs_sequence(50, "eur_afr_switch");
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];

    let mut params_d0 = make_3pop_params(0.01, 0.0);
    let _ll0 = params_d0.baum_welch(&obs_refs, 5, 1e-6);

    let mut params_d05 = make_3pop_params(0.01, 0.5);
    let prior_trans = params_d05.transitions.clone();
    let _ll05 = params_d05.baum_welch(&obs_refs, 5, 1e-6);

    // d=0.5 transitions should deviate from prior LESS than d=0.0
    let deviation_d0 = transition_deviation(&params_d0.transitions, &prior_trans);
    let deviation_d05 = transition_deviation(&params_d05.transitions, &prior_trans);

    assert!(deviation_d05 < deviation_d0,
        "dampening=0.5 should deviate less from prior than dampening=0.0: d05={deviation_d05:.6} >= d0={deviation_d0:.6}");
}

#[test]
fn dampening_monotonic_effect() {
    // Higher dampening → less deviation from prior
    let obs = make_obs_sequence(80, "noisy");
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];

    let prior_trans = make_3pop_params(0.01, 0.0).transitions.clone();

    let dampening_values = [0.0, 0.2, 0.5, 0.8, 1.0];
    let mut deviations = Vec::new();

    for &d in &dampening_values {
        let mut params = make_3pop_params(0.01, d);
        let _ll = params.baum_welch(&obs_refs, 5, 1e-6);

        let deviation = transition_deviation(&params.transitions, &prior_trans);
        deviations.push(deviation);
    }

    for i in 1..deviations.len() {
        assert!(deviations[i] <= deviations[i - 1] + 1e-6,
            "Higher dampening should mean less deviation: d={} gave {:.6} > d={} gave {:.6}",
            dampening_values[i], deviations[i],
            dampening_values[i-1], deviations[i-1]);
    }
}

#[test]
fn dampening_preserves_row_normalization() {
    // Transitions must still sum to 1 per row after dampened BW
    for &d in &[0.0, 0.3, 0.7, 1.0] {
        let mut params = make_3pop_params(0.01, d);
        let obs = make_obs_sequence(50, "eur_afr_switch");
        let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
        let _ll = params.baum_welch(&obs_refs, 3, 1e-6);

        for (i, row) in params.transitions.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "Row {i} should sum to 1.0 after BW with dampening={d}, got {sum}");
        }
    }
}

#[test]
fn dampening_all_positive_transitions() {
    // All transition values must be > 0 after dampened BW
    for &d in &[0.0, 0.5, 1.0] {
        let mut params = make_3pop_params(0.01, d);
        let obs = make_obs_sequence(50, "eur_afr_switch");
        let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
        let _ll = params.baum_welch(&obs_refs, 5, 1e-6);

        for (i, row) in params.transitions.iter().enumerate() {
            for (j, &t) in row.iter().enumerate() {
                assert!(t > 0.0,
                    "T[{i}][{j}] should be positive after BW with dampening={d}, got {t}");
            }
        }
    }
}

#[test]
fn dampening_likelihood_still_improves() {
    // Even with dampening, BW should improve or maintain log-likelihood
    for &d in &[0.0, 0.3, 0.7] {
        let mut params = make_3pop_params(0.01, d);
        let obs = make_obs_sequence(60, "eur_afr_switch");
        let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];

        // Run 1 iteration, check LL
        let ll1 = params.baum_welch(&obs_refs, 1, f64::NEG_INFINITY);
        let ll2 = params.baum_welch(&obs_refs, 1, f64::NEG_INFINITY);

        // LL should not decrease (may plateau with heavy dampening)
        assert!(ll2 >= ll1 - 1e-6,
            "LL should not decrease with dampening={d}: iter1={ll1:.4}, iter2={ll2:.4}");
    }
}

#[test]
fn dampening_baum_welch_full_also_dampens() {
    // baum_welch_full should also respect transition_dampening
    let mut params_d0 = make_3pop_params(0.01, 0.0);
    let mut params_d1 = make_3pop_params(0.01, 1.0);
    let prior_trans = params_d1.transitions.clone();

    let obs = make_obs_sequence(50, "eur_afr_switch");
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];

    let _ll0 = params_d0.baum_welch_full(&obs_refs, 5, 1e-6, false);
    let _ll1 = params_d1.baum_welch_full(&obs_refs, 5, 1e-6, false);

    // d=1.0 transitions should be close to prior
    for i in 0..3 {
        for j in 0..3 {
            let diff = (params_d1.transitions[i][j] - prior_trans[i][j]).abs();
            assert!(diff < 0.01,
                "baum_welch_full with dampening=1.0 should freeze transitions, T[{i}][{j}] diff={diff:.6}");
        }
    }
}

#[test]
fn dampening_two_state_model() {
    // 2-state model enforces symmetry; dampening should still work
    let pops = vec![
        make_pop("EUR", &["eur1", "eur2"]),
        make_pop("AFR", &["afr1", "afr2"]),
    ];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.transition_dampening = 0.8;
    let prior_trans = params.transitions.clone();

    let obs: Vec<AncestryObservation> = (0..40).map(|i| {
        let (eur, afr) = if i < 20 { (0.99, 0.93) } else { (0.93, 0.99) };
        make_obs("chr1", i * 10000, (i + 1) * 10000, &[
            ("eur1", eur), ("eur2", eur - 0.01),
            ("afr1", afr), ("afr2", afr - 0.01),
        ])
    }).collect();
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let _ll = params.baum_welch(&obs_refs, 5, 1e-6);

    // With d=0.8, transitions should barely change for 2-state
    let max_change = transition_deviation(&params.transitions, &prior_trans);
    assert!(max_change < 0.15,
        "2-state with dampening=0.8 should have small total deviation, got={max_change:.6}");
}
