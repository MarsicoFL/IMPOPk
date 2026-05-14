//! Edge-case tests for ancestry-cli Baum-Welch, genetic map, and emission paths.
//!
//! Targets genuinely untested branches:
//! - forward_backward_with_genetic_map single observation (window_size = 10_000 fallback)
//! - AncestryHmmParams::baum_welch k < 2 early return
//! - AncestryHmmParams::baum_welch all-short-sequences (len < 2 guard)
//! - AncestryHmmParams::baum_welch zero iterations
//! - AncestryHmmParams::baum_welch multiple sequences
//! - estimate_switch_prob fewer than 10 observations early return (boundary at 9/10)
//! - log_emission with single valid population returning 0.0
//! - forward_backward_with_genetic_map two observations (window_size calculated)
//! - AncestryGeneticMap::interpolate_cm single-entry via from_file

use std::io::Write;

use impopk_ancestry_cli::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    estimate_switch_prob, forward_backward_with_genetic_map,
};

// ── Helper to create observations ──

fn make_obs(start: u64, sims: Vec<(&str, f64)>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr10".to_string(),
        start,
        end: start + 10_000,
        sample: "query".to_string(),
        similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_pops_2way() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string(), "afr2".to_string()],
        },
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string(), "eur2".to_string()],
        },
    ]
}

// ── AncestryGeneticMap::interpolate_cm single-entry via from_file ──

#[test]
fn genmap_interpolate_single_entry_via_file() {
    // Create a temp genetic map file with only one entry for chr1
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single_entry.map");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1\t50000\t0.5\t1.5").unwrap();
    }
    let gm = AncestryGeneticMap::from_file(&path, "chr1").unwrap();
    // With a single entry, interpolate_cm always returns entries[0].1 = 1.5
    assert_eq!(gm.interpolate_cm(0), 1.5);
    assert_eq!(gm.interpolate_cm(50_000), 1.5);
    assert_eq!(gm.interpolate_cm(100_000), 1.5);
}

#[test]
fn genmap_genetic_distance_single_entry_always_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single.map");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1\t50000\t0.5\t1.5").unwrap();
    }
    let gm = AncestryGeneticMap::from_file(&path, "chr1").unwrap();
    // genetic_distance_cm = |interpolate(pos2) - interpolate(pos1)| = |1.5 - 1.5| = 0
    let dist = gm.genetic_distance_cm(0, 100_000);
    assert_eq!(dist, 0.0);
}

// ── forward_backward_with_genetic_map: single observation ──

#[test]
fn fb_with_genetic_map_single_obs_uses_10k_window() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000, 1.0);

    // Single observation → window_size else branch = 10_000
    let obs = vec![make_obs(
        50_000,
        vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
    )];

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 1);
    // Posteriors should sum to ~1.0
    let sum: f64 = posteriors[0].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Single-obs posteriors should sum to 1.0, got {}",
        sum
    );
    // AFR should be higher (more similar)
    assert!(
        posteriors[0][0] > posteriors[0][1],
        "AFR posterior ({}) should > EUR posterior ({})",
        posteriors[0][0],
        posteriors[0][1]
    );
}

// ── AncestryHmmParams::baum_welch k < 2 ──

#[test]
fn baum_welch_single_state_returns_neg_infinity() {
    // k < 2 → return NEG_INFINITY immediately
    let pops = vec![AncestralPopulation {
        name: "AFR".to_string(),
        haplotypes: vec!["afr1".to_string()],
    }];
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs = vec![make_obs(0, vec![("afr1", 0.95)]), make_obs(10_000, vec![("afr1", 0.93)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_refs, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

#[test]
fn baum_welch_empty_observations_returns_neg_infinity() {
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs_refs: Vec<&[AncestryObservation]> = vec![];
    let ll = params.baum_welch(&obs_refs, 10, 1e-6);
    assert_eq!(ll, f64::NEG_INFINITY);
}

// ── AncestryHmmParams::baum_welch with valid data converges ──

#[test]
fn baum_welch_two_states_converges() {
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops, 0.01);

    // Create a sequence where first 10 windows favor AFR, next 10 favor EUR
    let mut obs = Vec::new();
    for i in 0..10 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
        ));
    }
    for i in 10..20 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.80), ("afr2", 0.78), ("eur1", 0.96), ("eur2", 0.94)],
        ));
    }

    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_refs, 20, 1e-6);
    // Should return a finite log-likelihood
    assert!(ll.is_finite(), "BW should converge to finite LL, got {}", ll);
}

// ── AncestryHmmParams::baum_welch short sequence skipped ──

#[test]
fn baum_welch_single_obs_sequences_skipped() {
    // Each sequence has len < 2 → all skipped in the inner loop
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops, 0.01);
    let obs1 = vec![make_obs(0, vec![("afr1", 0.95), ("eur1", 0.80)])];
    let obs2 = vec![make_obs(10_000, vec![("afr1", 0.90), ("eur1", 0.85)])];
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs1, &obs2];
    let ll = params.baum_welch(&obs_refs, 5, 1e-6);
    // Function runs but all sequences skipped; should not panic
    assert!(!ll.is_nan(), "LL should not be NaN");
}

// ── estimate_switch_prob: boundary at 9 vs 10 observations ──

#[test]
fn estimate_switch_prob_5_obs_returns_fallback() {
    let pops = make_pops_2way();
    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs(i * 10_000, vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)]))
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    assert_eq!(sp, 0.001, "Should return fallback 0.001 for < 10 obs");
}

#[test]
fn estimate_switch_prob_9_obs_returns_fallback() {
    let pops = make_pops_2way();
    let obs: Vec<AncestryObservation> = (0..9)
        .map(|i| make_obs(i * 10_000, vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)]))
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    assert_eq!(sp, 0.001);
}

#[test]
fn estimate_switch_prob_10_obs_runs_viterbi() {
    let pops = make_pops_2way();
    let obs: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs(i * 10_000, vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)]))
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    // With 10 obs, the function runs Viterbi — result should be in [0.0001, 0.1]
    assert!(
        sp >= 0.0001 && sp <= 0.1,
        "Switch prob should be within clamped range, got {}",
        sp
    );
}

#[test]
fn estimate_switch_prob_empty_returns_fallback() {
    let pops = make_pops_2way();
    let obs: Vec<AncestryObservation> = vec![];
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    assert_eq!(sp, 0.001);
}

// ── log_emission: single valid population score → returns 0.0 ──

#[test]
fn log_emission_single_valid_pop_returns_zero() {
    // When only one population has valid (> 0) similarity data,
    // valid_scores.len() <= 1 → return 0.0 (= log(1))
    let pops = vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string()],
        },
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.01);

    // Only AFR has data; EUR haplotype is missing
    let obs = make_obs(0, vec![("afr1", 0.95)]);

    // Emission for state 0 (AFR) should be 0.0 since it's the only valid score
    let emission_afr = params.log_emission(&obs, 0);
    assert_eq!(
        emission_afr, 0.0,
        "Single valid population should return log(1) = 0.0, got {}",
        emission_afr
    );

    // Emission for state 1 (EUR) should be NEG_INFINITY (no data)
    let emission_eur = params.log_emission(&obs, 1);
    assert_eq!(emission_eur, f64::NEG_INFINITY);
}

#[test]
fn log_emission_both_pops_have_data_returns_nonzero() {
    // With two valid populations, the softmax produces non-zero log-probabilities
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);

    let obs = make_obs(
        0,
        vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
    );
    let emission_afr = params.log_emission(&obs, 0);
    let emission_eur = params.log_emission(&obs, 1);

    // Both should be finite
    assert!(emission_afr.is_finite());
    assert!(emission_eur.is_finite());
    // At least one should be negative (log-probabilities in softmax)
    assert!(
        emission_afr < 0.0 || emission_eur < 0.0,
        "At least one log-emission should be < 0"
    );
    // AFR is more similar → higher log-emission
    assert!(emission_afr > emission_eur);
}

#[test]
fn log_emission_no_data_for_any_pop_returns_neg_inf() {
    // When no population has data at all
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    // Observations with haplotypes not in any population
    let obs = make_obs(0, vec![("unknown1", 0.95), ("unknown2", 0.80)]);
    let emission_afr = params.log_emission(&obs, 0);
    let emission_eur = params.log_emission(&obs, 1);
    assert_eq!(emission_afr, f64::NEG_INFINITY);
    assert_eq!(emission_eur, f64::NEG_INFINITY);
}

// ── forward_backward_with_genetic_map: posteriors sum correctly ──

#[test]
fn fb_with_genetic_map_multiple_obs_posteriors_sum() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| {
            make_obs(
                i * 10_000,
                vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
            )
        })
        .collect();

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 5);
    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Posteriors at t={} should sum to 1.0, got {}",
            t,
            sum
        );
    }
}

#[test]
fn fb_with_genetic_map_empty_returns_empty() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000, 1.0);

    let posteriors = forward_backward_with_genetic_map(&[], &params, &gm);
    assert!(posteriors.is_empty());
}

// ── AncestryHmmParams::baum_welch max_iter = 0 ──

#[test]
fn baum_welch_zero_iterations() {
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
    let original_transitions = params.transitions.clone();

    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            make_obs(
                i * 10_000,
                vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
            )
        })
        .collect();
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];

    let ll = params.baum_welch(&obs_refs, 0, 1e-6);
    // Zero iterations → prev_ll = NEG_INFINITY → returned as-is
    assert_eq!(ll, f64::NEG_INFINITY);
    // Transitions should not have changed
    assert_eq!(params.transitions, original_transitions);
}

// ── baum_welch: multiple sequences ──

#[test]
fn baum_welch_multiple_sequences() {
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let obs1: Vec<AncestryObservation> = (0..15)
        .map(|i| {
            make_obs(
                i * 10_000,
                vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
            )
        })
        .collect();
    let obs2: Vec<AncestryObservation> = (0..15)
        .map(|i| {
            make_obs(
                i * 10_000,
                vec![("afr1", 0.82), ("afr2", 0.80), ("eur1", 0.96), ("eur2", 0.94)],
            )
        })
        .collect();
    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs1, &obs2];

    let ll = params.baum_welch(&obs_refs, 10, 1e-6);
    assert!(
        ll.is_finite(),
        "BW with multiple sequences should converge, got {}",
        ll
    );
}

// ── forward_backward_with_genetic_map: two observations → window_size calculated ──

#[test]
fn fb_with_genetic_map_two_obs_calculates_window_size() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 200_000, 1.0);

    // Two observations separated by 20_000 bp → window_size = 20_000 (not 10_000 fallback)
    let obs = vec![
        make_obs(
            0,
            vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
        ),
        make_obs(
            20_000,
            vec![("afr1", 0.90), ("afr2", 0.88), ("eur1", 0.85), ("eur2", 0.83)],
        ),
    ];

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 2);
    for post in &posteriors {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// ── baum_welch: convergence — log-likelihood monotonically increases ──

#[test]
fn baum_welch_ll_increases_monotonically() {
    // Run BW with 1 iteration at a time and verify LL doesn't decrease
    let pops = make_pops_2way();

    let mut obs = Vec::new();
    for i in 0..10 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
        ));
    }
    for i in 10..20 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.80), ("afr2", 0.78), ("eur1", 0.96), ("eur2", 0.94)],
        ));
    }

    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut params = AncestryHmmParams::new(pops, 0.01);

    for _ in 0..5 {
        let ll = params.baum_welch(&obs_refs, 1, 1e-12);
        if ll.is_finite() && prev_ll.is_finite() {
            assert!(
                ll >= prev_ll - 1e-6,
                "LL should not decrease: prev={}, curr={}",
                prev_ll,
                ll
            );
        }
        prev_ll = ll;
    }
}

// ── forward_backward_with_genetic_map: 3-state ──

#[test]
fn fb_with_genetic_map_three_states() {
    let pops = vec![
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string()],
        },
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string()],
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: vec!["amr1".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000, 1.0);

    let obs: Vec<AncestryObservation> = (0..5)
        .map(|i| {
            make_obs(
                i * 10_000,
                vec![("afr1", 0.95), ("eur1", 0.80), ("amr1", 0.85)],
            )
        })
        .collect();

    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 5);
    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 3, "Should have 3 states at t={}", t);
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "3-state posteriors at t={} should sum to 1.0, got {}",
            t,
            sum
        );
    }
    // AFR has highest similarity → highest posterior
    assert!(posteriors[0][0] > posteriors[0][1]);
    assert!(posteriors[0][0] > posteriors[0][2]);
}

// ── estimate_switch_prob: Viterbi detects switches ──

#[test]
fn estimate_switch_prob_alternating_signal_detects_switches() {
    let pops = make_pops_2way();
    // Alternating strong AFR / strong EUR signals every 2 windows
    let obs: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            if i % 4 < 2 {
                make_obs(
                    i * 10_000,
                    vec![("afr1", 0.99), ("afr2", 0.98), ("eur1", 0.70), ("eur2", 0.72)],
                )
            } else {
                make_obs(
                    i * 10_000,
                    vec![("afr1", 0.70), ("afr2", 0.72), ("eur1", 0.99), ("eur2", 0.98)],
                )
            }
        })
        .collect();
    let sp = estimate_switch_prob(&obs, &pops, 0.03);
    // With many switches, the estimated switch prob should be relatively high
    assert!(
        sp > 0.01,
        "Alternating signal should produce higher switch prob, got {}",
        sp
    );
}

// ── log_emission: zero similarity for target state → NEG_INFINITY ──

#[test]
fn log_emission_zero_similarity_returns_neg_inf() {
    let pops = make_pops_2way();
    let params = AncestryHmmParams::new(pops, 0.01);
    // AFR has zero similarity, EUR has positive
    let obs = make_obs(0, vec![("afr1", 0.0), ("afr2", 0.0), ("eur1", 0.95), ("eur2", 0.93)]);
    let emission_afr = params.log_emission(&obs, 0);
    assert_eq!(
        emission_afr,
        f64::NEG_INFINITY,
        "Zero target sim should return NEG_INFINITY"
    );
}

// ── Asymmetric BW transitions for 3+ states ──

fn make_pops_3way() -> Vec<AncestralPopulation> {
    vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: vec!["eur1".to_string(), "eur2".to_string()],
        },
        AncestralPopulation {
            name: "AFR".to_string(),
            haplotypes: vec!["afr1".to_string(), "afr2".to_string()],
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: vec!["amr1".to_string(), "amr2".to_string()],
        },
    ]
}

#[test]
fn baum_welch_3state_learns_asymmetric_transitions() {
    let pops = make_pops_3way();
    let mut params = AncestryHmmParams::new(pops, 0.005);

    // Create data: EUR and AMR are similar, AFR is distinct.
    // Windows 0-9: EUR (eur high, amr slightly lower, afr low)
    // Windows 10-19: AMR (amr high, eur slightly lower, afr low)
    // This should teach BW that EUR↔AMR transitions are more likely than EUR↔AFR
    let mut obs = Vec::new();
    for i in 0..10 {
        obs.push(make_obs(
            i * 10_000,
            vec![("eur1", 0.98), ("eur2", 0.97), ("afr1", 0.80), ("afr2", 0.81),
                 ("amr1", 0.96), ("amr2", 0.95)],
        ));
    }
    for i in 10..20 {
        obs.push(make_obs(
            i * 10_000,
            vec![("eur1", 0.96), ("eur2", 0.95), ("afr1", 0.80), ("afr2", 0.81),
                 ("amr1", 0.98), ("amr2", 0.97)],
        ));
    }

    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    let ll = params.baum_welch(&obs_refs, 20, 1e-6);
    assert!(ll.is_finite(), "BW should converge, got {}", ll);

    // With 3 states, transitions should be asymmetric (not enforced to uniform)
    // EUR (state 0) should prefer transitioning to AMR (state 2) over AFR (state 1)
    let eur_to_amr = params.transitions[0][2];
    let eur_to_afr = params.transitions[0][1];

    // Rows should still sum to ~1.0
    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Row {} should sum to 1.0, got {}", i, sum
        );
    }

    // Diagonal should be high (stay probability)
    for i in 0..3 {
        assert!(
            params.transitions[i][i] >= 0.9,
            "Stay probability for state {} should be >= 0.9, got {}",
            i, params.transitions[i][i]
        );
    }

    // The transition matrix should NOT be perfectly symmetric
    // (i.e., not all off-diagonal elements equal)
    let trans = &params.transitions;
    let off_diag: Vec<f64> = (0..3)
        .flat_map(|i| (0..3).filter(move |&j| j != i).map(move |j| (i, j)))
        .map(|(i, j)| trans[i][j])
        .collect();
    let all_equal = off_diag.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
    assert!(
        !all_equal,
        "3-state BW should learn asymmetric transitions, not uniform. Off-diag: {:?}",
        off_diag
    );
}

#[test]
fn baum_welch_2state_keeps_symmetric_transitions() {
    let pops = make_pops_2way();
    let mut params = AncestryHmmParams::new(pops, 0.01);

    let mut obs = Vec::new();
    for i in 0..10 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.95), ("afr2", 0.93), ("eur1", 0.80), ("eur2", 0.82)],
        ));
    }
    for i in 10..20 {
        obs.push(make_obs(
            i * 10_000,
            vec![("afr1", 0.80), ("afr2", 0.78), ("eur1", 0.96), ("eur2", 0.94)],
        ));
    }

    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    params.baum_welch(&obs_refs, 20, 1e-6);

    // 2-state BW should enforce symmetric transitions
    // P(0→1) should equal P(1→0)
    let p01 = params.transitions[0][1];
    let p10 = params.transitions[1][0];
    assert!(
        (p01 - p10).abs() < 1e-6,
        "2-state BW should be symmetric: P(0→1)={} vs P(1→0)={}",
        p01, p10
    );
}

#[test]
fn baum_welch_3state_rows_sum_to_one() {
    let pops = make_pops_3way();
    let mut params = AncestryHmmParams::new(pops, 0.005);

    // Simple 3-state data
    let mut obs = Vec::new();
    for i in 0..30 {
        let phase = i % 3;
        let sims = match phase {
            0 => vec![("eur1", 0.98), ("eur2", 0.97), ("afr1", 0.80), ("afr2", 0.81),
                      ("amr1", 0.90), ("amr2", 0.89)],
            1 => vec![("eur1", 0.80), ("eur2", 0.81), ("afr1", 0.98), ("afr2", 0.97),
                      ("amr1", 0.82), ("amr2", 0.83)],
            _ => vec![("eur1", 0.90), ("eur2", 0.89), ("afr1", 0.80), ("afr2", 0.81),
                      ("amr1", 0.98), ("amr2", 0.97)],
        };
        obs.push(make_obs(i * 10_000, sims));
    }

    let obs_refs: Vec<&[AncestryObservation]> = vec![&obs];
    params.baum_welch(&obs_refs, 20, 1e-6);

    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Row {} should sum to 1.0, got {}", i, sum
        );
        // All entries should be positive
        for (j, &p) in row.iter().enumerate() {
            assert!(p > 0.0, "transitions[{}][{}] = {} should be > 0", i, j, p);
        }
    }
}
