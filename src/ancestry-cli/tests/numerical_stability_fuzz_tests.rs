//! Fuzz-like numerical stability tests for the Ancestry HMM.
//!
//! Verifies that the ancestry HMM algorithms (viterbi, forward_backward,
//! posterior_decode, baum_welch, estimate_emissions, learn_normalization)
//! produce valid, finite outputs for adversarial and edge-case inputs.

use impopk_ancestry_cli::hmm::{
    forward_backward, posterior_decode, viterbi, AncestralPopulation, AncestryHmmParams,
    AncestryObservation, EmissionModel,
};
use std::collections::HashMap;

/// Simple deterministic pseudo-random number generator (xorshift64)
struct PseudoRng {
    state: u64,
}

impl PseudoRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 1_000_000_000) as f64 / 1_000_000_000.0
    }

    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

fn make_populations(n_pops: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n_pops)
        .map(|i| AncestralPopulation {
            name: names[i % names.len()].to_string(),
            haplotypes: (0..5)
                .map(|h| format!("{}_{}", names[i % names.len()], h))
                .collect(),
        })
        .collect()
}

fn make_observation(rng: &mut PseudoRng, pop_names: &[&str], idx: usize) -> AncestryObservation {
    let mut sims = HashMap::new();
    for &pop in pop_names {
        for h in 0..5 {
            let hap = format!("{}_{}", pop, h);
            sims.insert(hap, rng.next_range(0.990, 1.0));
        }
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: (idx as u64) * 10000,
        end: (idx as u64 + 1) * 10000,
        sample: "TEST#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_observation_biased(
    rng: &mut PseudoRng,
    pop_names: &[&str],
    true_pop_idx: usize,
    idx: usize,
) -> AncestryObservation {
    let mut sims = HashMap::new();
    for (i, &pop) in pop_names.iter().enumerate() {
        let base = if i == true_pop_idx { 0.998 } else { 0.993 };
        for h in 0..5 {
            let hap = format!("{}_{}", pop, h);
            sims.insert(hap, rng.next_range(base - 0.002, base + 0.002));
        }
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: (idx as u64) * 10000,
        end: (idx as u64 + 1) * 10000,
        sample: "TEST#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// ── Viterbi stability ──────────────────────────────────────────────────

#[test]
fn viterbi_random_states_in_range_3pop() {
    let mut rng = PseudoRng::new(42);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..200).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 200);
    for (i, &s) in states.iter().enumerate() {
        assert!(s < 3, "state[{}] = {} >= n_states=3", i, s);
    }
}

#[test]
fn viterbi_random_states_in_range_5pop() {
    let mut rng = PseudoRng::new(77);
    let pops = make_populations(5);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    let obs: Vec<_> = (0..300).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 300);
    for &s in &states {
        assert!(s < 5);
    }
}

#[test]
fn viterbi_empty_observations() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let states = viterbi(&[], &params);
    assert!(states.is_empty());
}

#[test]
fn viterbi_single_observation() {
    let mut rng = PseudoRng::new(99);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs = vec![make_observation(&mut rng, &pop_names, 0)];
    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 3);
}

// ── Forward-backward stability ─────────────────────────────────────────

#[test]
fn forward_backward_posteriors_sum_to_one() {
    let mut rng = PseudoRng::new(123);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..100).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let posteriors = forward_backward(&obs, &params);
    assert_eq!(posteriors.len(), 100);
    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), 3);
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "posteriors[{}] sum={} (expected 1.0)",
            t,
            sum
        );
        for (s, &p) in post.iter().enumerate() {
            assert!(
                p >= -1e-10 && p <= 1.0 + 1e-10,
                "posteriors[{}][{}] = {}",
                t,
                s,
                p
            );
            assert!(p.is_finite(), "posteriors[{}][{}] not finite", t, s);
        }
    }
}

#[test]
fn forward_backward_5pop_posteriors_sum_to_one() {
    let mut rng = PseudoRng::new(555);
    let pops = make_populations(5);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    let obs: Vec<_> = (0..150).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let posteriors = forward_backward(&obs, &params);
    for (t, post) in posteriors.iter().enumerate() {
        let sum: f64 = post.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "t={} sum={}",
            t,
            sum
        );
    }
}

#[test]
fn forward_backward_empty() {
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let posteriors = forward_backward(&[], &params);
    assert!(posteriors.is_empty());
}

// ── Posterior decode ───────────────────────────────────────────────────

#[test]
fn posterior_decode_random_states_in_range() {
    let mut rng = PseudoRng::new(321);
    let pops = make_populations(4);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.emission_model = EmissionModel::TopK(3);
    let pop_names = ["AFR", "EUR", "EAS", "CSA"];
    let obs: Vec<_> = (0..200).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let states = posterior_decode(&obs, &params);
    assert_eq!(states.len(), 200);
    for &s in &states {
        assert!(s < 4);
    }
}

// ── Viterbi vs posterior_decode consistency ─────────────────────────────

#[test]
fn viterbi_and_posterior_decode_same_length() {
    let mut rng = PseudoRng::new(888);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..100).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let viterbi_states = viterbi(&obs, &params);
    let pd_states = posterior_decode(&obs, &params);
    assert_eq!(viterbi_states.len(), pd_states.len());
    // Both produce valid state indices
    for (&vs, &ps) in viterbi_states.iter().zip(pd_states.iter()) {
        assert!(vs < 3);
        assert!(ps < 3);
    }
}

// ── Biased data: should favor the true population ──────────────────────

#[test]
fn viterbi_biased_favors_true_population() {
    let mut rng = PseudoRng::new(2024);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];

    // All windows biased toward population 1 (EUR)
    let obs: Vec<_> = (0..100)
        .map(|i| make_observation_biased(&mut rng, &pop_names, 1, i))
        .collect();
    let states = viterbi(&obs, &params);

    // Majority of windows should be called as state 1
    let eur_count = states.iter().filter(|&&s| s == 1).count();
    assert!(
        eur_count > 50,
        "Expected majority EUR, got {}/100",
        eur_count
    );
}

#[test]
fn forward_backward_biased_highest_posterior_for_true_pop() {
    let mut rng = PseudoRng::new(2025);
    let pops = make_populations(3);
    let params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];

    // All windows biased toward population 0 (AFR)
    let obs: Vec<_> = (0..50)
        .map(|i| make_observation_biased(&mut rng, &pop_names, 0, i))
        .collect();
    let posteriors = forward_backward(&obs, &params);

    // Mean posterior for AFR should be highest
    let mean_afr: f64 = posteriors.iter().map(|p| p[0]).sum::<f64>() / 50.0;
    let mean_eur: f64 = posteriors.iter().map(|p| p[1]).sum::<f64>() / 50.0;
    let mean_eas: f64 = posteriors.iter().map(|p| p[2]).sum::<f64>() / 50.0;
    assert!(
        mean_afr > mean_eur && mean_afr > mean_eas,
        "AFR={:.3}, EUR={:.3}, EAS={:.3}",
        mean_afr,
        mean_eur,
        mean_eas
    );
}

// ── EmissionModel variants stability ───────────────────────────────────

#[test]
fn all_emission_models_produce_valid_viterbi() {
    let models = [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(1),
        EmissionModel::TopK(3),
        EmissionModel::TopK(5),
    ];
    let mut rng = PseudoRng::new(6060);
    let pop_names = ["AFR", "EUR", "EAS"];

    for model in &models {
        let pops = make_populations(3);
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.emission_model = *model;
        let obs: Vec<_> = (0..80).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
        let states = viterbi(&obs, &params);
        assert_eq!(
            states.len(),
            80,
            "model={:?} produced wrong length",
            model
        );
        for &s in &states {
            assert!(s < 3, "model={:?} invalid state", model);
        }
    }
}

#[test]
fn all_emission_models_produce_valid_posteriors() {
    let models = [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(3),
    ];
    let mut rng = PseudoRng::new(7070);
    let pop_names = ["AFR", "EUR"];

    for model in &models {
        let pops = make_populations(2);
        let mut params = AncestryHmmParams::new(pops, 0.001);
        params.emission_model = *model;
        let obs: Vec<_> = (0..60).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
        let posteriors = forward_backward(&obs, &params);
        for (t, post) in posteriors.iter().enumerate() {
            let sum: f64 = post.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "model={:?} t={} sum={}",
                model,
                t,
                sum
            );
        }
    }
}

// ── set_switch_prob / set_temperature ──────────────────────────────────

#[test]
fn set_switch_prob_updates_transitions_correctly() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);

    params.set_switch_prob(0.01);
    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "row {} sum={} after set_switch_prob",
            i,
            sum
        );
        // Diagonal should be 1 - 0.01 = 0.99
        assert!((row[i] - 0.99).abs() < 1e-10);
        // Off-diagonal should be 0.01 / 2 = 0.005
        for (j, &p) in row.iter().enumerate() {
            if i != j {
                assert!((p - 0.005).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn set_switch_prob_extreme_values() {
    let pops = make_populations(4);
    let mut params = AncestryHmmParams::new(pops, 0.001);

    // Very small switch prob
    params.set_switch_prob(1e-10);
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // Large switch prob (0.5)
    params.set_switch_prob(0.5);
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn set_temperature_affects_emissions() {
    let mut rng = PseudoRng::new(1111);
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..50).map(|i| make_observation(&mut rng, &pop_names, i)).collect();

    // Low temperature → more peaky posteriors
    params.set_temperature(0.001);
    let posteriors_sharp = forward_backward(&obs, &params);

    // High temperature → more uniform posteriors
    params.set_temperature(1.0);
    let posteriors_smooth = forward_backward(&obs, &params);

    // Both should have valid posteriors summing to 1
    for post in &posteriors_sharp {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    for post in &posteriors_smooth {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // Sharp posteriors should have higher max on average (more concentrated)
    let mean_max_sharp: f64 =
        posteriors_sharp.iter().map(|p| p.iter().cloned().fold(0.0, f64::max)).sum::<f64>()
            / posteriors_sharp.len() as f64;
    let mean_max_smooth: f64 =
        posteriors_smooth.iter().map(|p| p.iter().cloned().fold(0.0, f64::max)).sum::<f64>()
            / posteriors_smooth.len() as f64;
    assert!(
        mean_max_sharp >= mean_max_smooth - 0.01,
        "sharp={:.3} smooth={:.3}",
        mean_max_sharp,
        mean_max_smooth
    );
}

// ── Multi-seed fuzz ────────────────────────────────────────────────────

#[test]
fn multi_seed_fuzz_ancestry_invariants() {
    for seed in 200..220 {
        let mut rng = PseudoRng::new(seed);
        let n_pops = 2 + (seed % 4) as usize; // 2-5 populations
        let pops = make_populations(n_pops);
        let params = AncestryHmmParams::new(pops, 0.001);
        let pop_names: Vec<&str> = ["AFR", "EUR", "EAS", "CSA", "AMR"]
            .iter()
            .take(n_pops)
            .cloned()
            .collect();

        let n_obs = 30 + (rng.next_u64() % 100) as usize;
        let obs: Vec<_> = (0..n_obs)
            .map(|i| make_observation(&mut rng, &pop_names, i))
            .collect();

        let states = viterbi(&obs, &params);
        let posteriors = forward_backward(&obs, &params);

        assert_eq!(states.len(), n_obs, "seed={}", seed);
        assert_eq!(posteriors.len(), n_obs, "seed={}", seed);

        for t in 0..n_obs {
            assert!(states[t] < n_pops, "seed={} t={}", seed, t);
            let sum: f64 = posteriors[t].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "seed={} t={} sum={}",
                seed,
                t,
                sum
            );
        }
    }
}

// ── learn_normalization stability ──────────────────────────────────────

#[test]
fn learn_normalization_then_viterbi_valid() {
    let mut rng = PseudoRng::new(333);
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..100).map(|i| make_observation(&mut rng, &pop_names, i)).collect();

    params.learn_normalization(&obs);
    assert!(params.normalization.is_some());

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 100);
    for &s in &states {
        assert!(s < 3);
    }

    let posteriors = forward_backward(&obs, &params);
    for post in &posteriors {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}

// ── Baum-Welch stability ───────────────────────────────────────────────

#[test]
fn baum_welch_random_data_preserves_transition_validity() {
    let mut rng = PseudoRng::new(444);
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR", "EAS"];
    let obs: Vec<_> = (0..150).map(|i| make_observation(&mut rng, &pop_names, i)).collect();
    let obs_ref: &[AncestryObservation] = &obs;

    params.baum_welch(&[obs_ref], 10, 1e-6);

    // After BW, transitions should still be valid stochastic matrix
    for (i, row) in params.transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "row {} sum={} after BW",
            i,
            sum
        );
        for &p in row {
            assert!(p >= 0.0 && p <= 1.0, "invalid transition prob {}", p);
        }
    }
}

#[test]
fn baum_welch_constant_observations_no_crash() {
    let pops = make_populations(2);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let pop_names = ["AFR", "EUR"];

    // All observations identical
    let mut sims = HashMap::new();
    for &pop in &pop_names {
        for h in 0..5 {
            sims.insert(format!("{}_{}", pop, h), 0.998);
        }
    }
    let obs: Vec<_> = (0..50)
        .map(|i| AncestryObservation {
            chrom: "chr1".to_string(),
            start: (i as u64) * 10000,
            end: (i as u64 + 1) * 10000,
            sample: "TEST#1".to_string(),
            similarities: sims.clone(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        })
        .collect();
    let obs_ref: &[AncestryObservation] = &obs;

    params.baum_welch(&[obs_ref], 5, 1e-6);
    // Should not crash; transitions still valid
    for row in &params.transitions {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}

// ── 2-population edge case ─────────────────────────────────────────────

#[test]
fn two_population_viterbi_and_fb_consistent() {
    let mut rng = PseudoRng::new(2222);
    let pops = make_populations(2);
    let params = AncestryHmmParams::new(pops, 0.01);
    let pop_names = ["AFR", "EUR"];
    let obs: Vec<_> = (0..50).map(|i| make_observation(&mut rng, &pop_names, i)).collect();

    let states = viterbi(&obs, &params);
    let posteriors = forward_backward(&obs, &params);

    assert_eq!(states.len(), posteriors.len());
    for (t, (post, &state)) in posteriors.iter().zip(states.iter()).enumerate() {
        // Posterior for state 0 and 1 should sum to 1
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "t={}", t);

        // States should be binary
        assert!(state < 2, "t={} state={}", t, state);
    }
}

// ── EmissionModel parsing roundtrip ────────────────────────────────────

#[test]
fn emission_model_display_parse_roundtrip() {
    let models = [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(3),
        EmissionModel::TopK(10),
    ];
    for model in &models {
        let s = model.to_string();
        let parsed: EmissionModel = s.parse().unwrap();
        assert_eq!(*model, parsed, "roundtrip failed for {:?}", model);
    }
}

#[test]
fn emission_model_parse_case_insensitive() {
    assert_eq!("MAX".parse::<EmissionModel>().unwrap(), EmissionModel::Max);
    assert_eq!(
        "Mean".parse::<EmissionModel>().unwrap(),
        EmissionModel::Mean
    );
    assert_eq!(
        "MEDIAN".parse::<EmissionModel>().unwrap(),
        EmissionModel::Median
    );
    assert_eq!(
        "Top5".parse::<EmissionModel>().unwrap(),
        EmissionModel::TopK(5)
    );
    assert_eq!(
        "TOP10".parse::<EmissionModel>().unwrap(),
        EmissionModel::TopK(10)
    );
}

#[test]
fn emission_model_parse_invalid() {
    assert!("foo".parse::<EmissionModel>().is_err());
    assert!("topX".parse::<EmissionModel>().is_err());
    assert!("top-5".parse::<EmissionModel>().is_err());
}

// ── Coverage-weighted emission ─────────────────────────────────────────

#[test]
fn coverage_weighted_emission_produces_valid_states() {
    let mut rng = PseudoRng::new(5050);
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.coverage_weight = 0.5;
    let pop_names = ["AFR", "EUR", "EAS"];

    let obs: Vec<_> = (0..80)
        .map(|i| {
            let mut sims = HashMap::new();
            let mut covs = HashMap::new();
            for &pop in &pop_names {
                for h in 0..5 {
                    let hap = format!("{}_{}", pop, h);
                    sims.insert(hap.clone(), rng.next_range(0.990, 1.0));
                    covs.insert(hap, rng.next_range(0.5, 1.0));
                }
            }
            AncestryObservation {
                chrom: "chr1".to_string(),
                start: (i as u64) * 10000,
                end: (i as u64 + 1) * 10000,
                sample: "TEST#1".to_string(),
                similarities: sims,
                coverage_ratios: Some(covs),
            haplotype_consistency_bonus: None,
            }
        })
        .collect();

    let states = viterbi(&obs, &params);
    assert_eq!(states.len(), 80);
    for &s in &states {
        assert!(s < 3);
    }

    let posteriors = forward_backward(&obs, &params);
    for post in &posteriors {
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
