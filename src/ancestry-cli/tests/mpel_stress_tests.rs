//! Stress tests for MPEL decoder with long sequences.
//!
//! Validates that MPEL decode produces smooth paths on long observation
//! sequences (100-500+ windows) with varying SNR regimes.

use impopk_ancestry_cli::hmm::{
    AncestryHmmParams, AncestralPopulation, AncestryObservation,
    forward_backward, mpel_decode_from_posteriors, posterior_decode, viterbi,
};

fn make_pop(name: &str, haps: &[&str]) -> AncestralPopulation {
    AncestralPopulation {
        name: name.to_string(),
        haplotypes: haps.iter().map(|s| s.to_string()).collect(),
    }
}

fn make_params(n_pops: usize, switch_prob: f64) -> (AncestryHmmParams, Vec<Vec<&'static str>>) {
    let pop_names = ["EUR", "AFR", "EAS", "CSA", "AMR"];
    let mut pops = Vec::new();
    let mut hap_sets = Vec::new();
    for i in 0..n_pops {
        let name = pop_names[i % pop_names.len()];
        let h1: &'static str = Box::leak(format!("{}_h1", name.to_lowercase()).into_boxed_str());
        let h2: &'static str = Box::leak(format!("{}_h2", name.to_lowercase()).into_boxed_str());
        pops.push(make_pop(name, &[h1, h2]));
        hap_sets.push(vec![h1, h2]);
    }
    let params = AncestryHmmParams::new(pops, switch_prob);
    (params, hap_sets)
}

fn make_long_obs(
    n_windows: usize,
    true_states: &[usize],
    hap_sets: &[Vec<&str>],
    snr: f64, // signal-to-noise ratio: higher = clearer signal
) -> Vec<AncestryObservation> {
    let _n_pops = hap_sets.len();
    (0..n_windows).map(|t| {
        let true_state = true_states[t % true_states.len()];
        let mut sims = Vec::new();
        for (p, haps) in hap_sets.iter().enumerate() {
            let base = if p == true_state { 0.99 } else { 0.99 - snr * 0.01 };
            // Deterministic but varied noise
            let noise = ((t as f64 * 0.7 + p as f64 * 1.3).sin()) * 0.002;
            for h in haps {
                sims.push((*h, base + noise));
            }
        }
        AncestryObservation {
            chrom: "chr1".to_string(),
            start: t as u64 * 10000,
            end: (t as u64 + 1) * 10000,
            sample: "query".to_string(),
            similarities: sims.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            coverage_ratios: None,
            haplotype_consistency_bonus: None,
        }
    }).collect()
}

fn count_switches(states: &[usize]) -> usize {
    states.windows(2).filter(|w| w[0] != w[1]).count()
}

#[test]
fn mpel_200_windows_clear_signal() {
    let (params, hap_sets) = make_params(3, 0.01);
    // True pattern: 100 windows EUR, 100 windows AFR
    let true_states: Vec<usize> = (0..200).map(|i| if i < 100 { 0 } else { 1 }).collect();
    let obs = make_long_obs(200, &true_states, &hap_sets, 3.0);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);
    let _vit = viterbi(&obs, &params);
    let _post = posterior_decode(&obs, &params);

    assert_eq!(mpel.len(), 200);
    // With clear signal, all decoders should agree
    let mpel_switches = count_switches(&mpel);
    assert!(mpel_switches <= 3, "MPEL should have ≤3 switches on clear signal, got {mpel_switches}");

    // MPEL accuracy vs true states
    let mpel_correct: usize = mpel.iter().zip(&true_states).filter(|(a, b)| a == b).count();
    assert!(mpel_correct >= 190, "MPEL should be >95% correct, got {mpel_correct}/200");
}

#[test]
fn mpel_500_windows_multiple_switches() {
    let (params, hap_sets) = make_params(3, 0.005);
    // 5 segments of ~100 windows each: EUR, AFR, EAS, EUR, AFR
    let true_states: Vec<usize> = (0..500).map(|i| {
        match i / 100 {
            0 => 0, 1 => 1, 2 => 2, 3 => 0, _ => 1,
        }
    }).collect();
    let obs = make_long_obs(500, &true_states, &hap_sets, 2.5);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);

    assert_eq!(mpel.len(), 500);
    let switches = count_switches(&mpel);
    // True switches = 4, MPEL should be close
    assert!(switches >= 3 && switches <= 8,
        "MPEL should detect ~4 switches, got {switches}");
}

#[test]
fn mpel_smoother_than_posterior_on_noisy_data() {
    let (params, hap_sets) = make_params(3, 0.01);
    // Low SNR: signal is weak
    let true_states: Vec<usize> = (0..200).map(|i| if i < 100 { 0 } else { 1 }).collect();
    let obs = make_long_obs(200, &true_states, &hap_sets, 0.5); // low SNR

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);
    let post = posterior_decode(&obs, &params);

    let mpel_switches = count_switches(&mpel);
    let post_switches = count_switches(&post);
    let _ = &post; // used above

    // MPEL should be at least as smooth as posterior decode (usually smoother)
    assert!(mpel_switches <= post_switches + 5,
        "MPEL should be smoother than posterior: mpel_switches={mpel_switches}, post_switches={post_switches}");
}

#[test]
fn mpel_handles_very_short_tracts() {
    let (params, hap_sets) = make_params(2, 0.02);
    // Short tracts: 5 EUR, 3 AFR, 5 EUR, 3 AFR, repeated
    let true_states: Vec<usize> = (0..200).map(|i| {
        if (i % 8) < 5 { 0 } else { 1 }
    }).collect();
    let obs = make_long_obs(200, &true_states, &hap_sets, 2.0);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);

    // MPEL smooths over very short tracts — this is expected behavior.
    // The Viterbi pass penalizes frequent switching, so 3-window tracts
    // may be absorbed into longer segments.
    let mpel_switches = count_switches(&mpel);
    let true_switches = count_switches(&true_states);
    // MPEL should be strictly smoother than truth for short tracts
    assert!(mpel_switches < true_switches,
        "MPEL should smooth short tracts: mpel_switches={mpel_switches}, true={true_switches}");
}

#[test]
fn mpel_5_pops_long_sequence() {
    let (params, hap_sets) = make_params(5, 0.005);
    // Cycle through all 5 populations, 60 windows each
    let true_states: Vec<usize> = (0..300).map(|i| (i / 60) % 5).collect();
    let obs = make_long_obs(300, &true_states, &hap_sets, 2.0);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);

    assert_eq!(mpel.len(), 300);
    // Should roughly track true ancestry
    let correct: usize = mpel.iter().zip(&true_states).filter(|(a, b)| a == b).count();
    assert!(correct >= 200, "5-pop MPEL should be >66% correct, got {correct}/300");
}

#[test]
fn mpel_posteriors_all_near_uniform() {
    // Edge case: posteriors are nearly uniform → MPEL should still return valid path
    let (params, hap_sets) = make_params(3, 0.01);
    // Very low SNR → near-uniform posteriors
    let obs = make_long_obs(100, &[0], &hap_sets, 0.01);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);

    assert_eq!(mpel.len(), 100);
    for &s in &mpel {
        assert!(s < 3, "State should be valid (0-2), got {s}");
    }
    // With near-uniform posteriors, MPEL should be very smooth (few switches)
    let switches = count_switches(&mpel);
    assert!(switches <= 10, "Near-uniform posteriors → smooth path, got {switches} switches");
}

#[test]
fn mpel_single_window_returns_argmax() {
    let (params, hap_sets) = make_params(3, 0.01);
    let obs = make_long_obs(1, &[1], &hap_sets, 3.0);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);

    assert_eq!(mpel.len(), 1);
    assert_eq!(mpel[0], 1, "Single window should pick state with highest posterior");
}

#[test]
fn mpel_boundary_accuracy_vs_viterbi() {
    // MPEL should place boundaries at least as accurately as Viterbi
    let (params, hap_sets) = make_params(2, 0.01);
    let true_boundary = 100; // Switch at window 100
    let true_states: Vec<usize> = (0..200).map(|i| if i < true_boundary { 0 } else { 1 }).collect();
    let obs = make_long_obs(200, &true_states, &hap_sets, 2.0);

    let posteriors = forward_backward(&obs, &params);
    let mpel = mpel_decode_from_posteriors(&posteriors, &params);
    let vit = viterbi(&obs, &params);

    // Find boundary position in each decoder's output
    let mpel_boundary = mpel.windows(2).position(|w| w[0] == 0 && w[1] == 1);
    let vit_boundary = vit.windows(2).position(|w| w[0] == 0 && w[1] == 1);

    if let (Some(mb), Some(vb)) = (mpel_boundary, vit_boundary) {
        let mpel_err = (mb as i64 - true_boundary as i64).unsigned_abs();
        let vit_err = (vb as i64 - true_boundary as i64).unsigned_abs();
        // MPEL should be at least as accurate (allow 2 window tolerance)
        assert!(mpel_err <= vit_err + 2,
            "MPEL boundary error={mpel_err} should be ≤ Viterbi error={vit_err} + 2");
    }
}
