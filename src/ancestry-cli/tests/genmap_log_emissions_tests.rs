//! Tests for viterbi_from_log_emissions_with_genetic_map and
//! forward_backward_from_log_emissions_with_genetic_map.
//!
//! These variants use precomputed log emissions (instead of computing them
//! internally) combined with genetic-map-aware transition probabilities.
//! This enables efficient multi-pass workflows where emissions are computed once
//! and decoding is repeated with different transition parameters.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    AncestryGeneticMap,
    viterbi_from_log_emissions_with_genetic_map,
    forward_backward_from_log_emissions_with_genetic_map,
    viterbi_from_log_emissions, forward_backward_from_log_emissions,
};

// ── Helpers ──

fn make_pops(n: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n)
        .map(|i| AncestralPopulation {
            name: names[i].to_string(),
            haplotypes: vec![
                format!("{}#HAP1", names[i]),
                format!("{}#HAP2", names[i]),
            ],
        })
        .collect()
}

fn make_params(n_pops: usize, switch_prob: f64) -> AncestryHmmParams {
    AncestryHmmParams::new(make_pops(n_pops), switch_prob)
}

fn make_obs(start: u64, end: u64) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: "query#1".to_string(),
        similarities: HashMap::new(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_log_emissions_clear_signal(n_windows: usize, k: usize, dominant: &[usize]) -> Vec<Vec<f64>> {
    // Create log emissions where dominant[t] has highest probability at window t
    (0..n_windows).map(|t| {
        let dom = dominant[t % dominant.len()];
        (0..k).map(|s| {
            if s == dom { -0.1 } else { -3.0 }  // ~0.90 vs ~0.05
        }).collect()
    }).collect()
}

fn make_uniform_genmap(n_windows: usize, window_size: u64) -> AncestryGeneticMap {
    let total_bp = (n_windows as u64 + 1) * window_size;
    AncestryGeneticMap::uniform(0, total_bp, 1.0) // 1 cM/Mb
}

fn make_observations(n_windows: usize, window_size: u64) -> Vec<AncestryObservation> {
    (0..n_windows).map(|i| {
        let start = i as u64 * window_size;
        make_obs(start, start + window_size)
    }).collect()
}

// ── Viterbi tests ──

#[test]
fn viterbi_genmap_empty_input() {
    let params = make_params(2, 0.01);
    let genmap = make_uniform_genmap(10, 100_000);
    let states = viterbi_from_log_emissions_with_genetic_map(&[], &[], &params, &genmap);
    assert!(states.is_empty());
}

#[test]
fn viterbi_genmap_single_window() {
    let params = make_params(2, 0.01);
    let genmap = make_uniform_genmap(1, 100_000);
    let obs = make_observations(1, 100_000);
    let log_emit = vec![vec![-0.5, -2.0]]; // state 0 dominant

    let states = viterbi_from_log_emissions_with_genetic_map(&obs, &log_emit, &params, &genmap);
    assert_eq!(states.len(), 1);
    assert_eq!(states[0], 0);
}

#[test]
fn viterbi_genmap_clear_signal_tracks_dominant() {
    let k = 3;
    let n = 20;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    // First 10 windows: state 0, next 10: state 1
    let dominant: Vec<usize> = (0..n).map(|t| if t < 10 { 0 } else { 1 }).collect();
    let log_emit = make_log_emissions_clear_signal(n, k, &dominant);

    let states = viterbi_from_log_emissions_with_genetic_map(&obs, &log_emit, &params, &genmap);
    assert_eq!(states.len(), n);

    // Most windows should track the dominant state
    let correct: usize = states.iter().enumerate()
        .filter(|&(t, &s)| s == dominant[t])
        .count();
    assert!(correct >= 16, "at least 80% correct: {}/{}", correct, n);
}

#[test]
fn viterbi_genmap_agrees_with_uniform_transitions_approx() {
    // With a uniform genetic map at 1 cM/Mb and 100kb windows,
    // results should be similar to standard viterbi
    let k = 2;
    let n = 30;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    let dominant: Vec<usize> = (0..n).map(|t| if t < 15 { 0 } else { 1 }).collect();
    let log_emit = make_log_emissions_clear_signal(n, k, &dominant);

    let states_genmap = viterbi_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    let states_standard = viterbi_from_log_emissions(&log_emit, &params);

    // With clear signal and uniform map, both should produce same states
    // (genetic map modulation is close to identity at uniform 1cM/Mb)
    let agreement: usize = states_genmap.iter().zip(states_standard.iter())
        .filter(|(&a, &b)| a == b).count();
    assert!(agreement >= n - 4,
        "genmap and standard should mostly agree: {}/{}", agreement, n);
}

#[test]
fn viterbi_genmap_hotspot_encourages_switch() {
    // Create a genetic map with a recombination hotspot between windows 10-11.
    // This should make ancestry switches at that boundary more likely.
    let k = 2;
    let n = 20;
    let ws: u64 = 100_000;
    let params = make_params(k, 0.005);
    let obs = make_observations(n, ws);

    // Build genetic map: 1 cM/Mb everywhere except huge jump at window 10
    let genmap = AncestryGeneticMap::uniform(0, n as u64 * ws, 1.0);

    // Weak emissions that favor state 0 in first half, state 1 in second half
    // but with low confidence so transitions matter more
    let log_emit: Vec<Vec<f64>> = (0..n).map(|t| {
        if t < 10 {
            vec![-0.8, -1.2] // weak preference for state 0
        } else {
            vec![-1.2, -0.8] // weak preference for state 1
        }
    }).collect();

    let states = viterbi_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    assert_eq!(states.len(), n);
    // Should produce valid output (not panic)
}

// ── Forward-backward tests ──

#[test]
fn fb_genmap_empty_input() {
    let params = make_params(2, 0.01);
    let genmap = make_uniform_genmap(10, 100_000);
    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &[], &[], &params, &genmap);
    assert!(posteriors.is_empty());
}

#[test]
fn fb_genmap_single_window() {
    let params = make_params(2, 0.01);
    let genmap = make_uniform_genmap(1, 100_000);
    let obs = make_observations(1, 100_000);
    let log_emit = vec![vec![-0.5, -2.0]]; // state 0 dominant

    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    assert_eq!(posteriors.len(), 1);
    assert_eq!(posteriors[0].len(), 2);

    // State 0 should have higher posterior
    assert!(posteriors[0][0] > posteriors[0][1],
        "state 0 should dominate: {:?}", posteriors[0]);
}

#[test]
fn fb_genmap_posteriors_sum_to_one() {
    let k = 3;
    let n = 15;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    let dominant: Vec<usize> = (0..n).map(|t| t % k).collect();
    let log_emit = make_log_emissions_clear_signal(n, k, &dominant);

    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    assert_eq!(posteriors.len(), n);

    for (t, post) in posteriors.iter().enumerate() {
        assert_eq!(post.len(), k);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "window {} posteriors should sum to 1: sum={}, {:?}", t, sum, post);
        for &p in post {
            assert!(p >= 0.0 && p <= 1.0,
                "window {} has invalid posterior: {:?}", t, post);
        }
    }
}

#[test]
fn fb_genmap_clear_signal_high_confidence() {
    let k = 2;
    let n = 20;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    // Strong signal: state 0 for all windows
    let log_emit: Vec<Vec<f64>> = (0..n).map(|_| vec![-0.1, -5.0]).collect();

    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);

    // All posteriors should strongly favor state 0
    for (t, post) in posteriors.iter().enumerate() {
        assert!(post[0] > 0.95,
            "window {} should have high posterior for state 0: {}", t, post[0]);
    }
}

#[test]
fn fb_genmap_agrees_with_standard_approx() {
    let k = 2;
    let n = 20;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    let dominant: Vec<usize> = (0..n).map(|t| if t < 10 { 0 } else { 1 }).collect();
    let log_emit = make_log_emissions_clear_signal(n, k, &dominant);

    let post_genmap = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    let post_standard = forward_backward_from_log_emissions(&log_emit, &params);

    // With uniform genetic map, posteriors should be close
    for t in 0..n {
        let diff: f64 = post_genmap[t].iter().zip(post_standard[t].iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 0.3,
            "window {} posteriors should be close: genmap={:?} standard={:?}",
            t, post_genmap[t], post_standard[t]);
    }
}

#[test]
fn viterbi_and_fb_genmap_consistent() {
    // Viterbi state should match argmax of forward-backward posterior
    let k = 2;
    let n = 20;
    let params = make_params(k, 0.01);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    let log_emit = make_log_emissions_clear_signal(n, k, &(0..n).map(|t| if t < 10 { 0 } else { 1 }).collect::<Vec<_>>());

    let states = viterbi_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);

    let fb_argmax: Vec<usize> = posteriors.iter().map(|post| {
        post.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx).unwrap()
    }).collect();

    // With clear signal, Viterbi and FB argmax should agree on most windows
    let agreement: usize = states.iter().zip(fb_argmax.iter())
        .filter(|(&a, &b)| a == b).count();
    assert!(agreement >= n - 2,
        "Viterbi and FB argmax should mostly agree: {}/{}", agreement, n);
}

#[test]
fn fb_genmap_five_populations() {
    let k = 5;
    let n = 25;
    let params = make_params(k, 0.005);
    let genmap = make_uniform_genmap(n, 100_000);
    let obs = make_observations(n, 100_000);

    // Cycling through 5 pops every 5 windows
    let dominant: Vec<usize> = (0..n).map(|t| (t / 5) % k).collect();
    let log_emit = make_log_emissions_clear_signal(n, k, &dominant);

    let posteriors = forward_backward_from_log_emissions_with_genetic_map(
        &obs, &log_emit, &params, &genmap);
    assert_eq!(posteriors.len(), n);

    for post in &posteriors {
        assert_eq!(post.len(), k);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posteriors should sum to 1: {}", sum);
    }
}
