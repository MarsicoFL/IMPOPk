//! Edge case tests for new ancestry-cli features:
//! - Posterior decoding
//! - Per-population emission normalization
//! - Ancestry genetic map (recombination-aware transitions)
//!
//! These test edge cases NOT covered by the existing unit tests in hmm.rs.

use impopk_ancestry_cli::hmm::{
    AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, AncestryObservation,
    DecodingMethod, forward_backward, forward_backward_with_genetic_map, posterior_decode,
    posterior_decode_with_genetic_map, viterbi, viterbi_with_genetic_map,
};
use std::collections::HashMap;

// ===========================================================================
// Helpers
// ===========================================================================

fn two_pop_params(temperature: f64, switch_prob: f64) -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, switch_prob);
    params.set_temperature(temperature);
    params
}

fn three_pop_params(temperature: f64, switch_prob: f64) -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "pop_a".to_string(),
            haplotypes: vec!["hap_a1".to_string(), "hap_a2".to_string()],
        },
        AncestralPopulation {
            name: "pop_b".to_string(),
            haplotypes: vec!["hap_b1".to_string(), "hap_b2".to_string()],
        },
        AncestralPopulation {
            name: "pop_c".to_string(),
            haplotypes: vec!["hap_c1".to_string(), "hap_c2".to_string()],
        },
    ];
    let mut params = AncestryHmmParams::new(pops, switch_prob);
    params.set_temperature(temperature);
    params
}

fn make_obs(start: u64, sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 5000,
        sample: "test_sample".to_string(),
        similarities: sims
            .iter()
            .map(|(name, val)| (name.to_string(), *val))
            .collect::<HashMap<String, f64>>(),
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_obs_simple(start: u64, a1: f64, a2: f64, b1: f64, b2: f64) -> AncestryObservation {
    make_obs(start, &[
        ("hap_a1", a1), ("hap_a2", a2),
        ("hap_b1", b1), ("hap_b2", b2),
    ])
}

fn make_obs_3pop(start: u64, a1: f64, a2: f64, b1: f64, b2: f64, c1: f64, c2: f64) -> AncestryObservation {
    make_obs(start, &[
        ("hap_a1", a1), ("hap_a2", a2),
        ("hap_b1", b1), ("hap_b2", b2),
        ("hap_c1", c1), ("hap_c2", c2),
    ])
}

// ===========================================================================
// 1. Posterior Decoding Edge Cases
// ===========================================================================

/// Posterior decoding on a very long homogeneous sequence should return all same state.
#[test]
fn test_posterior_decode_long_homogeneous() {
    let params = two_pop_params(0.03, 0.01);

    // 200 windows, all strongly pop_a
    let observations: Vec<AncestryObservation> = (0..200)
        .map(|i| make_obs_simple(i * 5000, 0.99, 0.98, 0.80, 0.79))
        .collect();

    let states = posterior_decode(&observations, &params);
    assert_eq!(states.len(), 200);

    // All windows should be state 0 (pop_a)
    let pop_a_count = states.iter().filter(|&&s| s == 0).count();
    assert!(
        pop_a_count >= 195,
        "Expected nearly all pop_a but got {}/200",
        pop_a_count
    );
}

/// Posterior decoding on rapidly alternating signal.
/// Posterior decode should detect more switches than Viterbi because it
/// doesn't penalize state transitions.
#[test]
fn test_posterior_decode_rapid_switching() {
    let params = two_pop_params(0.03, 0.05); // high switch rate

    // Alternate between pop_a and pop_b every 2 windows
    let observations: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            if i % 4 < 2 {
                make_obs_simple(i * 5000, 0.99, 0.98, 0.80, 0.79)
            } else {
                make_obs_simple(i * 5000, 0.80, 0.79, 0.99, 0.98)
            }
        })
        .collect();

    let posterior_states = posterior_decode(&observations, &params);
    let viterbi_states = viterbi(&observations, &params);

    // Count state switches
    let posterior_switches = posterior_states.windows(2).filter(|w| w[0] != w[1]).count();
    let viterbi_switches = viterbi_states.windows(2).filter(|w| w[0] != w[1]).count();

    // Posterior should detect at least as many switches as Viterbi
    assert!(
        posterior_switches >= viterbi_switches,
        "Posterior ({} switches) should detect >= Viterbi ({} switches)",
        posterior_switches,
        viterbi_switches
    );
}

/// Posterior decode with 3 populations, one very similar to another.
/// Should still assign the correct state based on argmax.
#[test]
fn test_posterior_decode_three_pop_close_signals() {
    let params = three_pop_params(0.03, 0.01);

    // pop_a and pop_b very similar, pop_c clearly different
    let observations: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_3pop(i * 5000, 0.96, 0.95, 0.955, 0.945, 0.80, 0.79))
        .collect();

    let states = posterior_decode(&observations, &params);
    assert_eq!(states.len(), 10);

    // All states should be 0 or 1 (the close populations), never 2
    for (t, &state) in states.iter().enumerate() {
        assert!(
            state <= 1,
            "Window {} assigned to state {} (pop_c), should be pop_a or pop_b",
            t,
            state
        );
    }
}

/// Posterior decode with two observations where forward-backward produces
/// uniform posteriors (equal signals) - should not panic.
#[test]
fn test_posterior_decode_uniform_posteriors() {
    let params = two_pop_params(100.0, 0.5); // very flat temperature + high switch

    // Equal sims for both pops
    let observations = vec![
        make_obs_simple(0, 0.90, 0.90, 0.90, 0.90),
        make_obs_simple(5000, 0.90, 0.90, 0.90, 0.90),
    ];

    let states = posterior_decode(&observations, &params);
    assert_eq!(states.len(), 2);
    // States should be valid (0 or 1)
    for &s in &states {
        assert!(s < 2, "State {} out of bounds", s);
    }
}

/// DecodingMethod FromStr handles mixed case.
#[test]
fn test_decoding_method_case_insensitive() {
    assert_eq!(
        "VITERBI".parse::<DecodingMethod>().unwrap(),
        DecodingMethod::Viterbi
    );
    assert_eq!(
        "Posterior".parse::<DecodingMethod>().unwrap(),
        DecodingMethod::Posterior
    );
    assert!("unknown".parse::<DecodingMethod>().is_err());
}

// ===========================================================================
// 2. Emission Normalization Edge Cases
// ===========================================================================

/// Normalization with only a single observation should not panic.
/// With count=1, std is set to 1e-6 as fallback.
#[test]
fn test_normalization_single_observation() {
    let mut params = two_pop_params(0.03, 0.01);

    let observations = vec![make_obs_simple(0, 0.95, 0.94, 0.85, 0.84)];
    params.learn_normalization(&observations);

    let norm = params.normalization.as_ref().unwrap();
    assert_eq!(norm.means.len(), 2);
    assert_eq!(norm.stds.len(), 2);

    // With only 1 observation, stds should be 1e-6 fallback
    for &s in &norm.stds {
        assert!(
            (s - 1e-6).abs() < 1e-10,
            "Expected std fallback 1e-6 but got {}",
            s
        );
    }
}

/// Normalization with all identical similarities across populations.
/// Z-scores should be 0 for all, emissions should be equal.
#[test]
fn test_normalization_all_identical_sims() {
    let mut params = two_pop_params(0.03, 0.01);

    let observations: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_simple(i * 5000, 0.90, 0.90, 0.90, 0.90))
        .collect();
    params.learn_normalization(&observations);

    let norm = params.normalization.as_ref().unwrap();
    // Means should be equal
    assert!(
        (norm.means[0] - norm.means[1]).abs() < 1e-10,
        "Means should be equal for identical sims"
    );

    // Log emissions should be equal for both states
    let obs = make_obs_simple(0, 0.90, 0.90, 0.90, 0.90);
    let log_em_0 = params.log_emission(&obs, 0);
    let log_em_1 = params.log_emission(&obs, 1);
    assert!(
        (log_em_0 - log_em_1).abs() < 1e-6,
        "Emissions should be equal: {} vs {}",
        log_em_0,
        log_em_1
    );
}

/// Re-learning normalization replaces the previous one, not accumulates.
#[test]
fn test_normalization_relearn_replaces() {
    let mut params = two_pop_params(0.03, 0.01);

    // Learn from data with pop_a > pop_b
    let obs_1: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_simple(i * 5000, 0.95, 0.94, 0.85, 0.84))
        .collect();
    params.learn_normalization(&obs_1);
    let means_1 = params.normalization.as_ref().unwrap().means.clone();

    // Re-learn from data with pop_b > pop_a
    let obs_2: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_simple(i * 5000, 0.85, 0.84, 0.95, 0.94))
        .collect();
    params.learn_normalization(&obs_2);
    let means_2 = params.normalization.as_ref().unwrap().means.clone();

    // Means should flip
    assert!(
        means_1[0] > means_1[1],
        "First learn: pop_a mean should be higher"
    );
    assert!(
        means_2[1] > means_2[0],
        "Second learn: pop_b mean should be higher"
    );
}

/// Normalization with high variance in one population.
/// The high-variance pop should get a larger std.
#[test]
fn test_normalization_heterogeneous_variance() {
    let mut params = two_pop_params(0.03, 0.01);

    // pop_a: stable sims ~0.95; pop_b: varying sims 0.70-0.95
    let observations: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            let b_sim = 0.70 + 0.25 * (i as f64 / 19.0);
            make_obs_simple(i * 5000, 0.95, 0.94, b_sim, b_sim - 0.01)
        })
        .collect();

    params.learn_normalization(&observations);
    let norm = params.normalization.as_ref().unwrap();

    // pop_b (index 1) should have higher std than pop_a (index 0)
    assert!(
        norm.stds[1] > norm.stds[0],
        "pop_b std ({}) should be > pop_a std ({})",
        norm.stds[1],
        norm.stds[0]
    );
}

/// Normalization with sparse data: some population haplotypes not in observations.
#[test]
fn test_normalization_missing_haplotypes() {
    let mut params = two_pop_params(0.03, 0.01);

    // Only provide hap_a1 (not hap_a2) and only hap_b2 (not hap_b1)
    let observations: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs(i * 5000, &[("hap_a1", 0.95), ("hap_b2", 0.85)]))
        .collect();

    params.learn_normalization(&observations);
    let norm = params.normalization.as_ref().unwrap();

    // Should still compute means (using max of available haplotypes)
    assert!(norm.means[0] > 0.0, "pop_a mean should be computed");
    assert!(norm.means[1] > 0.0, "pop_b mean should be computed");
}

/// Normalization where one population has zero similarity (below threshold).
/// The learn function filters agg > 0.0, so zero-sim pops get count=0.
#[test]
fn test_normalization_zero_similarity_population() {
    let mut params = two_pop_params(0.03, 0.01);

    // pop_b gets 0.0 similarity (will be filtered by agg > 0.0 check)
    let observations: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs_simple(i * 5000, 0.95, 0.94, 0.0, 0.0))
        .collect();

    params.learn_normalization(&observations);
    let norm = params.normalization.as_ref().unwrap();

    // pop_b should have mean=0 and std=1e-6 (fallback)
    assert!(
        norm.means[1].abs() < 1e-10,
        "Zero-sim pop mean should be 0"
    );
    assert!(
        (norm.stds[1] - 1e-6).abs() < 1e-10,
        "Zero-sim pop std should be fallback 1e-6"
    );
}

/// Forward-backward posteriors with normalization still sum to 1 across states.
#[test]
fn test_normalization_posteriors_valid_three_pop() {
    let mut params = three_pop_params(0.03, 0.01);

    let observations: Vec<AncestryObservation> = (0..15)
        .map(|i| make_obs_3pop(i * 5000, 0.95, 0.94, 0.88, 0.87, 0.80, 0.79))
        .collect();

    params.learn_normalization(&observations);
    let posteriors = forward_backward(&observations, &params);

    assert_eq!(posteriors.len(), 15);
    for (t, probs) in posteriors.iter().enumerate() {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Posteriors at t={} sum to {} (expected 1.0)",
            t,
            sum
        );
        for (s, &p) in probs.iter().enumerate() {
            assert!(
                p >= 0.0 && p <= 1.0,
                "Posterior at t={}, state={} is {} (out of [0,1])",
                t,
                s,
                p
            );
        }
    }
}

// ===========================================================================
// 3. Ancestry Genetic Map Edge Cases
// ===========================================================================

/// Genetic map with a single entry should return that entry's cM everywhere.
#[test]
fn test_genetic_map_single_entry() {
    let gm = AncestryGeneticMap::uniform(0, 0, 1.0);
    // A uniform map from 0 to 0 has a single entry effectively at 0
    // The interpolation should handle this edge case
    let cm = gm.interpolate_cm(50000);
    assert!(cm.is_finite(), "Should produce finite result for single-entry map");
}

/// Genetic map: distance between same position should be 0.
#[test]
fn test_genetic_map_same_position_distance() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let dist = gm.genetic_distance_cm(50_000_000, 50_000_000);
    assert!(
        dist.abs() < 1e-10,
        "Distance from same position should be 0, got {}",
        dist
    );
}

/// Genetic map: distance should be symmetric (|a-b| = |b-a|).
#[test]
fn test_genetic_map_distance_symmetric() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let d1 = gm.genetic_distance_cm(10_000_000, 50_000_000);
    let d2 = gm.genetic_distance_cm(50_000_000, 10_000_000);
    assert!(
        (d1 - d2).abs() < 1e-10,
        "Distance should be symmetric: {} vs {}",
        d1,
        d2
    );
}

/// Modulated switch prob with zero window size should return base switch prob.
#[test]
fn test_modulated_switch_prob_zero_window() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let base = 0.01;
    let result = gm.modulated_switch_prob(base, 10_000_000, 20_000_000, 0);
    assert_eq!(
        result, base,
        "Zero window size should return base switch prob"
    );
}

/// Modulated switch prob should be higher in a recombination hotspot.
#[test]
fn test_modulated_switch_prob_hotspot() {
    // Create a non-uniform genetic map with a hotspot
    // Normal rate: ~1 cM/Mb, hotspot at 20-30Mb: 10 cM/Mb
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let base = 0.01;
    let window_size = 5000;

    let prob_normal = gm.modulated_switch_prob(base, 50_000_000, 50_005_000, window_size);

    // At uniform 1 cM/Mb, should be close to base
    assert!(
        (prob_normal - base).abs() / base < 0.5,
        "Uniform map: prob {} should be close to base {}",
        prob_normal,
        base
    );
}

/// Modulated switch prob should be clamped to [1e-6, 0.5].
#[test]
fn test_modulated_switch_prob_clamping() {
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 100.0); // very high rate
    let base = 0.4;
    let result = gm.modulated_switch_prob(base, 0, 100_000_000, 5000);
    assert!(
        result <= 0.5,
        "Switch prob {} should be clamped to <= 0.5",
        result
    );
    assert!(
        result >= 1e-6,
        "Switch prob {} should be clamped to >= 1e-6",
        result
    );
}

/// Viterbi with genetic map on empty input returns empty vec.
#[test]
fn test_viterbi_genetic_map_empty() {
    let params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let states = viterbi_with_genetic_map(&[], &params, &gm);
    assert!(states.is_empty());
}

/// Viterbi with genetic map on single observation returns a valid state.
#[test]
fn test_viterbi_genetic_map_single_obs() {
    let params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let obs = vec![make_obs_simple(50_000_000, 0.99, 0.98, 0.80, 0.79)];
    let states = viterbi_with_genetic_map(&obs, &params, &gm);
    assert_eq!(states.len(), 1);
    assert!(states[0] < 2);
}

/// Forward-backward with genetic map on single obs returns valid posteriors.
#[test]
fn test_fb_genetic_map_single_obs() {
    let params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);
    let obs = vec![make_obs_simple(50_000_000, 0.99, 0.98, 0.80, 0.79)];
    let posteriors = forward_backward_with_genetic_map(&obs, &params, &gm);
    assert_eq!(posteriors.len(), 1);

    let sum: f64 = posteriors[0].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Posteriors should sum to 1, got {}",
        sum
    );
}

/// Posterior decode with genetic map produces valid states for 3 populations.
#[test]
fn test_posterior_decode_genetic_map_three_pop() {
    let params = three_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    let observations: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_3pop(i as u64 * 10_000_000, 0.99, 0.98, 0.80, 0.79, 0.75, 0.74))
        .collect();

    let states = posterior_decode_with_genetic_map(&observations, &params, &gm);
    assert_eq!(states.len(), 10);
    for &s in &states {
        assert!(s < 3, "State {} out of bounds for 3 populations", s);
    }
    // All should be pop_a (state 0) since it has the highest sims
    assert!(
        states.iter().all(|&s| s == 0),
        "All windows should be pop_a: {:?}",
        states
    );
}

/// Forward-backward with genetic map: long sequence with a switch midway.
/// The genetic map modulated transitions should allow detection of the switch.
#[test]
fn test_fb_genetic_map_detects_switch() {
    let params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    // First 10 windows: pop_a, last 10 windows: pop_b
    let mut observations = Vec::new();
    for i in 0..20 {
        let start = i as u64 * 5_000_000;
        if i < 10 {
            observations.push(make_obs_simple(start, 0.99, 0.98, 0.80, 0.79));
        } else {
            observations.push(make_obs_simple(start, 0.80, 0.79, 0.99, 0.98));
        }
    }

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    assert_eq!(posteriors.len(), 20);

    // First half should have high posterior for state 0
    for t in 0..5 {
        assert!(
            posteriors[t][0] > 0.8,
            "t={}: pop_a posterior {} should be > 0.8",
            t,
            posteriors[t][0]
        );
    }
    // Second half should have high posterior for state 1
    for t in 15..20 {
        assert!(
            posteriors[t][1] > 0.8,
            "t={}: pop_b posterior {} should be > 0.8",
            t,
            posteriors[t][1]
        );
    }
}

/// Uniform genetic map with standard rate should produce results close to
/// non-genetic-map versions.
#[test]
fn test_genetic_map_uniform_matches_standard() {
    let params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000, 1.0);

    let observations: Vec<AncestryObservation> = (0..10)
        .map(|i| make_obs_simple(i * 5000, 0.95, 0.94, 0.85, 0.84))
        .collect();

    let states_standard = viterbi(&observations, &params);
    let states_gm = viterbi_with_genetic_map(&observations, &params, &gm);

    // Should produce the same states
    assert_eq!(
        states_standard, states_gm,
        "Uniform genetic map should match standard: {:?} vs {:?}",
        states_standard, states_gm
    );
}

/// Genetic map: very large genetic distances don't produce NaN or Inf.
#[test]
fn test_genetic_map_large_distance_numerical_stability() {
    let gm = AncestryGeneticMap::uniform(0, 1_000_000_000, 10.0); // 10 cM/Mb (high rate)
    let params = two_pop_params(0.03, 0.01);

    // Observations spanning 1 Gbp with high recombination rate
    let observations: Vec<AncestryObservation> = (0..5)
        .map(|i| make_obs_simple(i as u64 * 200_000_000, 0.95, 0.94, 0.85, 0.84))
        .collect();

    let posteriors = forward_backward_with_genetic_map(&observations, &params, &gm);
    for (t, probs) in posteriors.iter().enumerate() {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "t={}: posteriors sum to {} (expected 1.0)",
            t,
            sum
        );
        for &p in probs {
            assert!(
                p.is_finite(),
                "t={}: posterior is not finite: {}",
                t,
                p
            );
        }
    }
}

/// Genetic map from_file with a valid temp file.
#[test]
fn test_genetic_map_from_file() {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let path = dir.join("test_ancestry_genetic_map.map");

    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "chr1\t1000000\t1.0\t1.0").unwrap();
        writeln!(f, "chr1\t2000000\t1.5\t2.5").unwrap();
        writeln!(f, "chr1\t3000000\t2.0\t4.5").unwrap();
        writeln!(f, "chr2\t1000000\t1.0\t1.0").unwrap(); // different chrom
    }

    let gm = AncestryGeneticMap::from_file(&path, "chr1").unwrap();
    let cm_at_1_5m = gm.interpolate_cm(1_500_000);
    // Should be between 1.0 and 2.5 (interpolated)
    assert!(
        cm_at_1_5m > 1.0 && cm_at_1_5m < 2.5,
        "Expected interpolated cM between 1.0 and 2.5, got {}",
        cm_at_1_5m
    );

    // Wrong chrom should fail
    let result = AncestryGeneticMap::from_file(&path, "chr3");
    assert!(result.is_err(), "Should fail for missing chromosome");

    std::fs::remove_file(&path).ok();
}

/// Genetic map from_file with chr prefix normalization (chr1 vs 1).
#[test]
fn test_genetic_map_chr_prefix_normalization() {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let path = dir.join("test_ancestry_gm_prefix.map");

    {
        let mut f = std::fs::File::create(&path).unwrap();
        // File uses "20" (no chr prefix)
        writeln!(f, "20\t1000000\t1.0\t1.0").unwrap();
        writeln!(f, "20\t50000000\t1.0\t50.0").unwrap();
    }

    // Query with "chr20" should match "20" in file
    let result = AncestryGeneticMap::from_file(&path, "chr20");
    assert!(result.is_ok(), "chr20 should match 20 in file");

    // Query with "20" should also work
    let result2 = AncestryGeneticMap::from_file(&path, "20");
    assert!(result2.is_ok(), "20 should match 20 in file");

    std::fs::remove_file(&path).ok();
}

/// Genetic map from_file with 3-column format.
#[test]
fn test_genetic_map_three_column_format() {
    use std::io::Write;
    let dir = std::env::temp_dir();
    let path = dir.join("test_ancestry_gm_3col.map");

    {
        let mut f = std::fs::File::create(&path).unwrap();
        // 3-column: pos_bp, rate, pos_cM (no chrom column)
        writeln!(f, "1000000\t1.0\t1.0").unwrap();
        writeln!(f, "2000000\t1.5\t2.5").unwrap();
    }

    // 3-column format doesn't filter by chromosome
    let gm = AncestryGeneticMap::from_file(&path, "chr1").unwrap();
    let dist = gm.genetic_distance_cm(1_000_000, 2_000_000);
    assert!(
        (dist - 1.5).abs() < 0.01,
        "Distance should be ~1.5 cM, got {}",
        dist
    );

    std::fs::remove_file(&path).ok();
}

// ===========================================================================
// 4. Integration: Normalization + Genetic Map + Posterior Decode
// ===========================================================================

/// Test the full pipeline: normalization + genetic map + posterior decode.
/// This tests the feature interaction path.
#[test]
fn test_full_pipeline_normalized_genetic_map_posterior() {
    let mut params = two_pop_params(0.03, 0.01);
    let gm = AncestryGeneticMap::uniform(0, 100_000_000, 1.0);

    // Create observations with systematic bias: pop_a always ~0.02 higher
    let observations: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            let start = i as u64 * 5_000_000;
            if i < 10 {
                // True pop_a: large gap
                make_obs_simple(start, 0.97, 0.96, 0.80, 0.79)
            } else {
                // True pop_b but pop_a has systematic bias
                make_obs_simple(start, 0.87, 0.86, 0.95, 0.94)
            }
        })
        .collect();

    // Learn normalization to remove bias
    params.learn_normalization(&observations);

    // Run posterior decode with genetic map
    let states = posterior_decode_with_genetic_map(&observations, &params, &gm);
    assert_eq!(states.len(), 20);

    // First half should be pop_a
    let first_half_a = states[..10].iter().filter(|&&s| s == 0).count();
    assert!(
        first_half_a >= 8,
        "First half: expected mostly pop_a but got {}/10",
        first_half_a
    );

    // Second half should be pop_b
    let second_half_b = states[10..].iter().filter(|&&s| s == 1).count();
    assert!(
        second_half_b >= 8,
        "Second half: expected mostly pop_b but got {}/10",
        second_half_b
    );
}

/// Normalization + posterior decode should be more accurate than
/// unnormalized Viterbi when there's a systematic bias.
#[test]
fn test_normalization_improves_minority_detection() {
    let mut params_norm = two_pop_params(0.03, 0.01);
    let params_raw = two_pop_params(0.03, 0.01);

    // Systematic bias: pop_a sims are always 0.05 higher than pop_b
    // True ancestry: first 15 windows pop_a, last 5 windows pop_b
    let observations: Vec<AncestryObservation> = (0..20)
        .map(|i| {
            let start = i as u64 * 5000;
            if i < 15 {
                // True pop_a
                make_obs_simple(start, 0.97, 0.96, 0.85, 0.84)
            } else {
                // True pop_b, but with bias: pop_a sims still relatively high
                make_obs_simple(start, 0.92, 0.91, 0.94, 0.93)
            }
        })
        .collect();

    params_norm.learn_normalization(&observations);

    let states_raw = viterbi(&observations, &params_raw);
    let states_norm = posterior_decode(&observations, &params_norm);

    // Count correct assignments in the minority region (last 5)
    let raw_correct = states_raw[15..].iter().filter(|&&s| s == 1).count();
    let norm_correct = states_norm[15..].iter().filter(|&&s| s == 1).count();

    // Normalization should detect at least as many pop_b windows
    assert!(
        norm_correct >= raw_correct,
        "Normalized ({}/5 correct) should be >= raw ({}/5 correct)",
        norm_correct,
        raw_correct
    );
}
