//! Tests for two-pass ancestry inference and population temperature scaling.

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    scale_temperature_for_populations,
    estimate_proportions_from_states, estimate_per_state_switch_rates,
    viterbi, forward_backward,
};

fn make_populations(n: usize) -> Vec<AncestralPopulation> {
    (0..n).map(|i| AncestralPopulation {
        name: format!("pop{}", i),
        haplotypes: vec![format!("pop{}#HAP1", i), format!("pop{}#HAP2", i)],
    }).collect()
}

fn make_obs(
    start: u64,
    sims: &[(usize, f64)],  // (pop_idx, similarity)
    n_pops: usize,
) -> AncestryObservation {
    let mut similarities = HashMap::new();
    for &(pop_idx, sim) in sims {
        similarities.insert(format!("pop{}#HAP1", pop_idx), sim);
        similarities.insert(format!("pop{}#HAP2", pop_idx), sim - 0.005);
    }
    // Fill missing populations with low similarity
    for i in 0..n_pops {
        similarities.entry(format!("pop{}#HAP1", i)).or_insert(0.90);
        similarities.entry(format!("pop{}#HAP2", i)).or_insert(0.895);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

// === scale_temperature_for_populations tests ===

#[test]
fn test_scale_temp_populations_reference_k3() {
    // At reference k=3, correction should be ~1.0
    let result = scale_temperature_for_populations(0.03, 3);
    assert!((result - 0.03).abs() < 0.001, "k=3 should be near no-op, got {}", result);
}

#[test]
fn test_scale_temp_populations_k5_decreases() {
    let base = 0.05;
    let result = scale_temperature_for_populations(base, 5);
    assert!(result < base, "k=5 should decrease temperature: {} < {}", result, base);
    assert!(result > base * 0.5, "k=5 reduction shouldn't be extreme: {}", result);
}

#[test]
fn test_scale_temp_populations_k2_increases() {
    let base = 0.05;
    let result = scale_temperature_for_populations(base, 2);
    assert!(result > base, "k=2 should increase temperature: {} > {}", result, base);
}

#[test]
fn test_scale_temp_populations_k1_unchanged() {
    let base = 0.05;
    let result = scale_temperature_for_populations(base, 1);
    assert_eq!(result, base, "k=1 should return unchanged temperature");
}

#[test]
fn test_scale_temp_populations_k10_further_decrease() {
    let base = 0.05;
    let k5 = scale_temperature_for_populations(base, 5);
    let k10 = scale_temperature_for_populations(base, 10);
    assert!(k10 < k5, "k=10 should be lower than k=5: {} < {}", k10, k5);
}

#[test]
fn test_scale_temp_populations_clamped_low() {
    let result = scale_temperature_for_populations(0.0005, 100);
    assert!(result >= 0.001, "result should be clamped >= 0.001, got {}", result);
}

#[test]
fn test_scale_temp_populations_clamped_high() {
    let result = scale_temperature_for_populations(5.0, 2);
    assert!(result <= 1.0, "result should be clamped <= 1.0, got {}", result);
}

#[test]
fn test_scale_temp_populations_monotonic() {
    let base = 0.05;
    let mut prev = scale_temperature_for_populations(base, 2);
    for k in 3..=20 {
        let curr = scale_temperature_for_populations(base, k);
        assert!(curr <= prev + 1e-10, "should be monotonically decreasing: k={}, {} > {}", k, curr, prev);
        prev = curr;
    }
}

// === estimate_proportions_from_states tests ===

#[test]
fn test_proportions_empty_states() {
    let props = estimate_proportions_from_states(&[], 3);
    assert_eq!(props.len(), 3);
    for &p in &props {
        assert!((p - 1.0 / 3.0).abs() < 1e-6);
    }
}

#[test]
fn test_proportions_uniform() {
    let states = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let props = estimate_proportions_from_states(&states, 3);
    assert_eq!(props.len(), 3);
    for &p in &props {
        assert!((p - 1.0 / 3.0).abs() < 0.01, "expected ~0.333, got {}", p);
    }
}

#[test]
fn test_proportions_dominant() {
    // 80% state 0, 10% each for 1 and 2
    let mut states = vec![0; 80];
    states.extend(vec![1; 10]);
    states.extend(vec![2; 10]);
    let props = estimate_proportions_from_states(&states, 3);
    assert!((props[0] - 0.80).abs() < 0.01);
    assert!((props[1] - 0.10).abs() < 0.01);
    assert!((props[2] - 0.10).abs() < 0.01);
}

#[test]
fn test_proportions_single_state() {
    let states = vec![1; 100];
    let props = estimate_proportions_from_states(&states, 3);
    assert!(props[1] > 0.99, "dominant state should be near 1.0");
    // Other states floored at ~1e-6 (then renormalized)
    assert!(props[0] < 0.001);
    assert!(props[2] < 0.001);
}

#[test]
fn test_proportions_sum_to_one() {
    let states = vec![0, 0, 0, 1, 2, 2];
    let props = estimate_proportions_from_states(&states, 3);
    let sum: f64 = props.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "proportions should sum to 1.0, got {}", sum);
}

#[test]
fn test_proportions_n_states_zero() {
    // n_states=0 uses .max(1) fallback, returns single-element uniform distribution
    let props = estimate_proportions_from_states(&[0, 1], 0);
    assert_eq!(props.len(), 1);
    assert!((props[0] - 1.0).abs() < 1e-10);
}

// === estimate_per_state_switch_rates tests ===

#[test]
fn test_switch_rates_empty() {
    let rates = estimate_per_state_switch_rates(&[], 3);
    assert_eq!(rates.len(), 3);
    for &r in &rates {
        assert_eq!(r, 0.001);
    }
}

#[test]
fn test_switch_rates_no_switches() {
    let states = vec![0; 100];
    let rates = estimate_per_state_switch_rates(&states, 3);
    // State 0 has 100 windows and 0 switches → clamped to 0.0001
    assert_eq!(rates[0], 0.0001);
    // States 1, 2 have 0 windows → fallback 0.001
    assert_eq!(rates[1], 0.001);
    assert_eq!(rates[2], 0.001);
}

#[test]
fn test_switch_rates_high_switch_rate() {
    // Alternating: 0,1,0,1,0,1,... → 50% switch rate, clamped to 0.05
    let states: Vec<usize> = (0..100).map(|i| i % 2).collect();
    let rates = estimate_per_state_switch_rates(&states, 2);
    assert_eq!(rates[0], 0.05, "high switch rate should be clamped to 0.05");
    assert_eq!(rates[1], 0.05);
}

#[test]
fn test_switch_rates_mixed() {
    // State 0: long tract (50 windows), State 1: short tracts (10 windows each), State 2: rare
    let mut states = vec![0; 50]; // 0 switches in 50 windows
    states.extend(vec![1; 10]);   // 0 switches in 10 windows
    states.extend(vec![0; 20]);
    states.extend(vec![2; 5]);
    states.extend(vec![0; 15]);
    let rates = estimate_per_state_switch_rates(&states, 3);
    // State 0 has most windows and few switches
    // State 1 has fewer windows
    assert!(rates[0] < rates[1] || rates[1] == 0.001,
        "state 0 should have lower switch rate or state 1 should be fallback");
}

#[test]
fn test_switch_rates_few_windows_fallback() {
    // State with < 10 windows should use fallback
    let states = vec![0, 0, 0, 1, 1, 0, 0, 0]; // state 1 has only 2 windows
    let rates = estimate_per_state_switch_rates(&states, 2);
    assert_eq!(rates[1], 0.001, "state with <10 windows should use fallback");
}

// === AncestryHmmParams::set_initial_probs tests ===

#[test]
fn test_set_initial_probs_normalizes() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_initial_probs(&[0.7, 0.2, 0.1]);
    let sum: f64 = params.initial.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "initial probs should sum to 1.0");
    assert!((params.initial[0] - 0.7).abs() < 1e-10);
    assert!((params.initial[1] - 0.2).abs() < 1e-10);
    assert!((params.initial[2] - 0.1).abs() < 1e-10);
}

#[test]
fn test_set_initial_probs_unnormalized_input() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_initial_probs(&[2.0, 1.0, 1.0]);
    assert!((params.initial[0] - 0.5).abs() < 1e-10);
    assert!((params.initial[1] - 0.25).abs() < 1e-10);
}

#[test]
fn test_set_initial_probs_wrong_length_no_op() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let original = params.initial.clone();
    params.set_initial_probs(&[0.5, 0.5]); // wrong length
    assert_eq!(params.initial, original, "wrong length should be no-op");
}

#[test]
fn test_set_initial_probs_all_zero_no_op() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let original = params.initial.clone();
    params.set_initial_probs(&[0.0, 0.0, 0.0]);
    assert_eq!(params.initial, original, "all-zero should be no-op");
}

// === AncestryHmmParams::set_proportional_transitions tests ===

#[test]
fn test_proportional_transitions_rows_sum_to_one() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let proportions = vec![0.7, 0.2, 0.1];
    let switch_rates = vec![0.001, 0.002, 0.003];
    params.set_proportional_transitions(&proportions, &switch_rates);

    for i in 0..3 {
        let sum: f64 = params.transitions[i].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "row {} should sum to 1.0, got {}", i, sum);
    }
}

#[test]
fn test_proportional_transitions_asymmetric() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let proportions = vec![0.8, 0.15, 0.05];
    let switch_rates = vec![0.01, 0.01, 0.01];
    params.set_proportional_transitions(&proportions, &switch_rates);

    // From state 1, switching to state 0 (prop=0.8) should be more likely than to state 2 (prop=0.05)
    assert!(params.transitions[1][0] > params.transitions[1][2],
        "T[1→0] ({}) should be > T[1→2] ({})", params.transitions[1][0], params.transitions[1][2]);

    // From state 2, switching to state 0 should dominate
    assert!(params.transitions[2][0] > params.transitions[2][1],
        "T[2→0] ({}) should be > T[2→1] ({})", params.transitions[2][0], params.transitions[2][1]);
}

#[test]
fn test_proportional_transitions_stays_correct() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let proportions = vec![0.5, 0.3, 0.2];
    let switch_rates = vec![0.01, 0.02, 0.03];
    params.set_proportional_transitions(&proportions, &switch_rates);

    // Self-transition = 1 - switch_rate
    assert!((params.transitions[0][0] - 0.99).abs() < 1e-10);
    assert!((params.transitions[1][1] - 0.98).abs() < 1e-10);
    assert!((params.transitions[2][2] - 0.97).abs() < 1e-10);
}

#[test]
fn test_proportional_transitions_per_state_rates() {
    let pops = make_populations(2);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let proportions = vec![0.5, 0.5];
    let switch_rates = vec![0.001, 0.01]; // state 1 switches 10x more
    params.set_proportional_transitions(&proportions, &switch_rates);

    assert!((params.transitions[0][0] - 0.999).abs() < 1e-10);
    assert!((params.transitions[1][1] - 0.99).abs() < 1e-10);
    // State 1 switches more often
    assert!(params.transitions[1][0] > params.transitions[0][1]);
}

#[test]
fn test_proportional_transitions_wrong_length_no_op() {
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    let original = params.transitions.clone();
    params.set_proportional_transitions(&[0.5, 0.5], &[0.01, 0.01]); // wrong length
    assert_eq!(params.transitions, original);
}

// === Two-pass integration tests ===

#[test]
fn test_two_pass_uniform_ancestry_near_identical() {
    // If ancestry is truly uniform (1/3 each), two-pass should be nearly identical to one-pass
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // Generate observations alternating ancestry
    let observations: Vec<AncestryObservation> = (0..30u64).map(|i| {
        let dominant = (i % 3) as usize;
        make_obs(i * 10000, &[(dominant, 0.98)], 3)
    }).collect();

    // One-pass
    let states_1pass = viterbi(&observations, &params);

    // Two-pass
    let proportions = estimate_proportions_from_states(&states_1pass, 3);
    let switch_rates = estimate_per_state_switch_rates(&states_1pass, 3);
    let mut params2 = params.clone();
    params2.set_initial_probs(&proportions);
    params2.set_proportional_transitions(&proportions, &switch_rates);
    let states_2pass = viterbi(&observations, &params2);

    // Should agree on most windows
    let agreement = states_1pass.iter().zip(&states_2pass)
        .filter(|(a, b)| a == b).count();
    let pct = agreement as f64 / states_1pass.len() as f64;
    assert!(pct > 0.8, "uniform ancestry should give high agreement: {:.1}%", pct * 100.0);
}

#[test]
fn test_two_pass_dominant_ancestry_improves() {
    // 90% pop0 + 10% pop1 → two-pass should set informative priors
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    // 90 windows of pop0, 10 windows of pop1
    let mut observations: Vec<AncestryObservation> = (0..90).map(|i| {
        make_obs(i * 10000, &[(0, 0.98)], 3)
    }).collect();
    observations.extend((90..100).map(|i| {
        make_obs(i * 10000, &[(1, 0.97)], 3)
    }));

    // One-pass: uses uniform prior 1/3
    let states_1pass = viterbi(&observations, &params);

    // Two-pass
    let proportions = estimate_proportions_from_states(&states_1pass, 3);
    assert!(proportions[0] > 0.7, "pass 1 should detect pop0 dominance: {:.2}", proportions[0]);

    let switch_rates = estimate_per_state_switch_rates(&states_1pass, 3);
    let mut params2 = params.clone();
    params2.set_initial_probs(&proportions);
    params2.set_proportional_transitions(&proportions, &switch_rates);

    // Verify the prior was updated
    assert!(params2.initial[0] > 0.5, "pop0 prior should be dominant");

    // Both passes should detect the pattern
    let pop0_count_1 = states_1pass.iter().filter(|&&s| s == 0).count();
    let states_2pass = viterbi(&observations, &params2);
    let pop0_count_2 = states_2pass.iter().filter(|&&s| s == 0).count();

    // Two-pass should assign at least as many windows to pop0
    assert!(pop0_count_2 >= pop0_count_1 - 2,
        "two-pass pop0 count ({}) should be >= one-pass ({})", pop0_count_2, pop0_count_1);
}

#[test]
fn test_two_pass_posteriors_valid() {
    // Verify posteriors from two-pass sum to 1.0
    let pops = make_populations(3);
    let mut params = AncestryHmmParams::new(pops, 0.001);
    params.set_temperature(0.03);

    let observations: Vec<AncestryObservation> = (0..20).map(|i| {
        make_obs(i * 10000, &[(0, 0.98)], 3)
    }).collect();

    let states = viterbi(&observations, &params);
    let proportions = estimate_proportions_from_states(&states, 3);
    let switch_rates = estimate_per_state_switch_rates(&states, 3);

    let mut params2 = params.clone();
    params2.set_initial_probs(&proportions);
    params2.set_proportional_transitions(&proportions, &switch_rates);

    let posteriors = forward_backward(&observations, &params2);
    for (t, probs) in posteriors.iter().enumerate() {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "posteriors at window {} should sum to 1.0, got {}", t, sum);
    }
}

#[test]
fn test_proportional_transitions_equal_proportions_equals_uniform() {
    // Equal proportions should give same result as uniform transitions
    let pops = make_populations(3);
    let params_uniform = AncestryHmmParams::new(pops.clone(), 0.001);
    let mut params_prop = AncestryHmmParams::new(pops, 0.001);

    let equal_props = vec![1.0 / 3.0; 3];
    let equal_rates = vec![0.001; 3];
    params_prop.set_proportional_transitions(&equal_props, &equal_rates);

    for i in 0..3 {
        for j in 0..3 {
            assert!((params_uniform.transitions[i][j] - params_prop.transitions[i][j]).abs() < 1e-10,
                "T[{}][{}]: uniform={}, prop={}", i, j,
                params_uniform.transitions[i][j], params_prop.transitions[i][j]);
        }
    }
}

// === Edge cases ===

#[test]
fn test_proportions_out_of_range_state() {
    // States with indices >= n_states should be ignored
    let states = vec![0, 1, 5, 10, 0, 1]; // 5 and 10 are out of range for n_states=3
    let props = estimate_proportions_from_states(&states, 3);
    assert_eq!(props.len(), 3);
    let sum: f64 = props.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    // Only states 0 and 1 counted (2 each out of 4 valid)
    assert!((props[0] - props[1]).abs() < 0.01);
}

#[test]
fn test_switch_rates_single_window() {
    let rates = estimate_per_state_switch_rates(&[0], 3);
    assert_eq!(rates, vec![0.001; 3]);
}

#[test]
fn test_scale_temp_populations_large_k() {
    // Should not panic for large k
    let result = scale_temperature_for_populations(0.05, 1000);
    assert!(result > 0.0);
    assert!(result <= 0.05);
}
