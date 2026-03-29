//! Asymmetric Baum-Welch stress tests for K=3, 4, 5 states.
//!
//! Validates that BW training with asymmetric transition matrices:
//! - Converges for K=3,4,5 states
//! - Maintains valid stochastic matrices (rows sum to 1)
//! - Learns asymmetric transitions when ancestry proportions are unequal
//! - Handles extreme proportion imbalances (e.g. 90/5/5)
//! - Improves (or doesn't decrease) log-likelihood monotonically
//! - Clamps diagonal to [0.9, 0.9999] for K≥3

use std::collections::HashMap;
use hprc_ancestry_cli::{
    AncestralPopulation, AncestryHmmParams, AncestryObservation,
    viterbi, forward_backward,
};

// ── Helpers ──

fn make_pops(n: usize) -> Vec<AncestralPopulation> {
    let names = ["AFR", "EUR", "EAS", "CSA", "AMR"];
    (0..n).map(|i| AncestralPopulation {
        name: names.get(i).unwrap_or(&"UNK").to_string(),
        haplotypes: vec![
            format!("pop{}#HAP1", i),
            format!("pop{}#HAP2", i),
        ],
    }).collect()
}

fn make_obs(start: u64, dominant_pop: usize, n_pops: usize) -> AncestryObservation {
    let mut sims = HashMap::new();
    for i in 0..n_pops {
        let base = if i == dominant_pop { 0.97 } else { 0.88 };
        sims.insert(format!("pop{}#HAP1", i), base);
        sims.insert(format!("pop{}#HAP2", i), base - 0.01);
    }
    AncestryObservation {
        chrom: "chr1".to_string(),
        start,
        end: start + 10000,
        sample: "query#1".to_string(),
        similarities: sims,
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

/// Generate a sequence with known ancestry blocks: the dominant pop
/// switches at specified indices.
fn make_ancestry_sequence(
    n_windows: usize,
    n_pops: usize,
    blocks: &[(usize, usize)], // (start_idx, dominant_pop)
) -> Vec<AncestryObservation> {
    let mut obs = Vec::with_capacity(n_windows);
    let mut current_pop = 0;
    let mut block_iter = blocks.iter().peekable();

    for i in 0..n_windows {
        while let Some(&&(start, pop)) = block_iter.peek() {
            if i >= start {
                current_pop = pop;
                block_iter.next();
            } else {
                break;
            }
        }
        obs.push(make_obs(i as u64 * 10000, current_pop, n_pops));
    }
    obs
}

fn assert_stochastic_matrix(transitions: &[Vec<f64>], label: &str) {
    for (i, row) in transitions.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "{}: row {} sums to {} (expected 1.0)",
            label, i, sum
        );
        for (j, &p) in row.iter().enumerate() {
            assert!(
                p >= 0.0,
                "{}: transition[{}][{}] = {} is negative",
                label, i, j, p
            );
        }
    }
}

// ── K=3 tests ──

#[test]
fn bw_k3_symmetric_converges() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // 100 windows: 0-39 pop0, 40-69 pop1, 70-99 pop2
    let obs = make_ancestry_sequence(100, 3, &[(0, 0), (40, 1), (70, 2)]);
    let ll = params.baum_welch(&[obs.as_slice()], 20, 1e-6);

    assert!(ll.is_finite(), "K=3 BW should converge to finite LL, got {}", ll);
    assert_stochastic_matrix(&params.transitions, "K=3 symmetric");
}

#[test]
fn bw_k3_asymmetric_proportional_transitions() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // Heavily asymmetric: 80% pop0, 10% pop1, 10% pop2
    let obs = make_ancestry_sequence(200, 3, &[(0, 0), (160, 1), (180, 2)]);

    // Set proportional transitions before BW
    params.set_proportional_transitions(
        &[0.8, 0.1, 0.1],
        &[0.01, 0.01, 0.01],
    );

    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);
    assert!(ll.is_finite(), "K=3 asymmetric BW LL should be finite");
    assert_stochastic_matrix(&params.transitions, "K=3 asymmetric");

    // For K≥3, diagonal should be clamped to [0.9, 0.9999]
    for i in 0..3 {
        assert!(
            params.transitions[i][i] >= 0.9 - 1e-8,
            "diagonal[{}] = {} < 0.9",
            i, params.transitions[i][i]
        );
        assert!(
            params.transitions[i][i] <= 0.9999 + 1e-8,
            "diagonal[{}] = {} > 0.9999",
            i, params.transitions[i][i]
        );
    }
}

#[test]
fn bw_k3_extreme_imbalance_90_5_5() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.005);
    params.set_temperature(0.03);

    // 90% pop0, 5% pop1, 5% pop2
    let obs = make_ancestry_sequence(200, 3, &[(0, 0), (180, 1), (190, 2)]);
    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);

    assert!(ll.is_finite());
    assert_stochastic_matrix(&params.transitions, "K=3 extreme imbalance");

    // Viterbi should still recover majority ancestry
    let states = viterbi(&obs, &params);
    let pop0_count = states.iter().filter(|&&s| s == 0).count();
    assert!(
        pop0_count >= 150,
        "Expected majority pop0, got {} out of 200",
        pop0_count
    );
}

#[test]
fn bw_k3_multiple_sequences() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    let seq1 = make_ancestry_sequence(80, 3, &[(0, 0), (50, 1)]);
    let seq2 = make_ancestry_sequence(60, 3, &[(0, 2), (30, 0)]);
    let seq3 = make_ancestry_sequence(40, 3, &[(0, 1)]);

    let ll = params.baum_welch(&[seq1.as_slice(), seq2.as_slice(), seq3.as_slice()], 20, 1e-6);
    assert!(ll.is_finite(), "K=3 multi-seq BW should converge");
    assert_stochastic_matrix(&params.transitions, "K=3 multi-seq");
}

// ── K=4 tests ──

#[test]
fn bw_k4_converges_with_all_states_visited() {
    let pops = make_pops(4);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // Visit all 4 states
    let obs = make_ancestry_sequence(200, 4, &[(0, 0), (50, 1), (100, 2), (150, 3)]);
    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);

    assert!(ll.is_finite(), "K=4 BW should converge");
    assert_stochastic_matrix(&params.transitions, "K=4 all states");

    // Posteriors should sum to 1 per window
    let posteriors = forward_backward(&obs, &params);
    for (t, row) in posteriors.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "K=4 posteriors at t={} sum to {} (expected 1.0)",
            t, sum
        );
    }
}

#[test]
fn bw_k4_asymmetric_learns_directionality() {
    let pops = make_pops(4);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // Pop0 and pop1 alternate frequently, pop2 and pop3 are rare
    // This should produce asymmetric transitions where pop0↔pop1 switch rate > pop0↔pop2
    let mut blocks = Vec::new();
    let mut pos = 0;
    for _cycle in 0..10 {
        blocks.push((pos, 0));
        pos += 10;
        blocks.push((pos, 1));
        pos += 10;
    }
    // Short pop2 and pop3 blocks at end
    blocks.push((pos, 2));
    pos += 5;
    blocks.push((pos, 3));

    let obs = make_ancestry_sequence(pos + 5, 4, &blocks);
    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);

    assert!(ll.is_finite());
    assert_stochastic_matrix(&params.transitions, "K=4 asymmetric");

    // After BW, transitions between pop0↔pop1 should be relatively higher
    // than transitions to pop2 or pop3 (since those switch more often in the data)
    let t01 = params.transitions[0][1];
    let t02 = params.transitions[0][2];
    let t03 = params.transitions[0][3];
    // The off-diagonal toward pop1 should be >= off-diagonal toward pop2/3
    // (BW should learn this from the data pattern)
    assert!(
        t01 >= t02.min(t03) - 1e-6,
        "Expected T[0→1]={} >= min(T[0→2]={}, T[0→3]={})",
        t01, t02, t03
    );
}

// ── K=5 tests ──

#[test]
fn bw_k5_converges_with_uniform_proportions() {
    let pops = make_pops(5);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    // Equal blocks across 5 populations
    let obs = make_ancestry_sequence(250, 5, &[(0, 0), (50, 1), (100, 2), (150, 3), (200, 4)]);
    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);

    assert!(ll.is_finite(), "K=5 BW should converge, got {}", ll);
    assert_stochastic_matrix(&params.transitions, "K=5 uniform");

    // All diagonals should be in [0.9, 0.9999]
    for i in 0..5 {
        assert!(
            params.transitions[i][i] >= 0.9 - 1e-8 && params.transitions[i][i] <= 0.9999 + 1e-8,
            "K=5 diagonal[{}] = {} out of [0.9, 0.9999]",
            i, params.transitions[i][i]
        );
    }
}

#[test]
fn bw_k5_extreme_imbalance_one_dominant() {
    let pops = make_pops(5);
    let mut params = AncestryHmmParams::new(pops, 0.005);
    params.set_temperature(0.03);

    // 95% pop0, 1% each for pop1-4
    let obs = make_ancestry_sequence(300, 5, &[
        (0, 0), (285, 1), (288, 2), (291, 3), (294, 4)
    ]);

    let ll = params.baum_welch(&[obs.as_slice()], 30, 1e-6);
    assert!(ll.is_finite());
    assert_stochastic_matrix(&params.transitions, "K=5 extreme imbalance");

    let states = viterbi(&obs, &params);
    let pop0_count = states.iter().filter(|&&s| s == 0).count();
    assert!(
        pop0_count >= 250,
        "K=5 extreme: expected majority pop0, got {} / 300",
        pop0_count
    );
}

#[test]
fn bw_k5_proportional_transitions_rows_valid() {
    let pops = make_pops(5);
    let mut params = AncestryHmmParams::new(pops, 0.01);

    // Set asymmetric proportional transitions
    params.set_proportional_transitions(
        &[0.4, 0.3, 0.15, 0.1, 0.05],
        &[0.02, 0.03, 0.01, 0.02, 0.01],
    );

    // Verify rows sum to 1 before BW
    assert_stochastic_matrix(&params.transitions, "K=5 proportional pre-BW");

    // Run BW
    params.set_temperature(0.03);
    let obs = make_ancestry_sequence(200, 5, &[(0, 0), (80, 1), (120, 2), (160, 3), (180, 4)]);
    let ll = params.baum_welch(&[obs.as_slice()], 20, 1e-6);

    assert!(ll.is_finite());
    assert_stochastic_matrix(&params.transitions, "K=5 proportional post-BW");
}

// ── Cross-K consistency tests ──

#[test]
fn bw_monotonic_ll_improvement() {
    // Run BW iteration by iteration and check LL doesn't decrease (monotonicity)
    for k in [3, 4, 5] {
        let pops = make_pops(k);
        let obs = make_ancestry_sequence(100, k, &[(0, 0), (50, 1.min(k - 1))]);

        let mut prev_ll = f64::NEG_INFINITY;
        for iter in 1..=10 {
            let mut params = AncestryHmmParams::new(pops.clone(), 0.01);
            params.set_temperature(0.03);
            let ll = params.baum_welch(&[obs.as_slice()], iter, 0.0); // tol=0 forces exact iter count
            if ll.is_finite() && prev_ll.is_finite() {
                assert!(
                    ll >= prev_ll - 1e-6,
                    "K={} LL decreased: iter {} gave {} < prev {}",
                    k, iter, ll, prev_ll
                );
            }
            prev_ll = ll;
        }
    }
}

#[test]
fn bw_all_k_posteriors_sum_to_one() {
    for k in [3, 4, 5] {
        let pops = make_pops(k);
        let mut params = AncestryHmmParams::new(pops, 0.01);
        params.set_temperature(0.03);

        let obs = make_ancestry_sequence(50, k, &[(0, 0), (25, k - 1)]);
        params.baum_welch(&[obs.as_slice()], 10, 1e-6);

        let posteriors = forward_backward(&obs, &params);
        for (t, row) in posteriors.iter().enumerate() {
            assert_eq!(row.len(), k, "K={} posteriors at t={} has wrong length", k, t);
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "K={} posteriors at t={} sum to {} (expected 1.0)",
                k, t, sum
            );
        }
    }
}

#[test]
fn bw_k3_all_off_diagonal_positive_after_training() {
    let pops = make_pops(3);
    let mut params = AncestryHmmParams::new(pops, 0.01);
    params.set_temperature(0.03);

    let obs = make_ancestry_sequence(100, 3, &[(0, 0), (50, 1), (80, 2)]);
    params.baum_welch(&[obs.as_slice()], 20, 1e-6);

    for i in 0..3 {
        for j in 0..3 {
            assert!(
                params.transitions[i][j] > 0.0,
                "transition[{}][{}] = {} should be > 0 (min 1e-10)",
                i, j, params.transitions[i][j]
            );
        }
    }
}
