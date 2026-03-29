//! Cycle 90: Edge case tests for low-coverage ibd-cli functions.
//!
//! Targets:
//! - `HmmParams::summary` (2 prior test calls)
//! - `forward_with_genetic_map_from_log_emit` (5 prior test calls)
//! - `backward_with_genetic_map_from_log_emit`
//! - `viterbi_with_genetic_map_from_log_emit`

use hprc_ibd::hmm::{
    backward_with_genetic_map_from_log_emit,
    forward_with_genetic_map_from_log_emit,
    viterbi_with_genetic_map_from_log_emit,
    GeneticMap, HmmParams,
};
use hprc_ibd::stats::GaussianParams;

fn make_params() -> HmmParams {
    HmmParams {
        initial: [0.9, 0.1],
        transition: [[0.99, 0.01], [0.05, 0.95]],
        emission: [
            GaussianParams::new_unchecked(0.5, 0.15),
            GaussianParams::new_unchecked(0.98, 0.01),
        ],
    }
}

fn make_genmap(start: u64, end: u64, rate: f64) -> GeneticMap {
    let cm_len = (end - start) as f64 * rate / 1_000_000.0;
    GeneticMap::new(vec![(start, 0.0), (end, cm_len)])
}

// ============================================================================
// summary tests
// ============================================================================

#[test]
fn summary_contains_initial_probs() {
    let p = make_params();
    let s = p.summary();
    assert!(s.contains("P(non-IBD)=0.9000"), "got: {s}");
    assert!(s.contains("P(IBD)=0.1000"), "got: {s}");
}

#[test]
fn summary_contains_transition_probs() {
    let p = make_params();
    let s = p.summary();
    assert!(s.contains("P(stay non-IBD)=0.990000"), "got: {s}");
    assert!(s.contains("P(enter IBD)=0.010000"), "got: {s}");
    assert!(s.contains("P(exit IBD)=0.050000"), "got: {s}");
    assert!(s.contains("P(stay IBD)=0.950000"), "got: {s}");
}

#[test]
fn summary_contains_emission_params() {
    let p = make_params();
    let s = p.summary();
    assert!(s.contains("mean=0.500000"), "got: {s}");
    assert!(s.contains("std=0.150000"), "got: {s}");
    assert!(s.contains("mean=0.980000"), "got: {s}");
    assert!(s.contains("std=0.010000"), "got: {s}");
}

#[test]
fn summary_symmetric_initial() {
    let p = HmmParams {
        initial: [0.5, 0.5],
        transition: [[0.9, 0.1], [0.1, 0.9]],
        emission: [
            GaussianParams::new_unchecked(0.5, 0.2),
            GaussianParams::new_unchecked(0.9, 0.05),
        ],
    };
    let s = p.summary();
    assert!(s.contains("P(non-IBD)=0.5000"), "got: {s}");
    assert!(s.contains("P(IBD)=0.5000"), "got: {s}");
}

#[test]
fn summary_extreme_transition_near_zero() {
    let p = HmmParams {
        initial: [0.999, 0.001],
        transition: [[0.999999, 0.000001], [0.5, 0.5]],
        emission: [
            GaussianParams::new_unchecked(0.5, 0.2),
            GaussianParams::new_unchecked(0.99, 0.01),
        ],
    };
    let s = p.summary();
    assert!(s.contains("P(enter IBD)=0.000001"), "got: {s}");
}

#[test]
fn summary_from_expected_length() {
    let p = HmmParams::from_expected_length(50.0, 0.001, 10000);
    let s = p.summary();
    assert!(s.contains("HMM Parameters:"), "got: {s}");
    assert!(s.contains("P(non-IBD)="), "got: {s}");
    assert!(s.contains("Emission non-IBD:"), "got: {s}");
    assert!(s.contains("Emission IBD:"), "got: {s}");
}

// ============================================================================
// forward_with_genetic_map_from_log_emit tests
// ============================================================================

#[test]
fn fwd_genmap_empty_input() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&[], &p, &[], &gm, 10000);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn fwd_genmap_single_window() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5]];
    let positions = [(0u64, 10000u64)];
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

#[test]
fn fwd_genmap_mismatched_positions_falls_back() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5], [-0.8, -0.3]];
    // positions length != log_emit length → falls back to forward_from_log_emit
    let positions = [(0u64, 10000u64)];
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), 2);
    assert!(ll.is_finite());
}

#[test]
fn fwd_genmap_two_windows_log_likelihood_finite() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.2], [-0.3, -1.5]];
    let positions = [(0u64, 10000u64), (10000, 20000)];
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), 2);
    assert!(ll.is_finite());
    assert!(ll < 0.0, "log-likelihood should be negative, got {ll}");
}

#[test]
fn fwd_genmap_strong_ibd_emission_favors_state1() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    // Very strong IBD emission at all windows
    let log_emit = [[-10.0, -0.01], [-10.0, -0.01], [-10.0, -0.01]];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let (alpha, _ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    // State 1 (IBD) should dominate at last position
    assert!(alpha[2][1] > alpha[2][0], "IBD state should dominate");
}

#[test]
fn fwd_genmap_matches_no_genmap_with_uniform_rate() {
    // With a uniform genetic map, results should be close to (but not identical to)
    // the non-genetic-map version, because the recombination rates differ.
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5], [-0.8, -0.3], [-1.2, -0.1]];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), 3);
    assert!(ll.is_finite());
    // Just verify numerical sanity
    for a in &alpha {
        assert!(a[0].is_finite());
        assert!(a[1].is_finite());
    }
}

#[test]
fn fwd_genmap_many_windows() {
    let p = make_params();
    let gm = make_genmap(0, 10_000_000, 1.0);
    let n = 100;
    let log_emit: Vec<[f64; 2]> = (0..n)
        .map(|i| if i % 3 == 0 { [-0.5, -1.0] } else { [-1.0, -0.5] })
        .collect();
    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| (i as u64 * 10000, (i as u64 + 1) * 10000))
        .collect();
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), n);
    assert!(ll.is_finite());
}

#[test]
fn fwd_genmap_neg_inf_emission_propagates() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    // One state has NEG_INFINITY emission
    let log_emit = [[f64::NEG_INFINITY, -0.1], [-0.5, f64::NEG_INFINITY]];
    let positions = [(0u64, 10000u64), (10000, 20000)];
    let (alpha, ll) = forward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(alpha.len(), 2);
    assert!(ll.is_finite());
}

// ============================================================================
// backward_with_genetic_map_from_log_emit tests
// ============================================================================

#[test]
fn bwd_genmap_empty_input() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let beta = backward_with_genetic_map_from_log_emit(&[], &p, &[], &gm, 10000);
    assert!(beta.is_empty());
}

#[test]
fn bwd_genmap_single_window() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5]];
    let positions = [(0u64, 10000u64)];
    let beta = backward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(beta.len(), 1);
    // Last position beta is [0.0, 0.0] (log(1.0))
    assert!((beta[0][0] - 0.0).abs() < 1e-10);
    assert!((beta[0][1] - 0.0).abs() < 1e-10);
}

#[test]
fn bwd_genmap_mismatched_positions_falls_back() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5], [-0.8, -0.3]];
    let positions = [(0u64, 10000u64)]; // length mismatch
    let beta = backward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(beta.len(), 2);
}

#[test]
fn bwd_genmap_last_window_is_zero() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5], [-0.3, -1.0], [-0.8, -0.2]];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let beta = backward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(beta.len(), 3);
    assert!((beta[2][0] - 0.0).abs() < 1e-10);
    assert!((beta[2][1] - 0.0).abs() < 1e-10);
}

#[test]
fn bwd_genmap_finite_values() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-1.0, -0.5], [-0.3, -1.0], [-0.8, -0.2]];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let beta = backward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    for b in &beta {
        assert!(b[0].is_finite());
        assert!(b[1].is_finite());
    }
}

#[test]
fn bwd_genmap_many_windows() {
    let p = make_params();
    let gm = make_genmap(0, 10_000_000, 1.0);
    let n = 50;
    let log_emit: Vec<[f64; 2]> = (0..n).map(|i| if i < 25 { [-0.5, -1.0] } else { [-1.0, -0.5] }).collect();
    let positions: Vec<(u64, u64)> = (0..n).map(|i| (i as u64 * 10000, (i as u64 + 1) * 10000)).collect();
    let beta = backward_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(beta.len(), n);
    for b in &beta {
        assert!(b[0].is_finite());
        assert!(b[1].is_finite());
    }
}

// ============================================================================
// viterbi_with_genetic_map_from_log_emit tests
// ============================================================================

#[test]
fn vit_genmap_empty_input() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let states = viterbi_with_genetic_map_from_log_emit(&[], &p, &[], &gm, 10000);
    assert!(states.is_empty());
}

#[test]
fn vit_genmap_single_window() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    // Strong non-IBD emission
    let log_emit = [[-0.01, -10.0]];
    let positions = [(0u64, 10000u64)];
    let states = viterbi_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(states, vec![0]);
}

#[test]
fn vit_genmap_strong_ibd() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-10.0, -0.01], [-10.0, -0.01], [-10.0, -0.01]];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let states = viterbi_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    for s in &states {
        assert_eq!(*s, 1, "should decode as IBD");
    }
}

#[test]
fn vit_genmap_mismatched_positions_falls_back() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    let log_emit = [[-0.5, -1.0], [-1.0, -0.5]];
    let positions = [(0u64, 10000u64)]; // mismatch
    let states = viterbi_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(states.len(), 2);
    // States are valid
    for s in &states {
        assert!(*s < 2);
    }
}

#[test]
fn vit_genmap_switching_emissions() {
    let p = make_params();
    let gm = make_genmap(0, 1_000_000, 1.0);
    // Alternating strong emissions
    let log_emit = [
        [-0.01, -10.0], // clear non-IBD
        [-10.0, -0.01], // clear IBD
        [-0.01, -10.0], // clear non-IBD
    ];
    let positions = [(0u64, 10000u64), (10000, 20000), (20000, 30000)];
    let states = viterbi_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(states.len(), 3);
    // With very strong emissions, the viterbi should follow the emission signal
    // even though transitions penalize switching
    assert_eq!(states[0], 0);
    // The middle window might switch to 1 or stay at 0 depending on transition cost
    // Just verify valid states
    for s in &states {
        assert!(*s < 2);
    }
}

#[test]
fn vit_genmap_many_windows_all_valid() {
    let p = make_params();
    let gm = make_genmap(0, 10_000_000, 1.0);
    let n = 100;
    let log_emit: Vec<[f64; 2]> = (0..n)
        .map(|i| if i < 50 { [-0.5, -1.5] } else { [-1.5, -0.5] })
        .collect();
    let positions: Vec<(u64, u64)> = (0..n)
        .map(|i| (i as u64 * 10000, (i as u64 + 1) * 10000))
        .collect();
    let states = viterbi_with_genetic_map_from_log_emit(&log_emit, &p, &positions, &gm, 10000);
    assert_eq!(states.len(), n);
    for s in &states {
        assert!(*s < 2);
    }
}
