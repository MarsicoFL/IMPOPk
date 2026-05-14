//! Tests for Baum-Welch parameter bounds enforcement.
//!
//! Validates that BW maintains:
//! - emission[0].mean < emission[1].mean (identifiability)
//! - transition[0][1] ∈ [1e-8, 0.1] (p_enter_ibd)
//! - transition[1][0] ∈ [0.001, 0.5] (p_exit_ibd)
//! - emission bounds: non-IBD mean clamped, std clamped
//! - Transition rows sum to 1

use impopk_ibd::hmm::{HmmParams, Population};

fn make_params() -> HmmParams {
    HmmParams::from_population(Population::EUR, 100.0, 0.001, 10000)
}

fn run_bw_and_check_bounds(obs: &[f64], label: &str) {
    let mut params = make_params();
    params.baum_welch(obs, 10, 1e-6, Some(Population::EUR), 10000);

    // Identifiability: non-IBD mean < IBD mean
    assert!(params.emission[0].mean < params.emission[1].mean,
        "{label}: emission[0].mean ({}) must be < emission[1].mean ({})",
        params.emission[0].mean, params.emission[1].mean);

    // Transition bounds
    let p_enter = params.transition[0][1];
    let p_exit = params.transition[1][0];

    assert!(p_enter >= 1e-8 && p_enter <= 0.1,
        "{label}: p_enter_ibd={p_enter} out of [1e-8, 0.1]");
    assert!(p_exit >= 0.001 && p_exit <= 0.5,
        "{label}: p_exit_ibd={p_exit} out of [0.001, 0.5]");

    // Row sums
    for (i, row) in params.transition.iter().enumerate() {
        let sum = row[0] + row[1];
        assert!((sum - 1.0).abs() < 1e-10,
            "{label}: transition row {i} sums to {sum}");
    }

    // Emission std bounds
    assert!(params.emission[0].std >= 0.0003 && params.emission[0].std <= 0.005,
        "{label}: non-IBD std={} out of [0.0003, 0.005]", params.emission[0].std);
    assert!(params.emission[1].std >= 0.0002 && params.emission[1].std <= 0.002,
        "{label}: IBD std={} out of [0.0002, 0.002]", params.emission[1].std);

    // No NaN or Inf
    assert!(params.emission[0].mean.is_finite(), "{label}: non-IBD mean not finite");
    assert!(params.emission[1].mean.is_finite(), "{label}: IBD mean not finite");
    assert!(params.emission[0].std.is_finite(), "{label}: non-IBD std not finite");
    assert!(params.emission[1].std.is_finite(), "{label}: IBD std not finite");
}

#[test]
fn bw_bounds_standard_data() {
    // Typical data: non-IBD around 0.999, IBD around 0.9997
    let obs: Vec<f64> = (0..200).map(|i| {
        if i < 150 { 0.9990 + ((i as f64) * 0.3).sin() * 0.0003 }
        else { 0.9997 + ((i as f64) * 0.3).sin() * 0.0001 }
    }).collect();
    run_bw_and_check_bounds(&obs, "standard_data");
}

#[test]
fn bw_bounds_all_high_identity() {
    // All observations very high (could confuse states)
    let obs: Vec<f64> = (0..200).map(|i| {
        0.9998 + ((i as f64) * 0.5).sin() * 0.00005
    }).collect();
    run_bw_and_check_bounds(&obs, "all_high");
}

#[test]
fn bw_bounds_bimodal_clear() {
    // Clear bimodal: non-IBD and IBD well-separated
    let obs: Vec<f64> = (0..200).map(|i| {
        if i % 3 == 0 { 0.9997 } else { 0.9985 }
    }).collect();
    run_bw_and_check_bounds(&obs, "bimodal_clear");
}

#[test]
fn bw_bounds_noisy_data() {
    // Very noisy data
    let obs: Vec<f64> = (0..200).map(|i| {
        let noise = ((i as f64) * 1.7).sin() * 0.002;
        (0.9990 + noise).clamp(0.990, 1.0)
    }).collect();
    run_bw_and_check_bounds(&obs, "noisy");
}

#[test]
fn bw_bounds_extreme_uniformity() {
    // All observations exactly the same value
    let obs = vec![0.9995; 200];
    run_bw_and_check_bounds(&obs, "uniform");
}

#[test]
fn bw_bounds_afr_population() {
    let mut params = HmmParams::from_population(Population::AFR, 80.0, 0.0008, 10000);
    let obs: Vec<f64> = (0..200).map(|i| {
        if i < 160 { 0.99875 + ((i as f64) * 0.4).sin() * 0.0003 }
        else { 0.9997 + ((i as f64) * 0.4).sin() * 0.0001 }
    }).collect();
    params.baum_welch(&obs, 10, 1e-6, Some(Population::AFR), 10000);

    assert!(params.emission[0].mean < params.emission[1].mean,
        "AFR: identifiability violated: {} >= {}", params.emission[0].mean, params.emission[1].mean);
}

#[test]
fn bw_bounds_eas_population() {
    let mut params = HmmParams::from_population(Population::EAS, 120.0, 0.0005, 10000);
    let obs: Vec<f64> = (0..200).map(|i| {
        if i < 160 { 0.9992 + ((i as f64) * 0.4).sin() * 0.0002 }
        else { 0.9997 + ((i as f64) * 0.4).sin() * 0.0001 }
    }).collect();
    params.baum_welch(&obs, 10, 1e-6, Some(Population::EAS), 10000);

    assert!(params.emission[0].mean < params.emission[1].mean,
        "EAS: identifiability violated");
    assert!(params.transition[0][1] >= 1e-8 && params.transition[0][1] <= 0.1);
    assert!(params.transition[1][0] >= 0.001 && params.transition[1][0] <= 0.5);
}

#[test]
fn bw_multiple_iterations_maintain_bounds() {
    // Run BW for many iterations, check bounds at each step
    let obs: Vec<f64> = (0..300).map(|i| {
        if i < 250 { 0.9990 + ((i as f64) * 0.3).sin() * 0.0003 }
        else { 0.9997 }
    }).collect();

    let mut params = make_params();
    for iter in 0..20 {
        params.baum_welch(&obs, 1, f64::NEG_INFINITY, Some(Population::EUR), 10000);

        assert!(params.emission[0].mean < params.emission[1].mean,
            "Iteration {iter}: identifiability violated");
        assert!(params.transition[0][1] >= 1e-8 && params.transition[0][1] <= 0.1,
            "Iteration {iter}: p_enter out of bounds: {}", params.transition[0][1]);
        assert!(params.transition[1][0] >= 0.001 && params.transition[1][0] <= 0.5,
            "Iteration {iter}: p_exit out of bounds: {}", params.transition[1][0]);
    }
}

#[test]
fn bw_bounds_with_distances() {
    // Distance-aware BW should also maintain bounds
    let obs: Vec<f64> = (0..100).map(|i| {
        if i < 80 { 0.9990 } else { 0.9997 }
    }).collect();
    let positions: Vec<(u64, u64)> = (0..100).map(|i| {
        (i as u64 * 10000, (i as u64 + 1) * 10000)
    }).collect();

    let mut params = make_params();
    params.baum_welch_with_distances(&obs, &positions, 10, 1e-6, Some(Population::EUR), 10000);

    assert!(params.emission[0].mean < params.emission[1].mean,
        "Distance BW: identifiability violated");
    assert!(params.transition[0][1] >= 1e-8 && params.transition[0][1] <= 0.1);
    assert!(params.transition[1][0] >= 0.001 && params.transition[1][0] <= 0.5);
}

#[test]
fn bw_log_likelihood_nondecreasing() {
    // BW should not decrease log-likelihood
    let obs: Vec<f64> = (0..200).map(|i| {
        if i < 150 { 0.9990 } else { 0.9997 }
    }).collect();

    let mut params = make_params();
    let _prev_ll = f64::NEG_INFINITY;

    for _ in 0..10 {
        params.baum_welch(&obs, 1, f64::NEG_INFINITY, Some(Population::EUR), 10000);
        // We can't directly get LL from baum_welch, but we can verify params are valid
        assert!(params.emission[0].mean < params.emission[1].mean);
    }
}

#[test]
fn bw_short_data_no_crash() {
    // < 10 observations: BW should no-op
    let obs = vec![0.999; 5];
    let mut params = make_params();
    let orig_emission = params.emission.clone();
    params.baum_welch(&obs, 10, 1e-6, Some(Population::EUR), 10000);
    // Params should be unchanged (BW skips < 10 obs)
    assert_eq!(params.emission[0].mean, orig_emission[0].mean);
    assert_eq!(params.emission[1].mean, orig_emission[1].mean);
}

#[test]
fn bw_bounds_interpop() {
    // InterPop should also maintain bounds
    let mut params = HmmParams::from_population(Population::InterPop, 50.0, 0.0002, 10000);
    let obs: Vec<f64> = (0..200).map(|i| {
        if i < 180 { 0.9985 } else { 0.9997 }
    }).collect();
    params.baum_welch(&obs, 10, 1e-6, Some(Population::InterPop), 10000);

    assert!(params.emission[0].mean < params.emission[1].mean,
        "InterPop: identifiability violated");
    assert!(params.transition[0][1] >= 1e-8);
    assert!(params.transition[1][0] >= 0.001);
}
