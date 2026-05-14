//! Edge-case tests for logit-space IBD pipeline (algo_dev cycles 3-4).
//!
//! Covers: logit/inv_logit boundary behavior, gaussian_to_logit_space edge cases,
//! from_population_logit parameter bounds, estimate_emissions_logit edge cases,
//! and baum_welch_logit stress/boundary scenarios.

use impopk_ibd::hmm::{
    infer_ibd, HmmParams, Population,
};
use impopk_ibd::stats::GaussianParams;
use impopk_ibd::stats::{gaussian_to_logit_space, inv_logit, logit, logit_transform_observations, LOGIT_CAP};

// =====================================================================
// logit() edge cases
// =====================================================================

#[test]
fn logit_exactly_half() {
    // logit(0.5) = ln(1) = 0.0
    let v = logit(0.5);
    assert!((v - 0.0).abs() < 1e-12, "logit(0.5) should be 0.0, got {}", v);
}

#[test]
fn logit_clamped_at_zero() {
    // Input 0.0 should be clamped to eps, giving a large negative value (but >= -LOGIT_CAP)
    let v = logit(0.0);
    assert!(v.is_finite(), "logit(0.0) must be finite");
    assert!(v >= -LOGIT_CAP, "logit(0.0) should be >= -LOGIT_CAP, got {}", v);
    assert!(v < 0.0, "logit(0.0) should be negative, got {}", v);
}

#[test]
fn logit_clamped_at_one() {
    // Input 1.0 should be clamped to 1-eps, giving a large positive value (but <= LOGIT_CAP)
    let v = logit(1.0);
    assert!(v.is_finite(), "logit(1.0) must be finite");
    assert!(v <= LOGIT_CAP, "logit(1.0) should be <= LOGIT_CAP, got {}", v);
    assert!(v > 0.0, "logit(1.0) should be positive, got {}", v);
}

#[test]
fn logit_negative_input_clamped() {
    // Negative identity doesn't exist biologically but shouldn't panic
    let v = logit(-1.0);
    assert!(v.is_finite());
    assert_eq!(v, -LOGIT_CAP, "logit(-1.0) should clamp to -LOGIT_CAP");
}

#[test]
fn logit_above_one_clamped() {
    // Identity >1.0 shouldn't happen but shouldn't panic
    let v = logit(1.5);
    assert!(v.is_finite());
    assert!(v <= LOGIT_CAP);
}

#[test]
fn logit_nan_propagates() {
    let v = logit(f64::NAN);
    // NaN clamped gives NaN (NaN.clamp returns NaN in Rust)
    // The behavior depends on implementation — just ensure no panic
    let _ = v; // No panic is the assertion
}

#[test]
fn logit_very_small_positive() {
    // Identity 1e-15 (near zero) should be clamped to eps=1e-10
    let v = logit(1e-15);
    assert!(v.is_finite());
    assert!(v < 0.0);
}

#[test]
fn logit_very_near_one() {
    // Identity 0.999999999999 (very near 1) should cap at LOGIT_CAP
    let v = logit(1.0 - 1e-12);
    assert!(v.is_finite());
    assert!(v <= LOGIT_CAP);
}

#[test]
fn logit_cap_symmetry() {
    // logit(0) and logit(1) should have symmetric absolute caps
    let v0 = logit(0.0);
    let v1 = logit(1.0);
    assert!((v0.abs() - v1.abs()).abs() < 1e-6,
        "caps should be symmetric: |logit(0)|={}, |logit(1)|={}", v0.abs(), v1.abs());
}

// =====================================================================
// inv_logit() edge cases
// =====================================================================

#[test]
fn inv_logit_zero() {
    // inv_logit(0) = 1/(1+1) = 0.5
    let v = inv_logit(0.0);
    assert!((v - 0.5).abs() < 1e-12);
}

#[test]
fn inv_logit_large_positive() {
    // inv_logit(100) should be very close to 1.0
    let v = inv_logit(100.0);
    assert!(v > 0.999999, "inv_logit(100) should be ~1.0, got {}", v);
    assert!(v <= 1.0);
}

#[test]
fn inv_logit_large_negative() {
    // inv_logit(-100) should be very close to 0.0
    let v = inv_logit(-100.0);
    assert!(v < 1e-6, "inv_logit(-100) should be ~0.0, got {}", v);
    assert!(v >= 0.0);
}

#[test]
fn inv_logit_positive_infinity() {
    let v = inv_logit(f64::INFINITY);
    assert_eq!(v, 1.0, "inv_logit(+inf) should be 1.0");
}

#[test]
fn inv_logit_negative_infinity() {
    let v = inv_logit(f64::NEG_INFINITY);
    assert_eq!(v, 0.0, "inv_logit(-inf) should be 0.0");
}

#[test]
fn inv_logit_nan() {
    let v = inv_logit(f64::NAN);
    assert!(v.is_nan(), "inv_logit(NaN) should be NaN");
}

// =====================================================================
// logit_transform_observations() edge cases
// =====================================================================

#[test]
fn logit_transform_single_element() {
    let obs = vec![0.999];
    let result = logit_transform_observations(&obs);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_finite());
}

#[test]
fn logit_transform_all_zeros() {
    let obs = vec![0.0; 5];
    let result = logit_transform_observations(&obs);
    assert_eq!(result.len(), 5);
    for &v in &result {
        assert!(v.is_finite());
        assert_eq!(v, -LOGIT_CAP);
    }
}

#[test]
fn logit_transform_all_ones() {
    let obs = vec![1.0; 5];
    let result = logit_transform_observations(&obs);
    assert_eq!(result.len(), 5);
    for &v in &result {
        assert!(v.is_finite());
        assert_eq!(v, LOGIT_CAP);
    }
}

#[test]
fn logit_transform_preserves_length() {
    for len in [0, 1, 10, 100, 1000] {
        let obs = vec![0.999; len];
        let result = logit_transform_observations(&obs);
        assert_eq!(result.len(), len);
    }
}

#[test]
fn logit_transform_mixed_extremes() {
    // Mix of 0, 1, and normal values — no panic, all finite
    let obs = vec![0.0, 0.5, 0.999, 1.0, 0.001, 0.9999];
    let result = logit_transform_observations(&obs);
    for &v in &result {
        assert!(v.is_finite(), "logit({}) should be finite", v);
    }
    // Check ordering: 0.0 < 0.001 < 0.5 < 0.999 < 0.9999 < 1.0
    assert!(result[0] < result[4]); // 0.0 < 0.001
    assert!(result[4] < result[1]); // 0.001 < 0.5
    assert!(result[1] < result[2]); // 0.5 < 0.999
    assert!(result[2] < result[5]); // 0.999 < 0.9999
    assert!(result[5] < result[3]); // 0.9999 < 1.0
}

// =====================================================================
// gaussian_to_logit_space() edge cases
// =====================================================================

#[test]
fn gaussian_to_logit_space_normal_case() {
    let gp = gaussian_to_logit_space(0.999, 0.001);
    assert!(gp.mean.is_finite());
    assert!(gp.std > 0.0);
    assert!(gp.mean > 5.0, "logit(0.999) should be > 5");
}

#[test]
fn gaussian_to_logit_space_mean_at_half() {
    // mean=0.5 → logit=0, denominator=0.25
    let gp = gaussian_to_logit_space(0.5, 0.1);
    assert!((gp.mean - 0.0).abs() < 1e-6, "logit(0.5)={}", gp.mean);
    // std = 0.1 / (0.5*0.5) = 0.4
    assert!((gp.std - 0.4).abs() < 1e-6, "expected std=0.4, got {}", gp.std);
}

#[test]
fn gaussian_to_logit_space_mean_near_zero() {
    // mean near 0: denominator ~ 0, should produce capped std
    let gp = gaussian_to_logit_space(1e-12, 0.001);
    assert!(gp.mean.is_finite());
    assert!(gp.std.is_finite());
    assert!(gp.std > 0.0);
}

#[test]
fn gaussian_to_logit_space_mean_near_one() {
    // mean near 1: denominator ~ 0, should produce capped std
    let gp = gaussian_to_logit_space(1.0 - 1e-12, 0.001);
    assert!(gp.mean.is_finite());
    assert!(gp.std.is_finite());
    assert!(gp.std > 0.0);
}

#[test]
fn gaussian_to_logit_space_zero_std() {
    // std=0 → logit_std = 0 / denom → 0, but clamped to max(0.01)
    let gp = gaussian_to_logit_space(0.999, 0.0);
    assert!(gp.std >= 0.01, "zero raw std should floor to 0.01, got {}", gp.std);
}

#[test]
fn gaussian_to_logit_space_very_large_std() {
    // Large std → logit_std gets capped at LOGIT_CAP * 0.5
    let gp = gaussian_to_logit_space(0.999, 100.0);
    assert!(gp.std <= LOGIT_CAP * 0.5 + 0.01,
        "large std should cap, got {}", gp.std);
}

#[test]
fn gaussian_to_logit_space_denominator_below_threshold() {
    // If mean*(1-mean) < 1e-15, use fallback LOGIT_CAP*0.5
    // mean = 1e-20 → mean*(1-mean) ≈ 1e-20 < 1e-15
    let gp = gaussian_to_logit_space(1e-20, 0.001);
    assert!((gp.std - LOGIT_CAP * 0.5).abs() < 0.1,
        "denominator ~0 should give std ≈ LOGIT_CAP*0.5={}, got {}",
        LOGIT_CAP * 0.5, gp.std);
}

#[test]
fn gaussian_to_logit_space_mean_exactly_zero() {
    let gp = gaussian_to_logit_space(0.0, 0.001);
    assert!(gp.mean.is_finite());
    assert!(gp.std > 0.0);
}

#[test]
fn gaussian_to_logit_space_mean_exactly_one() {
    let gp = gaussian_to_logit_space(1.0, 0.001);
    assert!(gp.mean.is_finite());
    assert!(gp.std > 0.0);
}

// =====================================================================
// from_population_logit() edge cases
// =====================================================================

#[test]
fn from_population_logit_tiny_p_enter() {
    // Very small p_enter_ibd
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 1e-8, 5000);
    assert!(params.emission[1].mean > params.emission[0].mean);
    assert!(params.transition[0][1] < 1e-7);
    // Row sums to 1
    assert!((params.transition[0][0] + params.transition[0][1] - 1.0).abs() < 1e-12);
}

#[test]
fn from_population_logit_large_expected_windows() {
    // expected_ibd_windows = 10000 → p_stay_ibd very high, clamped to 0.9999
    let params = HmmParams::from_population_logit(Population::EUR, 10000.0, 0.001, 5000);
    assert!(params.transition[1][1] >= 0.999);
}

#[test]
fn from_population_logit_small_expected_windows() {
    // expected_ibd_windows = 2 → p_stay_ibd = 0.5, clamped to 0.5
    let params = HmmParams::from_population_logit(Population::EUR, 2.0, 0.001, 5000);
    assert!((params.transition[1][1] - 0.5).abs() < 1e-10);
}

#[test]
fn from_population_logit_window_size_1() {
    // Window size 1 should still produce valid params
    let params = HmmParams::from_population_logit(Population::Generic, 50.0, 0.001, 1);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[1].mean > params.emission[0].mean);
}

#[test]
fn from_population_logit_interpop() {
    // InterPop should work and produce valid logit-space params
    let params = HmmParams::from_population_logit(Population::InterPop, 50.0, 0.001, 5000);
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean > params.emission[0].mean);
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn from_population_logit_p_enter_zero_panics() {
    HmmParams::from_population_logit(Population::EUR, 50.0, 0.0, 5000);
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn from_population_logit_p_enter_one_panics() {
    HmmParams::from_population_logit(Population::EUR, 50.0, 1.0, 5000);
}

// =====================================================================
// estimate_emissions_logit() edge cases
// =====================================================================

#[test]
fn estimate_emissions_logit_fewer_than_10_noop() {
    let raw = vec![0.999; 9];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let mean_before = params.emission[0].mean;
    let std_before = params.emission[0].std;

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    assert_eq!(params.emission[0].mean, mean_before);
    assert_eq!(params.emission[0].std, std_before);
}

#[test]
fn estimate_emissions_logit_exactly_10_runs() {
    let raw = vec![0.999; 10];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let mean_before = params.emission[0].mean;

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // With 10 identical observations, low variance branch should be triggered
    // Params may or may not change, but should remain valid
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    let _ = mean_before; // used above for comparison context
}

#[test]
fn estimate_emissions_logit_all_identical_low_variance() {
    // All identical → variance = 0 < 1e-6 → low-variance branch
    let raw = vec![0.999; 50];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // Should not panic, params remain valid
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

#[test]
fn estimate_emissions_logit_all_identical_high_value() {
    // All at IBD level → low variance, mean near IBD prior → updates state 1
    let raw = vec![0.9998; 30];
    let logit_obs = logit_transform_observations(&raw);
    let ibd_logit = gaussian_to_logit_space(0.9997, 0.0003);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // With all data near IBD, state 1 emission mean should be close to data mean
    let data_mean = logit_obs.iter().sum::<f64>() / logit_obs.len() as f64;
    // The function assigns to IBD if mean > ibd_prior - 1.0
    if data_mean > ibd_logit.mean - 1.0 {
        assert!((params.emission[1].mean - data_mean).abs() < 2.0,
            "IBD emission mean should be near data mean");
    }
}

#[test]
fn estimate_emissions_logit_bimodal_data() {
    // Clear bimodal: 70% non-IBD, 30% IBD
    let mut raw = vec![0.999; 70];
    raw.extend(vec![0.9997; 30]);
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // Should detect two clusters with IBD > non-IBD
    assert!(params.emission[1].mean > params.emission[0].mean,
        "After bimodal estimation: IBD mean ({}) should > non-IBD mean ({})",
        params.emission[1].mean, params.emission[0].mean);
}

#[test]
fn estimate_emissions_logit_no_population_prior() {
    // None population → defaults to Generic
    let raw = vec![0.999; 30];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::Generic, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, None, 5000);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
}

#[test]
fn estimate_emissions_logit_afr_high_prior_strength() {
    // AFR uses prior_strength=15 in EM branch
    let raw = vec![0.998; 50];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::AFR, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, Some(Population::AFR), 5000);

    assert!(params.emission[0].mean.is_finite());
}

#[test]
fn estimate_emissions_logit_interpop_high_prior_strength() {
    // InterPop uses prior_strength=10 in EM branch
    let raw = vec![0.998; 50];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::InterPop, 50.0, 0.001, 5000);

    params.estimate_emissions_logit(&logit_obs, Some(Population::InterPop), 5000);

    assert!(params.emission[0].mean.is_finite());
}

// =====================================================================
// baum_welch_logit() edge cases
// =====================================================================

#[test]
fn bw_logit_exactly_10_obs() {
    // Boundary: exactly 10 observations should run (n >= 10)
    let raw = vec![0.999; 10];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let mean_before = params.emission[0].mean;

    params.baum_welch_logit(&logit_obs, 5, 1e-6, Some(Population::EUR), 5000);

    // Should have run (may or may not change params with identical data)
    assert!(params.emission[0].mean.is_finite());
    let _ = mean_before;
}

#[test]
fn bw_logit_nine_obs_noop() {
    // Boundary: 9 observations → early return (n < 10)
    let raw = vec![0.999; 9];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    let mean_before = params.emission[0].mean;

    params.baum_welch_logit(&logit_obs, 50, 1e-6, Some(Population::EUR), 5000);

    assert_eq!(params.emission[0].mean, mean_before);
}

#[test]
fn bw_logit_single_iteration() {
    // max_iter=1: runs exactly one E-M step
    let mut raw = vec![0.999; 60];
    for i in 20..40 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    params.baum_welch_logit(&logit_obs, 1, 1e-6, Some(Population::EUR), 5000);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean > params.emission[0].mean);
}

#[test]
fn bw_logit_identifiability_constraint() {
    // If BW M-step would swap means (state0 >= state1), it resets to priors
    let raw = vec![0.999; 30];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    // Manually swap emissions to test identifiability reset
    let tmp = params.emission[0];
    params.emission[0] = params.emission[1];
    params.emission[1] = tmp;
    assert!(params.emission[0].mean >= params.emission[1].mean, "precondition: swapped");

    params.baum_welch_logit(&logit_obs, 5, 1e-6, Some(Population::EUR), 5000);

    // After BW, identifiability should be restored: IBD mean > non-IBD mean
    assert!(params.emission[1].mean > params.emission[0].mean,
        "BW should restore identifiability: IBD mean ({}) > non-IBD mean ({})",
        params.emission[1].mean, params.emission[0].mean);
}

#[test]
fn bw_logit_constant_obs_no_divergence() {
    // All identical observations: BW shouldn't diverge
    let raw = vec![0.999; 50];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    // All params should remain finite
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std.is_finite() && params.emission[0].std > 0.0);
    assert!(params.emission[1].std.is_finite() && params.emission[1].std > 0.0);
    for row in &params.transition {
        assert!((row[0] + row[1] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn bw_logit_emission_mean_bounds_respected() {
    // After BW, non-IBD mean should be within [prior-2, prior+2] logit range
    let mut raw = vec![0.999; 80];
    for i in 30..50 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    let prior_logit = gaussian_to_logit_space(
        Population::EUR.non_ibd_emission(5000).mean,
        Population::EUR.non_ibd_emission(5000).std,
    );

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    assert!(params.emission[0].mean >= prior_logit.mean - 2.0 - 1e-6,
        "non-IBD mean {} below bound {}", params.emission[0].mean, prior_logit.mean - 2.0);
    assert!(params.emission[0].mean <= prior_logit.mean + 2.0 + 1e-6,
        "non-IBD mean {} above bound {}", params.emission[0].mean, prior_logit.mean + 2.0);
}

#[test]
fn bw_logit_emission_std_bounds_respected() {
    // After BW, std should be in [0.2, 3.0]
    let mut raw = vec![0.999; 80];
    for i in 30..50 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    assert!(params.emission[0].std >= 0.2 - 1e-6, "std0={} below 0.2", params.emission[0].std);
    assert!(params.emission[0].std <= 3.0 + 1e-6, "std0={} above 3.0", params.emission[0].std);
    assert!(params.emission[1].std >= 0.2 - 1e-6, "std1={} below 0.2", params.emission[1].std);
    assert!(params.emission[1].std <= 3.0 + 1e-6, "std1={} above 3.0", params.emission[1].std);
}

#[test]
fn bw_logit_transition_bounds_after_training() {
    // P(enter IBD) in [1e-8, 0.1], P(exit IBD) in [0.001, 0.5]
    let mut raw = vec![0.999; 100];
    for i in 40..60 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    assert!(params.transition[0][1] >= 1e-8);
    assert!(params.transition[0][1] <= 0.1);
    assert!(params.transition[1][0] >= 0.001);
    assert!(params.transition[1][0] <= 0.5);
}

#[test]
fn bw_logit_many_iterations_converges() {
    // 100 iterations should converge (tol=1e-6) without diverging
    let mut raw = vec![0.999; 100];
    for i in 40..60 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    params.baum_welch_logit(&logit_obs, 100, 1e-6, Some(Population::EUR), 5000);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[1].mean > params.emission[0].mean);
}

#[test]
fn bw_logit_no_population_prior_defaults_generic() {
    let raw = vec![0.999; 30];
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::Generic, 50.0, 0.001, 5000);

    // None should default to Generic
    params.baum_welch_logit(&logit_obs, 5, 1e-6, None, 5000);

    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean > params.emission[0].mean);
}

#[test]
fn bw_logit_ibd_mean_capped_at_logit_cap() {
    // IBD mean should be capped at LOGIT_CAP
    let raw = vec![0.99999; 50]; // extremely high identity
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    assert!(params.emission[1].mean <= LOGIT_CAP + 1e-6,
        "IBD mean should be capped at LOGIT_CAP={}, got {}", LOGIT_CAP, params.emission[1].mean);
}

// =====================================================================
// End-to-end: logit pipeline stress tests
// =====================================================================

#[test]
fn logit_pipeline_short_ibd_segment() {
    // Very short IBD segment (3 windows) — harder to detect
    let mut raw = vec![0.999; 50];
    for i in 20..23 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);
    assert_eq!(result.states.len(), 50);
    // Short segment may or may not be detected — just ensure no panic
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

#[test]
fn logit_pipeline_two_ibd_segments() {
    // Two separate IBD segments
    let mut raw = vec![0.999; 120];
    for i in 20..40 { raw[i] = 0.9997; }
    for i in 80..100 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    params.baum_welch_logit(&logit_obs, 10, 1e-6, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);

    // Both segments should have some IBD windows
    let seg1_ibd: usize = result.states[20..40].iter().filter(|&&s| s == 1).count();
    let seg2_ibd: usize = result.states[80..100].iter().filter(|&&s| s == 1).count();
    assert!(seg1_ibd > 5, "First segment should have IBD, got {}/20", seg1_ibd);
    assert!(seg2_ibd > 5, "Second segment should have IBD, got {}/20", seg2_ibd);

    // Gap between segments should be mostly non-IBD
    let gap_ibd: usize = result.states[50..70].iter().filter(|&&s| s == 1).count();
    assert!(gap_ibd < 5, "Gap should be non-IBD, got {}/20", gap_ibd);
}

#[test]
fn logit_pipeline_all_populations_consistent() {
    let populations = vec![
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];
    let mut raw = vec![0.999; 80];
    for i in 30..50 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);

    for pop in &populations {
        let mut params = HmmParams::from_population_logit(*pop, 50.0, 0.001, 5000);
        params.estimate_emissions_logit(&logit_obs, Some(*pop), 5000);
        params.baum_welch_logit(&logit_obs, 10, 1e-6, Some(*pop), 5000);

        let result = infer_ibd(&logit_obs, &params);
        assert_eq!(result.states.len(), 80, "Population {:?}", pop);
        for &p in &result.posteriors {
            assert!(p >= 0.0 && p <= 1.0, "Population {:?}: posterior {} out of range", pop, p);
        }
    }
}

#[test]
fn logit_pipeline_noisy_data() {
    // Non-IBD with random noise — should not produce excessive false positives
    let raw: Vec<f64> = (0..100).map(|i| {
        // Deterministic "noise": slight variation around 0.999
        0.999 + (i as f64 * 0.0001).sin() * 0.0003
    }).collect();
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);
    let ibd_count: usize = result.states.iter().filter(|&&s| s == 1).count();
    assert!(ibd_count < 30,
        "Noisy non-IBD data should have few false positives, got {}/100", ibd_count);
}

#[test]
fn logit_pipeline_high_identity_background() {
    // Background at 0.9995 (high identity population) with IBD at 0.9999
    let mut raw = vec![0.9995; 80];
    for i in 30..50 { raw[i] = 0.9999; }
    let logit_obs = logit_transform_observations(&raw);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    params.baum_welch_logit(&logit_obs, 10, 1e-6, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);

    // Should produce valid posteriors
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
    assert_eq!(result.states.len(), 80);
}

#[test]
fn logit_vs_raw_both_produce_valid_results() {
    // Compare logit pipeline vs raw pipeline: both should produce valid output
    let mut raw = vec![0.999; 80];
    for i in 30..50 { raw[i] = 0.9997; }
    let logit_obs = logit_transform_observations(&raw);

    // Logit pipeline
    let mut logit_params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    logit_params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    let logit_result = infer_ibd(&logit_obs, &logit_params);

    // Raw pipeline
    let mut raw_params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    raw_params.estimate_emissions_robust(&raw, Some(Population::EUR), 5000);
    let raw_result = infer_ibd(&raw, &raw_params);

    // Both should have same length and valid posteriors
    assert_eq!(logit_result.states.len(), raw_result.states.len());
    for &p in &logit_result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
    for &p in &raw_result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}
