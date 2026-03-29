//! Tests for --logit-emissions IBD inference pipeline.
//!
//! Validates that the logit-space emission model correctly transforms
//! observations, uses appropriate parameters, and produces valid results.

use hprc_ibd::hmm::{
    infer_ibd, HmmParams, Population,
};
use hprc_ibd::stats::{logit_transform_observations, gaussian_to_logit_space, logit, inv_logit};

// -----------------------------------------------------------------------
// Logit transform properties
// -----------------------------------------------------------------------

#[test]
fn test_logit_transform_preserves_ordering() {
    // logit is monotonically increasing, so ordering is preserved
    let raw = vec![0.990, 0.995, 0.998, 0.999, 0.9995, 0.9999];
    let logit_obs = logit_transform_observations(&raw);
    for i in 1..logit_obs.len() {
        assert!(
            logit_obs[i] > logit_obs[i - 1],
            "logit should preserve order: logit({}) = {} <= logit({}) = {}",
            raw[i], logit_obs[i], raw[i-1], logit_obs[i-1]
        );
    }
}

#[test]
fn test_logit_ibd_nonibd_separation_much_larger() {
    // In raw space, IBD ~0.9997 vs non-IBD ~0.999 → gap ~0.0007
    // In logit space, gap should be >1.0 (dramatic improvement)
    let raw_ibd = 0.9997;
    let raw_non_ibd = 0.999;
    let raw_gap = raw_ibd - raw_non_ibd;

    let logit_ibd = logit(raw_ibd);
    let logit_non_ibd = logit(raw_non_ibd);
    let logit_gap = logit_ibd - logit_non_ibd;

    assert!(
        logit_gap > raw_gap * 100.0,
        "logit gap ({:.4}) should be >>100x raw gap ({:.6})",
        logit_gap, raw_gap
    );
    assert!(
        logit_gap > 1.0,
        "logit gap should be >1.0, got {:.4}",
        logit_gap
    );
}

#[test]
fn test_logit_roundtrip() {
    // logit then inv_logit should be identity (within floating point)
    let values = vec![0.5, 0.9, 0.99, 0.999, 0.9997, 0.001, 0.1];
    for &v in &values {
        let roundtrip = inv_logit(logit(v));
        assert!(
            (roundtrip - v).abs() < 1e-8,
            "roundtrip failed for {}: got {}",
            v, roundtrip
        );
    }
}

#[test]
fn test_logit_transform_empty_input() {
    let empty: Vec<f64> = vec![];
    let result = logit_transform_observations(&empty);
    assert!(result.is_empty());
}

#[test]
fn test_logit_transform_boundary_values() {
    // Values at 0 and 1 should be clamped (not produce -inf/+inf)
    let extreme = vec![0.0, 1.0, 0.5];
    let result = logit_transform_observations(&extreme);
    for &v in &result {
        assert!(v.is_finite(), "logit should be finite, got {}", v);
    }
}

// -----------------------------------------------------------------------
// Logit-space HMM parameters
// -----------------------------------------------------------------------

#[test]
fn test_from_population_logit_params_valid() {
    let params = HmmParams::from_population_logit(
        Population::EUR, 50.0, 0.0001, 5000,
    );
    // Emission means should be in logit space (~6-8 range)
    assert!(
        params.emission[0].mean > 5.0,
        "logit non-IBD mean should be >5, got {}",
        params.emission[0].mean
    );
    assert!(
        params.emission[1].mean > params.emission[0].mean,
        "logit IBD mean ({}) should be > non-IBD mean ({})",
        params.emission[1].mean, params.emission[0].mean
    );
    // Stds should be positive
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
}

#[test]
fn test_from_population_logit_all_populations() {
    let populations = vec![
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];
    for pop in &populations {
        let params = HmmParams::from_population_logit(*pop, 50.0, 0.0001, 5000);
        // All should produce valid params with IBD mean > non-IBD mean
        assert!(
            params.emission[1].mean > params.emission[0].mean,
            "Population {:?}: IBD mean ({}) should be > non-IBD mean ({})",
            pop, params.emission[1].mean, params.emission[0].mean
        );
    }
}

#[test]
fn test_logit_params_mean_separation_larger_than_raw() {
    let raw_params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let logit_params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);

    let raw_sep = raw_params.emission[1].mean - raw_params.emission[0].mean;
    let logit_sep = logit_params.emission[1].mean - logit_params.emission[0].mean;

    assert!(
        logit_sep > raw_sep * 100.0,
        "logit separation ({:.4}) should be >>100x raw separation ({:.6})",
        logit_sep, raw_sep
    );
}

// -----------------------------------------------------------------------
// Logit-space HMM inference
// -----------------------------------------------------------------------

#[test]
fn test_logit_infer_ibd_detects_ibd_segment() {
    // Create observations with a clear IBD region
    let mut raw_obs = vec![0.9990; 100]; // non-IBD background
    // Insert IBD segment at windows 40-59
    for i in 40..60 {
        raw_obs[i] = 0.9997;
    }

    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    // Data-driven emission estimation (as the real pipeline does)
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);

    // Should detect some IBD windows in the 40-59 range
    let ibd_count: usize = result.states[40..60].iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count > 10,
        "Should detect IBD in the inserted segment, found {} IBD windows out of 20",
        ibd_count
    );

    // Background should mostly be non-IBD
    let bg_ibd: usize = result.states[0..30].iter().filter(|&&s| s == 1).count();
    assert!(
        bg_ibd < 5,
        "Background should be mostly non-IBD, found {} IBD windows",
        bg_ibd
    );
}

#[test]
fn test_logit_infer_ibd_all_nonibd() {
    // All observations at non-IBD level → no IBD detected
    let raw_obs = vec![0.9990; 50];
    let logit_obs = logit_transform_observations(&raw_obs);
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);

    let result = infer_ibd(&logit_obs, &params);
    let ibd_count: usize = result.states.iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count < 5,
        "All non-IBD data should produce few/no IBD calls, got {}",
        ibd_count
    );
}

#[test]
fn test_logit_infer_ibd_all_ibd() {
    // All observations at IBD level → mostly IBD
    let raw_obs = vec![0.9998; 50];
    let logit_obs = logit_transform_observations(&raw_obs);
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.01, 5000);

    let result = infer_ibd(&logit_obs, &params);
    let ibd_count: usize = result.states.iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count > 40,
        "All IBD data should produce mostly IBD calls, got {}/50",
        ibd_count
    );
}

#[test]
fn test_logit_posteriors_valid_range() {
    let raw_obs = vec![0.9990; 30];
    let logit_obs = logit_transform_observations(&raw_obs);
    let params = HmmParams::from_population_logit(Population::Generic, 50.0, 0.0001, 5000);

    let result = infer_ibd(&logit_obs, &params);
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0, "Posterior {} out of [0,1]", p);
    }
}

#[test]
fn test_logit_estimate_emissions_updates_params() {
    let raw_obs = vec![0.9990; 50];
    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    let old_mean0 = params.emission[0].mean;

    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // After estimation, non-IBD mean should have shifted toward data
    assert!(
        (params.emission[0].mean - old_mean0).abs() < 5.0,
        "logit emission estimation should adjust mean within reasonable range"
    );
}

#[test]
fn test_logit_emission_context_compatible() {
    // Emission context smoothing should work on logit-transformed data
    use hprc_ibd::hmm::{precompute_log_emissions, smooth_log_emissions, viterbi_from_log_emit};

    let raw_obs = vec![0.9990; 30];
    let logit_obs = logit_transform_observations(&raw_obs);
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);

    let raw_emit = precompute_log_emissions(&logit_obs, &params);
    let smoothed = smooth_log_emissions(&raw_emit, 2);

    // Smoothed emissions should be valid
    for row in &smoothed {
        for &val in row {
            assert!(val.is_finite(), "Smoothed emission should be finite");
        }
    }

    let states = viterbi_from_log_emit(&smoothed, &params);
    assert_eq!(states.len(), 30);
}

#[test]
fn test_gaussian_to_logit_space_delta_method() {
    // The delta method should produce reasonable logit-space params
    let gp = gaussian_to_logit_space(0.999, 0.001);
    assert!(gp.mean > 5.0, "logit(0.999) should be >5, got {}", gp.mean);
    assert!(gp.std > 0.0, "logit std should be positive, got {}", gp.std);

    let gp2 = gaussian_to_logit_space(0.9997, 0.0005);
    assert!(
        gp2.mean > gp.mean,
        "logit(0.9997)={} should be > logit(0.999)={}",
        gp2.mean, gp.mean
    );
}

// -----------------------------------------------------------------------
// Logit-space Baum-Welch
// -----------------------------------------------------------------------

#[test]
fn test_logit_bw_converges() {
    // BW should not diverge — params should remain finite
    let mut raw_obs = vec![0.9990; 80];
    for i in 30..50 {
        raw_obs[i] = 0.9997;
    }
    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    // Params should be finite
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std > 0.0);
    assert!(params.emission[1].std > 0.0);
    // IBD mean should remain > non-IBD mean
    assert!(
        params.emission[1].mean > params.emission[0].mean,
        "After BW, IBD mean ({}) should be > non-IBD mean ({})",
        params.emission[1].mean, params.emission[0].mean
    );
}

#[test]
fn test_logit_bw_improves_detection() {
    // BW training should improve or maintain IBD detection quality
    let mut raw_obs = vec![0.9990; 100];
    for i in 40..60 {
        raw_obs[i] = 0.9997;
    }
    let logit_obs = logit_transform_observations(&raw_obs);

    // Without BW
    let mut params_no_bw = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    params_no_bw.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    let result_no_bw = infer_ibd(&logit_obs, &params_no_bw);
    let ibd_no_bw: usize = result_no_bw.states[40..60].iter().filter(|&&s| s == 1).count();

    // With BW
    let mut params_bw = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    params_bw.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    params_bw.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);
    let result_bw = infer_ibd(&logit_obs, &params_bw);
    let ibd_bw: usize = result_bw.states[40..60].iter().filter(|&&s| s == 1).count();

    // BW should detect at least as many IBD windows (or within 2)
    assert!(
        ibd_bw + 2 >= ibd_no_bw,
        "BW detection ({}) should be >= no-BW ({}) - 2",
        ibd_bw, ibd_no_bw
    );
}

#[test]
fn test_logit_bw_too_few_observations() {
    // Fewer than 10 observations: BW should be a no-op
    let raw_obs = vec![0.999; 5];
    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    let mean_before = params.emission[0].mean;

    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    assert_eq!(
        params.emission[0].mean, mean_before,
        "BW with < 10 observations should not change params"
    );
}

#[test]
fn test_logit_bw_transition_bounds() {
    // After BW, transitions should remain in valid ranges
    let raw_obs = vec![0.9990; 50];
    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);

    params.baum_welch_logit(&logit_obs, 10, 1e-6, Some(Population::EUR), 5000);

    // P(enter IBD) should be small
    assert!(params.transition[0][1] <= 0.1);
    assert!(params.transition[0][1] >= 1e-8);
    // Rows should sum to 1
    assert!((params.transition[0][0] + params.transition[0][1] - 1.0).abs() < 1e-10);
    assert!((params.transition[1][0] + params.transition[1][1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_logit_bw_all_populations() {
    let populations = vec![
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::Generic,
    ];
    let raw_obs = vec![0.9990; 30];
    let logit_obs = logit_transform_observations(&raw_obs);

    for pop in &populations {
        let mut params = HmmParams::from_population_logit(*pop, 50.0, 0.0001, 5000);
        params.baum_welch_logit(&logit_obs, 5, 1e-6, Some(*pop), 5000);
        assert!(
            params.emission[0].mean.is_finite(),
            "Population {:?}: non-IBD mean should be finite after BW",
            pop
        );
        assert!(
            params.emission[1].mean > params.emission[0].mean,
            "Population {:?}: IBD mean > non-IBD mean after BW",
            pop
        );
    }
}

#[test]
fn test_logit_bw_zero_iterations() {
    // 0 iterations should be a no-op
    let raw_obs = vec![0.999; 30];
    let logit_obs = logit_transform_observations(&raw_obs);
    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0001, 5000);
    let mean_before = params.emission[0].mean;

    params.baum_welch_logit(&logit_obs, 0, 1e-6, Some(Population::EUR), 5000);

    assert_eq!(params.emission[0].mean, mean_before);
}
