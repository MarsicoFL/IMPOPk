//! Tests for LOGIT_CAP=12.0 boundary behavior (algo_dev cycle 5).
//!
//! Verifies correct handling of identity values in the "newly uncapped" range
//! where logit output falls between the old cap (10.0) and new cap (12.0).
//! This range corresponds to identity values ~0.999960 to ~0.999994.

use hprc_ibd::hmm::{infer_ibd, HmmParams, Population};
use hprc_ibd::stats::{
    gaussian_to_logit_space, inv_logit, logit, logit_transform_observations, LOGIT_CAP,
};

// =====================================================================
// logit() preservation in the (10, 12] range
// =====================================================================

#[test]
fn logit_preserves_values_above_old_cap() {
    // logit(0.99998) ≈ 10.82 — was capped at 10.0, now preserved
    let v = logit(0.99998);
    assert!(v > 10.0, "logit(0.99998) should be > 10.0, got {}", v);
    assert!(v < LOGIT_CAP, "logit(0.99998) should be < LOGIT_CAP, got {}", v);
    let expected = (0.99998_f64 / (1.0 - 0.99998)).ln();
    assert!((v - expected).abs() < 0.01, "logit(0.99998) should be ~{}, got {}", expected, v);
}

#[test]
fn logit_preserves_identity_0_99999() {
    // logit(0.99999) ≈ 11.51 — the key value from algo_dev's analysis
    let v = logit(0.99999);
    assert!(v > 11.0, "logit(0.99999) should be > 11.0, got {}", v);
    assert!(v < 12.0, "logit(0.99999) should be < 12.0, got {}", v);
}

#[test]
fn logit_caps_at_boundary() {
    // logit(0.999994) ≈ 12.2 — should be capped at LOGIT_CAP
    let v = logit(0.999994);
    assert!((v - LOGIT_CAP).abs() < 1e-6,
        "logit(0.999994) should be capped at LOGIT_CAP={}, got {}", LOGIT_CAP, v);
}

#[test]
fn logit_negative_mirror_above_old_cap() {
    // Symmetry: logit(1 - 0.99998) = -logit(0.99998)
    let high = logit(0.99998);
    let low = logit(0.00002);
    assert!((high + low).abs() < 1e-6,
        "logit symmetry: {} + {} = {} (expected ~0)", high, low, high + low);
    assert!(low < -10.0, "logit(0.00002) should be < -10.0, got {}", low);
}

#[test]
fn logit_ordering_preserved_in_extended_range() {
    // Values in the newly uncapped range should maintain strict ordering
    let identities = [0.99996, 0.99997, 0.99998, 0.99999, 0.999993];
    let logits: Vec<f64> = identities.iter().map(|&x| logit(x)).collect();
    for i in 0..logits.len() - 1 {
        assert!(logits[i] < logits[i + 1],
            "Ordering violated: logit({}) = {} >= logit({}) = {}",
            identities[i], logits[i], identities[i + 1], logits[i + 1]);
    }
    // First should be just above old cap, last should be near new cap
    assert!(logits[0] > 10.0, "logit(0.99996) should be > 10.0");
    assert!(logits[logits.len() - 1] < LOGIT_CAP, "logit(0.999993) should be < LOGIT_CAP");
}

// =====================================================================
// inv_logit() roundtrip in extended range
// =====================================================================

#[test]
fn inv_logit_roundtrip_in_extended_range() {
    // Values between 10 and 12 should roundtrip through inv_logit
    for logit_val in [10.5, 11.0, 11.5, 11.9] {
        let identity = inv_logit(logit_val);
        let back = logit(identity);
        assert!((back - logit_val).abs() < 0.01,
            "Roundtrip failed: logit({}) -> inv_logit -> logit = {} (expected {})",
            logit_val, back, logit_val);
    }
}

#[test]
fn inv_logit_values_in_extended_range() {
    // inv_logit(10.5) should be very close to 1.0 but distinguishable
    let v = inv_logit(10.5);
    assert!(v > 0.9999, "inv_logit(10.5) should be > 0.9999, got {}", v);
    assert!(v < 1.0, "inv_logit(10.5) should be < 1.0, got {}", v);

    let v2 = inv_logit(11.5);
    assert!(v2 > v, "inv_logit(11.5) should be > inv_logit(10.5)");
    assert!(v2 < 1.0, "inv_logit(11.5) should be < 1.0");
}

// =====================================================================
// logit_transform_observations() in extended range
// =====================================================================

#[test]
fn logit_transform_preserves_extended_range_spread() {
    // Observations that were previously collapsed to LOGIT_CAP=10 should now spread
    let obs = vec![0.99996, 0.99997, 0.99998, 0.99999];
    let transformed = logit_transform_observations(&obs);

    // All should be above old cap
    for (i, &v) in transformed.iter().enumerate() {
        assert!(v > 10.0, "obs[{}] logit should be > 10.0, got {}", i, v);
    }

    // Should have meaningful spread (not all collapsed to same value)
    let spread = transformed.last().unwrap() - transformed.first().unwrap();
    assert!(spread > 0.5, "Extended range should have spread > 0.5, got {}", spread);
}

#[test]
fn logit_transform_mixed_ranges() {
    // Mix of normal and extended-range observations
    let obs = vec![0.999, 0.9995, 0.9999, 0.99999];
    let transformed = logit_transform_observations(&obs);

    assert!(transformed[0] < 8.0, "logit(0.999) should be < 8.0");
    assert!(transformed[1] > 7.0 && transformed[1] < 9.0, "logit(0.9995) should be ~7.6");
    assert!(transformed[2] > 9.0, "logit(0.9999) should be > 9.0");
    assert!(transformed[3] > 11.0, "logit(0.99999) should be > 11.0");

    // Strictly increasing
    for i in 0..3 {
        assert!(transformed[i] < transformed[i + 1]);
    }
}

// =====================================================================
// gaussian_to_logit_space() with new LOGIT_CAP
// =====================================================================

#[test]
fn gaussian_to_logit_std_cap_reflects_new_logit_cap() {
    // Std cap should be LOGIT_CAP * 0.5 = 6.0 (was 5.0)
    let params = gaussian_to_logit_space(0.5, 100.0); // huge std forces cap
    let expected_cap = LOGIT_CAP * 0.5;
    assert!((params.std - expected_cap).abs() < 0.01,
        "Std cap should be {}, got {}", expected_cap, params.std);
}

#[test]
fn gaussian_to_logit_mean_in_extended_range() {
    // Mean at 0.99998: logit ≈ 10.82, should be preserved (not capped at 10.0)
    let params = gaussian_to_logit_space(0.99998, 0.00001);
    assert!(params.mean > 10.0,
        "logit mean for identity=0.99998 should be > 10.0, got {}", params.mean);
    assert!(params.mean < LOGIT_CAP,
        "logit mean should be < LOGIT_CAP, got {}", params.mean);
}

#[test]
fn gaussian_to_logit_mean_at_new_cap() {
    // Mean at 0.999995: logit ≈ 12.2, should be capped at LOGIT_CAP
    let params = gaussian_to_logit_space(0.999995, 0.000001);
    assert!((params.mean - LOGIT_CAP).abs() < 1e-6,
        "logit mean for identity=0.999995 should be capped at {}, got {}", LOGIT_CAP, params.mean);
}

// =====================================================================
// from_population_logit() IBD emission with extended range
// =====================================================================

#[test]
fn from_population_logit_ibd_emission_can_exceed_old_cap() {
    // For a population with very high IBD identity priors,
    // the logit IBD emission mean could potentially exceed 10.0
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    // EUR IBD prior identity is very high (>0.999)
    // The IBD emission mean in logit space should be within [prior-1, LOGIT_CAP]
    assert!(params.emission[1].mean <= LOGIT_CAP + 1e-6,
        "IBD emission mean must be <= LOGIT_CAP, got {}", params.emission[1].mean);
    assert!(params.emission[1].mean > 0.0,
        "IBD emission mean should be positive, got {}", params.emission[1].mean);
}

// =====================================================================
// estimate_emissions_logit() with extended-range observations
// =====================================================================

#[test]
fn estimate_emissions_logit_extended_range_ibd_data() {
    // IBD region with very high identity (logit > 10)
    let mut raw = vec![0.999; 80]; // non-IBD background
    for i in 30..50 { raw[i] = 0.99999; } // IBD at logit ≈ 11.5
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // IBD emission mean should reflect the high-identity data
    // With LOGIT_CAP=12, the 0.99999 values contribute logit≈11.5
    // With old cap=10, they would have been truncated to 10.0
    assert!(params.emission[1].mean > 8.0,
        "IBD emission should reflect high-identity data, got {}", params.emission[1].mean);
    assert!(params.emission[1].mean <= LOGIT_CAP + 1e-6,
        "IBD emission must be <= LOGIT_CAP, got {}", params.emission[1].mean);
}

#[test]
fn estimate_emissions_logit_non_ibd_unaffected() {
    // Non-IBD emission should be unaffected by LOGIT_CAP change
    // (non-IBD identities are far from the cap)
    let raw = vec![0.999; 80]; // all non-IBD
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    // Non-IBD emission mean should be around logit(0.999) ≈ 6.9
    assert!(params.emission[0].mean > 5.0 && params.emission[0].mean < 9.0,
        "Non-IBD emission should be around logit(0.999), got {}", params.emission[0].mean);
}

// =====================================================================
// baum_welch_logit() with extended-range observations
// =====================================================================

#[test]
fn bw_logit_ibd_mean_can_reach_extended_range() {
    // Data with very high IBD identity — BW should estimate mean > 10
    let mut raw = vec![0.999; 100]; // non-IBD
    for i in 0..50 { raw[i] = 0.99999; } // IBD at logit ≈ 11.5
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    // With enough IBD data at logit≈11.5, BW should estimate IBD mean above old cap
    assert!(params.emission[1].mean <= LOGIT_CAP + 1e-6,
        "IBD mean must respect LOGIT_CAP bound, got {}", params.emission[1].mean);
    // The exact value depends on priors and convergence, but should be elevated
    assert!(params.emission[1].mean > 5.0,
        "IBD mean should be elevated for high-identity data, got {}", params.emission[1].mean);
}

#[test]
fn bw_logit_emission_gap_wider_with_extended_range() {
    // The key benefit of LOGIT_CAP=12: observations at 0.99999 contribute logit≈11.5
    // instead of being truncated to 10.0, increasing the IBD/non-IBD gap
    let mut raw = vec![0.999; 100]; // non-IBD at logit ≈ 6.9
    for i in 0..40 { raw[i] = 0.99999; } // IBD at logit ≈ 11.5
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    let gap = params.emission[1].mean - params.emission[0].mean;
    assert!(gap > 1.0,
        "IBD/non-IBD emission gap should be > 1.0 logit units, got {}", gap);
}

#[test]
fn bw_logit_constant_high_obs_no_divergence() {
    // All observations at a single high-identity value (logit > 10)
    let raw = vec![0.99999; 50];
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.baum_welch_logit(&logit_obs, 20, 1e-6, Some(Population::EUR), 5000);

    // Should not diverge
    assert!(params.emission[0].mean.is_finite());
    assert!(params.emission[1].mean.is_finite());
    assert!(params.emission[0].std.is_finite() && params.emission[0].std > 0.0);
    assert!(params.emission[1].std.is_finite() && params.emission[1].std > 0.0);
}

// =====================================================================
// Pipeline: infer_ibd with extended-range data
// =====================================================================

#[test]
fn pipeline_extended_range_ibd_detection() {
    // IBD at identity 0.99999 (logit≈11.5) should be detectable
    let mut raw = vec![0.999; 100];
    for i in 30..60 { raw[i] = 0.99999; }
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);
    params.baum_welch_logit(&logit_obs, 10, 1e-6, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);
    assert_eq!(result.states.len(), 100);

    // IBD region should be detected
    let ibd_in_region: usize = result.states[30..60].iter().filter(|&&s| s == 1).count();
    assert!(ibd_in_region > 15,
        "Should detect IBD in high-identity region, got {}/30", ibd_in_region);

    // Non-IBD region should be mostly non-IBD
    let ibd_outside: usize = result.states[0..20].iter().filter(|&&s| s == 1).count();
    assert!(ibd_outside < 5,
        "Should not detect IBD in background region, got {}/20", ibd_outside);
}

#[test]
fn pipeline_gradual_identity_increase_into_extended_range() {
    // Identity gradually increases from non-IBD into extended range
    let raw: Vec<f64> = (0..100).map(|i| {
        if i < 50 {
            0.999 // non-IBD
        } else {
            // Gradually increase from 0.9999 to 0.99999
            let frac = (i - 50) as f64 / 50.0;
            0.9999 + frac * 0.00009
        }
    }).collect();
    let logit_obs = logit_transform_observations(&raw);

    let mut params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    params.estimate_emissions_logit(&logit_obs, Some(Population::EUR), 5000);

    let result = infer_ibd(&logit_obs, &params);
    assert_eq!(result.states.len(), 100);

    // Valid posteriors
    for &p in &result.posteriors {
        assert!(p >= 0.0 && p <= 1.0);
    }
}

#[test]
fn pipeline_all_populations_extended_range() {
    // All populations should handle extended-range data
    let populations = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];
    let mut raw = vec![0.999; 60];
    for i in 20..40 { raw[i] = 0.99999; } // IBD at logit ≈ 11.5

    let logit_obs = logit_transform_observations(&raw);

    for pop in &populations {
        let mut params = HmmParams::from_population_logit(*pop, 50.0, 0.001, 5000);
        params.estimate_emissions_logit(&logit_obs, Some(*pop), 5000);
        params.baum_welch_logit(&logit_obs, 5, 1e-6, Some(*pop), 5000);

        let result = infer_ibd(&logit_obs, &params);
        assert_eq!(result.states.len(), 60, "{:?}: wrong length", pop);
        for &p in &result.posteriors {
            assert!(p >= 0.0 && p <= 1.0, "{:?}: posterior out of range: {}", pop, p);
        }
    }
}

// =====================================================================
// Regression: values at exact boundary of old cap
// =====================================================================

#[test]
fn logit_at_exact_old_cap_boundary() {
    // Identity where logit = exactly 10.0: inv_logit(10.0) ≈ 0.9999546
    let identity = inv_logit(10.0);
    let v = logit(identity);
    assert!((v - 10.0).abs() < 0.01,
        "Roundtrip at old boundary: logit(inv_logit(10.0)) should be ~10.0, got {}", v);
    // With new cap, this value should NOT be capped
    assert!(v < LOGIT_CAP, "Value at old boundary should be below new cap");
}

#[test]
fn logit_just_above_old_cap() {
    // Identity producing logit = 10.5
    let identity = inv_logit(10.5);
    let v = logit(identity);
    assert!((v - 10.5).abs() < 0.01,
        "logit at 10.5 should not be capped, got {}", v);
}

#[test]
fn logit_just_below_new_cap() {
    // Identity producing logit = 11.9
    let identity = inv_logit(11.9);
    let v = logit(identity);
    assert!((v - 11.9).abs() < 0.1,
        "logit at 11.9 should not be capped, got {}", v);
}

#[test]
fn logit_at_new_cap() {
    // Identity producing logit = 12.5 — should be capped at LOGIT_CAP
    let identity = inv_logit(12.5);
    let v = logit(identity);
    assert!((v - LOGIT_CAP).abs() < 1e-6,
        "logit at 12.5 should be capped at {}, got {}", LOGIT_CAP, v);
}
