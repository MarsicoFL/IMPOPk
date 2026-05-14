use impopk_ibd::hmm::{
    aggregate_observations, infer_ibd, infer_ibd_multi_scale,
    extract_ibd_segments, HmmParams, Population,
};

// ============================================================================
// aggregate_observations tests
// ============================================================================

#[test]
fn aggregate_factor_1_returns_copy() {
    let obs = vec![0.998, 0.997, 0.999];
    let agg = aggregate_observations(&obs, 1);
    assert_eq!(agg.len(), 3);
    assert!((agg[0] - 0.998).abs() < 1e-10);
    assert!((agg[1] - 0.997).abs() < 1e-10);
    assert!((agg[2] - 0.999).abs() < 1e-10);
}

#[test]
fn aggregate_factor_0_returns_copy() {
    let obs = vec![0.998, 0.997];
    let agg = aggregate_observations(&obs, 0);
    assert_eq!(agg.len(), 2);
}

#[test]
fn aggregate_empty_returns_empty() {
    let obs: Vec<f64> = vec![];
    let agg = aggregate_observations(&obs, 2);
    assert!(agg.is_empty());
}

#[test]
fn aggregate_factor_2_even_length() {
    let obs = vec![0.998, 0.996, 0.999, 0.997];
    let agg = aggregate_observations(&obs, 2);
    assert_eq!(agg.len(), 2);
    assert!((agg[0] - 0.997).abs() < 1e-10); // avg(0.998, 0.996)
    assert!((agg[1] - 0.998).abs() < 1e-10); // avg(0.999, 0.997)
}

#[test]
fn aggregate_factor_2_odd_length() {
    let obs = vec![0.998, 0.996, 0.994];
    let agg = aggregate_observations(&obs, 2);
    assert_eq!(agg.len(), 2);
    assert!((agg[0] - 0.997).abs() < 1e-10); // avg(0.998, 0.996)
    assert!((agg[1] - 0.994).abs() < 1e-10); // only one element in last bin
}

#[test]
fn aggregate_factor_3() {
    let obs = vec![0.990, 0.993, 0.996, 0.998, 0.997];
    let agg = aggregate_observations(&obs, 3);
    assert_eq!(agg.len(), 2);
    let expected_0 = (0.990 + 0.993 + 0.996) / 3.0;
    let expected_1 = (0.998 + 0.997) / 2.0;
    assert!((agg[0] - expected_0).abs() < 1e-10);
    assert!((agg[1] - expected_1).abs() < 1e-10);
}

#[test]
fn aggregate_factor_4_single_bin() {
    let obs = vec![0.990, 0.995, 0.997, 0.993];
    let agg = aggregate_observations(&obs, 4);
    assert_eq!(agg.len(), 1);
    let expected = (0.990 + 0.995 + 0.997 + 0.993) / 4.0;
    assert!((agg[0] - expected).abs() < 1e-10);
}

#[test]
fn aggregate_factor_larger_than_length() {
    let obs = vec![0.998, 0.996];
    let agg = aggregate_observations(&obs, 10);
    assert_eq!(agg.len(), 1);
    assert!((agg[0] - 0.997).abs() < 1e-10);
}

#[test]
fn aggregate_single_element() {
    let obs = vec![0.995];
    let agg = aggregate_observations(&obs, 2);
    assert_eq!(agg.len(), 1);
    assert!((agg[0] - 0.995).abs() < 1e-10);
}

#[test]
fn aggregate_preserves_mean() {
    // Aggregation should preserve the overall mean
    let obs = vec![0.990, 0.992, 0.994, 0.996, 0.998, 0.997];
    let overall_mean: f64 = obs.iter().sum::<f64>() / obs.len() as f64;
    let agg = aggregate_observations(&obs, 2);
    let agg_mean: f64 = agg.iter().sum::<f64>() / agg.len() as f64;
    assert!((overall_mean - agg_mean).abs() < 1e-10);
}

// ============================================================================
// infer_ibd_multi_scale tests
// ============================================================================

fn make_test_params() -> HmmParams {
    HmmParams::from_population(Population::EUR, 50.0, 0.0001, 10000)
}

/// Create params fitted to observations (like the real pipeline)
fn make_fitted_params(obs: &[f64]) -> HmmParams {
    let mut params = make_test_params();
    params.estimate_emissions_robust(obs, Some(Population::EUR), 10000);
    params
}

#[test]
fn multi_scale_empty_observations() {
    let params = make_test_params();
    let result = infer_ibd_multi_scale(&[], &params, &[1, 2, 4]);
    assert!(result.states.is_empty());
    assert!(result.posteriors.is_empty());
}

#[test]
fn multi_scale_short_observations_fallback() {
    // With <6 observations, multi-scale falls back to base
    let params = make_test_params();
    let obs = vec![0.998, 0.997, 0.999, 0.998, 0.997];
    let multi = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    let base = infer_ibd(&obs, &params);
    assert_eq!(multi.states, base.states);
}

#[test]
fn multi_scale_single_scale_is_base() {
    let params = make_test_params();
    let obs = vec![0.998, 0.997, 0.999, 0.998, 0.997, 0.996, 0.998, 0.999];
    let multi = infer_ibd_multi_scale(&obs, &params, &[1]);
    let base = infer_ibd(&obs, &params);
    assert_eq!(multi.states, base.states);
    assert_eq!(multi.posteriors, base.posteriors);
}

#[test]
fn multi_scale_returns_same_length() {
    let params = make_test_params();
    let obs: Vec<f64> = (0..20).map(|i| if i >= 8 && i <= 14 { 0.9999 } else { 0.997 }).collect();
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    assert_eq!(result.states.len(), obs.len());
    assert_eq!(result.posteriors.len(), obs.len());
}

#[test]
fn multi_scale_posteriors_from_base() {
    // Posteriors should come from base resolution (finest scale)
    let params = make_test_params();
    let obs: Vec<f64> = (0..20).map(|i| if i >= 8 && i <= 14 { 0.9999 } else { 0.997 }).collect();
    let base = infer_ibd(&obs, &params);
    let multi = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    assert_eq!(multi.posteriors.len(), base.posteriors.len());
    // Posteriors are identical (from base)
    for (a, b) in multi.posteriors.iter().zip(base.posteriors.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn multi_scale_all_non_ibd() {
    let params = make_test_params();
    let obs = vec![0.997; 20];
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    // No IBD detected at any scale
    assert!(result.states.iter().all(|&s| s == 0));
}

#[test]
fn multi_scale_strong_ibd_preserved() {
    // Clear IBD signal should be detected at all scales
    let mut obs = vec![0.997; 30];
    // Strong IBD block in the middle (windows 10-19)
    for i in 10..20 {
        obs[i] = 0.9999;
    }
    let params = make_fitted_params(&obs);
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    let segments = extract_ibd_segments(&result.states);
    // Should detect at least the core IBD region
    assert!(!segments.is_empty(), "Strong IBD should be detected");
    // Core region should be called IBD
    let any_ibd_in_core = (12..18).any(|i| result.states[i] == 1);
    assert!(any_ibd_in_core, "Core IBD windows should be called IBD");
}

#[test]
fn multi_scale_weak_fine_only_pruned() {
    // A weak signal at fine scale should be pruned if not confirmed at coarse scale.
    // This is the pruning behavior: fine-only segments with mean posterior <0.7 are removed.
    let params = make_test_params();
    let obs = vec![0.997; 20]; // All non-IBD background
    let base = infer_ibd(&obs, &params);
    let multi = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    // Both should agree: no IBD
    let base_ibd_count: usize = base.states.iter().filter(|&&s| s == 1).count();
    let multi_ibd_count: usize = multi.states.iter().filter(|&&s| s == 1).count();
    assert!(multi_ibd_count <= base_ibd_count,
        "Multi-scale should not add false positives: base={} multi={}", base_ibd_count, multi_ibd_count);
}

#[test]
fn multi_scale_states_are_binary() {
    let params = make_test_params();
    let mut obs = vec![0.997; 20];
    for i in 5..15 {
        obs[i] = 0.9998;
    }
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    for &s in &result.states {
        assert!(s == 0 || s == 1, "States must be 0 or 1, got {}", s);
    }
}

#[test]
fn multi_scale_log_likelihood_finite() {
    let params = make_test_params();
    let mut obs = vec![0.997; 20];
    for i in 10..15 {
        obs[i] = 0.9999;
    }
    let result = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    assert!(result.log_likelihood.is_finite(), "Log-likelihood should be finite");
}

#[test]
fn multi_scale_with_custom_scales() {
    let params = make_test_params();
    let obs = vec![0.997; 24];
    // Test with different scale configurations
    let r1 = infer_ibd_multi_scale(&obs, &params, &[1, 2]);
    let r2 = infer_ibd_multi_scale(&obs, &params, &[1, 3]);
    let r3 = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    // All should return valid results
    assert_eq!(r1.states.len(), 24);
    assert_eq!(r2.states.len(), 24);
    assert_eq!(r3.states.len(), 24);
}

#[test]
fn multi_scale_recovery_moderate_posterior() {
    // Construct a scenario where base resolution misses IBD but coarse detects it.
    // If fine posteriors are > 0.3, recovery should happen.
    // Create observations with a noisy IBD block
    let mut obs = vec![0.997; 20];
    // Moderate signal: not strong enough for fine-scale detection alone
    for i in 8..16 {
        obs[i] = 0.9990 + 0.0002 * (i as f64 - 8.0);
    }
    let params = make_fitted_params(&obs);
    let base = infer_ibd(&obs, &params);
    let multi = infer_ibd_multi_scale(&obs, &params, &[1, 2, 4]);
    // Multi-scale should have at least as many IBD windows as base in the signal region
    let base_ibd: usize = base.states[8..16].iter().filter(|&&s| s == 1).count();
    let multi_ibd: usize = multi.states[8..16].iter().filter(|&&s| s == 1).count();
    assert!(multi_ibd >= base_ibd,
        "Multi-scale should recover or match base: base={} multi={}", base_ibd, multi_ibd);
}

#[test]
fn multi_scale_empty_scales_array() {
    let params = make_test_params();
    let obs = vec![0.997; 10];
    // Empty scales should fall back to base (len <= 1 check)
    let result = infer_ibd_multi_scale(&obs, &params, &[]);
    let base = infer_ibd(&obs, &params);
    assert_eq!(result.states, base.states);
}
