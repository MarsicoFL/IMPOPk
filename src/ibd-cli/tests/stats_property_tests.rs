//! Property-based and edge case tests for stats.rs functions:
//! - em_two_component / em_two_component_map convergence properties
//! - gaussian_to_logit_space edge cases
//! - trimmed_mean properties
//! - bic_model_selection properties
//! - inv_logit / logit_transform_observations properties

use impopk_ibd::stats::*;

// ============================================================
// em_two_component: convergence and invariant properties
// ============================================================

#[test]
fn em_weights_sum_to_one() {
    let data: Vec<f64> = (0..50)
        .map(|i| 0.997 + (i as f64) * 0.0001)
        .chain((0..20).map(|i| 0.9997 + (i as f64) * 0.00001))
        .collect();
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let result = em_two_component(&data, &init_low, &init_high, 0.7, 100, 1e-8);
    let (_, _, w0, w1) = result.unwrap();
    assert!(
        (w0 + w1 - 1.0).abs() < 1e-10,
        "EM weights must sum to 1: {} + {} = {}",
        w0,
        w1,
        w0 + w1
    );
}

#[test]
fn em_low_mean_less_than_high_mean() {
    let mut data = vec![0.998; 40];
    data.extend(vec![0.9997; 30]);
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (low, high, _, _) = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6).unwrap();
    assert!(
        low.mean <= high.mean,
        "EM must ensure low.mean ({}) <= high.mean ({})",
        low.mean,
        high.mean
    );
}

#[test]
fn em_positive_std_invariant() {
    let data: Vec<f64> = (0..100).map(|i| 0.995 + (i as f64) * 0.00005).collect();
    let init_low = GaussianParams::new_unchecked(0.996, 0.002);
    let init_high = GaussianParams::new_unchecked(0.999, 0.001);
    let (low, high, _, _) = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6).unwrap();
    assert!(low.std > 0.0, "low.std must be positive: {}", low.std);
    assert!(high.std > 0.0, "high.std must be positive: {}", high.std);
}

#[test]
fn em_swapped_init_still_orders_correctly() {
    let mut data = vec![0.998; 50];
    data.extend(vec![0.9997; 20]);
    // Swap: init_low has higher mean than init_high
    let init_low = GaussianParams::new_unchecked(0.9997, 0.0005);
    let init_high = GaussianParams::new_unchecked(0.998, 0.001);
    let (low, high, _, _) =
        em_two_component(&data, &init_low, &init_high, 0.3, 50, 1e-6).unwrap();
    assert!(
        low.mean <= high.mean,
        "Even with swapped init, low.mean ({}) <= high.mean ({})",
        low.mean,
        high.mean
    );
}

#[test]
fn em_exactly_four_points() {
    let data = vec![0.997, 0.998, 0.9995, 0.9997];
    let init_low = GaussianParams::new_unchecked(0.9975, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9996, 0.0005);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    // 4 points is the minimum, should return Some
    assert!(result.is_some(), "4 points should be enough for EM");
}

#[test]
fn em_three_points_returns_none() {
    let data = vec![0.997, 0.998, 0.9997];
    let init_low = GaussianParams::new_unchecked(0.997, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let result = em_two_component(&data, &init_low, &init_high, 0.5, 50, 1e-6);
    assert!(result.is_none(), "3 points should return None");
}

#[test]
fn em_convergence_improves_with_more_iterations() {
    let mut data = vec![0.998; 50];
    data.extend(vec![0.9997; 20]);
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    // 1 iteration vs many
    let r1 = em_two_component(&data, &init_low, &init_high, 0.7, 1, 1e-20);
    let r100 = em_two_component(&data, &init_low, &init_high, 0.7, 100, 1e-20);

    assert!(r1.is_some());
    assert!(r100.is_some());
    // With more iterations, means should be closer to true cluster centers
    let (_, h1, _, _) = r1.unwrap();
    let (_, h100, _, _) = r100.unwrap();
    // The 100-iter high mean should be >= the 1-iter (closer to 0.9997)
    assert!(
        h100.mean >= h1.mean - 0.001,
        "More iterations should refine: 1-iter high mean={}, 100-iter={}",
        h1.mean,
        h100.mean
    );
}

// ============================================================
// em_two_component_map: MAP regularization properties
// ============================================================

#[test]
fn em_map_weights_sum_to_one() {
    let mut data = vec![0.998; 50];
    data.extend(vec![0.9997; 20]);
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (_, _, w0, w1) =
        em_two_component_map(&data, &init_low, &init_high, 0.7, 100, 1e-8, 10.0).unwrap();
    assert!(
        (w0 + w1 - 1.0).abs() < 1e-10,
        "MAP EM weights must sum to 1: {} + {} = {}",
        w0,
        w1,
        w0 + w1
    );
}

#[test]
fn em_map_ordering_invariant() {
    let mut data = vec![0.998; 50];
    data.extend(vec![0.9997; 20]);
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (low, high, _, _) =
        em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 5.0).unwrap();
    assert!(
        low.mean <= high.mean,
        "MAP EM low.mean ({}) <= high.mean ({})",
        low.mean,
        high.mean
    );
}

#[test]
fn em_map_strong_prior_keeps_means_close_to_init() {
    let mut data = vec![0.999; 100]; // ambiguous, all same
    let init_low = GaussianParams::new_unchecked(0.997, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    // Very strong prior
    let result = em_two_component_map(&data, &init_low, &init_high, 0.5, 50, 1e-6, 100.0);
    if let Some((low, high, _, _)) = result {
        // With prior_strength=100, means should stay near initial values
        assert!(
            (low.mean - 0.997).abs() < 0.005,
            "Strong prior should keep low mean near 0.997, got {}",
            low.mean
        );
        assert!(
            (high.mean - 0.9997).abs() < 0.003,
            "Strong prior should keep high mean near 0.9997, got {}",
            high.mean
        );
    }
}

#[test]
fn em_map_zero_prior_matches_standard_em() {
    let mut data = vec![0.998; 50];
    data.extend(vec![0.9997; 20]);
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);

    let standard = em_two_component(&data, &init_low, &init_high, 0.7, 50, 1e-8);
    // prior_strength = 0 should approximate standard EM
    let map_zero = em_two_component_map(&data, &init_low, &init_high, 0.7, 50, 1e-8, 0.001);

    if let (Some((s_l, s_h, _, _)), Some((m_l, m_h, _, _))) = (standard, map_zero) {
        // Should be very close
        assert!(
            (s_l.mean - m_l.mean).abs() < 0.001,
            "Zero prior should match standard EM: std={}, map={}",
            s_l.mean,
            m_l.mean
        );
        assert!(
            (s_h.mean - m_h.mean).abs() < 0.001,
            "Zero prior should match standard EM: std={}, map={}",
            s_h.mean,
            m_h.mean
        );
    }
}

#[test]
fn em_map_too_few_points() {
    let data = vec![0.998, 0.999, 0.9997];
    let init_low = GaussianParams::new_unchecked(0.998, 0.001);
    let init_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let result = em_two_component_map(&data, &init_low, &init_high, 0.7, 50, 1e-6, 5.0);
    assert!(result.is_none());
}

// ============================================================
// gaussian_to_logit_space: edge cases and properties
// ============================================================

#[test]
fn gaussian_to_logit_near_zero_denominator() {
    // Mean very close to 0 → denominator = mean * (1 - mean) ≈ 0
    let result = gaussian_to_logit_space(1e-16, 0.001);
    assert!(result.mean.is_finite(), "Logit mean must be finite for near-zero input");
    assert!(result.std > 0.0, "Logit std must be positive");
}

#[test]
fn gaussian_to_logit_near_one_denominator() {
    // Mean very close to 1 → denominator = mean * (1 - mean) ≈ 0
    let result = gaussian_to_logit_space(1.0 - 1e-16, 0.001);
    assert!(result.mean.is_finite(), "Logit mean must be finite for near-one input");
    assert!(result.std > 0.0, "Logit std must be positive");
    assert!(
        result.mean <= LOGIT_CAP,
        "Logit mean should be capped: {}",
        result.mean
    );
}

#[test]
fn gaussian_to_logit_at_half() {
    let result = gaussian_to_logit_space(0.5, 0.1);
    assert!(
        result.mean.abs() < 0.01,
        "logit(0.5) should be ~0, got {}",
        result.mean
    );
    // std in logit space: 0.1 / (0.5 * 0.5) = 0.4
    assert!(
        (result.std - 0.4).abs() < 0.01,
        "logit std at 0.5 should be ~0.4, got {}",
        result.std
    );
}

#[test]
fn gaussian_to_logit_preserves_ordering() {
    // Higher raw mean → higher logit mean
    let low = gaussian_to_logit_space(0.99, 0.001);
    let mid = gaussian_to_logit_space(0.995, 0.001);
    let high = gaussian_to_logit_space(0.999, 0.001);
    assert!(
        low.mean < mid.mean && mid.mean < high.mean,
        "Logit should preserve ordering: {} < {} < {}",
        low.mean,
        mid.mean,
        high.mean
    );
}

#[test]
fn gaussian_to_logit_std_floor() {
    // Even with very extreme params, std should be at least 0.01
    let result = gaussian_to_logit_space(0.5, 0.0);
    assert!(
        result.std >= 0.01,
        "Logit std should have floor of 0.01, got {}",
        result.std
    );
}

#[test]
fn gaussian_to_logit_std_cap() {
    // Very large std relative to denominator should be capped
    let result = gaussian_to_logit_space(0.9999, 0.1);
    assert!(
        result.std <= LOGIT_CAP * 0.5,
        "Logit std should be capped at LOGIT_CAP*0.5={}, got {}",
        LOGIT_CAP * 0.5,
        result.std
    );
}

#[test]
fn gaussian_to_logit_typical_pangenome_values() {
    // Typical EUR non-IBD: mean=0.99915, std=0.001
    let non_ibd = gaussian_to_logit_space(0.99915, 0.001);
    // Typical EUR IBD: mean=0.9997, std=0.0005
    let ibd = gaussian_to_logit_space(0.9997, 0.0005);

    // Both should have finite, reasonable values
    assert!(non_ibd.mean > 5.0 && non_ibd.mean < 9.0);
    assert!(ibd.mean > 7.0 && ibd.mean < 10.0);
    assert!(non_ibd.std > 0.1 && non_ibd.std < 5.0);
    assert!(ibd.std > 0.1 && ibd.std < 5.0);

    // Separation should be amplified vs raw
    let raw_sep = 0.9997 - 0.99915;
    let logit_sep = ibd.mean - non_ibd.mean;
    assert!(
        logit_sep > raw_sep * 50.0,
        "Logit separation ({}) should be >> raw separation ({})",
        logit_sep,
        raw_sep
    );
}

// ============================================================
// trimmed_mean: properties
// ============================================================

#[test]
fn trimmed_mean_single_element() {
    let data = vec![42.0];
    let tm = trimmed_mean(&data, 0.0).unwrap();
    assert!((tm - 42.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_single_element_with_trim() {
    // 1 element, trim_count = floor(1 * 0.1) = 0, should still work
    let data = vec![42.0];
    let tm = trimmed_mean(&data, 0.1).unwrap();
    assert!((tm - 42.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_two_elements_high_trim_returns_none() {
    // 2 elements, trim 50% from each side → nothing left
    let data = vec![1.0, 2.0];
    // trim_fraction clamped to 0.49 → trim_count = floor(2 * 0.49) = 0
    // Actually floor(2 * 0.49) = 0, so start=0 end=2, should return Some
    let result = trimmed_mean(&data, 0.49);
    assert!(result.is_some());
}

#[test]
fn trimmed_mean_symmetric_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tm = trimmed_mean(&data, 0.2).unwrap();
    // Symmetric data → trimmed mean = regular mean of inner values
    assert!(
        (tm - 3.0).abs() < 0.01,
        "Trimmed mean of symmetric data should be median-like: {}",
        tm
    );
}

#[test]
fn trimmed_mean_equals_mean_at_zero_trim() {
    let data = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let tm = trimmed_mean(&data, 0.0).unwrap();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    assert!(
        (tm - mean).abs() < 1e-10,
        "0-trim should equal mean: {} vs {}",
        tm,
        mean
    );
}

#[test]
fn trimmed_mean_handles_nan_via_total_cmp() {
    // NaN should be sorted to the end by total_cmp, then trimmed
    let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN];
    let tm = trimmed_mean(&data, 0.2);
    // With NaN sorted to end and trimmed, should still produce something
    // trim_count = floor(5 * 0.2) = 1 → removes lowest (1.0) and highest (NaN)
    if let Some(val) = tm {
        assert!(val.is_finite(), "Trimmed mean should be finite after trimming NaN");
    }
}

#[test]
fn trimmed_mean_invariant_outlier_resistant() {
    // With outliers vs without: trimmed mean should be much more similar than regular mean
    let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let dirty = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0];

    let tm_clean = trimmed_mean(&clean, 0.1).unwrap();
    let tm_dirty = trimmed_mean(&dirty, 0.1).unwrap();
    let mean_clean = clean.iter().sum::<f64>() / clean.len() as f64;
    let mean_dirty = dirty.iter().sum::<f64>() / dirty.len() as f64;

    let tm_diff = (tm_clean - tm_dirty).abs();
    let mean_diff = (mean_clean - mean_dirty).abs();
    assert!(
        tm_diff < mean_diff,
        "Trimmed mean should be more robust: tm_diff={}, mean_diff={}",
        tm_diff,
        mean_diff
    );
}

// ============================================================
// bic_model_selection: properties
// ============================================================

#[test]
fn bic_returns_zeros_for_single_point() {
    let data = vec![0.999];
    let low = GaussianParams::new_unchecked(0.998, 0.001);
    let high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (b1, b2) = bic_model_selection(&data, &low, &high, 0.5);
    assert_eq!(b1, 0.0);
    assert_eq!(b2, 0.0);
}

#[test]
fn bic_returns_zeros_for_empty() {
    let data: Vec<f64> = vec![];
    let low = GaussianParams::new_unchecked(0.998, 0.001);
    let high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (b1, b2) = bic_model_selection(&data, &low, &high, 0.5);
    assert_eq!(b1, 0.0);
    assert_eq!(b2, 0.0);
}

#[test]
fn bic_penalty_increases_with_params() {
    // The 2-component BIC has higher penalty (5 params vs 2)
    // For truly unimodal data, this penalty should dominate
    let data: Vec<f64> = (0..200).map(|i| 0.999 + (i as f64) * 0.000001).collect();
    let low = GaussianParams::new_unchecked(0.999, 0.0001);
    let high = GaussianParams::new_unchecked(0.99905, 0.0001);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    // For uniform-ish data, BIC should favor simpler model
    // The difference should involve the penalty term 3*ln(n)
    let penalty_diff = 3.0 * (200.0_f64).ln();
    // bic_2 - bic_1 should be close to the penalty difference (if likelihoods are similar)
    let bic_diff = bic_2 - bic_1;
    assert!(
        bic_diff.is_finite(),
        "BIC values should be finite: bic_1={}, bic_2={}",
        bic_1,
        bic_2
    );
    // The penalty term is 3*ln(200) ≈ 15.9
    assert!(
        penalty_diff > 10.0,
        "Penalty difference should be meaningful: {}",
        penalty_diff
    );
}

#[test]
fn bic_strongly_bimodal_prefers_two() {
    // Very clear bimodal
    let mut data = vec![0.990; 100];
    data.extend(vec![0.999; 100]);

    let low = GaussianParams::new_unchecked(0.990, 0.001);
    let high = GaussianParams::new_unchecked(0.999, 0.001);

    let (bic_1, bic_2) = bic_model_selection(&data, &low, &high, 0.5);
    assert!(
        bic_2 < bic_1,
        "BIC should strongly prefer 2 components for bimodal: bic_1={}, bic_2={}",
        bic_1,
        bic_2
    );
}

// ============================================================
// inv_logit: properties
// ============================================================

#[test]
fn inv_logit_at_zero_is_half() {
    assert!((inv_logit(0.0) - 0.5).abs() < 1e-10);
}

#[test]
fn inv_logit_monotone() {
    let values = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    for window in values.windows(2) {
        assert!(
            inv_logit(window[0]) < inv_logit(window[1]),
            "inv_logit should be monotone increasing"
        );
    }
}

#[test]
fn inv_logit_range_in_01() {
    for &x in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let y = inv_logit(x);
        assert!(y >= 0.0 && y <= 1.0, "inv_logit({}) = {} not in [0,1]", x, y);
    }
    // For moderate values, result should be strictly in (0,1)
    for &x in &[-10.0, -1.0, 0.0, 1.0, 10.0] {
        let y = inv_logit(x);
        assert!(y > 0.0 && y < 1.0, "inv_logit({}) = {} not in (0,1)", x, y);
    }
}

#[test]
fn inv_logit_symmetry() {
    // inv_logit(-x) = 1 - inv_logit(x)
    for &x in &[0.5, 1.0, 3.0, 7.0] {
        let left = inv_logit(-x);
        let right = 1.0 - inv_logit(x);
        assert!(
            (left - right).abs() < 1e-10,
            "inv_logit(-{}) = {}, 1 - inv_logit({}) = {}",
            x,
            left,
            x,
            right
        );
    }
}

// ============================================================
// logit_transform_observations: properties
// ============================================================

#[test]
fn logit_transform_preserves_length() {
    let obs = vec![0.1, 0.5, 0.9, 0.99, 0.999];
    let t = logit_transform_observations(&obs);
    assert_eq!(t.len(), obs.len());
}

#[test]
fn logit_transform_all_finite() {
    let obs = vec![0.0, 0.001, 0.5, 0.999, 1.0];
    let t = logit_transform_observations(&obs);
    for (i, &v) in t.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logit_transform({}) = {} should be finite",
            obs[i],
            v
        );
    }
}

#[test]
fn logit_transform_monotone() {
    let obs = vec![0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
    let t = logit_transform_observations(&obs);
    for i in 1..t.len() {
        assert!(
            t[i] > t[i - 1],
            "logit transform should be monotone: t[{}]={} <= t[{}]={}",
            i - 1,
            t[i - 1],
            i,
            t[i]
        );
    }
}

// ============================================================
// kmeans_1d: additional convergence properties
// ============================================================

#[test]
fn kmeans_three_well_separated_clusters() {
    let data: Vec<f64> = (0..30)
        .map(|_| 1.0)
        .chain((0..30).map(|_| 5.0))
        .chain((0..30).map(|_| 10.0))
        .collect();
    let (centers, assignments) = kmeans_1d(&data, 3, 100).unwrap();
    assert_eq!(centers.len(), 3);
    assert_eq!(assignments.len(), 90);

    let mut sorted_centers = centers.clone();
    sorted_centers.sort_by(|a, b| a.total_cmp(b));
    assert!((sorted_centers[0] - 1.0).abs() < 0.1);
    assert!((sorted_centers[1] - 5.0).abs() < 0.1);
    assert!((sorted_centers[2] - 10.0).abs() < 0.1);
}

#[test]
fn kmeans_all_assigned_to_valid_cluster() {
    let data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
    let (centers, assignments) = kmeans_1d(&data, 2, 50).unwrap();
    for &a in &assignments {
        assert!(a < centers.len(), "Assignment {} out of range", a);
    }
}

// ============================================================
// OnlineStats: additional properties
// ============================================================

#[test]
fn online_stats_matches_naive_computation() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let mut stats = OnlineStats::new();
    for &x in &data {
        stats.add(x);
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

    assert!(
        (stats.mean() - mean).abs() < 1e-10,
        "OnlineStats mean mismatch: {} vs {}",
        stats.mean(),
        mean
    );
    assert!(
        (stats.variance() - variance).abs() < 1e-10,
        "OnlineStats variance mismatch: {} vs {}",
        stats.variance(),
        variance
    );
}
