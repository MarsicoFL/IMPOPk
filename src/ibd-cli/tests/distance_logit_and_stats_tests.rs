//! Tests for distance-dependent transitions, population-adaptive/logit constructors,
//! trimmed_mean, bic_model_selection, and forward/backward/viterbi_with_distances.

use impopk_ibd::hmm::{
    distance_dependent_log_transition, forward_backward_with_distances, forward_with_distances,
    backward_with_distances, viterbi_with_distances, forward, backward, viterbi,
    forward_backward, recombination_aware_log_transition,
    GeneticMap, HmmParams, Population,
};
use impopk_ibd::stats::{bic_model_selection, trimmed_mean, GaussianParams};

// ── distance_dependent_log_transition ───────────────────────────────────

#[test]
fn distance_dep_zero_distance_returns_base_log_transitions() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let log_trans = distance_dependent_log_transition(&params, 0, 5000);

    let expected_00 = params.transition[0][0].ln();
    let expected_01 = params.transition[0][1].ln();
    assert!((log_trans[0][0] - expected_00).abs() < 1e-10);
    assert!((log_trans[0][1] - expected_01).abs() < 1e-10);
}

#[test]
fn distance_dep_zero_window_size_returns_base_log_transitions() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let log_trans = distance_dependent_log_transition(&params, 5000, 0);
    let expected_00 = params.transition[0][0].ln();
    assert!((log_trans[0][0] - expected_00).abs() < 1e-10);
}

#[test]
fn distance_dep_nominal_distance_equals_base_rates() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // When distance == window_size, scale == 1 → should approximate base rates
    let log_trans = distance_dependent_log_transition(&params, 5000, 5000);
    // p_enter scaled with scale=1 → 1-exp(-rate*1) ≈ p_enter_base for small p
    let p_enter = log_trans[0][1].exp();
    // Should be close to base rate
    assert!(p_enter > 0.0 && p_enter < 1.0);
    assert!((p_enter - params.transition[0][1]).abs() < 0.01);
}

#[test]
fn distance_dep_larger_distance_increases_switch_prob() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let log_trans_1x = distance_dependent_log_transition(&params, 5000, 5000);
    let log_trans_5x = distance_dependent_log_transition(&params, 25000, 5000);

    let p_enter_1x = log_trans_1x[0][1].exp();
    let p_enter_5x = log_trans_5x[0][1].exp();
    assert!(p_enter_5x > p_enter_1x, "Larger distance should increase switch probability");

    let p_exit_1x = log_trans_1x[1][0].exp();
    let p_exit_5x = log_trans_5x[1][0].exp();
    assert!(p_exit_5x > p_exit_1x, "Larger distance should increase exit probability");
}

#[test]
fn distance_dep_rows_sum_to_one() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    for distance in [1000, 5000, 20000, 100000] {
        let log_trans = distance_dependent_log_transition(&params, distance, 5000);
        for row in 0..2 {
            let sum = log_trans[row][0].exp() + log_trans[row][1].exp();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "Row {} doesn't sum to 1.0 for distance={}: got {}",
                row,
                distance,
                sum
            );
        }
    }
}

#[test]
fn distance_dep_very_large_distance_saturates_near_stationary() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // 10 Mb gap → should be near-stationary
    let log_trans = distance_dependent_log_transition(&params, 10_000_000, 5000);
    let p_enter = log_trans[0][1].exp();
    // For very large distance, p_enter → high (near 1 minus clamping)
    assert!(p_enter > 0.5, "Very large distance should have high switch probability");
}

#[test]
fn distance_dep_all_populations_produce_valid_transitions() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
        Population::Generic,
    ];
    for pop in &pops {
        let params = HmmParams::from_population(*pop, 50.0, 0.001, 5000);
        let log_trans = distance_dependent_log_transition(&params, 7500, 5000);
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    log_trans[row][col].is_finite(),
                    "Non-finite log transition for {:?}",
                    pop
                );
                assert!(log_trans[row][col] <= 0.0, "Log prob > 0 for {:?}", pop);
            }
        }
    }
}

// ── recombination_aware_log_transition ───────────────────────────────────

#[test]
fn recomb_aware_same_position_returns_base_rates() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let gmap = GeneticMap::uniform(0, 1_000_000, 1.0);
    let log_trans = recombination_aware_log_transition(&params, 5000, 5000, &gmap, 5000);
    // Same position → no change, base rates in log space
    let expected_00 = params.transition[0][0].ln();
    assert!((log_trans[0][0] - expected_00).abs() < 1e-10);
}

#[test]
fn recomb_aware_hotspot_increases_exit_rate() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let gmap_low = GeneticMap::uniform(0, 1_000_000, 0.5); // Low recombination
    let gmap_high = GeneticMap::uniform(0, 1_000_000, 5.0); // High recombination

    let log_trans_low = recombination_aware_log_transition(&params, 0, 10000, &gmap_low, 5000);
    let log_trans_high = recombination_aware_log_transition(&params, 0, 10000, &gmap_high, 5000);

    let p_exit_low = log_trans_low[1][0].exp();
    let p_exit_high = log_trans_high[1][0].exp();
    // Higher recombination should increase exit probability
    assert!(
        p_exit_high >= p_exit_low,
        "Hotspot should have higher exit rate: high={} low={}",
        p_exit_high,
        p_exit_low
    );
}

#[test]
fn recomb_aware_rows_always_sum_to_one() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let gmap = GeneticMap::uniform(0, 10_000_000, 1.0);
    for pos2 in [5000, 10000, 50000, 500000] {
        let log_trans =
            recombination_aware_log_transition(&params, 0, pos2, &gmap, 5000);
        for row in 0..2 {
            let sum = log_trans[row][0].exp() + log_trans[row][1].exp();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "Row sum != 1.0 for pos2={}: {}",
                pos2,
                sum
            );
        }
    }
}

// ── forward_with_distances ──────────────────────────────────────────────

#[test]
fn forward_with_distances_empty_returns_empty() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let (alpha, ll) = forward_with_distances(&[], &params, &[]);
    assert!(alpha.is_empty());
    assert_eq!(ll, 0.0);
}

#[test]
fn forward_with_distances_uniform_matches_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.998, 0.9995, 0.997, 0.9993, 0.998];
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

    let (alpha_std, ll_std) = forward(&obs, &params);
    let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);

    assert!(
        (ll_std - ll_dist).abs() < 0.1,
        "Uniform distances should closely match standard forward: std={} dist={}",
        ll_std,
        ll_dist
    );
    assert_eq!(alpha_std.len(), alpha_dist.len());
}

#[test]
fn forward_with_distances_mismatched_positions_falls_back() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.998, 0.9995, 0.997];
    let positions = [(0, 4999), (5000, 9999)]; // Mismatch: 2 positions for 3 obs

    let (alpha_dist, ll_dist) = forward_with_distances(&obs, &params, &positions);
    let (alpha_std, ll_std) = forward(&obs, &params);

    // Should fall back to standard forward
    assert_eq!(alpha_dist.len(), alpha_std.len());
    assert!((ll_dist - ll_std).abs() < 1e-10);
}

#[test]
fn forward_with_distances_single_observation() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.9995];
    let positions = [(0, 4999)];
    let (alpha, ll) = forward_with_distances(&obs, &params, &positions);
    assert_eq!(alpha.len(), 1);
    assert!(ll.is_finite());
}

// ── backward_with_distances ─────────────────────────────────────────────

#[test]
fn backward_with_distances_empty_returns_empty() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let beta = backward_with_distances(&[], &params, &[]);
    assert!(beta.is_empty());
}

#[test]
fn backward_with_distances_uniform_matches_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.998, 0.9995, 0.997, 0.9993, 0.998];
    let positions: Vec<(u64, u64)> = (0..5).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

    let beta_std = backward(&obs, &params);
    let beta_dist = backward_with_distances(&obs, &params, &positions);

    assert_eq!(beta_std.len(), beta_dist.len());
    // Last entry should be similar (both initialized to [0, 0] in log space)
    for s in 0..2 {
        assert!(
            (beta_std[4][s] - beta_dist[4][s]).abs() < 0.1,
            "Last beta differs: std={} dist={}",
            beta_std[4][s],
            beta_dist[4][s]
        );
    }
}

// ── forward_backward_with_distances ─────────────────────────────────────

#[test]
fn fb_with_distances_posteriors_in_range() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.998, 0.9995, 0.997, 0.9993, 0.998, 0.9996, 0.997];
    let positions: Vec<(u64, u64)> = (0..7).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

    let (posteriors, ll) = forward_backward_with_distances(&obs, &params, &positions);
    assert_eq!(posteriors.len(), obs.len());
    assert!(ll.is_finite());
    for (i, &p) in posteriors.iter().enumerate() {
        assert!(
            p >= 0.0 && p <= 1.0,
            "Posterior {} out of range at idx {}: {}",
            p,
            i,
            p
        );
    }
}

#[test]
fn fb_with_distances_gap_shifts_posteriors() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    // Non-IBD followed by IBD-like region
    let obs = [0.997, 0.997, 0.997, 0.9998, 0.9997, 0.9999, 0.997, 0.997];

    // Uniform positions
    let pos_uniform: Vec<(u64, u64)> = (0..8).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();
    let (post_uniform, _) = forward_backward_with_distances(&obs, &params, &pos_uniform);

    // Gap before IBD region (100kb gap)
    let pos_gap = vec![
        (0, 4999),
        (5000, 9999),
        (10000, 14999),
        (115000, 119999), // 100kb gap
        (120000, 124999),
        (125000, 129999),
        (130000, 134999),
        (135000, 139999),
    ];
    let (post_gap, _) = forward_backward_with_distances(&obs, &params, &pos_gap);

    // Gap should make the IBD region posteriors different from uniform
    let diff: f64 = post_uniform
        .iter()
        .zip(post_gap.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.001,
        "Gap should affect posteriors (diff={})",
        diff
    );
}

// ── viterbi_with_distances ──────────────────────────────────────────────

#[test]
fn viterbi_with_distances_empty_returns_empty() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let states = viterbi_with_distances(&[], &params, &[]);
    assert!(states.is_empty());
}

#[test]
fn viterbi_with_distances_single_obs() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let states = viterbi_with_distances(&[0.9995], &params, &[(0, 4999)]);
    assert_eq!(states.len(), 1);
}

#[test]
fn viterbi_with_distances_uniform_matches_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs: Vec<f64> = (0..10).map(|_| 0.998).collect();
    let positions: Vec<(u64, u64)> = (0..10).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();

    let states_std = viterbi(&obs, &params);
    let states_dist = viterbi_with_distances(&obs, &params, &positions);
    assert_eq!(states_std, states_dist);
}

#[test]
fn viterbi_with_distances_mismatched_falls_back_to_standard() {
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.001, 5000);
    let obs = [0.998, 0.9995, 0.997];
    let positions = [(0, 4999), (5000, 9999)]; // Mismatch

    let states_dist = viterbi_with_distances(&obs, &params, &positions);
    let states_std = viterbi(&obs, &params);
    assert_eq!(states_dist, states_std);
}

#[test]
fn viterbi_with_distances_detects_ibd_region() {
    // Use well-separated emission parameters for reliable detection
    let params = HmmParams {
        initial: [0.999, 0.001],
        transition: [[0.999, 0.001], [0.02, 0.98]],
        emission: [
            GaussianParams::new_unchecked(0.990, 0.003), // Non-IBD: mean=0.990
            GaussianParams::new_unchecked(0.999, 0.001), // IBD: mean=0.999
        ],
    };
    // Non-IBD (0.990) ... IBD (0.999+) ... Non-IBD
    let mut obs = vec![0.990; 20];
    for i in 8..14 {
        obs[i] = 0.999;
    }
    let positions: Vec<(u64, u64)> = (0..20).map(|i| (i * 5000, (i + 1) * 5000 - 1)).collect();
    let states = viterbi_with_distances(&obs, &params, &positions);

    // The IBD region should have state=1
    let ibd_count: usize = states[8..14].iter().filter(|&&s| s == 1).count();
    assert!(
        ibd_count >= 4,
        "Should detect most IBD windows, got {} IBD out of 6",
        ibd_count
    );
}

// ── from_population_adaptive ────────────────────────────────────────────

#[test]
fn adaptive_afr_has_lower_entry_than_eur() {
    let params_afr = HmmParams::from_population_adaptive(Population::AFR, 50.0, 0.001, 5000);
    let params_eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.001, 5000);
    assert!(
        params_afr.transition[0][1] < params_eur.transition[0][1],
        "AFR should have lower IBD entry rate"
    );
}

#[test]
fn adaptive_interpop_has_very_low_entry() {
    let params_inter =
        HmmParams::from_population_adaptive(Population::InterPop, 50.0, 0.001, 5000);
    let params_generic =
        HmmParams::from_population_adaptive(Population::Generic, 50.0, 0.001, 5000);
    assert!(
        params_inter.transition[0][1] < params_generic.transition[0][1] * 0.5,
        "InterPop should have much lower IBD entry rate"
    );
}

#[test]
fn adaptive_all_populations_produce_valid_params() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::InterPop,
        Population::Generic,
    ];
    for pop in &pops {
        let params = HmmParams::from_population_adaptive(*pop, 50.0, 0.001, 5000);
        // Transitions should sum to 1 per row
        for row in 0..2 {
            let sum = params.transition[row][0] + params.transition[row][1];
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Transition row {} doesn't sum to 1 for {:?}",
                row,
                pop
            );
        }
        // Emission params should be valid
        assert!(params.emission[0].mean > 0.0);
        assert!(params.emission[1].mean > 0.0);
    }
}

#[test]
fn adaptive_eas_has_longer_segments_than_eur() {
    let params_eas = HmmParams::from_population_adaptive(Population::EAS, 50.0, 0.001, 5000);
    let params_eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.001, 5000);
    // EAS has 1.1x expected windows → higher stay probability (less exit)
    assert!(
        params_eas.transition[1][1] >= params_eur.transition[1][1],
        "EAS should have higher stay-in-IBD probability"
    );
}

#[test]
fn adaptive_extreme_p_enter_gets_clamped() {
    // Very small p_enter_ibd → should get clamped to 1e-8
    let params = HmmParams::from_population_adaptive(Population::InterPop, 50.0, 1e-20, 5000);
    // InterPop scales by 0.1 → 1e-21 → clamped to 1e-8
    assert!(params.transition[0][1] >= 1e-8);
}

// ── from_population_logit ───────────────────────────────────────────────

#[test]
fn logit_creates_valid_params_all_populations() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
        Population::Generic,
    ];
    for pop in &pops {
        let params = HmmParams::from_population_logit(*pop, 50.0, 0.001, 5000);
        // Logit-space means should be > 5 (since raw values are near 1.0)
        assert!(
            params.emission[0].mean > 5.0,
            "Non-IBD logit mean should be > 5 for {:?}, got {}",
            pop,
            params.emission[0].mean
        );
        assert!(
            params.emission[1].mean > 5.0,
            "IBD logit mean should be > 5 for {:?}, got {}",
            pop,
            params.emission[1].mean
        );
        // IBD mean should be higher than non-IBD (higher identity → higher logit)
        assert!(
            params.emission[1].mean > params.emission[0].mean,
            "IBD logit mean should exceed non-IBD for {:?}",
            pop
        );
    }
}

#[test]
fn logit_separation_exceeds_raw_space() {
    let pop = Population::EUR;
    let raw_params = HmmParams::from_population(pop, 50.0, 0.001, 5000);
    let logit_params = HmmParams::from_population_logit(pop, 50.0, 0.001, 5000);

    let raw_sep = raw_params.emission[1].mean - raw_params.emission[0].mean;
    let logit_sep = logit_params.emission[1].mean - logit_params.emission[0].mean;

    assert!(
        logit_sep > raw_sep,
        "Logit separation ({}) should exceed raw separation ({})",
        logit_sep,
        raw_sep
    );
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn logit_panics_on_invalid_p_enter() {
    let _ = HmmParams::from_population_logit(Population::EUR, 50.0, 0.0, 5000);
}

#[test]
#[should_panic(expected = "p_enter_ibd must be in range")]
fn logit_panics_on_p_enter_one() {
    let _ = HmmParams::from_population_logit(Population::EUR, 50.0, 1.0, 5000);
}

#[test]
fn logit_transitions_are_valid_probabilities() {
    let params = HmmParams::from_population_logit(Population::EUR, 50.0, 0.001, 5000);
    for row in 0..2 {
        let sum = params.transition[row][0] + params.transition[row][1];
        assert!((sum - 1.0).abs() < 1e-10);
        for col in 0..2 {
            assert!(params.transition[row][col] >= 0.0);
            assert!(params.transition[row][col] <= 1.0);
        }
    }
}

// ── trimmed_mean ────────────────────────────────────────────────────────

#[test]
fn trimmed_mean_empty_returns_none() {
    assert!(trimmed_mean(&[], 0.1).is_none());
}

#[test]
fn trimmed_mean_single_element() {
    assert_eq!(trimmed_mean(&[42.0], 0.1).unwrap(), 42.0);
}

#[test]
fn trimmed_mean_no_trim() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = trimmed_mean(&data, 0.0).unwrap();
    assert!((result - 3.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_full_trim_returns_median() {
    let data = [1.0, 100.0, 3.0, 200.0, 5.0];
    // 49% trim: trims 2 from each end, leaving just the median
    let result = trimmed_mean(&data, 0.49).unwrap();
    // Sorted: [1.0, 3.0, 5.0, 100.0, 200.0], trim_count = floor(5*0.49)=2 → leaves [5.0]
    assert!((result - 5.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_robust_to_outliers() {
    let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    data.push(1000.0); // Outlier
    data.push(-500.0); // Outlier

    let untrimmed = trimmed_mean(&data, 0.0).unwrap();
    let trimmed = trimmed_mean(&data, 0.15).unwrap();
    // Trimmed mean should be closer to the true center (5.5)
    assert!((trimmed - 5.5).abs() < (untrimmed - 5.5).abs());
}

#[test]
fn trimmed_mean_negative_trim_clamped_to_zero() {
    let data = [1.0, 2.0, 3.0];
    // Negative trim fraction → clamped to 0
    let result = trimmed_mean(&data, -0.5).unwrap();
    assert!((result - 2.0).abs() < 1e-10);
}

#[test]
fn trimmed_mean_over_49_percent_clamped() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    // >0.49 clamped to 0.49
    let result = trimmed_mean(&data, 0.9).unwrap();
    // Same as 0.49 trim
    let result2 = trimmed_mean(&data, 0.49).unwrap();
    assert_eq!(result, result2);
}

#[test]
fn trimmed_mean_with_nan_sorts_nan_to_end() {
    // NaN sorts last with total_cmp
    let data = [1.0, f64::NAN, 3.0, 2.0, f64::NAN];
    // Sorted: [1.0, 2.0, 3.0, NaN, NaN], trim 1 each side → [2.0, 3.0, NaN]
    let result = trimmed_mean(&data, 0.2);
    // Result includes NaN → will be NaN
    assert!(result.is_some());
}

#[test]
fn trimmed_mean_identical_values() {
    let data = [5.0; 100];
    let result = trimmed_mean(&data, 0.25).unwrap();
    assert!((result - 5.0).abs() < 1e-10);
}

// ── bic_model_selection ─────────────────────────────────────────────────

#[test]
fn bic_too_few_data_returns_zeros() {
    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (bic_1, bic_2) = bic_model_selection(&[0.998], &p_low, &p_high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

#[test]
fn bic_empty_data_returns_zeros() {
    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.9997, 0.0005);
    let (bic_1, bic_2) = bic_model_selection(&[], &p_low, &p_high, 0.5);
    assert_eq!(bic_1, 0.0);
    assert_eq!(bic_2, 0.0);
}

#[test]
fn bic_prefers_two_components_clear_separation() {
    // Generate bimodal data
    let mut data = Vec::new();
    for i in 0..50 {
        data.push(0.996 + 0.001 * (i as f64 / 50.0));
    }
    for i in 0..50 {
        data.push(0.9995 + 0.0003 * (i as f64 / 50.0));
    }

    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.9997, 0.0002);
    let (bic_1, bic_2) = bic_model_selection(&data, &p_low, &p_high, 0.5);

    assert!(
        bic_2 < bic_1,
        "BIC should prefer 2 components for bimodal data: bic_1={} bic_2={}",
        bic_1,
        bic_2
    );
}

#[test]
fn bic_prefers_one_component_unimodal() {
    // Unimodal data
    let data: Vec<f64> = (0..100).map(|i| 0.997 + 0.001 * (i as f64 / 100.0)).collect();

    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.998, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&data, &p_low, &p_high, 0.5);

    // When components overlap heavily, 1-component should win due to lower penalty
    assert!(
        bic_1 <= bic_2,
        "BIC should prefer 1 component for unimodal data: bic_1={} bic_2={}",
        bic_1,
        bic_2
    );
}

#[test]
fn bic_returns_finite_values_for_normal_data() {
    let data: Vec<f64> = (0..50).map(|i| 0.997 + 0.0001 * i as f64).collect();
    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.999, 0.001);
    let (bic_1, bic_2) = bic_model_selection(&data, &p_low, &p_high, 0.5);
    assert!(bic_1.is_finite());
    assert!(bic_2.is_finite());
}

#[test]
fn bic_weight_extremes() {
    let data: Vec<f64> = (0..100).map(|i| 0.997 + 0.002 * (i as f64 / 100.0)).collect();
    let p_low = GaussianParams::new_unchecked(0.997, 0.001);
    let p_high = GaussianParams::new_unchecked(0.999, 0.001);

    // Weight = 0 → all data assigned to high component
    let (_, bic_2_w0) = bic_model_selection(&data, &p_low, &p_high, 0.0001);
    // Weight = 1 → all data assigned to low component
    let (_, bic_2_w1) = bic_model_selection(&data, &p_low, &p_high, 0.9999);
    // Both should be finite
    assert!(bic_2_w0.is_finite());
    assert!(bic_2_w1.is_finite());
}
