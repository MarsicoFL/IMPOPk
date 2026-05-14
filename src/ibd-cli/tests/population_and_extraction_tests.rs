//! Tests for Population methods, compute_per_window_lod, and segment extraction
//! with posterior/LOD filtering. These cover previously untested public functions.

use impopk_ibd::hmm::{
    compute_per_window_lod, extract_ibd_segments_with_lod, extract_ibd_segments_with_posteriors,
    segment_lod_score, HmmParams, Population,
};

// ===========================================================================
// Population::diversity() tests
// ===========================================================================

#[test]
fn test_population_diversity_afr_highest() {
    // AFR has the highest nucleotide diversity
    let afr = Population::AFR.diversity();
    assert!(afr > Population::EUR.diversity());
    assert!(afr > Population::EAS.diversity());
    assert!(afr > Population::CSA.diversity());
    assert!(afr > Population::AMR.diversity());
}

#[test]
fn test_population_diversity_eas_lowest() {
    // EAS has the lowest diversity among specific populations
    let eas = Population::EAS.diversity();
    assert!(eas < Population::AFR.diversity());
    assert!(eas < Population::EUR.diversity());
    assert!(eas <= Population::CSA.diversity());
    assert!(eas < Population::AMR.diversity());
}

#[test]
fn test_population_diversity_all_positive() {
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
        let d = pop.diversity();
        assert!(d > 0.0, "{:?} diversity should be positive, got {}", pop, d);
        assert!(d < 1.0, "{:?} diversity should be < 1, got {}", pop, d);
    }
}

#[test]
fn test_population_diversity_known_values() {
    assert!((Population::AFR.diversity() - 0.00125).abs() < 1e-10);
    assert!((Population::EUR.diversity() - 0.00085).abs() < 1e-10);
    assert!((Population::EAS.diversity() - 0.00080).abs() < 1e-10);
    assert!((Population::CSA.diversity() - 0.00095).abs() < 1e-10);
    assert!((Population::AMR.diversity() - 0.00100).abs() < 1e-10);
    assert!((Population::InterPop.diversity() - 0.00110).abs() < 1e-10);
    assert!((Population::Generic.diversity() - 0.00100).abs() < 1e-10);
}

// ===========================================================================
// Population::non_ibd_emission() tests
// ===========================================================================

#[test]
fn test_non_ibd_emission_mean_is_one_minus_diversity() {
    let pops = [
        Population::AFR,
        Population::EUR,
        Population::EAS,
        Population::CSA,
        Population::AMR,
    ];
    for pop in &pops {
        let emission = pop.non_ibd_emission(10000);
        let expected_mean = 1.0 - pop.diversity();
        assert!(
            (emission.mean - expected_mean).abs() < 1e-10,
            "{:?}: mean {} != expected {}",
            pop,
            emission.mean,
            expected_mean
        );
    }
}

#[test]
fn test_non_ibd_emission_std_positive() {
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
        let emission = pop.non_ibd_emission(5000);
        assert!(
            emission.std > 0.0,
            "{:?}: std should be positive, got {}",
            pop,
            emission.std
        );
    }
}

#[test]
fn test_non_ibd_emission_std_decreases_with_window_size() {
    // Larger windows => less variance (more SNPs averaged)
    let pop = Population::EUR;
    let small_window = pop.non_ibd_emission(1000);
    let large_window = pop.non_ibd_emission(100000);
    assert!(
        small_window.std > large_window.std,
        "std for 1kb ({}) should be > std for 100kb ({})",
        small_window.std,
        large_window.std
    );
}

#[test]
fn test_non_ibd_emission_afr_wider_than_eur() {
    // AFR has higher diversity, so non-IBD emission should have lower mean
    let afr = Population::AFR.non_ibd_emission(10000);
    let eur = Population::EUR.non_ibd_emission(10000);
    assert!(afr.mean < eur.mean);
    // AFR should also have wider std (more diversity = more variance)
    assert!(afr.std > eur.std);
}

#[test]
fn test_non_ibd_emission_window_size_scaling() {
    // std scales as sqrt(1/window_size), so doubling window_size should reduce std by sqrt(2)
    let pop = Population::EUR;
    let e1 = pop.non_ibd_emission(5000);
    let e2 = pop.non_ibd_emission(20000);
    let ratio = e1.std / e2.std;
    let expected_ratio = (20000.0_f64 / 5000.0).sqrt(); // sqrt(4) = 2
    assert!(
        (ratio - expected_ratio).abs() < 1e-6,
        "std ratio {} != expected {}",
        ratio,
        expected_ratio
    );
}

// ===========================================================================
// compute_per_window_lod() tests
// ===========================================================================

#[test]
fn test_compute_per_window_lod_empty() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let lods = compute_per_window_lod(&[], &params);
    assert!(lods.is_empty());
}

#[test]
fn test_compute_per_window_lod_ibd_observation_positive() {
    // An observation near the IBD emission mean should give a positive LOD
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.9997]; // near IBD emission mean
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    assert!(
        lods[0] > 0.0,
        "LOD at IBD mean should be positive, got {}",
        lods[0]
    );
}

#[test]
fn test_compute_per_window_lod_non_ibd_observation_negative() {
    // An observation near the non-IBD emission mean should give a negative LOD
    let params = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    let non_ibd_mean = params.emission[0].mean;
    let obs = vec![non_ibd_mean];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 1);
    assert!(
        lods[0] < 0.0,
        "LOD at non-IBD mean should be negative, got {}",
        lods[0]
    );
}

#[test]
fn test_compute_per_window_lod_length_matches_observations() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.999, 0.9997, 0.9999, 0.995];
    let lods = compute_per_window_lod(&obs, &params);
    assert_eq!(lods.len(), 5);
}

#[test]
fn test_compute_per_window_lod_monotonic_with_identity() {
    // Higher identity => higher LOD (closer to IBD emission)
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.990, 0.995, 0.998, 0.9997, 0.99999];
    let lods = compute_per_window_lod(&obs, &params);
    for i in 1..lods.len() {
        assert!(
            lods[i] >= lods[i - 1],
            "LOD should increase with identity: lod[{}]={} < lod[{}]={}",
            i,
            lods[i],
            i - 1,
            lods[i - 1]
        );
    }
}

#[test]
fn test_compute_per_window_lod_sum_equals_segment_lod() {
    // The sum of per-window LODs should equal the segment LOD score
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9997, 0.9999, 0.9998, 0.997];
    let lods = compute_per_window_lod(&obs, &params);
    let lod_sum: f64 = lods.iter().sum();
    let seg_lod = segment_lod_score(&obs, 0, obs.len() - 1, &params);
    assert!(
        (lod_sum - seg_lod).abs() < 1e-10,
        "Sum of per-window LODs ({}) should equal segment LOD ({})",
        lod_sum,
        seg_lod
    );
}

// ===========================================================================
// extract_ibd_segments_with_posteriors() tests
// ===========================================================================

#[test]
fn test_extract_segments_with_posteriors_empty() {
    let segments = extract_ibd_segments_with_posteriors(&[], &[], 1, 0.5);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_with_posteriors_mismatched_lengths() {
    let states = vec![0, 1, 1, 0];
    let posteriors = vec![0.1, 0.9]; // shorter than states
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_with_posteriors_no_ibd() {
    let states = vec![0, 0, 0, 0, 0];
    let posteriors = vec![0.1, 0.05, 0.15, 0.1, 0.08];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_with_posteriors_single_ibd_segment() {
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.98, 0.92, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 1);
    assert_eq!(segments[0].end_idx, 3);
    assert_eq!(segments[0].n_windows, 3);
    let expected_mean = (0.95 + 0.98 + 0.92) / 3.0;
    assert!((segments[0].mean_posterior - expected_mean).abs() < 1e-10);
    assert!((segments[0].min_posterior - 0.92).abs() < 1e-10);
    assert!((segments[0].max_posterior - 0.98).abs() < 1e-10);
}

#[test]
fn test_extract_segments_with_posteriors_min_windows_filter() {
    // Segment with 2 windows should be filtered if min_windows=3
    let states = vec![0, 1, 1, 0, 0];
    let posteriors = vec![0.1, 0.95, 0.98, 0.1, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 3, 0.5);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_with_posteriors_min_posterior_filter() {
    // Segment with low mean posterior should be filtered
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.3, 0.4, 0.35, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert!(segments.is_empty());
}

#[test]
fn test_extract_segments_with_posteriors_multiple_segments() {
    let states = vec![1, 1, 0, 0, 1, 1, 1, 0];
    let posteriors = vec![0.9, 0.95, 0.1, 0.05, 0.88, 0.92, 0.9, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 1);
    assert_eq!(segments[1].start_idx, 4);
    assert_eq!(segments[1].end_idx, 6);
}

#[test]
fn test_extract_segments_with_posteriors_segment_at_end() {
    let states = vec![0, 0, 1, 1, 1];
    let posteriors = vec![0.1, 0.1, 0.92, 0.95, 0.93];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 2);
    assert_eq!(segments[0].end_idx, 4);
}

#[test]
fn test_extract_segments_with_posteriors_all_ibd() {
    let states = vec![1, 1, 1, 1, 1];
    let posteriors = vec![0.9, 0.95, 0.98, 0.92, 0.91];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 4);
    assert_eq!(segments[0].n_windows, 5);
}

// ===========================================================================
// extract_ibd_segments_with_lod() tests
// ===========================================================================

#[test]
fn test_extract_segments_with_lod_no_obs_params() {
    // Without observations and params, LOD should be 0
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.98, 0.92, 0.1];
    let segments = extract_ibd_segments_with_lod(
        &states, &posteriors, 1, 0.5, None, None,
    );
    assert_eq!(segments.len(), 1);
    assert!((segments[0].lod_score - 0.0).abs() < 1e-10);
}

#[test]
fn test_extract_segments_with_lod_with_obs_params() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    let obs = vec![0.998, 0.9997, 0.9999, 0.9998, 0.998];
    // All IBD
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.99, 0.93, 0.1];
    let segments = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );
    assert_eq!(segments.len(), 1);
    // LOD should be computed for window indices 1-3
    let expected_lod = segment_lod_score(&obs, 1, 3, &params);
    assert!(
        (segments[0].lod_score - expected_lod).abs() < 1e-10,
        "LOD {} != expected {}",
        segments[0].lod_score,
        expected_lod
    );
}

#[test]
fn test_extract_segments_with_lod_min_lod_filter() {
    let params = HmmParams::from_expected_length(50.0, 0.0001, 5000);
    // Use values near non-IBD mean — low LOD
    let obs = vec![0.998, 0.999, 0.999, 0.999, 0.998];
    let states = vec![0, 1, 1, 1, 0];
    let posteriors = vec![0.1, 0.8, 0.85, 0.82, 0.1];

    // Without min_lod filter
    let segments_no_filter = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        None,
    );

    // With very high min_lod filter
    let segments_strict = extract_ibd_segments_with_lod(
        &states,
        &posteriors,
        1,
        0.5,
        Some((&obs, &params)),
        Some(1000.0), // unreasonably high
    );

    assert_eq!(segments_no_filter.len(), 1);
    assert!(segments_strict.is_empty());
}

#[test]
fn test_extract_segments_with_lod_single_window_ibd() {
    let states = vec![0, 1, 0];
    let posteriors = vec![0.1, 0.95, 0.1];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].n_windows, 1);
    assert_eq!(segments[0].start_idx, 0 + 1);
    assert_eq!(segments[0].end_idx, 1);
}

#[test]
fn test_extract_segments_alternating_ibd_non_ibd() {
    // Alternating: IBD(1), non-IBD(1), IBD(1), non-IBD(1), IBD(1)
    let states = vec![1, 0, 1, 0, 1];
    let posteriors = vec![0.9, 0.1, 0.92, 0.05, 0.88];
    let segments = extract_ibd_segments_with_posteriors(&states, &posteriors, 1, 0.5);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].start_idx, 0);
    assert_eq!(segments[0].end_idx, 0);
    assert_eq!(segments[1].start_idx, 2);
    assert_eq!(segments[1].end_idx, 2);
    assert_eq!(segments[2].start_idx, 4);
    assert_eq!(segments[2].end_idx, 4);
}

// ===========================================================================
// Population::from_str edge cases
// ===========================================================================

#[test]
fn test_population_from_str_mixed_case() {
    assert_eq!(Population::from_str("afr"), Some(Population::AFR));
    assert_eq!(Population::from_str("Afr"), Some(Population::AFR));
    assert_eq!(Population::from_str("eur"), Some(Population::EUR));
    assert_eq!(Population::from_str("Eas"), Some(Population::EAS));
}

#[test]
fn test_population_from_str_all_variants() {
    assert_eq!(Population::from_str("INTERPOP"), Some(Population::InterPop));
    assert_eq!(Population::from_str("INTER"), Some(Population::InterPop));
    assert_eq!(Population::from_str("GENERIC"), Some(Population::Generic));
    assert_eq!(Population::from_str("UNKNOWN"), Some(Population::Generic));
}

#[test]
fn test_population_from_str_invalid_returns_none() {
    assert_eq!(Population::from_str(""), None);
    assert_eq!(Population::from_str("FOO"), None);
    assert_eq!(Population::from_str("african"), None);
    assert_eq!(Population::from_str("AFRICAN"), None);
}

// ===========================================================================
// Integration: population-specific HMM pipeline
// ===========================================================================

#[test]
fn test_population_emission_creates_valid_hmm_params() {
    // Ensure that non_ibd_emission produces valid GaussianParams for all populations
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
        let emission = pop.non_ibd_emission(5000);
        assert!(emission.mean.is_finite(), "{:?}: mean not finite", pop);
        assert!(emission.std.is_finite(), "{:?}: std not finite", pop);
        assert!(emission.std > 0.0, "{:?}: std not positive", pop);
        // mean should be close to 1.0 (high identity)
        assert!(emission.mean > 0.99, "{:?}: mean {} not close to 1", pop, emission.mean);
        assert!(emission.mean < 1.0, "{:?}: mean {} not below 1", pop, emission.mean);
    }
}

#[test]
fn test_from_population_uses_correct_non_ibd_emission() {
    // Verify that HmmParams::from_population correctly sets non-IBD emission
    let pop = Population::EUR;
    let window_size = 5000;
    let params = HmmParams::from_population(pop, 50.0, 0.0001, window_size);
    let expected_emission = pop.non_ibd_emission(window_size as u64);
    assert!(
        (params.emission[0].mean - expected_emission.mean).abs() < 1e-10,
        "HmmParams non-IBD mean {} != expected {}",
        params.emission[0].mean,
        expected_emission.mean
    );
}
