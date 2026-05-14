//! Tests for HmmParams::from_population_adaptive across all populations,
//! Segment::fraction_called edge cases, IdentityTrack methods, and
//! Population-specific parameter ordering invariants.

use impopk_ibd::hmm::{HmmParams, Population};
use impopk_ibd::segment::{IdentityTrack, Segment, segment_length_distribution};

// ============================================================
// from_population_adaptive: per-population parameter ordering
// ============================================================

#[test]
fn test_from_population_adaptive_eur_same_as_base() {
    // EUR uses no scaling factors (multiplier = 1.0 on both)
    let adaptive = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.0001, 5000);
    let base = HmmParams::from_population(Population::EUR, 50.0, 0.0001, 5000);
    // Transitions should be identical since EUR has no adjustment
    assert!((adaptive.transition[0][1] - base.transition[0][1]).abs() < 1e-12,
        "EUR adaptive p_enter should match base: {} vs {}", adaptive.transition[0][1], base.transition[0][1]);
    assert!((adaptive.transition[1][1] - base.transition[1][1]).abs() < 1e-12,
        "EUR adaptive p_stay should match base: {} vs {}", adaptive.transition[1][1], base.transition[1][1]);
}

#[test]
fn test_from_population_adaptive_eas_longer_segments() {
    // EAS: expected_ibd_windows * 1.1, p_enter same
    let eas = HmmParams::from_population_adaptive(Population::EAS, 50.0, 0.0001, 5000);
    let eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.0001, 5000);
    // EAS should have higher p_stay_ibd (longer segments)
    assert!(eas.transition[1][1] > eur.transition[1][1],
        "EAS p_stay_ibd ({}) should be > EUR p_stay_ibd ({})",
        eas.transition[1][1], eur.transition[1][1]);
    // p_enter_ibd should be identical (same multiplier)
    assert!((eas.transition[0][1] - eur.transition[0][1]).abs() < 1e-12,
        "EAS and EUR should have same p_enter_ibd");
}

#[test]
fn test_from_population_adaptive_csa_intermediate() {
    // CSA: expected * 0.9, p_enter * 0.8
    let csa = HmmParams::from_population_adaptive(Population::CSA, 50.0, 0.0001, 5000);
    let eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.0001, 5000);
    // CSA should have lower p_enter_ibd than EUR
    assert!(csa.transition[0][1] < eur.transition[0][1],
        "CSA p_enter ({}) should be < EUR p_enter ({})", csa.transition[0][1], eur.transition[0][1]);
    // CSA should have slightly shorter segments (expected * 0.9)
    assert!(csa.transition[1][1] < eur.transition[1][1],
        "CSA p_stay ({}) should be < EUR p_stay ({})", csa.transition[1][1], eur.transition[1][1]);
}

#[test]
fn test_from_population_adaptive_amr_admixed() {
    // AMR: expected * 0.8, p_enter * 0.7
    let amr = HmmParams::from_population_adaptive(Population::AMR, 50.0, 0.0001, 5000);
    let eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.0001, 5000);
    // AMR should have lower p_enter than EUR
    assert!(amr.transition[0][1] < eur.transition[0][1],
        "AMR p_enter ({}) should be < EUR p_enter ({})", amr.transition[0][1], eur.transition[0][1]);
}

#[test]
fn test_from_population_adaptive_generic_same_as_base() {
    // Generic uses multiplier = 1.0 on both (same as base)
    let adaptive = HmmParams::from_population_adaptive(Population::Generic, 50.0, 0.0001, 5000);
    let base = HmmParams::from_population(Population::Generic, 50.0, 0.0001, 5000);
    assert!((adaptive.transition[0][1] - base.transition[0][1]).abs() < 1e-12);
    assert!((adaptive.transition[1][1] - base.transition[1][1]).abs() < 1e-12);
}

#[test]
fn test_from_population_adaptive_ordering_afr_least_ibd() {
    // AFR should have the lowest p_enter_ibd of all populations
    let afr = HmmParams::from_population_adaptive(Population::AFR, 50.0, 0.0001, 5000);
    let eur = HmmParams::from_population_adaptive(Population::EUR, 50.0, 0.0001, 5000);
    let eas = HmmParams::from_population_adaptive(Population::EAS, 50.0, 0.0001, 5000);
    let csa = HmmParams::from_population_adaptive(Population::CSA, 50.0, 0.0001, 5000);
    let amr = HmmParams::from_population_adaptive(Population::AMR, 50.0, 0.0001, 5000);

    let afr_enter = afr.transition[0][1];
    assert!(afr_enter < eur.transition[0][1], "AFR < EUR for p_enter");
    assert!(afr_enter < eas.transition[0][1], "AFR < EAS for p_enter");
    assert!(afr_enter < csa.transition[0][1], "AFR < CSA for p_enter");
    assert!(afr_enter < amr.transition[0][1], "AFR < AMR for p_enter");
}

#[test]
fn test_from_population_adaptive_interpop_rarest_ibd() {
    // InterPop should have even lower p_enter than AFR
    let inter = HmmParams::from_population_adaptive(Population::InterPop, 50.0, 0.0001, 5000);
    let afr = HmmParams::from_population_adaptive(Population::AFR, 50.0, 0.0001, 5000);
    assert!(inter.transition[0][1] < afr.transition[0][1],
        "InterPop p_enter ({}) should be < AFR p_enter ({})",
        inter.transition[0][1], afr.transition[0][1]);
}

#[test]
fn test_from_population_adaptive_all_valid_params() {
    // All populations should produce valid HMM parameters
    let pops = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];

    for pop in &pops {
        let params = HmmParams::from_population_adaptive(*pop, 50.0, 0.0001, 5000);
        // Transition row sums should be ~1.0
        let row0_sum = params.transition[0][0] + params.transition[0][1];
        let row1_sum = params.transition[1][0] + params.transition[1][1];
        assert!((row0_sum - 1.0).abs() < 1e-10, "{:?} row0 sum: {}", pop, row0_sum);
        assert!((row1_sum - 1.0).abs() < 1e-10, "{:?} row1 sum: {}", pop, row1_sum);

        // All probabilities should be in (0, 1)
        assert!(params.transition[0][1] > 0.0, "{:?} p_enter > 0", pop);
        assert!(params.transition[0][1] < 1.0, "{:?} p_enter < 1", pop);
        assert!(params.transition[1][1] > 0.0, "{:?} p_stay > 0", pop);
        assert!(params.transition[1][1] < 1.0, "{:?} p_stay < 1", pop);

        // Emission means should be reasonable
        assert!(params.emission[0].mean > 0.99 && params.emission[0].mean < 1.0,
            "{:?} non-IBD mean should be ~0.999", pop);
        assert!(params.emission[1].mean > 0.99 && params.emission[1].mean < 1.001,
            "{:?} IBD mean should be ~0.9997", pop);
    }
}

#[test]
fn test_from_population_adaptive_clamping() {
    // With very small p_enter_ibd, clamping should prevent it from going to zero
    let inter = HmmParams::from_population_adaptive(Population::InterPop, 50.0, 1e-7, 5000);
    assert!(inter.transition[0][1] > 0.0, "p_enter should be clamped above 0");
    assert!(inter.transition[0][1] >= 1e-8, "p_enter should be at least 1e-8 after clamping");
}

// ============================================================
// Segment::fraction_called edge cases
// ============================================================

fn make_test_segment(n_windows: usize, n_called: usize) -> Segment {
    Segment {
        chrom: "chr1".to_string(),
        start: 1000,
        end: 2000,
        hap_a: "A".to_string(),
        hap_b: "B".to_string(),
        n_windows,
        mean_identity: 0.999,
        min_identity: 0.998,
        identity_sum: 0.999 * n_called as f64,
        n_called,
        start_idx: 0,
        end_idx: n_windows.saturating_sub(1),
    }
}

#[test]
fn test_fraction_called_zero_windows() {
    let seg = make_test_segment(0, 0);
    assert_eq!(seg.fraction_called(), 0.0);
}

#[test]
fn test_fraction_called_all_called() {
    let seg = make_test_segment(10, 10);
    assert!((seg.fraction_called() - 1.0).abs() < 1e-10);
}

#[test]
fn test_fraction_called_none_called() {
    let seg = make_test_segment(10, 0);
    assert_eq!(seg.fraction_called(), 0.0);
}

#[test]
fn test_fraction_called_half_called() {
    let seg = make_test_segment(10, 5);
    assert!((seg.fraction_called() - 0.5).abs() < 1e-10);
}

#[test]
fn test_fraction_called_one_of_many() {
    let seg = make_test_segment(100, 1);
    assert!((seg.fraction_called() - 0.01).abs() < 1e-10);
}

#[test]
fn test_length_bp_start_equals_end() {
    let seg = make_test_segment(1, 1);
    // start=1000, end=2000, length = 2000 - 1000 + 1 = 1001
    assert_eq!(seg.length_bp(), 1001);
}

#[test]
fn test_length_bp_zero_length() {
    let mut seg = make_test_segment(1, 1);
    seg.start = 5000;
    seg.end = 5000;
    assert_eq!(seg.length_bp(), 1); // 5000 - 5000 + 1 = 1
}

#[test]
fn test_length_bp_saturating_sub() {
    let mut seg = make_test_segment(1, 1);
    seg.start = 10000;
    seg.end = 5000; // start > end
    // saturating_sub: 5000 - 10000 = 0, + 1 = 1
    assert_eq!(seg.length_bp(), 1);
}

// ============================================================
// IdentityTrack methods
// ============================================================

#[test]
fn test_identity_track_get_existing() {
    let track = IdentityTrack {
        windows: vec![(0, 0.99), (2, 0.999), (5, 0.9999)],
        n_total_windows: 10,
    };
    assert_eq!(track.get(0), Some(0.99));
    assert_eq!(track.get(2), Some(0.999));
    assert_eq!(track.get(5), Some(0.9999));
}

#[test]
fn test_identity_track_get_missing() {
    let track = IdentityTrack {
        windows: vec![(0, 0.99), (2, 0.999)],
        n_total_windows: 10,
    };
    assert_eq!(track.get(1), None);
    assert_eq!(track.get(3), None);
    assert_eq!(track.get(100), None);
}

#[test]
fn test_identity_track_get_empty() {
    let track = IdentityTrack {
        windows: vec![],
        n_total_windows: 0,
    };
    assert_eq!(track.get(0), None);
}

#[test]
fn test_identity_track_to_map() {
    let track = IdentityTrack {
        windows: vec![(0, 0.99), (3, 0.999), (7, 0.9999)],
        n_total_windows: 10,
    };
    let map = track.to_map();
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&0), Some(&0.99));
    assert_eq!(map.get(&3), Some(&0.999));
    assert_eq!(map.get(&7), Some(&0.9999));
    assert_eq!(map.get(&1), None);
}

#[test]
fn test_identity_track_to_map_empty() {
    let track = IdentityTrack {
        windows: vec![],
        n_total_windows: 0,
    };
    let map = track.to_map();
    assert!(map.is_empty());
}

#[test]
fn test_identity_track_to_map_single() {
    let track = IdentityTrack {
        windows: vec![(42, 0.5)],
        n_total_windows: 100,
    };
    let map = track.to_map();
    assert_eq!(map.len(), 1);
    assert_eq!(map[&42], 0.5);
}

// ============================================================
// segment_length_distribution with single segment
// ============================================================

#[test]
fn test_segment_length_distribution_single() {
    let seg = make_test_segment(10, 10);
    let stats = segment_length_distribution(&[seg]);
    assert_eq!(stats.count, 1);
    assert!((stats.median_bp - 1001.0).abs() < 0.1); // 2000 - 1000 + 1
    assert_eq!(stats.min_bp, 1001);
    assert_eq!(stats.max_bp, 1001);
    assert!((stats.mean_bp - 1001.0).abs() < 0.1);
}

// ============================================================
// HmmParams::summary output format
// ============================================================

#[test]
fn test_hmm_params_summary_all_populations() {
    let pops = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];

    for pop in &pops {
        let params = HmmParams::from_population_adaptive(*pop, 50.0, 0.0001, 5000);
        let summary = params.summary();
        assert!(summary.contains("HMM Parameters:"), "{:?}: missing header", pop);
        assert!(summary.contains("Initial:"), "{:?}: missing Initial", pop);
        assert!(summary.contains("Transition:"), "{:?}: missing Transition", pop);
        assert!(summary.contains("Emission non-IBD:"), "{:?}: missing non-IBD emission", pop);
        assert!(summary.contains("Emission IBD:"), "{:?}: missing IBD emission", pop);
    }
}

// ============================================================
// from_population_logit: all populations produce valid params
// ============================================================

#[test]
fn test_from_population_logit_all_populations_valid() {
    let pops = [
        Population::AFR, Population::EUR, Population::EAS,
        Population::CSA, Population::AMR, Population::InterPop,
        Population::Generic,
    ];

    for pop in &pops {
        let params = HmmParams::from_population_logit(*pop, 50.0, 0.0001, 5000);
        // Row sums should be ~1.0
        let row0_sum = params.transition[0][0] + params.transition[0][1];
        let row1_sum = params.transition[1][0] + params.transition[1][1];
        assert!((row0_sum - 1.0).abs() < 1e-10, "{:?} row0 sum: {}", pop, row0_sum);
        assert!((row1_sum - 1.0).abs() < 1e-10, "{:?} row1 sum: {}", pop, row1_sum);

        // Emission means should be finite
        assert!(params.emission[0].mean.is_finite(), "{:?} non-IBD mean should be finite", pop);
        assert!(params.emission[1].mean.is_finite(), "{:?} IBD mean should be finite", pop);
        // In logit space, IBD mean should be > non-IBD mean (since logit(0.9997) > logit(0.999))
        assert!(params.emission[1].mean > params.emission[0].mean,
            "{:?}: logit IBD mean ({}) should be > logit non-IBD mean ({})",
            pop, params.emission[1].mean, params.emission[0].mean);
    }
}
