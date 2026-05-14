//! Tests for posterior-based ancestry boundary refinement

use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestryObservation, AncestryHmmParams, AncestralPopulation,
    extract_ancestry_segments, refine_ancestry_boundaries,
};

fn make_obs(chrom: &str, start: u64, end: u64, sims: Vec<(&str, f64)>) -> AncestryObservation {
    let mut similarities = HashMap::new();
    for (name, sim) in sims {
        similarities.insert(name.to_string(), sim);
    }
    AncestryObservation {
        chrom: chrom.to_string(),
        start,
        end,
        sample: "TEST#1".to_string(),
        similarities,
        coverage_ratios: None,
            haplotype_consistency_bonus: None,
    }
}

fn make_params_2pop() -> AncestryHmmParams {
    let pops = vec![
        AncestralPopulation {
            name: "POP_A".to_string(),
            haplotypes: vec!["refA1".to_string(), "refA2".to_string()],
        },
        AncestralPopulation {
            name: "POP_B".to_string(),
            haplotypes: vec!["refB1".to_string(), "refB2".to_string()],
        },
    ];
    AncestryHmmParams::new(pops, 0.001)
}

fn observations_10kb(n: usize, high_pop: &[(usize, usize, &str)]) -> Vec<AncestryObservation> {
    // Create n observations with 10kb windows
    // high_pop: (from_idx, to_idx, pop_prefix) — windows where that pop has high similarity
    let mut obs = Vec::new();
    for i in 0..n {
        let start = i as u64 * 10000 + 1;
        let end = (i as u64 + 1) * 10000;

        // Default: low similarity to both
        let mut sim_a = 0.99;
        let mut sim_b = 0.99;

        for &(from, to, pop) in high_pop {
            if i >= from && i <= to {
                if pop == "A" {
                    sim_a = 0.999;
                    sim_b = 0.990;
                } else {
                    sim_a = 0.990;
                    sim_b = 0.999;
                }
            }
        }

        obs.push(make_obs("chr1", start, end, vec![
            ("refA1", sim_a), ("refA2", sim_a - 0.001),
            ("refB1", sim_b), ("refB2", sim_b - 0.001),
        ]));
    }
    obs
}

#[test]
fn basic_ancestry_boundary_refinement() {
    let params = make_params_2pop();
    let obs = observations_10kb(6, &[(0, 2, "A"), (3, 5, "B")]);

    // States: A A A B B B — switch at window 3
    let states = vec![0, 0, 0, 1, 1, 1];

    // Posteriors: state 0 goes from high to low at boundary
    let posteriors = vec![
        vec![0.95, 0.05],
        vec![0.90, 0.10],
        vec![0.80, 0.20],
        vec![0.15, 0.85],
        vec![0.05, 0.95],
        vec![0.05, 0.95],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
    assert_eq!(segments.len(), 2);

    let refined = refine_ancestry_boundaries(&segments, &posteriors, &obs, 0.5);
    assert_eq!(refined.len(), 2);

    // First segment (POP_A, windows 0-2):
    // End boundary: P_A at window 2 = 0.80 > 0.5, P_A at window 3 = 0.15 < 0.5
    // t = (0.80 - 0.50) / (0.80 - 0.15) ≈ 0.462
    // center[2] = 25000.5, center[3] = 35000.5
    // boundary ≈ 25000.5 + 0.462 * 10000 ≈ 29615
    assert!(refined[0].end_bp > 25000, "end should be past center of window 2");
    assert!(refined[0].end_bp < 35000, "end should be before center of window 3");

    // Second segment (POP_B, windows 3-5):
    // Start boundary: P_B at window 2 = 0.20 < 0.5, P_B at window 3 = 0.85 > 0.5
    // t = (0.5 - 0.20) / (0.85 - 0.20) ≈ 0.462
    // boundary ≈ 25000.5 + 0.462 * 10000 ≈ 29615
    assert!(refined[1].start_bp > 25000, "start should be past center of window 2");
    assert!(refined[1].start_bp < 35000, "start should be before center of window 3");
}

#[test]
fn no_refinement_at_edges() {
    let params = make_params_2pop();
    let obs = observations_10kb(4, &[(0, 3, "A")]);
    let states = vec![0, 0, 0, 0]; // all same state
    let posteriors = vec![
        vec![0.95, 0.05],
        vec![0.95, 0.05],
        vec![0.95, 0.05],
        vec![0.95, 0.05],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
    assert_eq!(segments.len(), 1);

    let refined = refine_ancestry_boundaries(&segments, &posteriors, &obs, 0.5);
    // Single segment covering all windows — no adjacent non-assigned windows
    assert_eq!(refined[0].start_bp, 1); // first window start
    assert_eq!(refined[0].end_bp, 40000); // last window end
}

#[test]
fn empty_segments_empty_result() {
    let posteriors: Vec<Vec<f64>> = vec![vec![0.5, 0.5]];
    let obs = observations_10kb(1, &[]);

    let refined = refine_ancestry_boundaries(&[], &posteriors, &obs, 0.5);
    assert!(refined.is_empty());
}

#[test]
fn empty_posteriors_fallback() {
    let params = make_params_2pop();
    let obs = observations_10kb(3, &[(0, 2, "A")]);
    let states = vec![0, 0, 0];
    let posteriors_for_extract = vec![
        vec![0.9, 0.1],
        vec![0.9, 0.1],
        vec![0.9, 0.1],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors_for_extract));

    // Pass empty posteriors → fallback to original positions
    let refined = refine_ancestry_boundaries(&segments, &[], &obs, 0.5);
    assert_eq!(refined.len(), 1);
    assert_eq!(refined[0].start_bp, segments[0].start);
    assert_eq!(refined[0].end_bp, segments[0].end);
}

#[test]
fn three_state_boundary_refinement() {
    // A A B B C C — two switch points
    let pops = vec![
        AncestralPopulation {
            name: "A".to_string(),
            haplotypes: vec!["rA".to_string()],
        },
        AncestralPopulation {
            name: "B".to_string(),
            haplotypes: vec!["rB".to_string()],
        },
        AncestralPopulation {
            name: "C".to_string(),
            haplotypes: vec!["rC".to_string()],
        },
    ];
    let params = AncestryHmmParams::new(pops, 0.001);

    let mut obs = Vec::new();
    for i in 0..6 {
        let start = i as u64 * 10000 + 1;
        let end = (i as u64 + 1) * 10000;
        let (sa, sb, sc) = match i {
            0 | 1 => (0.999, 0.990, 0.990),
            2 | 3 => (0.990, 0.999, 0.990),
            _ => (0.990, 0.990, 0.999),
        };
        obs.push(make_obs("chr1", start, end, vec![("rA", sa), ("rB", sb), ("rC", sc)]));
    }

    let states = vec![0, 0, 1, 1, 2, 2];
    let posteriors = vec![
        vec![0.90, 0.05, 0.05],
        vec![0.80, 0.15, 0.05],
        vec![0.10, 0.85, 0.05],
        vec![0.05, 0.80, 0.15],
        vec![0.05, 0.10, 0.85],
        vec![0.05, 0.05, 0.90],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));
    assert_eq!(segments.len(), 3);

    let refined = refine_ancestry_boundaries(&segments, &posteriors, &obs, 0.5);
    assert_eq!(refined.len(), 3);

    // All refined boundaries should maintain ordering
    assert!(refined[0].end_bp >= refined[0].start_bp);
    assert!(refined[1].end_bp >= refined[1].start_bp);
    assert!(refined[2].end_bp >= refined[2].start_bp);

    // First segment end should be refined (A→B transition)
    assert!(refined[0].end_bp > 15000, "end of A segment should be refined past center of window 1");
    assert!(refined[0].end_bp < 25000, "end of A segment should be before center of window 2");
}

#[test]
fn custom_crossover_ancestry() {
    let params = make_params_2pop();
    let obs = observations_10kb(4, &[(0, 1, "A"), (2, 3, "B")]);
    let states = vec![0, 0, 1, 1];
    let posteriors = vec![
        vec![0.95, 0.05],
        vec![0.80, 0.20],
        vec![0.15, 0.85],
        vec![0.05, 0.95],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));

    let refined_low = refine_ancestry_boundaries(&segments, &posteriors, &obs, 0.3);
    let refined_high = refine_ancestry_boundaries(&segments, &posteriors, &obs, 0.7);

    // Lower crossover → wider segments (boundaries pushed out)
    // For the first segment (state 0), end boundary:
    // With crossover=0.3: t = (0.80 - 0.30) / (0.80 - 0.15) ≈ 0.769 → farther right
    // With crossover=0.7: t = (0.80 - 0.70) / (0.80 - 0.15) ≈ 0.154 → closer to start
    assert!(refined_low[0].end_bp > refined_high[0].end_bp,
        "lower crossover should give later end: {} vs {}", refined_low[0].end_bp, refined_high[0].end_bp);
}

#[test]
fn refinement_end_always_gte_start() {
    let params = make_params_2pop();
    let obs = observations_10kb(3, &[(1, 1, "A")]);
    let states = vec![1, 0, 1]; // single window segment at idx 1

    let posteriors = vec![
        vec![0.40, 0.60],
        vec![0.60, 0.40],
        vec![0.40, 0.60],
    ];

    let segments = extract_ancestry_segments(&obs, &states, &params, Some(&posteriors));

    for seg in &segments {
        let refined = refine_ancestry_boundaries(&[seg.clone()], &posteriors, &obs, 0.5);
        assert!(refined[0].end_bp >= refined[0].start_bp,
            "end {} must be >= start {}", refined[0].end_bp, refined[0].start_bp);
    }
}
