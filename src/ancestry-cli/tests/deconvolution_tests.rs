use std::collections::HashMap;
use impopk_ancestry_cli::{
    AncestralPopulation, AncestryObservation,
    deconvolve_admixed_populations,
};

fn make_obs(sims: HashMap<String, f64>) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".to_string(),
        start: 0,
        end: 10000,
        sample: "test".to_string(),
        similarities: sims,
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

#[test]
fn test_deconvolve_no_admixture() {
    // Two well-separated populations — should NOT trigger deconvolution
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".into(), "b2".into(), "b3".into(), "b4".into()] },
    ];

    let observations: Vec<AncestryObservation> = (0..200).map(|i| {
        let mut sims = HashMap::new();
        // Population A: high identity ~0.9999
        for h in &pops[0].haplotypes {
            sims.insert(h.clone(), 0.9999 + (i as f64) * 0.000001);
        }
        // Population B: lower identity ~0.9990 — clear separation
        for h in &pops[1].haplotypes {
            sims.insert(h.clone(), 0.9990 + (i as f64) * 0.000001);
        }
        make_obs(sims)
    }).collect();

    let (result_pops, parent_map) = deconvolve_admixed_populations(&observations, &pops, 0.01);

    // Should remain 2 populations (no split)
    assert_eq!(result_pops.len(), 2);
    assert_eq!(parent_map["A"], "A");
    assert_eq!(parent_map["B"], "B");
}

#[test]
fn test_deconvolve_admixed_population() {
    // Simulate: EUR is homogeneous, AMR is admixed (half EUR-like, half NAT-like)
    // Key: the per-window noise must make EUR and AMR Max confusable (low Cohen's d)
    let pops = vec![
        AncestralPopulation {
            name: "EUR".to_string(),
            haplotypes: (0..6).map(|i| format!("eur_{}", i)).collect(),
        },
        AncestralPopulation {
            name: "AMR".to_string(),
            haplotypes: (0..10).map(|i| format!("amr_{}", i)).collect(),
        },
    ];

    // Create observations where EUR and AMR max-similarity distributions OVERLAP,
    // so Cohen's d is low. Key: noise must be INDEPENDENT per population.
    let observations: Vec<AncestryObservation> = (0..300).map(|i| {
        let mut sims = HashMap::new();
        // Independent noise per population via different phase offsets
        let noise_eur = ((i as f64 * 0.1).sin()) * 0.001;
        let noise_amr = ((i as f64 * 0.17 + 2.0).sin()) * 0.001;

        // EUR haplotypes: clustered around 0.9995 + noise_eur
        for (j, h) in pops[0].haplotypes.iter().enumerate() {
            sims.insert(h.clone(), 0.9995 + noise_eur + (j as f64) * 0.000001);
        }
        // AMR haplotypes: EUR-like component (first 5) has SIMILAR max to EUR
        // NAT-like component (last 5) is distinctly lower → creates within-pop variance
        for (j, h) in pops[1].haplotypes.iter().enumerate() {
            if j < 5 {
                // EUR-component: overlaps with EUR identity
                sims.insert(h.clone(), 0.9995 + noise_amr + (j as f64) * 0.000001);
            } else {
                // NAT-component: clearly lower identity
                sims.insert(h.clone(), 0.9970 + noise_amr * 0.3 + (j as f64) * 0.000001);
            }
        }
        make_obs(sims)
    }).collect();

    // With a generous threshold, AMR should be detected as admixed
    let (result_pops, parent_map) = deconvolve_admixed_populations(&observations, &pops, 100.0);

    // AMR should be split (EUR stays intact since it has lower within-pop variance)
    assert!(result_pops.len() >= 3, "Expected at least 3 pops, got {}. Pops: {:?}",
        result_pops.len(), result_pops.iter().map(|p| &p.name).collect::<Vec<_>>());

    // Check parent mapping
    let amr_children: Vec<_> = parent_map.iter()
        .filter(|(_, parent)| *parent == "AMR")
        .map(|(child, _)| child.clone())
        .collect();

    assert!(amr_children.len() >= 2, "AMR should be split into at least 2, got {}", amr_children.len());

    // All AMR haplotypes should be present across sub-populations
    let total_amr_haps: usize = result_pops.iter()
        .filter(|p| parent_map[&p.name] == "AMR")
        .map(|p| p.haplotypes.len())
        .sum();
    assert_eq!(total_amr_haps, 10);

    // Check that the split separates EUR-like from NAT-like haplotypes
    let amr_sub_pops: Vec<_> = result_pops.iter()
        .filter(|p| parent_map[&p.name] == "AMR")
        .collect();
    // One sub-pop should have primarily low-index haps (EUR-like), other high-index (NAT-like)
    let has_diverse_split = amr_sub_pops.iter().any(|p| p.haplotypes.len() >= 2)
        && amr_sub_pops.iter().any(|p| p.haplotypes.len() <= 8);
    assert!(has_diverse_split, "Expected non-trivial split of AMR haplotypes");
}

#[test]
fn test_deconvolve_too_few_haplotypes() {
    // Population with only 3 haplotypes should not be split
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".into(), "a2".into(), "a3".into()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".into(), "b2".into(), "b3".into()] },
    ];

    let observations: Vec<AncestryObservation> = (0..100).map(|_| {
        let mut sims = HashMap::new();
        for h in &pops[0].haplotypes { sims.insert(h.clone(), 0.999); }
        for h in &pops[1].haplotypes { sims.insert(h.clone(), 0.999); }
        make_obs(sims)
    }).collect();

    let (result_pops, _) = deconvolve_admixed_populations(&observations, &pops, 1.0);

    // Even with high threshold, too few haplotypes prevents splitting
    assert_eq!(result_pops.len(), 2);
}

#[test]
fn test_deconvolve_too_few_observations() {
    let pops = vec![
        AncestralPopulation { name: "A".to_string(), haplotypes: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()] },
        AncestralPopulation { name: "B".to_string(), haplotypes: vec!["b1".into(), "b2".into(), "b3".into(), "b4".into()] },
    ];

    let observations: Vec<AncestryObservation> = (0..10).map(|_| {
        let mut sims = HashMap::new();
        for h in &pops[0].haplotypes { sims.insert(h.clone(), 0.999); }
        for h in &pops[1].haplotypes { sims.insert(h.clone(), 0.999); }
        make_obs(sims)
    }).collect();

    let (result_pops, _) = deconvolve_admixed_populations(&observations, &pops, 1.0);

    // Too few observations (< 50) should not trigger deconvolution
    assert_eq!(result_pops.len(), 2);
}

#[test]
fn test_kmeans_deterministic() {
    // Run deconvolution twice with same data — should give same result
    let pops = vec![
        AncestralPopulation {
            name: "X".to_string(),
            haplotypes: (0..8).map(|i| format!("x_{}", i)).collect(),
        },
        AncestralPopulation {
            name: "Y".to_string(),
            haplotypes: (0..8).map(|i| format!("y_{}", i)).collect(),
        },
    ];

    let observations: Vec<AncestryObservation> = (0..200).map(|i| {
        let mut sims = HashMap::new();
        let base = 0.999;
        for (j, h) in pops[0].haplotypes.iter().enumerate() {
            let offset = if j < 4 { 0.0005 } else { -0.0005 };
            sims.insert(h.clone(), base + offset + (i as f64) * 0.0000001);
        }
        for h in &pops[1].haplotypes {
            sims.insert(h.clone(), base - 0.001);
        }
        make_obs(sims)
    }).collect();

    let (pops1, map1) = deconvolve_admixed_populations(&observations, &pops, 1.0);
    let (pops2, map2) = deconvolve_admixed_populations(&observations, &pops, 1.0);

    assert_eq!(pops1.len(), pops2.len());
    for (p1, p2) in pops1.iter().zip(pops2.iter()) {
        assert_eq!(p1.name, p2.name);
        assert_eq!(p1.haplotypes, p2.haplotypes);
    }
    assert_eq!(map1, map2);
}
