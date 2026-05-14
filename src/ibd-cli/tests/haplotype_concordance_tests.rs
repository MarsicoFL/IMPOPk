//! Tests for haplotype_level_concordance() and related concordance edge cases.
//!
//! Covers:
//! - haplotype_level_concordance: both-empty, single-source, matching, mismatched,
//!   reversed sample order, multiple haplotype combos, sample-level metrics
//! - segment_ancestry_lod (via integration)
//! - load_population_samples / load_populations_from_dir (file I/O with temp dirs)

use impopk_ibd::concordance::*;

// =============================================================================
// haplotype_level_concordance tests
// =============================================================================

/// When both our and hapibd segment lists are empty, returns None.
#[test]
fn test_haplotype_concordance_both_empty() {
    let ours: Vec<(String, String, u64, u64)> = vec![];
    let hapibd: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result = haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1000000));
    assert!(result.is_none());
}

/// When only our segments exist, should return Some with hapibd combos empty.
#[test]
fn test_haplotype_concordance_only_ours() {
    let ours = vec![(
        "HG00280#1#CONTIG:0-100".to_string(),
        "HG00323#2#CONTIG:0-100".to_string(),
        100_000u64,
        500_000u64,
    )];
    let hapibd: Vec<(String, u8, String, u8, u64, u64)> = vec![];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 1);
    assert_eq!(r.n_hapibd_hap_combos, 0);
    // With no hapibd segments, Jaccard should be 0
    assert!(r.best_jaccard < 1e-9);
    assert_eq!(r.sample1, "HG00280");
    assert_eq!(r.sample2, "HG00323");
}

/// When only hapibd segments exist, should return Some with our combos empty.
#[test]
fn test_haplotype_concordance_only_hapibd() {
    let ours: Vec<(String, String, u64, u64)> = vec![];
    let hapibd = vec![(
        "HG00280".to_string(),
        1u8,
        "HG00323".to_string(),
        2u8,
        100_000u64,
        500_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 0);
    assert_eq!(r.n_hapibd_hap_combos, 1);
    assert!(r.best_jaccard < 1e-9);
}

/// Perfect overlap on same haplotype combination: Jaccard and F1 should be 1.0.
#[test]
fn test_haplotype_concordance_perfect_overlap() {
    let ours = vec![(
        "HG00280#1#CONTIG:0-100".to_string(),
        "HG00323#2#CONTIG:0-100".to_string(),
        100_000u64,
        500_000u64,
    )];
    let hapibd = vec![(
        "HG00280".to_string(),
        1u8,
        "HG00323".to_string(),
        2u8,
        100_000u64,
        500_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 1);
    assert_eq!(r.n_hapibd_hap_combos, 1);
    assert!(
        (r.best_jaccard - 1.0).abs() < 1e-6,
        "Expected Jaccard=1.0, got {}",
        r.best_jaccard
    );
    assert!(
        (r.best_f1 - 1.0).abs() < 1e-6,
        "Expected F1=1.0, got {}",
        r.best_f1
    );
    assert!(
        (r.sample_level_jaccard - 1.0).abs() < 1e-6,
        "Expected sample Jaccard=1.0, got {}",
        r.sample_level_jaccard
    );
    assert!(
        (r.sample_level_f1 - 1.0).abs() < 1e-6,
        "Expected sample F1=1.0, got {}",
        r.sample_level_f1
    );
}

/// Reversed sample order in the call: should still match correctly.
#[test]
fn test_haplotype_concordance_reversed_sample_order() {
    let ours = vec![(
        "HG00280#1#CONTIG:0-100".to_string(),
        "HG00323#2#CONTIG:0-100".to_string(),
        100_000u64,
        500_000u64,
    )];
    let hapibd = vec![(
        "HG00323".to_string(),
        2u8,
        "HG00280".to_string(),
        1u8,
        100_000u64,
        500_000u64,
    )];
    // Call with reversed sample order
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00323", "HG00280", (0, 1_000_000));
    let r = result.unwrap();
    assert!(r.n_our_hap_combos >= 1);
    assert!(r.n_hapibd_hap_combos >= 1);
    // Should still produce Jaccard=1.0 for perfect overlap
    assert!(
        (r.best_jaccard - 1.0).abs() < 1e-6,
        "Reversed order should still match, got Jaccard={}",
        r.best_jaccard
    );
}

/// Segments from different sample pairs are filtered out.
#[test]
fn test_haplotype_concordance_wrong_pair_filtered() {
    let ours = vec![(
        "HG99999#1#CONTIG:0-100".to_string(),
        "HG88888#2#CONTIG:0-100".to_string(),
        100_000u64,
        500_000u64,
    )];
    let hapibd = vec![(
        "HG77777".to_string(),
        1u8,
        "HG66666".to_string(),
        2u8,
        100_000u64,
        500_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    // Neither set has segments for the requested pair
    assert!(result.is_none());
}

/// Multiple haplotype combinations: (1,1), (1,2), (2,1), (2,2).
#[test]
fn test_haplotype_concordance_multiple_hap_combos() {
    let ours = vec![
        (
            "HG00280#1#C:0-100".to_string(),
            "HG00323#1#C:0-100".to_string(),
            100_000u64,
            300_000u64,
        ),
        (
            "HG00280#2#C:0-100".to_string(),
            "HG00323#2#C:0-100".to_string(),
            400_000u64,
            600_000u64,
        ),
    ];
    let hapibd = vec![
        (
            "HG00280".to_string(),
            1u8,
            "HG00323".to_string(),
            1u8,
            100_000u64,
            300_000u64,
        ),
        (
            "HG00280".to_string(),
            2u8,
            "HG00323".to_string(),
            2u8,
            400_000u64,
            600_000u64,
        ),
    ];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    assert_eq!(r.n_our_hap_combos, 2);
    assert_eq!(r.n_hapibd_hap_combos, 2);
    assert_eq!(r.per_hap_combo.len(), 2);
    // Both combos have perfect overlap
    assert!(
        (r.best_jaccard - 1.0).abs() < 1e-6,
        "Each combo matches perfectly, got Jaccard={}",
        r.best_jaccard
    );
    assert!(
        (r.best_f1 - 1.0).abs() < 1e-6,
        "Expected best F1=1.0, got {}",
        r.best_f1
    );
}

/// Partial overlap: our segment covers half the hapibd segment.
#[test]
fn test_haplotype_concordance_partial_overlap() {
    let ours = vec![(
        "HG00280#1#C:0-100".to_string(),
        "HG00323#2#C:0-100".to_string(),
        100_000u64,
        300_000u64,
    )];
    let hapibd = vec![(
        "HG00280".to_string(),
        1u8,
        "HG00323".to_string(),
        2u8,
        200_000u64,
        400_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    // Overlap = 100k, union = 300k, so Jaccard = 100k/300k ≈ 0.333
    // Actually for segments_jaccard with region, it uses the whole region as denominator
    assert!(r.best_jaccard > 0.0 && r.best_jaccard < 1.0);
    assert!(r.best_f1 > 0.0 && r.best_f1 < 1.0);
}

/// No overlap between our and hapibd segments (disjoint intervals).
#[test]
fn test_haplotype_concordance_no_overlap() {
    let ours = vec![(
        "HG00280#1#C:0-100".to_string(),
        "HG00323#2#C:0-100".to_string(),
        100_000u64,
        200_000u64,
    )];
    let hapibd = vec![(
        "HG00280".to_string(),
        1u8,
        "HG00323".to_string(),
        2u8,
        500_000u64,
        600_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    assert!(
        r.best_jaccard < 1e-9,
        "Disjoint segments should have Jaccard=0, got {}",
        r.best_jaccard
    );
}

/// Haplotype index extraction from our segments with no hash should default to 0.
#[test]
fn test_haplotype_concordance_no_hash_in_hap_id() {
    let ours = vec![(
        "HG00280".to_string(), // no # separator
        "HG00323".to_string(), // no # separator
        100_000u64,
        500_000u64,
    )];
    let hapibd = vec![(
        "HG00280".to_string(),
        0u8, // default hap index
        "HG00323".to_string(),
        0u8,
        100_000u64,
        500_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    // Both use hap index 0 (our defaults to 0 via unwrap_or)
    assert!(
        (r.best_jaccard - 1.0).abs() < 1e-6,
        "Default hap index 0 should match, got Jaccard={}",
        r.best_jaccard
    );
}

/// Per-hap-combo results contain correct n_ours and n_theirs counts.
#[test]
fn test_haplotype_concordance_segment_counts() {
    let ours = vec![
        (
            "HG00280#1#C:0-100".to_string(),
            "HG00323#2#C:0-100".to_string(),
            100_000u64,
            200_000u64,
        ),
        (
            "HG00280#1#C:0-100".to_string(),
            "HG00323#2#C:0-100".to_string(),
            300_000u64,
            400_000u64,
        ),
    ];
    let hapibd = vec![(
        "HG00280".to_string(),
        1u8,
        "HG00323".to_string(),
        2u8,
        100_000u64,
        400_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    // Find the combo (1,2)
    let combo = r.per_hap_combo.iter().find(|c| c.hap1 == 1 && c.hap2 == 2).unwrap();
    assert_eq!(combo.n_ours, 2, "We have 2 segments");
    assert_eq!(combo.n_theirs, 1, "hapibd has 1 segment");
}

/// sample_level metrics merge segments across haplotype combinations.
#[test]
fn test_haplotype_concordance_sample_level_merges() {
    // Two segments on different hap combos, both matching hapibd
    let ours = vec![
        (
            "HG00280#1#C:0-100".to_string(),
            "HG00323#1#C:0-100".to_string(),
            100_000u64,
            200_000u64,
        ),
        (
            "HG00280#2#C:0-100".to_string(),
            "HG00323#2#C:0-100".to_string(),
            300_000u64,
            400_000u64,
        ),
    ];
    let hapibd = vec![
        (
            "HG00280".to_string(),
            1u8,
            "HG00323".to_string(),
            1u8,
            100_000u64,
            200_000u64,
        ),
        (
            "HG00280".to_string(),
            2u8,
            "HG00323".to_string(),
            2u8,
            300_000u64,
            400_000u64,
        ),
    ];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    // Sample-level should reflect both segments matching
    assert!(r.sample_level_f1 > 0.5, "Sample F1 should be high with matching segments");
    assert!(
        (r.sample_level_jaccard - r.sample_level_jaccard).abs() < 1e-9,
        "Sample Jaccard should be finite"
    );
}

/// Haplotype misphasing: our segments on (1,2) but hapibd on (2,1).
#[test]
fn test_haplotype_concordance_misphased() {
    let ours = vec![(
        "HG00280#1#C:0-100".to_string(),
        "HG00323#2#C:0-100".to_string(),
        100_000u64,
        500_000u64,
    )];
    let hapibd = vec![(
        "HG00280".to_string(),
        2u8, // different hap index
        "HG00323".to_string(),
        1u8, // different hap index
        100_000u64,
        500_000u64,
    )];
    let result =
        haplotype_level_concordance(&ours, &hapibd, "HG00280", "HG00323", (0, 1_000_000));
    let r = result.unwrap();
    // Per-hap-combo Jaccard should be 0 (different combos)
    // But sample-level should be 1.0 since merged intervals overlap perfectly
    assert_eq!(r.per_hap_combo.len(), 2, "Two distinct hap combos: (1,2) and (2,1)");
    // best_jaccard is 0 because (1,2) vs nothing and (2,1) vs nothing
    assert!(
        r.best_jaccard < 1e-9,
        "Misphased hap combos have 0 per-combo overlap"
    );
    // But sample-level should merge and find overlap
    assert!(
        (r.sample_level_jaccard - 1.0).abs() < 1e-6,
        "Sample-level should still find perfect overlap for misphased, got {}",
        r.sample_level_jaccard
    );
}
