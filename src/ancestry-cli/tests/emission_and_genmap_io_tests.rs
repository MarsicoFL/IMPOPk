//! Tests for EmissionModel::TopK edge cases, AncestryGeneticMap::from_file error paths,
//! ancestry_confusion_matrix edge cases, and switch_point_accuracy_bp additional coverage.

use impopk_ancestry_cli::concordance::{ancestry_confusion_matrix, switch_point_accuracy_bp};
use impopk_ancestry_cli::hmm::{AncestralPopulation, AncestryGeneticMap, AncestryHmmParams, EmissionModel};

// ============================================================================
// EmissionModel::aggregate edge cases
// ============================================================================

// Note: aggregate() is private, so we test it indirectly via log_emission
// or by testing EmissionModel Display/FromStr which are already covered.
// However, we can test TopK(0) behavior via the public API if it surfaces.
// Since aggregate is private, test the model configuration and Display/FromStr.

/// TopK Display and FromStr round-trip for various k values.
#[test]
fn emission_model_topk_display_roundtrip() {
    for k in [1, 2, 3, 5, 10, 100] {
        let model = EmissionModel::TopK(k);
        let display = format!("{}", model);
        assert_eq!(display, format!("top{}", k));
        let parsed: EmissionModel = display.parse().unwrap();
        assert_eq!(parsed, model);
    }
}

#[test]
fn emission_model_all_variants_display() {
    assert_eq!(format!("{}", EmissionModel::Max), "max");
    assert_eq!(format!("{}", EmissionModel::Mean), "mean");
    assert_eq!(format!("{}", EmissionModel::Median), "median");
    assert_eq!(format!("{}", EmissionModel::TopK(3)), "top3");
}

#[test]
fn emission_model_from_str_all_variants() {
    assert_eq!("max".parse::<EmissionModel>().unwrap(), EmissionModel::Max);
    assert_eq!("mean".parse::<EmissionModel>().unwrap(), EmissionModel::Mean);
    assert_eq!("median".parse::<EmissionModel>().unwrap(), EmissionModel::Median);
    assert_eq!("top3".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(3));
    assert_eq!("top10".parse::<EmissionModel>().unwrap(), EmissionModel::TopK(10));
}

#[test]
fn emission_model_from_str_invalid() {
    assert!("invalid".parse::<EmissionModel>().is_err());
    assert!("top".parse::<EmissionModel>().is_err());
    assert!("topXYZ".parse::<EmissionModel>().is_err());
    assert!("".parse::<EmissionModel>().is_err());
}

/// Verify that AncestryHmmParams can be configured with various EmissionModels.
#[test]
fn ancestry_hmm_params_with_different_emission_models() {
    let pops = vec![
        AncestralPopulation { name: "AFR".into(), haplotypes: vec!["h1".into()] },
        AncestralPopulation { name: "EUR".into(), haplotypes: vec!["h2".into()] },
        AncestralPopulation { name: "AMR".into(), haplotypes: vec!["h3".into()] },
    ];

    for model in [
        EmissionModel::Max,
        EmissionModel::Mean,
        EmissionModel::Median,
        EmissionModel::TopK(1),
        EmissionModel::TopK(3),
        EmissionModel::TopK(5),
    ] {
        let mut params = AncestryHmmParams::new(pops.clone(), 0.001);
        params.set_emission_model(model);
        // Should not panic and emission model should be set
        assert_eq!(format!("{}", params.emission_model), format!("{}", model));
    }
}

// ============================================================================
// AncestryGeneticMap::from_file — error path tests
// ============================================================================

#[test]
fn ancestry_genetic_map_from_file_nonexistent() {
    let result = AncestryGeneticMap::from_file("/nonexistent/genetic_map.txt", "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.map");
    std::fs::write(&path, "").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_only_comments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("comments.map");
    std::fs::write(&path, "# comment\n# another comment\n").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_wrong_chrom() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chr2.map");
    std::fs::write(&path, "chr2\t1000\t1.0\t0.001\nchr2\t2000\t1.0\t0.002\n").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_valid() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid.map");
    std::fs::write(
        &path,
        "chr10\t1000\t1.0\t0.001\nchr10\t5000\t1.0\t0.005\nchr10\t10000\t1.0\t0.010\n",
    )
    .unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr10");
    assert!(result.is_ok());
}

#[test]
fn ancestry_genetic_map_chr_prefix_normalization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prefix.map");
    std::fs::write(&path, "chr10\t1000\t1.0\t0.001\nchr10\t2000\t1.0\t0.002\n").unwrap();
    // Query "10" should match "chr10"
    let result = AncestryGeneticMap::from_file(&path, "10");
    assert!(result.is_ok());
}

#[test]
fn ancestry_genetic_map_from_file_bad_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_pos.map");
    std::fs::write(&path, "chr1\tNOTNUM\t1.0\t0.001\n").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_bad_cm() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_cm.map");
    std::fs::write(&path, "chr1\t1000\t1.0\tBADCM\n").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "chr1");
    assert!(result.is_err());
}

#[test]
fn ancestry_genetic_map_from_file_3col_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("3col.map");
    std::fs::write(&path, "1000\t1.0\t0.001\n5000\t1.0\t0.005\n").unwrap();
    let result = AncestryGeneticMap::from_file(&path, "anything");
    assert!(result.is_ok());
}

#[test]
fn ancestry_genetic_map_interpolation_outside_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("narrow.map");
    std::fs::write(&path, "chr1\t50000\t1.0\t0.05\nchr1\t100000\t1.0\t0.10\n").unwrap();
    let map = AncestryGeneticMap::from_file(&path, "chr1").unwrap();
    // Before range
    let cm_before = map.interpolate_cm(0);
    assert!(cm_before.is_finite());
    // After range
    let cm_after = map.interpolate_cm(200_000);
    assert!(cm_after.is_finite());
    // Within range — should be between 0.05 and 0.10
    let cm_mid = map.interpolate_cm(75_000);
    assert!(cm_mid.is_finite());
    assert!(cm_mid > 0.04 && cm_mid < 0.11);
}

// ============================================================================
// ancestry_confusion_matrix — additional edge cases
// ============================================================================

/// Out-of-range indices on both truth and ours sides should be silently ignored.
#[test]
fn confusion_matrix_both_sides_oob() {
    let ours = vec![0, 5, 1, 99];
    let truth = vec![10, 1, 1, 0];
    let matrix = ancestry_confusion_matrix(&ours, &truth, 3);
    // Only index (truth=1, ours=1) is valid
    let total: u64 = matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 1);
    assert_eq!(matrix[1][1], 1);
}

/// With n_pops=1, only index 0 is valid.
#[test]
fn confusion_matrix_single_pop() {
    let ours = vec![0, 0, 0];
    let truth = vec![0, 0, 0];
    let matrix = ancestry_confusion_matrix(&ours, &truth, 1);
    assert_eq!(matrix.len(), 1);
    assert_eq!(matrix[0].len(), 1);
    assert_eq!(matrix[0][0], 3);
}

/// Mismatched lengths — should use min(ours.len(), truth.len()).
#[test]
fn confusion_matrix_mismatched_lengths() {
    let ours = vec![0, 1, 0, 1, 0]; // len=5
    let truth = vec![0, 1, 1]; // len=3
    let matrix = ancestry_confusion_matrix(&ours, &truth, 2);
    let total: u64 = matrix.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, 3); // Only first 3 compared
    assert_eq!(matrix[0][0], 1); // truth=0, ours=0
    assert_eq!(matrix[1][1], 1); // truth=1, ours=1
    assert_eq!(matrix[1][0], 1); // truth=1, ours=0
}

// ============================================================================
// switch_point_accuracy_bp — additional edge cases
// ============================================================================

/// When all our predictions are far from truth, none should be detected.
#[test]
fn switch_accuracy_bp_all_missed() {
    let our = vec![1_000_000u64];
    let truth = vec![100u64];
    // tolerance = 500bp, distance is 999900, so no match
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 500);
    assert!((frac - 0.0).abs() < 1e-10);
    assert!((dist - 999_900.0).abs() < 1e-5);
}

/// Multiple true switches, some detected and some not.
#[test]
fn switch_accuracy_bp_partial_detection() {
    let our = vec![1000u64, 5000];
    let truth = vec![1005u64, 3000, 4990];
    // tolerance = 20bp
    // truth=1005 → nearest our=1000, dist=5, detected
    // truth=3000 → nearest our=1000(dist=2000) or 5000(dist=2000), not detected
    // truth=4990 → nearest our=5000, dist=10, detected
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 20);
    assert!((frac - 2.0 / 3.0).abs() < 1e-10);
    assert!(dist.is_finite());
}

/// Single true switch with exact match.
#[test]
fn switch_accuracy_bp_single_exact() {
    let our = vec![42000u64];
    let truth = vec![42000u64];
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 0);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 0.0).abs() < 1e-10);
}

/// Multiple our predictions, single truth — tests nearest-neighbor matching.
#[test]
fn switch_accuracy_bp_many_predictions_one_truth() {
    let our = vec![100u64, 200, 300, 400, 500];
    let truth = vec![250u64];
    let (frac, dist) = switch_point_accuracy_bp(&our, &truth, 50);
    assert!((frac - 1.0).abs() < 1e-10);
    assert!((dist - 50.0).abs() < 1e-10);
}

/// Truth with no predictions should have zero detection rate.
#[test]
fn switch_accuracy_bp_no_predictions_multiple_truth() {
    let (frac, dist) = switch_point_accuracy_bp(&[], &[100, 200, 300], 1000);
    assert!((frac - 0.0).abs() < 1e-10);
    assert!(dist.is_infinite());
}
