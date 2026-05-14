//! Integration tests for cross-chromosome parameter transfer (--save-params / --load-params).
//!
//! Targets: LearnedParams serialization, validation, edge cases, and the B76 bug
//! (nondeterministic population ordering from multithreaded parsing).
//!
//! Cycle 62 — testing agent.

use impopk_ancestry_cli::LearnedParams;
use std::fs;

// ── Construction & defaults ──

#[test]
fn learned_params_version_is_current() {
    let p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    assert_eq!(p.version, LearnedParams::CURRENT_VERSION);
}

#[test]
fn learned_params_optional_fields_none_by_default() {
    let p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    assert!(p.transitions.is_none());
    assert!(p.initial_probs.is_none());
    assert!(p.source.is_none());
}

// ── Serialization roundtrip properties ──

#[test]
fn roundtrip_preserves_all_scalar_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scalars.json");

    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into(), "EAS".into(), "AMR".into(), "CSA".into()],
        0.0045, 0.00032, 0.12, 15, 0.9, 0.5,
    );
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();

    assert_eq!(loaded.version, p.version);
    assert_eq!(loaded.population_names, p.population_names);
    assert!((loaded.temperature - p.temperature).abs() < 1e-15);
    assert!((loaded.switch_prob - p.switch_prob).abs() < 1e-15);
    assert!((loaded.pairwise_weight - p.pairwise_weight).abs() < 1e-15);
    assert_eq!(loaded.emission_context, p.emission_context);
    assert!((loaded.identity_floor - p.identity_floor).abs() < 1e-15);
    assert!((loaded.transition_dampening - p.transition_dampening).abs() < 1e-15);
}

#[test]
fn roundtrip_preserves_transition_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trans.json");

    let mut p = LearnedParams::new(
        vec!["A".into(), "B".into(), "C".into()],
        0.03, 0.001, 0.3, 0, 0.0, 0.0,
    );
    p.transitions = Some(vec![
        vec![0.998, 0.001, 0.001],
        vec![0.0005, 0.999, 0.0005],
        vec![0.002, 0.003, 0.995],
    ]);
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();

    let trans = loaded.transitions.unwrap();
    assert_eq!(trans.len(), 3);
    for row in &trans {
        assert_eq!(row.len(), 3);
        let row_sum: f64 = row.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10, "row sum={}", row_sum);
    }
}

#[test]
fn roundtrip_preserves_initial_probs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("init.json");

    let mut p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p.initial_probs = Some(vec![0.6, 0.4]);
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();

    let init = loaded.initial_probs.unwrap();
    assert_eq!(init.len(), 2);
    assert!((init[0] - 0.6).abs() < 1e-15);
    assert!((init[1] - 0.4).abs() < 1e-15);
}

#[test]
fn roundtrip_preserves_source_string() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("src.json");

    let mut p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p.source = Some("chr12:0-133000000".into());
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();
    assert_eq!(loaded.source, Some("chr12:0-133000000".into()));
}

#[test]
fn optional_fields_skip_serialization_when_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minimal.json");

    let p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p.save(&path).unwrap();

    let json = fs::read_to_string(&path).unwrap();
    // serde skip_serializing_if means None fields are absent
    assert!(!json.contains("transitions"));
    assert!(!json.contains("initial_probs"));
    assert!(!json.contains("source"));
}

// ── Extreme parameter values ──

#[test]
fn roundtrip_extreme_temperature_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extreme_temp.json");

    let p = LearnedParams::new(
        vec!["A".into(), "B".into()],
        1e-10, // very small temperature
        1e-8,
        0.0,   // zero pairwise_weight
        0,     // zero emission_context
        0.0,   // zero identity_floor
        0.0,   // zero dampening
    );
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();
    assert!((loaded.temperature - 1e-10).abs() < 1e-20);
    assert!((loaded.switch_prob - 1e-8).abs() < 1e-20);
}

#[test]
fn roundtrip_large_population_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_pops.json");

    let names: Vec<String> = (0..50).map(|i| format!("POP_{:03}", i)).collect();
    let p = LearnedParams::new(names.clone(), 0.01, 0.001, 0.2, 5, 0.9, 0.5);
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();
    assert_eq!(loaded.population_names, names);
}

#[test]
fn roundtrip_large_transition_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_trans.json");

    let k = 10;
    let names: Vec<String> = (0..k).map(|i| format!("POP_{}", i)).collect();
    let mut p = LearnedParams::new(names, 0.01, 0.001, 0.2, 5, 0.9, 0.5);

    // K×K transition matrix with realistic values
    let mut trans = vec![vec![0.0; k]; k];
    for i in 0..k {
        let off_diag = 0.001 / (k - 1) as f64;
        for j in 0..k {
            trans[i][j] = if i == j { 1.0 - 0.001 } else { off_diag };
        }
    }
    p.transitions = Some(trans.clone());
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();

    let loaded_trans = loaded.transitions.unwrap();
    for i in 0..k {
        for j in 0..k {
            assert!(
                (loaded_trans[i][j] - trans[i][j]).abs() < 1e-12,
                "mismatch at [{},{}]", i, j
            );
        }
    }
}

// ── Population validation ──

#[test]
fn validate_populations_reorder_accepted() {
    // B76 fix: validate_and_reorder handles nondeterministic population ordering
    // by permuting transitions/initial_probs to match current run's order.
    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into(), "EAS".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );

    // Same names, different order → accepted and reordered
    let result = p.validate_and_reorder(&["AFR".into(), "EUR".into(), "EAS".into()]);
    assert!(result.is_ok());
    assert_eq!(p.population_names, vec!["AFR", "EUR", "EAS"]);
}

#[test]
fn validate_populations_empty_vs_nonempty() {
    let mut p = LearnedParams::new(
        vec!["A".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let result = p.validate_and_reorder(&[]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("has 0"));
}

#[test]
fn validate_populations_both_empty() {
    let mut p = LearnedParams::new(
        vec![],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    // Empty vs empty — matches
    assert!(p.validate_and_reorder(&[]).is_ok());
}

#[test]
fn validate_populations_subset_fails() {
    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into(), "EAS".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    // Subset of populations → count mismatch
    let result = p.validate_and_reorder(&["EUR".into(), "AFR".into()]);
    assert!(result.is_err());
}

#[test]
fn validate_populations_case_sensitive() {
    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    // Case mismatch → fails
    let result = p.validate_and_reorder(&["eur".into(), "afr".into()]);
    assert!(result.is_err());
}

#[test]
fn validate_populations_superset_fails() {
    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let result = p.validate_and_reorder(&["EUR".into(), "AFR".into(), "AMR".into()]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("has 3"));
}

// ── Error handling ──

#[test]
fn load_truncated_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.json");
    fs::write(&path, r#"{"version": 1, "population_names": ["A""#).unwrap();
    let result = LearnedParams::load(&path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("failed to parse"));
}

#[test]
fn load_missing_required_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.json");
    // Missing temperature field
    let json = r#"{
        "version": 1,
        "population_names": ["A", "B"],
        "switch_prob": 0.001,
        "pairwise_weight": 0.2,
        "emission_context": 5,
        "identity_floor": 0.9,
        "transition_dampening": 0.5
    }"#;
    fs::write(&path, json).unwrap();
    let result = LearnedParams::load(&path);
    assert!(result.is_err());
}

#[test]
fn load_wrong_type_for_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrong_type.json");
    let json = r#"{
        "version": 1,
        "population_names": ["A", "B"],
        "temperature": "not_a_number",
        "switch_prob": 0.001,
        "pairwise_weight": 0.2,
        "emission_context": 5,
        "identity_floor": 0.9,
        "transition_dampening": 0.5
    }"#;
    fs::write(&path, json).unwrap();
    let result = LearnedParams::load(&path);
    assert!(result.is_err());
}

#[test]
fn load_version_0_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v0.json");
    let json = r#"{
        "version": 0,
        "population_names": ["A", "B"],
        "temperature": 0.01,
        "switch_prob": 0.001,
        "pairwise_weight": 0.2,
        "emission_context": 5,
        "identity_floor": 0.9,
        "transition_dampening": 0.5
    }"#;
    fs::write(&path, json).unwrap();
    let result = LearnedParams::load(&path);
    assert!(result.is_ok());
}

#[test]
fn save_to_nonexistent_directory_fails() {
    let p = LearnedParams::new(
        vec!["A".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let result = p.save(std::path::Path::new("/nonexistent/dir/params.json"));
    assert!(result.is_err());
}

// ── JSON forward compatibility ──

#[test]
fn extra_json_fields_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extra.json");
    // Future version might add fields; serde should ignore unknown fields
    let json = r#"{
        "version": 1,
        "population_names": ["A", "B"],
        "temperature": 0.01,
        "switch_prob": 0.001,
        "pairwise_weight": 0.2,
        "emission_context": 5,
        "identity_floor": 0.9,
        "transition_dampening": 0.5,
        "unknown_future_field": 42,
        "another_unknown": [1, 2, 3]
    }"#;
    fs::write(&path, json).unwrap();
    let result = LearnedParams::load(&path);
    // serde default behavior: deny_unknown_fields is NOT set, so this should work
    assert!(result.is_ok(), "extra fields should be ignored: {:?}", result.err());
}

#[test]
fn json_is_human_readable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readable.json");

    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p.save(&path).unwrap();

    let content = fs::read_to_string(&path).unwrap();
    // Pretty-printed JSON should have newlines
    assert!(content.contains('\n'));
    // Should contain field names for human readability
    assert!(content.contains("\"temperature\""));
    assert!(content.contains("\"population_names\""));
}

// ── B76 fix: population ordering reordering ──

#[test]
fn population_ordering_reordered_for_transfer() {
    // B76 fix: validate_and_reorder handles nondeterministic population ordering
    // by permuting transitions/initial_probs to match current run's order.

    // All rotations should be accepted and reordered correctly
    let rotations: Vec<Vec<String>> = vec![
        vec!["AMR".into(), "CSA".into(), "EAS".into(), "EUR".into(), "AFR".into()],
        vec!["EUR".into(), "AFR".into(), "AMR".into(), "CSA".into(), "EAS".into()],
        vec!["EAS".into(), "EUR".into(), "AFR".into(), "AMR".into(), "CSA".into()],
    ];

    for rotation in &rotations {
        let mut saved = LearnedParams::new(
            vec!["AFR".into(), "AMR".into(), "CSA".into(), "EAS".into(), "EUR".into()],
            0.01, 0.001, 0.2, 5, 0.9, 0.5,
        );
        assert!(
            saved.validate_and_reorder(rotation).is_ok(),
            "rotation {:?} should be accepted and reordered", rotation
        );
        assert_eq!(&saved.population_names, rotation);
    }

    // Exact match should also pass (trivial case)
    let mut saved = LearnedParams::new(
        vec!["AFR".into(), "AMR".into(), "CSA".into(), "EAS".into(), "EUR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let exact = vec!["AFR".into(), "AMR".into(), "CSA".into(), "EAS".into(), "EUR".into()];
    assert!(saved.validate_and_reorder(&exact).is_ok());
}

#[test]
fn transitions_carry_population_identity() {
    // Verify that transition matrix rows/cols are tied to population indices.
    // If populations get reordered, the entire matrix becomes wrong.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trans_identity.json");

    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    // Asymmetric: EUR→AFR rate (0.002) ≠ AFR→EUR rate (0.005)
    p.transitions = Some(vec![
        vec![0.998, 0.002],
        vec![0.005, 0.995],
    ]);
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();

    let t = loaded.transitions.unwrap();
    // EUR→AFR (position [0][1]) should be 0.002
    assert!((t[0][1] - 0.002).abs() < 1e-15);
    // AFR→EUR (position [1][0]) should be 0.005
    assert!((t[1][0] - 0.005).abs() < 1e-15);
}

// ── Unicode / special characters in population names ──

#[test]
fn unicode_population_names_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unicode.json");

    let names = vec![
        "São_Paulo".into(),
        "Großglockner".into(),
        "日本".into(),
    ];
    let p = LearnedParams::new(names.clone(), 0.01, 0.001, 0.2, 5, 0.9, 0.5);
    p.save(&path).unwrap();
    let loaded = LearnedParams::load(&path).unwrap();
    assert_eq!(loaded.population_names, names);
}

// ── Overwrite safety ──

#[test]
fn save_overwrites_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("overwrite.json");

    let p1 = LearnedParams::new(
        vec!["A".into(), "B".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p1.save(&path).unwrap();

    let p2 = LearnedParams::new(
        vec!["X".into(), "Y".into(), "Z".into()],
        0.05, 0.002, 0.4, 10, 0.8, 0.3,
    );
    p2.save(&path).unwrap();

    let loaded = LearnedParams::load(&path).unwrap();
    assert_eq!(loaded.population_names, vec!["X", "Y", "Z"]);
    assert!((loaded.temperature - 0.05).abs() < 1e-15);
}

// ── validate_populations (legacy wrapper) ──

#[test]
fn validate_populations_wrapper_same_order_ok() {
    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into(), "AMR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    assert!(p.validate_populations(&["EUR".into(), "AFR".into(), "AMR".into()]).is_ok());
}

#[test]
fn validate_populations_wrapper_reordered_ok() {
    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into(), "AMR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    // Different order should still validate successfully
    assert!(p.validate_populations(&["AMR".into(), "EUR".into(), "AFR".into()]).is_ok());
}

#[test]
fn validate_populations_wrapper_count_mismatch() {
    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let result = p.validate_populations(&["EUR".into(), "AFR".into(), "AMR".into()]);
    assert!(result.is_err());
}

#[test]
fn validate_populations_wrapper_name_not_found() {
    let p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    let result = p.validate_populations(&["EUR".into(), "EAS".into()]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn validate_populations_wrapper_does_not_mutate_original() {
    let mut p = LearnedParams::new(
        vec!["EUR".into(), "AFR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    p.transitions = Some(vec![
        vec![0.99, 0.01],
        vec![0.02, 0.98],
    ]);
    p.initial_probs = Some(vec![0.6, 0.4]);

    // Call validate_populations (immutable wrapper)
    assert!(p.validate_populations(&["AFR".into(), "EUR".into()]).is_ok());

    // Original should NOT be reordered (it works on a clone)
    assert_eq!(p.population_names, vec!["EUR", "AFR"]);
    let trans = p.transitions.as_ref().unwrap();
    assert!((trans[0][0] - 0.99).abs() < 1e-10); // EUR→EUR still first
    let init = p.initial_probs.as_ref().unwrap();
    assert!((init[0] - 0.6).abs() < 1e-10); // EUR still first
}

#[test]
fn validate_populations_wrapper_empty_pops() {
    let p = LearnedParams::new(
        vec![],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    assert!(p.validate_populations(&[]).is_ok());
}

#[test]
fn validate_populations_wrapper_single_pop() {
    let p = LearnedParams::new(
        vec!["EUR".into()],
        0.01, 0.001, 0.2, 5, 0.9, 0.5,
    );
    assert!(p.validate_populations(&["EUR".into()]).is_ok());
}
