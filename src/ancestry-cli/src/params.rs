//! Cross-chromosome parameter transfer for ancestry HMM.
//!
//! Enables saving learned HMM parameters from one chromosome and loading them
//! for inference on another. Validated by T68: cross-chromosome transfer incurs
//! <1pp accuracy loss because HMM parameters (temperature, switch probability,
//! pairwise weight, emission context) are genome-wide properties that generalize
//! across chromosomes.
//!
//! ## Usage
//!
//! ```bash
//! # Learn parameters on chr12 (lots of data, good signal):
//! ancestry --similarity-file chr12.tsv --auto-configure --save-params chr12_params.json ...
//!
//! # Apply to chr1 without re-estimation:
//! ancestry --similarity-file chr1.tsv --load-params chr12_params.json ...
//! ```

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Learned HMM parameters that can be transferred across chromosomes.
///
/// Contains all parameters estimated during auto-configure and Baum-Welch
/// training, enabling parameter reuse without re-estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedParams {
    /// Schema version for forward compatibility.
    pub version: u32,

    /// Population names used during parameter estimation.
    /// Used to validate that loaded params match the current population set.
    pub population_names: Vec<String>,

    /// Softmax temperature (final value after all scalings).
    pub temperature: f64,

    /// Ancestry switch probability per window.
    pub switch_prob: f64,

    /// Weight for pairwise contrast emissions (0.0-1.0).
    pub pairwise_weight: f64,

    /// Emission context radius (number of neighboring windows for smoothing).
    pub emission_context: usize,

    /// Identity floor threshold.
    pub identity_floor: f64,

    /// Transition dampening factor for Baum-Welch (T52).
    pub transition_dampening: f64,

    /// Full K×K transition matrix (if Baum-Welch trained).
    /// None if transitions were not learned (symmetric from switch_prob).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitions: Option<Vec<Vec<f64>>>,

    /// Initial state distribution (if Baum-Welch full trained).
    /// None if uniform priors were used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_probs: Option<Vec<f64>>,

    /// Source chromosome/region where parameters were learned (informational).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl LearnedParams {
    /// Current schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new LearnedParams with the given values.
    pub fn new(
        population_names: Vec<String>,
        temperature: f64,
        switch_prob: f64,
        pairwise_weight: f64,
        emission_context: usize,
        identity_floor: f64,
        transition_dampening: f64,
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            population_names,
            temperature,
            switch_prob,
            pairwise_weight,
            emission_context,
            identity_floor,
            transition_dampening,
            transitions: None,
            initial_probs: None,
            source: None,
        }
    }

    /// Save parameters to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(std::io::Error::other)?;
        fs::write(path, json)
    }

    /// Load parameters from a JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read params file {:?}: {}", path, e))?;
        let params: Self = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse params file {:?}: {}", path, e))?;
        if params.version > Self::CURRENT_VERSION {
            return Err(format!(
                "params file version {} is newer than supported version {}",
                params.version,
                Self::CURRENT_VERSION
            ));
        }
        Ok(params)
    }

    /// Validate that loaded params are compatible with the given populations.
    /// Order-independent: checks that the same set of names exists.
    /// If valid, reorders internal state (transitions, initial_probs, population_names)
    /// to match the target ordering, so HMM state indices are consistent.
    pub fn validate_and_reorder(&mut self, target_names: &[String]) -> Result<(), String> {
        if self.population_names.len() != target_names.len() {
            return Err(format!(
                "loaded params have {} populations ({}) but current run has {} ({})",
                self.population_names.len(),
                self.population_names.join(", "),
                target_names.len(),
                target_names.join(", "),
            ));
        }
        // Build permutation: perm[new_idx] = old_idx
        // For each population in target order, find its index in saved order
        let mut perm = Vec::with_capacity(target_names.len());
        for target in target_names {
            match self.population_names.iter().position(|s| s == target) {
                Some(old_idx) => perm.push(old_idx),
                None => {
                    return Err(format!(
                        "population '{}' not found in loaded params (saved: {})",
                        target,
                        self.population_names.join(", "),
                    ));
                }
            }
        }
        // Check for duplicates in saved names (would cause perm issues)
        let is_identity = perm.iter().enumerate().all(|(i, &j)| i == j);
        if is_identity {
            return Ok(());
        }
        // Reorder population names
        self.population_names = perm.iter().map(|&i| self.population_names[i].clone()).collect();
        // Reorder transition matrix rows and columns
        if let Some(ref old_trans) = self.transitions {
            let k = old_trans.len();
            let mut new_trans = vec![vec![0.0; k]; k];
            for (new_i, &old_i) in perm.iter().enumerate() {
                for (new_j, &old_j) in perm.iter().enumerate() {
                    new_trans[new_i][new_j] = old_trans[old_i][old_j];
                }
            }
            self.transitions = Some(new_trans);
        }
        // Reorder initial probabilities
        if let Some(ref old_init) = self.initial_probs {
            self.initial_probs = Some(perm.iter().map(|&i| old_init[i]).collect());
        }
        eprintln!("  Reordered loaded params to match current population order: {:?}",
            self.population_names);
        Ok(())
    }

    /// Legacy validation (order-sensitive). Prefer validate_and_reorder.
    pub fn validate_populations(&self, population_names: &[String]) -> Result<(), String> {
        let mut clone = self.clone();
        clone.validate_and_reorder(population_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_roundtrip_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("params.json");

        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into(), "AMR".into()],
            0.0045,
            0.00032,
            0.12,
            15,
            0.9,
            0.5,
        );
        params.transitions = Some(vec![
            vec![0.9990, 0.0005, 0.0005],
            vec![0.0003, 0.9994, 0.0003],
            vec![0.0008, 0.0002, 0.9990],
        ]);
        params.initial_probs = Some(vec![0.4, 0.3, 0.3]);
        params.source = Some("chr12:0-133000000".into());

        params.save(&path).unwrap();

        let loaded = LearnedParams::load(&path).unwrap();
        assert_eq!(loaded.version, LearnedParams::CURRENT_VERSION);
        assert_eq!(loaded.population_names, vec!["EUR", "AFR", "AMR"]);
        assert!((loaded.temperature - 0.0045).abs() < 1e-10);
        assert!((loaded.switch_prob - 0.00032).abs() < 1e-10);
        assert!((loaded.pairwise_weight - 0.12).abs() < 1e-10);
        assert_eq!(loaded.emission_context, 15);
        assert!((loaded.identity_floor - 0.9).abs() < 1e-10);
        assert!((loaded.transition_dampening - 0.5).abs() < 1e-10);
        assert!(loaded.transitions.is_some());
        assert!(loaded.initial_probs.is_some());
        assert_eq!(loaded.source, Some("chr12:0-133000000".into()));
    }

    #[test]
    fn test_roundtrip_minimal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("params_min.json");

        let params = LearnedParams::new(
            vec!["POP_A".into(), "POP_B".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );

        params.save(&path).unwrap();

        let loaded = LearnedParams::load(&path).unwrap();
        assert_eq!(loaded.population_names, vec!["POP_A", "POP_B"]);
        assert!(loaded.transitions.is_none());
        assert!(loaded.initial_probs.is_none());
        assert!(loaded.source.is_none());
    }

    #[test]
    fn test_validate_populations_ok_same_order() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        assert!(params
            .validate_and_reorder(&["EUR".into(), "AFR".into()])
            .is_ok());
        assert_eq!(params.population_names, vec!["EUR", "AFR"]);
    }

    #[test]
    fn test_validate_populations_reorder_2pop() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        // Transition matrix: EUR→EUR=0.99, EUR→AFR=0.01, AFR→EUR=0.02, AFR→AFR=0.98
        params.transitions = Some(vec![
            vec![0.99, 0.01],
            vec![0.02, 0.98],
        ]);
        params.initial_probs = Some(vec![0.6, 0.4]);

        // Reorder to [AFR, EUR]
        assert!(params
            .validate_and_reorder(&["AFR".into(), "EUR".into()])
            .is_ok());
        assert_eq!(params.population_names, vec!["AFR", "EUR"]);
        // After reorder: AFR→AFR=0.98, AFR→EUR=0.02, EUR→AFR=0.01, EUR→EUR=0.99
        let trans = params.transitions.unwrap();
        assert!((trans[0][0] - 0.98).abs() < 1e-10);
        assert!((trans[0][1] - 0.02).abs() < 1e-10);
        assert!((trans[1][0] - 0.01).abs() < 1e-10);
        assert!((trans[1][1] - 0.99).abs() < 1e-10);
        let init = params.initial_probs.unwrap();
        assert!((init[0] - 0.4).abs() < 1e-10); // AFR was 0.4
        assert!((init[1] - 0.6).abs() < 1e-10); // EUR was 0.6
    }

    #[test]
    fn test_validate_populations_reorder_3pop() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into(), "AMR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        // Distinct values so we can verify correctness
        params.transitions = Some(vec![
            vec![0.90, 0.05, 0.05],  // EUR→EUR, EUR→AFR, EUR→AMR
            vec![0.03, 0.94, 0.03],  // AFR→EUR, AFR→AFR, AFR→AMR
            vec![0.08, 0.02, 0.90],  // AMR→EUR, AMR→AFR, AMR→AMR
        ]);
        params.initial_probs = Some(vec![0.5, 0.3, 0.2]);

        // Reorder to [AMR, EUR, AFR] — a full rotation
        assert!(params
            .validate_and_reorder(&["AMR".into(), "EUR".into(), "AFR".into()])
            .is_ok());
        assert_eq!(params.population_names, vec!["AMR", "EUR", "AFR"]);
        let trans = params.transitions.unwrap();
        // AMR→AMR=0.90, AMR→EUR=0.08, AMR→AFR=0.02
        assert!((trans[0][0] - 0.90).abs() < 1e-10);
        assert!((trans[0][1] - 0.08).abs() < 1e-10);
        assert!((trans[0][2] - 0.02).abs() < 1e-10);
        // EUR→AMR=0.05, EUR→EUR=0.90, EUR→AFR=0.05
        assert!((trans[1][0] - 0.05).abs() < 1e-10);
        assert!((trans[1][1] - 0.90).abs() < 1e-10);
        assert!((trans[1][2] - 0.05).abs() < 1e-10);
        // AFR→AMR=0.03, AFR→EUR=0.03, AFR→AFR=0.94
        assert!((trans[2][0] - 0.03).abs() < 1e-10);
        assert!((trans[2][1] - 0.03).abs() < 1e-10);
        assert!((trans[2][2] - 0.94).abs() < 1e-10);
        let init = params.initial_probs.unwrap();
        assert!((init[0] - 0.2).abs() < 1e-10); // AMR
        assert!((init[1] - 0.5).abs() < 1e-10); // EUR
        assert!((init[2] - 0.3).abs() < 1e-10); // AFR
    }

    #[test]
    fn test_validate_populations_reorder_no_transitions() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        // No transitions or initial_probs — should still work
        assert!(params
            .validate_and_reorder(&["AFR".into(), "EUR".into()])
            .is_ok());
        assert_eq!(params.population_names, vec!["AFR", "EUR"]);
        assert!(params.transitions.is_none());
        assert!(params.initial_probs.is_none());
    }

    #[test]
    fn test_validate_populations_count_mismatch() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        let result = params.validate_and_reorder(&["EUR".into(), "AFR".into(), "AMR".into()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("has 3"));
    }

    #[test]
    fn test_validate_populations_name_not_found() {
        let mut params = LearnedParams::new(
            vec!["EUR".into(), "AFR".into()],
            0.03,
            0.001,
            0.3,
            0,
            0.0,
            0.0,
        );
        let result = params.validate_and_reorder(&["EUR".into(), "EAS".into()]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found in loaded params"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = LearnedParams::load(&PathBuf::from("/nonexistent/params.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        fs::write(&path, "not valid json").unwrap();
        let result = LearnedParams::load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to parse"));
    }

    #[test]
    fn test_future_version_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.json");
        let json = r#"{
            "version": 999,
            "population_names": ["A", "B"],
            "temperature": 0.03,
            "switch_prob": 0.001,
            "pairwise_weight": 0.3,
            "emission_context": 0,
            "identity_floor": 0.0,
            "transition_dampening": 0.0
        }"#;
        fs::write(&path, json).unwrap();
        let result = LearnedParams::load(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("newer than supported"));
    }
}
