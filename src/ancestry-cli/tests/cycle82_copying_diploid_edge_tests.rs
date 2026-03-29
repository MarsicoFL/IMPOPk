//! Edge case tests for estimate_copying_params, write_diploid_gcta_grm,
//! and apply_posterior_feedback — cycle 82, testing agent.
//!
//! Targets gap coverage for:
//! - estimate_copying_params: single observation, all-identical sims,
//!   single haplotype per pop, NaN sims, high switch rate, constant best hap
//! - write_diploid_gcta_grm integration: NaN in accumulator, triploid individual,
//!   3+ individuals scaling, missing cross-individual data, centering row-sum zero
//! - apply_posterior_feedback: mismatched dimensions, NaN posteriors,
//!   negative lambda, large lambda saturation, single state

use hprc_ancestry_cli::{
    AncestralPopulation, AncestryObservation,
    apply_posterior_feedback,
    estimate_copying_params,
    parse_similarity_for_egrm, write_diploid_gcta_grm,
};
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================
// Helpers
// ============================================================

fn make_pops(n_haps_per_pop: &[usize]) -> Vec<AncestralPopulation> {
    let names = ["EUR", "AFR", "EAS", "AMR", "CSA"];
    n_haps_per_pop
        .iter()
        .enumerate()
        .map(|(i, &nh)| AncestralPopulation {
            name: names[i % names.len()].into(),
            haplotypes: (0..nh).map(|h| format!("{}_{}", names[i % names.len()], h)).collect(),
        })
        .collect()
}

fn make_obs_map(hap_sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".into(),
        start: 0,
        end: 10000,
        sample: "TEST#1".into(),
        similarities: hap_sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn make_obs_at(start: u64, hap_sims: &[(&str, f64)]) -> AncestryObservation {
    AncestryObservation {
        chrom: "chr1".into(),
        start,
        end: start + 10000,
        sample: "TEST#1".into(),
        similarities: hap_sims.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        coverage_ratios: None,
        haplotype_consistency_bonus: None,
    }
}

fn create_sim_file(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    write!(f, "{}", content).unwrap();
    f.flush().unwrap();
    f
}

fn read_f32(data: &[u8], index: usize) -> f32 {
    f32::from_le_bytes(data[index * 4..(index + 1) * 4].try_into().unwrap())
}

// ============================================================
// estimate_copying_params edge cases
// ============================================================

#[test]
fn estimate_copying_params_single_observation() {
    let pops = make_pops(&[2, 2]);
    let obs = vec![make_obs_map(&[
        ("EUR_0", 0.998),
        ("EUR_1", 0.997),
        ("AFR_0", 0.993),
        ("AFR_1", 0.992),
    ])];

    let (temp, switch, default) = estimate_copying_params(&obs, &pops);
    // Single observation → no consecutive pairs → fallback switch_rate
    assert!((switch - 0.005).abs() < 1e-10, "single obs should give default switch_rate, got {}", switch);
    assert!(temp > 0.0, "temperature should be positive");
    assert!(default > 0.0 && default < 1.0, "default_sim should be in (0,1)");
}

#[test]
fn estimate_copying_params_all_identical_sims() {
    let pops = make_pops(&[2, 2]);
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.995),
            ("EUR_1", 0.995),
            ("AFR_0", 0.995),
            ("AFR_1", 0.995),
        ]))
        .collect();

    let (temp, _switch, default) = estimate_copying_params(&obs, &pops);
    // All identical → max-min=0 for every window → no diffs → fallback temperature
    assert!((temp - 0.003).abs() < 1e-10, "all identical should give default temp, got {}", temp);
    assert!(default > 0.0, "default_sim should be positive");
}

#[test]
fn estimate_copying_params_empty_populations() {
    let pops: Vec<AncestralPopulation> = vec![];
    let obs = vec![make_obs_map(&[("H1", 0.99)])];
    let (temp, switch, default) = estimate_copying_params(&obs, &pops);
    // Empty pops → fallback defaults
    assert!((temp - 0.003).abs() < 1e-10);
    assert!((switch - 0.005).abs() < 1e-10);
    assert!((default - 0.99).abs() < 1e-10);
}

#[test]
fn estimate_copying_params_single_hap_per_pop() {
    let pops = make_pops(&[1, 1]);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.998),
            ("AFR_0", 0.993),
        ]))
        .collect();

    let (temp, switch, _default) = estimate_copying_params(&obs, &pops);
    assert!(temp > 0.0, "should compute valid temperature");
    // Same best hap every window → zero raw switches → clamped to minimum
    assert!((switch - 0.001).abs() < 1e-10,
        "constant best hap should give minimum switch_rate, got {}", switch);
}

#[test]
fn estimate_copying_params_high_switch_rate() {
    let pops = make_pops(&[2, 2]);
    // Alternate best haplotype every window
    let obs: Vec<_> = (0..20)
        .map(|i| {
            if i % 2 == 0 {
                make_obs_at(i * 10000, &[
                    ("EUR_0", 0.999),
                    ("EUR_1", 0.990),
                    ("AFR_0", 0.991),
                    ("AFR_1", 0.990),
                ])
            } else {
                make_obs_at(i * 10000, &[
                    ("EUR_0", 0.990),
                    ("EUR_1", 0.990),
                    ("AFR_0", 0.999),
                    ("AFR_1", 0.990),
                ])
            }
        })
        .collect();

    let (_temp, switch, _default) = estimate_copying_params(&obs, &pops);
    // Every consecutive pair switches → raw_rate ≈ 1.0, clamped at 0.05
    assert!((switch - 0.05).abs() < 1e-10,
        "frequent switches should be clamped to max, got {}", switch);
}

#[test]
fn estimate_copying_params_nan_sims_filtered() {
    let pops = make_pops(&[2, 2]);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", f64::NAN),
            ("EUR_1", 0.995),
            ("AFR_0", 0.993),
            ("AFR_1", f64::NAN),
        ]))
        .collect();

    let (temp, switch, default) = estimate_copying_params(&obs, &pops);
    // NaN values are > 0 and < 1 both false, so they should be filtered from all_sims
    // The function should not panic
    assert!(temp > 0.0 || (temp - 0.003).abs() < 1e-10);
    assert!(switch > 0.0);
    assert!(default > 0.0);
}

#[test]
fn estimate_copying_params_missing_haps_in_obs() {
    // Population declares haplotypes not present in observations
    let pops = make_pops(&[3, 3]);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.998),
            // EUR_1, EUR_2 missing
            ("AFR_0", 0.993),
            // AFR_1, AFR_2 missing
        ]))
        .collect();

    let (temp, _switch, default) = estimate_copying_params(&obs, &pops);
    // Should not panic despite missing haplotypes
    assert!(temp > 0.0 || (temp - 0.003).abs() < 1e-10);
    assert!(default > 0.0);
}

#[test]
fn estimate_copying_params_two_observations_no_switch() {
    let pops = make_pops(&[2, 2]);
    let obs = vec![
        make_obs_at(0, &[
            ("EUR_0", 0.999),
            ("EUR_1", 0.997),
            ("AFR_0", 0.990),
            ("AFR_1", 0.989),
        ]),
        make_obs_at(10000, &[
            ("EUR_0", 0.998),
            ("EUR_1", 0.996),
            ("AFR_0", 0.991),
            ("AFR_1", 0.990),
        ]),
    ];

    let (_temp, switch, _default) = estimate_copying_params(&obs, &pops);
    // Same best hap in both windows → 0 switches / 1 comparison → clamped
    assert!((switch - 0.001).abs() < 1e-10,
        "no switches should give min switch_rate, got {}", switch);
}

#[test]
fn estimate_copying_params_temperature_clamped_below() {
    // Very tight spread → temperature clamps to 0.0005
    let pops = make_pops(&[2, 2]);
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.999999),
            ("EUR_1", 0.999998),
            ("AFR_0", 0.999997),
            ("AFR_1", 0.999996),
        ]))
        .collect();

    let (temp, _, _) = estimate_copying_params(&obs, &pops);
    assert!((temp - 0.0005).abs() < 1e-10,
        "very tight spread should clamp temp to 0.0005, got {}", temp);
}

#[test]
fn estimate_copying_params_temperature_clamped_above() {
    // Very wide spread → temperature clamps to 0.05
    let pops = make_pops(&[2, 2]);
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.99),
            ("EUR_1", 0.50),
            ("AFR_0", 0.30),
            ("AFR_1", 0.10),
        ]))
        .collect();

    let (temp, _, _) = estimate_copying_params(&obs, &pops);
    assert!((temp - 0.05).abs() < 1e-10,
        "wide spread should clamp temp to 0.05, got {}", temp);
}

// ============================================================
// write_diploid_gcta_grm integration edge cases
// ============================================================

#[test]
fn diploid_grm_triploid_individual() {
    // Individual with 3 haplotypes (A#1, A#2, A#3)
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tA#2#c\t0.995\n\
                   chr1\t0\t10000\tA#1#c\tA#3#c\t0.993\n\
                   chr1\t0\t10000\tA#2#c\tA#3#c\t0.991\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.980\n\
                   chr1\t0\t10000\tA#2#c\tB#1#c\t0.978\n\
                   chr1\t0\t10000\tA#3#c\tB#1#c\t0.976\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("triploid");
    let prefix_str = prefix.to_str().unwrap();

    // Should not panic — A gets 3 haplotypes, B gets 1
    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
    let lines: Vec<&str> = id_content.trim().lines().collect();
    assert_eq!(lines.len(), 2, "should have 2 individuals (A and B)");

    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    assert_eq!(bin_data.len(), 3 * 4, "2 individuals → 3 lower-triangle entries");

    // Verify values are finite and reasonable
    for i in 0..3 {
        let v = read_f32(&bin_data, i);
        assert!(v.is_finite(), "entry {} should be finite, got {}", i, v);
        assert!(v >= 0.0 && v <= 1.0, "entry {} should be in [0,1], got {}", i, v);
    }
}

#[test]
fn diploid_grm_four_individuals_scaling() {
    // 4 diploid individuals (8 haplotypes)
    let mut content = String::from(
        "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n",
    );
    let indivs = ["A", "B", "C", "D"];
    for i in 0..4u32 {
        // Within-individual
        content.push_str(&format!(
            "chr1\t0\t10000\t{}#1#c\t{}#2#c\t{}\n",
            indivs[i as usize], indivs[i as usize],
            0.99 - 0.001 * i as f64
        ));
        // Cross-individual
        for j in (i + 1)..4 {
            for h1 in 1..=2u32 {
                for h2 in 1..=2u32 {
                    content.push_str(&format!(
                        "chr1\t0\t10000\t{}#{}#c\t{}#{}#c\t{}\n",
                        indivs[i as usize], h1,
                        indivs[j as usize], h2,
                        0.98 - 0.002 * (i * 4 + j) as f64
                    ));
                }
            }
        }
    }

    let f = create_sim_file(&content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("four_indiv");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let n = 4usize;
    let entries = n * (n + 1) / 2; // 10
    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    assert_eq!(bin_data.len(), entries * 4);

    let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
    assert_eq!(id_content.trim().lines().count(), n);
}

#[test]
fn diploid_grm_missing_cross_individual_data() {
    // Some cross-individual pairs have no data
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tA#2#c\t0.995\n\
                   chr1\t0\t10000\tB#1#c\tB#2#c\t0.993\n\
                   chr1\t0\t10000\tC#1#c\tC#2#c\t0.991\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.980\n";
    // Only A-B has cross data; A-C and B-C have no cross data

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("missing");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    let n = 3usize;
    let entries = n * (n + 1) / 2; // 6
    assert_eq!(bin_data.len(), entries * 4);

    // Pairs with no cross-data should have 0.0
    // A-C: pos(2,0) = 2*3/2+0 = 3
    let v_ac = read_f32(&bin_data, 3);
    assert!((v_ac - 0.0).abs() < 1e-5, "missing pair A-C should be 0.0, got {}", v_ac);
}

#[test]
fn diploid_grm_centering_with_3_individuals_rows_sum_zero() {
    // Fully populated 3-individual diploid GRM, centered
    let mut content = String::from(
        "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n",
    );
    let indivs = ["X", "Y", "Z"];
    for i in 0..3u32 {
        content.push_str(&format!(
            "chr1\t0\t10000\t{}#1#c\t{}#2#c\t{}\n",
            indivs[i as usize], indivs[i as usize],
            0.992 - 0.001 * i as f64
        ));
        for j in (i + 1)..3 {
            for h1 in 1..=2u32 {
                for h2 in 1..=2u32 {
                    content.push_str(&format!(
                        "chr1\t0\t10000\t{}#{}#c\t{}#{}#c\t{}\n",
                        indivs[i as usize], h1,
                        indivs[j as usize], h2,
                        0.97 - 0.003 * (i + j) as f64
                    ));
                }
            }
        }
    }

    let f = create_sim_file(&content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("centered_3");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, true).unwrap();

    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    // 3 individuals → 6 entries
    assert_eq!(bin_data.len(), 6 * 4);

    // Row sums should be ~0 after centering
    let g = |i: usize, j: usize| -> f64 {
        let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
        read_f32(&bin_data, ii * (ii + 1) / 2 + jj) as f64
    };

    for i in 0..3 {
        let row_sum: f64 = (0..3).map(|j| g(i, j)).sum();
        assert!(row_sum.abs() < 1e-3, "row {} sum = {:.6}, expected ~0", i, row_sum);
    }
}

#[test]
fn diploid_grm_centering_vs_uncentered_differ() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tA#2#c\t0.995\n\
                   chr1\t0\t10000\tB#1#c\tB#2#c\t0.993\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.985\n\
                   chr1\t0\t10000\tA#1#c\tB#2#c\t0.983\n\
                   chr1\t0\t10000\tA#2#c\tB#1#c\t0.984\n\
                   chr1\t0\t10000\tA#2#c\tB#2#c\t0.982\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let p_uc = dir.path().join("uc");
    let p_c = dir.path().join("c");

    write_diploid_gcta_grm(&acc, p_uc.to_str().unwrap(), false).unwrap();
    write_diploid_gcta_grm(&acc, p_c.to_str().unwrap(), true).unwrap();

    let uc = std::fs::read(format!("{}.grm.bin", p_uc.to_str().unwrap())).unwrap();
    let c = std::fs::read(format!("{}.grm.bin", p_c.to_str().unwrap())).unwrap();

    assert_ne!(uc, c, "centered and uncentered diploid GRM should differ");
}

#[test]
fn diploid_grm_id_file_contains_haplotype_info() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tHG00733#1#c\tHG00733#2#c\t0.996\n\
                   chr1\t0\t10000\tNA19240#1#c\tNA19240#2#c\t0.994\n\
                   chr1\t0\t10000\tHG00733#1#c\tNA19240#1#c\t0.985\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("id_fmt");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let id_content = std::fs::read_to_string(format!("{}.grm.id", prefix_str)).unwrap();
    let lines: Vec<&str> = id_content.trim().lines().collect();
    assert_eq!(lines.len(), 2);

    // Each line should be: individual_name\thaplotype_list
    for line in &lines {
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 2, "expected name\\thaps: {}", line);
    }
    // First individual should be HG00733
    assert!(lines[0].starts_with("HG00733\t"));
    assert!(lines[1].starts_with("NA19240\t"));
}

#[test]
fn diploid_grm_single_haplotype_per_individual() {
    // Individuals with only hap #1 (no #2)
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.985\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("haploid_indiv");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();
    // 2 individuals → 3 entries
    assert_eq!(bin_data.len(), 3 * 4);

    // Diagonal: only self-identity (1.0), no within-individual comparison
    let g_aa = read_f32(&bin_data, 0);
    assert!((g_aa - 1.0).abs() < 1e-5, "single-hap diagonal should be 1.0, got {}", g_aa);

    let g_ba = read_f32(&bin_data, 1);
    assert!((g_ba - 0.985).abs() < 1e-5, "cross should be 0.985, got {}", g_ba);
}

#[test]
fn diploid_grm_multiple_windows_averaged() {
    let content = "query.name\tstart\tend\tgroup.a\tgroup.b\testimated.identity\n\
                   chr1\t0\t10000\tA#1#c\tA#2#c\t0.990\n\
                   chr1\t10000\t20000\tA#1#c\tA#2#c\t0.996\n\
                   chr1\t0\t10000\tA#1#c\tB#1#c\t0.980\n\
                   chr1\t10000\t20000\tA#1#c\tB#1#c\t0.984\n\
                   chr1\t0\t10000\tB#1#c\tB#2#c\t0.992\n\
                   chr1\t10000\t20000\tB#1#c\tB#2#c\t0.994\n";

    let f = create_sim_file(content);
    let acc = parse_similarity_for_egrm(f.path(), "estimated.identity").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("multi_win");
    let prefix_str = prefix.to_str().unwrap();

    write_diploid_gcta_grm(&acc, prefix_str, false).unwrap();

    let bin_data = std::fs::read(format!("{}.grm.bin", prefix_str)).unwrap();

    // A within: A#1-A#2 avg = (0.990+0.996)/2 = 0.993
    // G[A,A] = (1.0 + 0.993 + 0.993 + 1.0)/4 = 0.9965
    let g_aa = read_f32(&bin_data, 0);
    assert!((g_aa - 0.9965).abs() < 1e-4, "G[A,A] = {}, expected 0.9965", g_aa);
}

// ============================================================
// apply_posterior_feedback edge cases
// ============================================================

#[test]
fn posterior_feedback_negative_lambda_returns_original() {
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let posteriors = vec![vec![0.5, 0.3, 0.2]];
    let result = apply_posterior_feedback(&emissions, &posteriors, -1.0);
    assert_eq!(result, emissions, "negative lambda should return original emissions");
}

#[test]
fn posterior_feedback_empty_posteriors_returns_original() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors: Vec<Vec<f64>> = vec![];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    assert_eq!(result, emissions, "empty posteriors should return original emissions");
}

#[test]
fn posterior_feedback_single_state() {
    let emissions = vec![vec![-0.5]];
    let posteriors = vec![vec![1.0]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 1);
    // feedback = 1.0 * ln(1.0) = 0.0, so result should equal original
    assert!((result[0][0] - (-0.5)).abs() < 1e-10);
}

#[test]
fn posterior_feedback_large_lambda_saturates() {
    let emissions = vec![vec![-1.0, -1.0, -1.0]];
    let posteriors = vec![vec![0.9, 0.09, 0.01]];
    let result_small = apply_posterior_feedback(&emissions, &posteriors, 0.1);
    let result_large = apply_posterior_feedback(&emissions, &posteriors, 10.0);

    // With larger lambda, the spread between states should be wider
    let spread_small = result_small[0][0] - result_small[0][2];
    let spread_large = result_large[0][0] - result_large[0][2];
    assert!(spread_large > spread_small * 5.0,
        "large lambda should give much wider spread: small={:.4}, large={:.4}",
        spread_small, spread_large);
}

#[test]
fn posterior_feedback_all_equal_posteriors() {
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let posteriors = vec![vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);

    // Equal posteriors → same feedback for all states → ranking preserved
    assert!(result[0][0] > result[0][1]);
    assert!(result[0][1] > result[0][2]);

    // All offsets should be equal (same log(1/3) added to each)
    let offset0 = result[0][0] - (-1.0);
    let offset1 = result[0][1] - (-2.0);
    let offset2 = result[0][2] - (-3.0);
    assert!((offset0 - offset1).abs() < 1e-10);
    assert!((offset1 - offset2).abs() < 1e-10);
}

#[test]
fn posterior_feedback_nan_posteriors_do_not_corrupt() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![f64::NAN, 0.5]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // NaN posterior: Rust's f64::max(NaN, epsilon) returns epsilon (1e-10)
    // So feedback = 1.0 * ln(1e-10) ≈ -23.03 → state gets heavily suppressed
    // This is the correct behavior: NaN is treated as "no information" = minimum
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
    let epsilon: f64 = 1e-10;
    let expected = -1.0 + epsilon.ln(); // ≈ -24.026
    assert!((result[0][0] - expected).abs() < 1e-5,
        "NaN posterior should be treated as epsilon, got {}", result[0][0]);
    // State 1 with valid posterior 0.5 should be less suppressed
    assert!(result[0][1] > result[0][0],
        "valid posterior 0.5 should be less suppressed than NaN");
}

#[test]
fn posterior_feedback_inf_posteriors_handled() {
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![f64::INFINITY, 0.5]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // Inf posterior → ln(Inf) = Inf → feedback = Inf → e + Inf = Inf
    // e.is_finite() && feedback.is_finite() → false (feedback not finite) → returns e
    assert_eq!(result[0].len(), 2);
    // The code checks if both e AND feedback are finite; Inf feedback → returns original
    assert!((result[0][0] - (-1.0)).abs() < 1e-10 || result[0][0].is_finite(),
        "Inf posterior should be handled gracefully");
}

#[test]
fn posterior_feedback_mismatched_length_shorter_posteriors() {
    // Posteriors shorter than emissions → zip truncates
    let emissions = vec![
        vec![-1.0, -2.0],
        vec![-1.5, -2.5],
        vec![-1.8, -2.8],
    ];
    let posteriors = vec![
        vec![0.8, 0.2],
    ];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // zip of 3 emissions × 1 posterior = 1 window output
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);
}

#[test]
fn posterior_feedback_mismatched_states_shorter_posteriors() {
    // Per-window: posteriors have fewer states than emissions
    let emissions = vec![vec![-1.0, -2.0, -3.0]];
    let posteriors = vec![vec![0.8, 0.2]]; // only 2 states, emissions has 3
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // Inner zip truncates → only first 2 states modified
    assert_eq!(result[0].len(), 2);
}

#[test]
fn posterior_feedback_very_small_posteriors() {
    let emissions = vec![vec![-1.0, -1.0, -1.0]];
    let posteriors = vec![vec![1e-15, 1e-15, 1e-15]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // Very small posteriors → log(epsilon) ≈ -23 added to all equally
    // All results should be equal (same suppression)
    assert!((result[0][0] - result[0][1]).abs() < 1e-10);
    assert!((result[0][1] - result[0][2]).abs() < 1e-10);
    // And strongly suppressed
    assert!(result[0][0] < -20.0, "small posteriors should suppress heavily: {}", result[0][0]);
}

#[test]
fn posterior_feedback_preserves_neg_infinity_emissions() {
    let emissions = vec![vec![f64::NEG_INFINITY, f64::NEG_INFINITY, -1.0]];
    let posteriors = vec![vec![0.9, 0.05, 0.05]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    // NEG_INFINITY is not finite → returns original
    assert_eq!(result[0][0], f64::NEG_INFINITY);
    assert_eq!(result[0][1], f64::NEG_INFINITY);
    assert!(result[0][2].is_finite());
}

#[test]
fn posterior_feedback_lambda_exactly_zero_boundary() {
    // Boundary: lambda = 0.0 exactly (not negative, not positive)
    let emissions = vec![vec![-1.0, -2.0]];
    let posteriors = vec![vec![0.7, 0.3]];
    let result = apply_posterior_feedback(&emissions, &posteriors, 0.0);
    // The guard is `lambda <= 0.0` → should return original
    assert_eq!(result, emissions);
}

#[test]
fn posterior_feedback_many_windows_independence() {
    // Each window should be processed independently
    let n = 10;
    let emissions: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![-(i as f64 + 1.0), -(i as f64 + 2.0)])
        .collect();
    let posteriors: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                vec![0.9, 0.1]
            } else {
                vec![0.1, 0.9]
            }
        })
        .collect();

    let result = apply_posterior_feedback(&emissions, &posteriors, 1.0);
    assert_eq!(result.len(), n);

    // Even windows: state 0 boosted (high posterior)
    for i in (0..n).step_by(2) {
        assert!(result[i][0] > result[i][1], "window {} should favor state 0", i);
    }
    // Odd windows: state 1 boosted (high posterior) — but emissions also differ,
    // so we check the RELATIVE boost, not absolute ranking
    for i in (1..n).step_by(2) {
        let diff_original = emissions[i][0] - emissions[i][1]; // positive (state 0 has higher emission)
        let diff_result = result[i][0] - result[i][1]; // should be reduced or reversed
        assert!(diff_result < diff_original,
            "window {}: posterior feedback should reduce state-0 advantage", i);
    }
}

// ============================================================
// Additional edge cases for copying model integration
// ============================================================

#[test]
fn estimate_copying_params_three_populations() {
    let pops = make_pops(&[2, 2, 2]);
    let obs: Vec<_> = (0..10)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.998),
            ("EUR_1", 0.997),
            ("AFR_0", 0.993),
            ("AFR_1", 0.992),
            ("EAS_0", 0.990),
            ("EAS_1", 0.989),
        ]))
        .collect();

    let (temp, switch, default) = estimate_copying_params(&obs, &pops);
    assert!(temp > 0.0 && temp <= 0.05, "temp={}", temp);
    assert!(switch > 0.0 && switch <= 0.05, "switch={}", switch);
    assert!(default > 0.0 && default < 1.0, "default={}", default);
}

#[test]
fn estimate_copying_params_all_zero_sims() {
    let pops = make_pops(&[2, 2]);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.0),
            ("EUR_1", 0.0),
            ("AFR_0", 0.0),
            ("AFR_1", 0.0),
        ]))
        .collect();

    let (temp, _switch, default) = estimate_copying_params(&obs, &pops);
    // All zeros → filtered from all_sims (> 0.0 && < 1.0 is false for 0.0) → fallback default
    assert!((default - 0.99).abs() < 1e-10, "all-zero sims should give default sim");
    // Temperature: all sims are 0.0, max=min=0.0 → no diffs → fallback temp
    assert!((temp - 0.003).abs() < 1e-10, "all-zero sims should give default temp");
}

#[test]
fn estimate_copying_params_single_population() {
    let pops = make_pops(&[3]);
    let obs: Vec<_> = (0..5)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.998),
            ("EUR_1", 0.996),
            ("EUR_2", 0.994),
        ]))
        .collect();

    let (temp, switch, default) = estimate_copying_params(&obs, &pops);
    assert!(temp > 0.0, "single pop should still compute valid temperature");
    assert!(switch > 0.0, "switch should be positive");
    assert!(default > 0.0 && default < 1.0, "default should be in (0,1)");
}

#[test]
fn estimate_copying_params_default_sim_is_p25() {
    let pops = make_pops(&[2, 2]);
    // Create observations with known distribution
    // Sims: 0.990, 0.992, 0.994, 0.996 — P25 = 0.990 (index len/4 = 0)
    let obs: Vec<_> = (0..1)
        .map(|i| make_obs_at(i * 10000, &[
            ("EUR_0", 0.990),
            ("EUR_1", 0.992),
            ("AFR_0", 0.994),
            ("AFR_1", 0.996),
        ]))
        .collect();

    let (_, _, default) = estimate_copying_params(&obs, &pops);
    // P25 of sorted [0.990, 0.992, 0.994, 0.996] = index 1 = 0.992
    assert!((default - 0.992).abs() < 1e-10,
        "default should be P25 of sims, got {}", default);
}
