//! Cycle 85: Edge case tests for infer_per_sample_demography, write_demography_tsv,
//! and format_demography_report — previously untested public functions in demography.rs.

use hprc_ancestry_cli::{
    AncestrySegment, DemographicResult, DemographyParams, PulseEstimate,
    SampleDemographicResult, extract_tract_lengths, format_demography_report,
    infer_demography, infer_per_sample_demography, write_demography_tsv,
};

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn make_seg(sample: &str, ancestry: &str, start: u64, end: u64) -> AncestrySegment {
    AncestrySegment {
        chrom: "chr1".to_string(),
        start,
        end,
        sample: sample.to_string(),
        ancestry_idx: 0,
        ancestry_name: ancestry.to_string(),
        n_windows: ((end - start) / 10_000).max(1) as usize,
        mean_similarity: 0.95,
        mean_posterior: Some(0.9),
        discriminability: 0.1,
        lod_score: 5.0,
    }
}

fn default_params() -> DemographyParams {
    DemographyParams::default()
}

/// Generate deterministic exponential tract segments for a sample/ancestry
fn make_exponential_segments(
    sample: &str,
    ancestry: &str,
    n: usize,
    lambda: f64,
    l_min: f64,
) -> Vec<AncestrySegment> {
    let mut pos = 0u64;
    (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / n as f64;
            let length = (-((1.0 - p).ln()) / lambda + l_min) as u64;
            let start = pos;
            let end = pos + length;
            pos = end + 1000; // gap between segments
            make_seg(sample, ancestry, start, end)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// infer_per_sample_demography — edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn per_sample_demography_empty_segments() {
    let results = infer_per_sample_demography(
        &[],
        &["EUR".to_string()],
        &default_params(),
    );
    assert!(results.is_empty(), "no segments → no samples → empty results");
}

#[test]
fn per_sample_demography_single_sample_single_pop() {
    let segs = make_exponential_segments("HG001", "EUR", 50, 1e-5, 25_000.0);
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();
    let results = infer_per_sample_demography(
        &seg_refs,
        &["EUR".to_string()],
        &default_params(),
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].sample, "HG001");
    assert_eq!(results[0].results.len(), 1);
    assert_eq!(results[0].results[0].population, "EUR");
    assert!(results[0].results[0].n_tracts > 0);
}

#[test]
fn per_sample_demography_multiple_samples() {
    let mut all_segs = Vec::new();
    all_segs.extend(make_exponential_segments("HG001", "EUR", 30, 1e-5, 25_000.0));
    all_segs.extend(make_exponential_segments("HG002", "EUR", 20, 2e-5, 25_000.0));
    all_segs.extend(make_exponential_segments("HG001", "AFR", 10, 5e-6, 25_000.0));

    let seg_refs: Vec<&AncestrySegment> = all_segs.iter().collect();
    let pops = vec!["EUR".to_string(), "AFR".to_string()];
    let results = infer_per_sample_demography(&seg_refs, &pops, &default_params());

    assert_eq!(results.len(), 2, "two distinct samples");
    // Results should be sorted by sample name (BTreeMap)
    assert_eq!(results[0].sample, "HG001");
    assert_eq!(results[1].sample, "HG002");
    // Each sample gets results for both populations
    assert_eq!(results[0].results.len(), 2);
    assert_eq!(results[1].results.len(), 2);
}

#[test]
fn per_sample_demography_sample_with_no_tracts_for_pop() {
    // HG001 has EUR tracts but NO AFR tracts
    let segs = make_exponential_segments("HG001", "EUR", 30, 1e-5, 25_000.0);
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();
    let pops = vec!["EUR".to_string(), "AFR".to_string()];
    let results = infer_per_sample_demography(&seg_refs, &pops, &default_params());

    assert_eq!(results.len(), 1);
    // AFR result should have 0 tracts
    let afr_result = &results[0].results[1];
    assert_eq!(afr_result.population, "AFR");
    assert_eq!(afr_result.n_tracts, 0);
    assert_eq!(afr_result.n_pulses, 0);
}

#[test]
fn per_sample_demography_too_few_tracts() {
    // Only 2 tracts — below minimum (3) for inference
    let segs = vec![
        make_seg("HG001", "EUR", 0, 100_000),
        make_seg("HG001", "EUR", 200_000, 350_000),
    ];
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();
    let results = infer_per_sample_demography(
        &seg_refs,
        &["EUR".to_string()],
        &default_params(),
    );
    assert_eq!(results.len(), 1);
    // Should return warning for insufficient tracts
    let eur = &results[0].results[0];
    // Tracts below l_min may also be filtered
    assert!(eur.n_pulses == 0 || eur.low_tract_warning.is_some());
}

#[test]
fn per_sample_demography_empty_population_names() {
    let segs = make_exponential_segments("HG001", "EUR", 30, 1e-5, 25_000.0);
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();
    let results = infer_per_sample_demography(&seg_refs, &[], &default_params());
    assert_eq!(results.len(), 1);
    assert!(results[0].results.is_empty(), "no populations → no results per pop");
}

// ═══════════════════════════════════════════════════════════════════
// write_demography_tsv — file output edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn write_demography_tsv_empty_results() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_empty.tsv");
    write_demography_tsv(&path, &[], &[]).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "header only");
    assert!(lines[0].starts_with("sample\tpopulation"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_demography_tsv_pooled_only() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_pooled.tsv");

    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 100,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 500.0,
            proportion: 1.0,
            rate: 5e-6,
            cv: 0.1,
        }],
        bic: 1234.5,
        log_likelihood: -600.0,
        ks_statistic: Some(0.05),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];

    write_demography_tsv(&path, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "header + 1 data line");
    assert!(lines[1].starts_with("POOLED\tEUR"));
    assert!(lines[1].contains("500.0"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_demography_tsv_no_pulses_writes_na() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_no_pulses.tsv");

    let pooled = vec![DemographicResult {
        population: "AFR".to_string(),
        n_tracts: 2,
        n_pulses: 0,
        pulses: vec![],
        bic: f64::INFINITY,
        log_likelihood: f64::NEG_INFINITY,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: Some("insufficient".to_string()),
    }];

    write_demography_tsv(&path, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2);
    // Should contain NA for pulse fields, BIC, LL, KS
    assert!(lines[1].contains("NA"), "no-pulse row should have NA fields: {}", lines[1]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_demography_tsv_per_sample() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_persample.tsv");

    let per_sample = vec![
        SampleDemographicResult {
            sample: "HG001".to_string(),
            results: vec![DemographicResult {
                population: "EUR".to_string(),
                n_tracts: 50,
                n_pulses: 1,
                pulses: vec![PulseEstimate {
                    generations: 300.0,
                    proportion: 1.0,
                    rate: 3e-6,
                    cv: 0.14,
                }],
                bic: 800.0,
                log_likelihood: -380.0,
                ks_statistic: Some(0.03),
                single_pulse_rejected: false,
                low_tract_warning: None,
            }],
        },
        SampleDemographicResult {
            sample: "HG002".to_string(),
            results: vec![DemographicResult {
                population: "EUR".to_string(),
                n_tracts: 0,
                n_pulses: 0,
                pulses: vec![],
                bic: f64::INFINITY,
                log_likelihood: f64::NEG_INFINITY,
                ks_statistic: None,
                single_pulse_rejected: false,
                low_tract_warning: None,
            }],
        },
    ];

    write_demography_tsv(&path, &[], &per_sample).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 samples");
    assert!(lines[1].starts_with("HG001"));
    assert!(lines[2].starts_with("HG002"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_demography_tsv_multi_pulse() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_multipulse.tsv");

    let pooled = vec![DemographicResult {
        population: "AFR".to_string(),
        n_tracts: 200,
        n_pulses: 2,
        pulses: vec![
            PulseEstimate {
                generations: 50.0,
                proportion: 0.6,
                rate: 5e-7,
                cv: 0.09,
            },
            PulseEstimate {
                generations: 500.0,
                proportion: 0.4,
                rate: 5e-6,
                cv: 0.11,
            },
        ],
        bic: 2345.6,
        log_likelihood: -1100.0,
        ks_statistic: Some(0.12),
        single_pulse_rejected: true,
        low_tract_warning: None,
    }];

    write_demography_tsv(&path, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 pulse rows");
    // First pulse row: pulse_idx=1
    assert!(lines[1].contains("\t1\t"));
    // Second pulse row: pulse_idx=2
    assert!(lines[2].contains("\t2\t"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_demography_tsv_pooled_and_per_sample() {
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_both.tsv");

    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 100,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 400.0,
            proportion: 1.0,
            rate: 4e-6,
            cv: 0.1,
        }],
        bic: 1000.0,
        log_likelihood: -480.0,
        ks_statistic: Some(0.02),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];

    let per_sample = vec![SampleDemographicResult {
        sample: "S1".to_string(),
        results: vec![DemographicResult {
            population: "EUR".to_string(),
            n_tracts: 50,
            n_pulses: 1,
            pulses: vec![PulseEstimate {
                generations: 350.0,
                proportion: 1.0,
                rate: 3.5e-6,
                cv: 0.14,
            }],
            bic: 500.0,
            log_likelihood: -230.0,
            ks_statistic: Some(0.04),
            single_pulse_rejected: false,
            low_tract_warning: None,
        }],
    }];

    write_demography_tsv(&path, &pooled, &per_sample).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "header + POOLED + S1");
    assert!(lines[1].starts_with("POOLED"), "pooled first");
    assert!(lines[2].starts_with("S1"), "per-sample second");
    std::fs::remove_file(&path).ok();
}

// ═══════════════════════════════════════════════════════════════════
// format_demography_report — formatting edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn format_demography_empty_results() {
    let out = format_demography_report(&[]);
    assert!(out.contains("Demographic Inference"));
}

#[test]
fn format_demography_insufficient_tracts() {
    let results = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 2,
        n_pulses: 0,
        pulses: vec![],
        bic: f64::INFINITY,
        log_likelihood: f64::NEG_INFINITY,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: Some("too few".to_string()),
    }];
    let out = format_demography_report(&results);
    assert!(out.contains("insufficient"));
    assert!(out.contains("EUR"));
}

#[test]
fn format_demography_single_pulse() {
    let results = vec![DemographicResult {
        population: "AFR".to_string(),
        n_tracts: 100,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 250.0,
            proportion: 1.0,
            rate: 2.5e-6,
            cv: 0.1,
        }],
        bic: 1500.0,
        log_likelihood: -700.0,
        ks_statistic: Some(0.04),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];
    let out = format_demography_report(&results);
    assert!(out.contains("AFR"));
    assert!(out.contains("1 pulse"));
    assert!(out.contains("KS=0.040"));
    assert!(out.contains("single-pulse OK"));
    assert!(out.contains("Pulse 1"));
    assert!(out.contains("BIC="));
}

#[test]
fn format_demography_multi_pulse_rejected() {
    let results = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 200,
        n_pulses: 2,
        pulses: vec![
            PulseEstimate {
                generations: 50.0,
                proportion: 0.7,
                rate: 5e-7,
                cv: 0.08,
            },
            PulseEstimate {
                generations: 500.0,
                proportion: 0.3,
                rate: 5e-6,
                cv: 0.12,
            },
        ],
        bic: 3000.0,
        log_likelihood: -1400.0,
        ks_statistic: Some(0.15),
        single_pulse_rejected: true,
        low_tract_warning: None,
    }];
    let out = format_demography_report(&results);
    assert!(out.contains("2 pulses"));
    assert!(out.contains("single-pulse REJECTED"));
    assert!(out.contains("Pulse 1"));
    assert!(out.contains("Pulse 2"));
    assert!(out.contains("70.0%")); // 0.7 * 100
    assert!(out.contains("30.0%")); // 0.3 * 100
}

#[test]
fn format_demography_multiple_populations() {
    let results = vec![
        DemographicResult {
            population: "EUR".to_string(),
            n_tracts: 100,
            n_pulses: 1,
            pulses: vec![PulseEstimate {
                generations: 400.0,
                proportion: 1.0,
                rate: 4e-6,
                cv: 0.1,
            }],
            bic: 1000.0,
            log_likelihood: -480.0,
            ks_statistic: Some(0.03),
            single_pulse_rejected: false,
            low_tract_warning: None,
        },
        DemographicResult {
            population: "AFR".to_string(),
            n_tracts: 0,
            n_pulses: 0,
            pulses: vec![],
            bic: f64::INFINITY,
            log_likelihood: f64::NEG_INFINITY,
            ks_statistic: None,
            single_pulse_rejected: false,
            low_tract_warning: None,
        },
    ];
    let out = format_demography_report(&results);
    assert!(out.contains("EUR"));
    assert!(out.contains("AFR"));
}

#[test]
fn format_demography_no_ks_statistic() {
    let results = vec![DemographicResult {
        population: "X".to_string(),
        n_tracts: 5,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 100.0,
            proportion: 1.0,
            rate: 1e-6,
            cv: 0.45,
        }],
        bic: 50.0,
        log_likelihood: -20.0,
        ks_statistic: None,
        single_pulse_rejected: false,
        low_tract_warning: Some("few tracts".to_string()),
    }];
    let out = format_demography_report(&results);
    // No KS string when ks_statistic is None
    assert!(!out.contains("KS="));
}

// ═══════════════════════════════════════════════════════════════════
// infer_demography — additional edge cases not in existing tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn infer_demography_exactly_three_tracts() {
    // Minimum viable input
    let tracts = vec![30_000.0, 40_000.0, 50_000.0];
    let result = infer_demography(&tracts, "POP", &default_params());
    assert!(result.n_tracts <= 3);
    // May or may not have enough valid tracts depending on l_min filtering
}

#[test]
fn infer_demography_all_tracts_below_l_min() {
    // All tracts below detection limit (20000bp default)
    let tracts = vec![10_000.0, 15_000.0, 18_000.0, 5_000.0];
    let result = infer_demography(&tracts, "POP", &default_params());
    assert_eq!(result.n_pulses, 0, "all filtered out → no inference");
}

#[test]
fn infer_demography_custom_params() {
    let mut params = default_params();
    params.l_min_bp = 5_000.0; // Lower detection limit
    params.max_pulses = 1; // Force single pulse
    params.recomb_rate = 2e-8; // Different recombination rate

    // Generate tracts above the custom l_min
    let tracts: Vec<f64> = (0..50)
        .map(|i| 10_000.0 + (i as f64 * 1_000.0))
        .collect();

    let result = infer_demography(&tracts, "TEST", &params);
    assert!(result.n_tracts > 0);
    assert_eq!(result.n_pulses, 1);
    // Generation estimate depends on recomb_rate
    let gen = result.pulses[0].generations;
    assert!(gen > 0.0 && gen.is_finite());
}

#[test]
fn infer_demography_low_tract_warning_threshold() {
    // 10 tracts — enough for inference but triggers warning for multi-pulse
    let tracts: Vec<f64> = (0..10)
        .map(|i| 25_000.0 + (i as f64 * 5_000.0))
        .collect();
    let result = infer_demography(&tracts, "POP", &default_params());
    // Should have a low_tract_warning since n < 20
    if result.n_tracts < 20 && result.n_tracts >= 3 {
        assert!(result.low_tract_warning.is_some());
    }
}

#[test]
fn infer_demography_cv_finite_for_valid_result() {
    let tracts: Vec<f64> = (0..100)
        .map(|i| 25_000.0 + (i as f64 * 2_000.0))
        .collect();
    let result = infer_demography(&tracts, "POP", &default_params());
    for pulse in &result.pulses {
        assert!(pulse.cv.is_finite(), "CV should be finite: {}", pulse.cv);
        assert!(pulse.cv > 0.0, "CV should be positive: {}", pulse.cv);
    }
}

#[test]
fn infer_demography_bic_finite_for_valid() {
    let tracts: Vec<f64> = (0..50)
        .map(|i| 30_000.0 + (i as f64 * 3_000.0))
        .collect();
    let result = infer_demography(&tracts, "POP", &default_params());
    if result.n_pulses > 0 {
        assert!(result.bic.is_finite(), "BIC should be finite");
        assert!(result.log_likelihood.is_finite(), "LL should be finite");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cross-function consistency
// ═══════════════════════════════════════════════════════════════════

#[test]
fn per_sample_results_match_pooled_for_single_sample() {
    // With one sample, per-sample should give same results as pooled
    let segs = make_exponential_segments("HG001", "EUR", 80, 1e-5, 25_000.0);
    let seg_refs: Vec<&AncestrySegment> = segs.iter().collect();
    let params = default_params();
    let pops = vec!["EUR".to_string()];

    let per_sample = infer_per_sample_demography(&seg_refs, &pops, &params);
    let tracts: Vec<f64> = segs.iter().map(|s| (s.end - s.start) as f64).collect();
    let pooled = infer_demography(&tracts, "EUR", &params);

    assert_eq!(per_sample.len(), 1);
    let sample_eur = &per_sample[0].results[0];
    // Same number of tracts
    assert_eq!(sample_eur.n_tracts, pooled.n_tracts);
    // Same number of pulses
    assert_eq!(sample_eur.n_pulses, pooled.n_pulses);
}

#[test]
fn write_then_read_tsv_roundtrip() {
    // Write TSV and verify it can be read back as valid TSV
    let dir = std::env::temp_dir();
    let path = dir.join("test_dem_roundtrip.tsv");

    let pooled = vec![DemographicResult {
        population: "EUR".to_string(),
        n_tracts: 50,
        n_pulses: 1,
        pulses: vec![PulseEstimate {
            generations: 123.4,
            proportion: 1.0,
            rate: 1.234e-6,
            cv: 0.14,
        }],
        bic: 999.9,
        log_likelihood: -444.4,
        ks_statistic: Some(0.0321),
        single_pulse_rejected: false,
        low_tract_warning: None,
    }];

    write_demography_tsv(&path, &pooled, &[]).unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    // Verify header has expected columns
    let header_fields: Vec<&str> = lines[0].split('\t').collect();
    assert_eq!(header_fields.len(), 12);
    assert_eq!(header_fields[0], "sample");
    assert_eq!(header_fields[11], "ks_rejected");

    // Verify data line has same number of fields
    let data_fields: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(data_fields.len(), 12, "data should have 12 tab-separated fields");

    std::fs::remove_file(&path).ok();
}
