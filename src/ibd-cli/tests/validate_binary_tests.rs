//! Tests for the ibd-validate binary (src/bin/validate.rs).
//!
//! Since validate.rs functions are private, pure functions are reimplemented
//! for unit verification, and the binary is exercised via CLI integration tests
//! for I/O and end-to-end paths.

use assert_cmd::Command;
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

// =====================================================================
// Re-implemented pure functions from validate.rs for unit testing
// =====================================================================

/// Reimplementation of validate_probability from validate.rs:27
fn validate_probability(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(format!("probability must be in (0.0, 1.0), got {}", v))
    }
}

/// Reimplementation of extract_haplotype_id from validate.rs:180
fn extract_haplotype_id(full_id: &str) -> String {
    if let Some(colon_pos) = full_id.rfind(':') {
        let after_colon = &full_id[colon_pos + 1..];
        if after_colon.contains('-')
            && after_colon.chars().all(|c| c.is_ascii_digit() || c == '-')
        {
            return full_id[..colon_pos].to_string();
        }
    }
    full_id.to_string()
}

/// Reimplementation of extract_sample_id from validate.rs:196
fn extract_sample_id(hap_id: &str) -> String {
    hap_id.split('#').next().unwrap_or(hap_id).to_string()
}

/// Reimplementation of ExcludeRegion from validate.rs:499
#[derive(Debug, Clone)]
struct ExcludeRegion {
    _chrom: String,
    start: u64,
    end: u64,
}

/// Reimplementation of overlaps_excluded from validate.rs:535
fn overlaps_excluded(seg_start: u64, seg_end: u64, regions: &[ExcludeRegion]) -> bool {
    for r in regions {
        if seg_start < r.end && seg_end > r.start {
            return true;
        }
    }
    false
}

/// Reimplementation of parse_exclude_regions from validate.rs:507
fn parse_exclude_regions(path: &std::path::Path) -> anyhow::Result<Vec<ExcludeRegion>> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }
        let start: u64 = fields[1].parse()?;
        let end: u64 = fields[2].parse()?;
        regions.push(ExcludeRegion {
            _chrom: fields[0].to_string(),
            start,
            end,
        });
    }
    Ok(regions)
}

/// Reimplementation of IbsRecord from validate.rs:167
#[derive(Debug, Clone)]
struct IbsRecord {
    chrom: String,
    start: u64,
    end: u64,
    identity: f64,
    a_length: u64,
    b_length: u64,
}

/// Reimplementation of WindowBackgroundStats from validate.rs:545
struct WindowBackgroundStats {
    high_identity_counts: HashMap<(u64, u64), (usize, usize)>,
    identity_sums: HashMap<(u64, u64), (f64, usize)>,
}

/// Reimplementation of compute_window_background from validate.rs:554
fn compute_window_background(
    pair_data: &HashMap<(String, String), Vec<IbsRecord>>,
    identity_threshold: f64,
    identity_floor: f64,
) -> WindowBackgroundStats {
    let mut high_counts: HashMap<(u64, u64), (usize, usize)> = HashMap::new();
    let mut id_sums: HashMap<(u64, u64), (f64, usize)> = HashMap::new();
    for records in pair_data.values() {
        for r in records {
            if identity_floor > 0.0 && r.identity < identity_floor {
                continue;
            }
            let key = (r.start, r.end);
            let hc = high_counts.entry(key).or_insert((0, 0));
            hc.1 += 1;
            if r.identity >= identity_threshold {
                hc.0 += 1;
            }
            let is = id_sums.entry(key).or_insert((0.0, 0));
            is.0 += r.identity;
            is.1 += 1;
        }
    }
    WindowBackgroundStats {
        high_identity_counts: high_counts,
        identity_sums: id_sums,
    }
}

/// Reimplementation of apply_background_filter from validate.rs:587
fn apply_background_filter(
    records: &mut Vec<IbsRecord>,
    background: &WindowBackgroundStats,
    ratio_threshold: f64,
) {
    records.retain(|r| {
        let key = (r.start, r.end);
        match background.high_identity_counts.get(&key) {
            Some(&(n_high, n_total)) if n_total > 0 => {
                let ratio = n_high as f64 / n_total as f64;
                ratio <= ratio_threshold
            }
            _ => true,
        }
    });
}

/// Reimplementation of WindowMeanVar from validate.rs:605
struct WindowMeanVar {
    mean: f64,
    std: f64,
}

/// Reimplementation of compute_window_mean_var from validate.rs:611
fn compute_window_mean_var(
    background: &WindowBackgroundStats,
    pair_data: &HashMap<(String, String), Vec<IbsRecord>>,
    identity_floor: f64,
) -> HashMap<(u64, u64), WindowMeanVar> {
    let means: HashMap<(u64, u64), f64> = background
        .identity_sums
        .iter()
        .filter(|(_, &(_, count))| count > 1)
        .map(|(&key, &(sum, count))| (key, sum / count as f64))
        .collect();

    let mut var_accum: HashMap<(u64, u64), (f64, usize)> = HashMap::new();
    for records in pair_data.values() {
        for r in records {
            if identity_floor > 0.0 && r.identity < identity_floor {
                continue;
            }
            let key = (r.start, r.end);
            if let Some(&mean) = means.get(&key) {
                let entry = var_accum.entry(key).or_insert((0.0, 0));
                entry.0 += (r.identity - mean).powi(2);
                entry.1 += 1;
            }
        }
    }

    var_accum
        .into_iter()
        .filter_map(|(key, (sum_sq, count))| {
            if count > 1 {
                let mean = *means.get(&key)?;
                let std = (sum_sq / count as f64).sqrt().max(1e-8);
                Some((key, WindowMeanVar { mean, std }))
            } else {
                None
            }
        })
        .collect()
}

/// Reimplementation of normalize_identity_by_background from validate.rs:653
fn normalize_identity_by_background(
    records: &mut [IbsRecord],
    window_stats: &HashMap<(u64, u64), WindowMeanVar>,
) {
    for r in records.iter_mut() {
        let key = (r.start, r.end);
        if let Some(stats) = window_stats.get(&key) {
            let z = (r.identity - stats.mean) / stats.std;
            r.identity = 1.0 / (1.0 + (-z).exp());
        }
    }
}

// =====================================================================
// Helper to build IbsRecord
// =====================================================================
fn make_record(chrom: &str, start: u64, end: u64, identity: f64) -> IbsRecord {
    IbsRecord {
        chrom: chrom.to_string(),
        start,
        end,
        identity,
        a_length: 0,
        b_length: 0,
    }
}

fn make_pair_data(
    pairs: Vec<(&str, &str, Vec<IbsRecord>)>,
) -> HashMap<(String, String), Vec<IbsRecord>> {
    let mut map = HashMap::new();
    for (a, b, records) in pairs {
        map.insert((a.to_string(), b.to_string()), records);
    }
    map
}

// =====================================================================
// validate_probability tests
// =====================================================================

#[test]
fn validate_probability_valid_mid() {
    assert_eq!(validate_probability("0.5").unwrap(), 0.5);
}

#[test]
fn validate_probability_valid_small() {
    assert_eq!(validate_probability("0.001").unwrap(), 0.001);
}

#[test]
fn validate_probability_valid_near_one() {
    assert_eq!(validate_probability("0.999").unwrap(), 0.999);
}

#[test]
fn validate_probability_zero_rejected() {
    assert!(validate_probability("0.0").is_err());
}

#[test]
fn validate_probability_one_rejected() {
    assert!(validate_probability("1.0").is_err());
}

#[test]
fn validate_probability_negative_rejected() {
    assert!(validate_probability("-0.1").is_err());
}

#[test]
fn validate_probability_above_one_rejected() {
    assert!(validate_probability("1.1").is_err());
}

#[test]
fn validate_probability_nan_rejected() {
    assert!(validate_probability("NaN").is_err());
}

#[test]
fn validate_probability_not_a_number() {
    let err = validate_probability("abc").unwrap_err();
    assert!(err.contains("not a valid number"));
}

#[test]
fn validate_probability_empty_rejected() {
    assert!(validate_probability("").is_err());
}

// =====================================================================
// extract_haplotype_id tests
// =====================================================================

#[test]
fn extract_haplotype_id_with_coords() {
    assert_eq!(
        extract_haplotype_id("HG00280#2#JBHDWB010000002.1:130787850-130792849"),
        "HG00280#2#JBHDWB010000002.1"
    );
}

#[test]
fn extract_haplotype_id_without_coords() {
    assert_eq!(
        extract_haplotype_id("HG00280#2#JBHDWB010000002.1"),
        "HG00280#2#JBHDWB010000002.1"
    );
}

#[test]
fn extract_haplotype_id_simple() {
    assert_eq!(extract_haplotype_id("HG00280"), "HG00280");
}

#[test]
fn extract_haplotype_id_colon_but_not_coords() {
    // Colon followed by text that isn't coordinates — should keep as-is
    assert_eq!(
        extract_haplotype_id("HG00280#2:scaffold_name"),
        "HG00280#2:scaffold_name"
    );
}

#[test]
fn extract_haplotype_id_multiple_colons() {
    // Only the last colon should be checked
    assert_eq!(
        extract_haplotype_id("HG00280#2#contig:12345-67890"),
        "HG00280#2#contig"
    );
}

#[test]
fn extract_haplotype_id_just_coords() {
    // Edge case: entire string is "coords" format
    assert_eq!(extract_haplotype_id(":100-200"), "");
}

#[test]
fn extract_haplotype_id_empty() {
    assert_eq!(extract_haplotype_id(""), "");
}

#[test]
fn extract_haplotype_id_colon_at_end_no_coords() {
    // Colon at end with nothing after — not coords format
    assert_eq!(extract_haplotype_id("HG00280:"), "HG00280:");
}

// =====================================================================
// extract_sample_id tests
// =====================================================================

#[test]
fn extract_sample_id_full_haplotype() {
    assert_eq!(
        extract_sample_id("HG00280#2#JBHDWB010000002.1"),
        "HG00280"
    );
}

#[test]
fn extract_sample_id_with_coords() {
    assert_eq!(
        extract_sample_id("HG00280#2#contig:100-200"),
        "HG00280"
    );
}

#[test]
fn extract_sample_id_no_hash() {
    assert_eq!(extract_sample_id("HG00280"), "HG00280");
}

#[test]
fn extract_sample_id_empty() {
    assert_eq!(extract_sample_id(""), "");
}

#[test]
fn extract_sample_id_hash_at_start() {
    assert_eq!(extract_sample_id("#2#contig"), "");
}

// =====================================================================
// overlaps_excluded tests
// =====================================================================

#[test]
fn overlaps_excluded_no_regions() {
    assert!(!overlaps_excluded(100, 200, &[]));
}

#[test]
fn overlaps_excluded_no_overlap() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 300,
        end: 400,
    }];
    assert!(!overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_full_overlap() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 50,
        end: 250,
    }];
    assert!(overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_partial_left() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 50,
        end: 150,
    }];
    assert!(overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_partial_right() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 150,
        end: 250,
    }];
    assert!(overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_adjacent_no_overlap() {
    // Adjacent but not overlapping (half-open intervals)
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 200,
        end: 300,
    }];
    assert!(!overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_segment_inside_region() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 50,
        end: 500,
    }];
    assert!(overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_multiple_regions_second_hits() {
    let regions = vec![
        ExcludeRegion {
            _chrom: "chr1".into(),
            start: 0,
            end: 50,
        },
        ExcludeRegion {
            _chrom: "chr1".into(),
            start: 150,
            end: 250,
        },
    ];
    assert!(overlaps_excluded(100, 200, &regions));
}

#[test]
fn overlaps_excluded_single_base_overlap() {
    let regions = vec![ExcludeRegion {
        _chrom: "chr1".into(),
        start: 199,
        end: 300,
    }];
    // seg_start(100) < r.end(300) && seg_end(200) > r.start(199) → true
    assert!(overlaps_excluded(100, 200, &regions));
}

// =====================================================================
// parse_exclude_regions tests (file I/O)
// =====================================================================

#[test]
fn parse_exclude_regions_basic() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chr1\t100\t200").unwrap();
    writeln!(f, "chr1\t300\t400").unwrap();
    f.flush().unwrap();

    let regions = parse_exclude_regions(f.path()).unwrap();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].start, 100);
    assert_eq!(regions[0].end, 200);
    assert_eq!(regions[1].start, 300);
    assert_eq!(regions[1].end, 400);
}

#[test]
fn parse_exclude_regions_comments_and_blanks() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "# header").unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "chr1\t100\t200").unwrap();
    writeln!(f, "  ").unwrap();
    writeln!(f, "chr2\t500\t600").unwrap();
    f.flush().unwrap();

    let regions = parse_exclude_regions(f.path()).unwrap();
    assert_eq!(regions.len(), 2);
}

#[test]
fn parse_exclude_regions_too_few_fields_skipped() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chr1\t100").unwrap(); // too few — skipped
    writeln!(f, "chr1\t200\t300").unwrap(); // valid
    f.flush().unwrap();

    let regions = parse_exclude_regions(f.path()).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].start, 200);
}

#[test]
fn parse_exclude_regions_nonexistent_file() {
    assert!(parse_exclude_regions(std::path::Path::new("/nonexistent/file.bed")).is_err());
}

#[test]
fn parse_exclude_regions_empty_file() {
    let f = NamedTempFile::new().unwrap();
    let regions = parse_exclude_regions(f.path()).unwrap();
    assert!(regions.is_empty());
}

#[test]
fn parse_exclude_regions_invalid_coordinate() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chr1\tabc\t200").unwrap();
    f.flush().unwrap();

    assert!(parse_exclude_regions(f.path()).is_err());
}

#[test]
fn parse_exclude_regions_extra_columns_ok() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(f, "chr1\t100\t200\tname\tscore").unwrap();
    f.flush().unwrap();

    let regions = parse_exclude_regions(f.path()).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].start, 100);
}

// =====================================================================
// compute_window_background tests
// =====================================================================

#[test]
fn compute_background_single_pair_single_window() {
    let pair_data = make_pair_data(vec![(
        "A",
        "B",
        vec![make_record("chr1", 0, 5000, 0.999)],
    )]);

    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let counts = bg.high_identity_counts.get(&(0, 5000)).unwrap();
    assert_eq!(*counts, (1, 1)); // 1 high out of 1 total
    let sums = bg.identity_sums.get(&(0, 5000)).unwrap();
    assert!((sums.0 - 0.999).abs() < 1e-10);
    assert_eq!(sums.1, 1);
}

#[test]
fn compute_background_two_pairs_same_window() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.999)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.990)]),
    ]);

    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let counts = bg.high_identity_counts.get(&(0, 5000)).unwrap();
    // Pair A-B has identity 0.999 >= threshold, pair C-D has 0.990 < threshold
    assert_eq!(counts.0, 1); // only 1 high
    assert_eq!(counts.1, 2); // 2 total
}

#[test]
fn compute_background_identity_floor_filters() {
    let pair_data = make_pair_data(vec![(
        "A",
        "B",
        vec![
            make_record("chr1", 0, 5000, 0.3), // below floor
            make_record("chr1", 5000, 10000, 0.999),
        ],
    )]);

    let bg = compute_window_background(&pair_data, 0.999, 0.5);
    // Window (0, 5000) should be absent because its identity < floor
    assert!(bg.high_identity_counts.get(&(0, 5000)).is_none());
    // Window (5000, 10000) should be present
    assert!(bg.high_identity_counts.get(&(5000, 10000)).is_some());
}

#[test]
fn compute_background_empty_pairs() {
    let pair_data: HashMap<(String, String), Vec<IbsRecord>> = HashMap::new();
    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    assert!(bg.high_identity_counts.is_empty());
    assert!(bg.identity_sums.is_empty());
}

#[test]
fn compute_background_all_below_threshold() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.990)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.995)]),
    ]);

    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let counts = bg.high_identity_counts.get(&(0, 5000)).unwrap();
    assert_eq!(counts.0, 0); // none high
    assert_eq!(counts.1, 2); // 2 total
}

#[test]
fn compute_background_all_above_threshold() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.9999)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.9995)]),
    ]);

    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let counts = bg.high_identity_counts.get(&(0, 5000)).unwrap();
    assert_eq!(counts.0, 2); // both high
    assert_eq!(counts.1, 2);
}

// =====================================================================
// apply_background_filter tests
// =====================================================================

#[test]
fn apply_background_filter_removes_high_ratio_windows() {
    // 3 pairs, all with identity >= 0.999 at window (0, 5000) → ratio = 1.0 > 0.5
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.9999)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.9995)]),
        ("E", "F", vec![make_record("chr1", 0, 5000, 0.9998)]),
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);

    let mut records = vec![
        make_record("chr1", 0, 5000, 0.9999),
        make_record("chr1", 5000, 10000, 0.998),
    ];
    apply_background_filter(&mut records, &bg, 0.5);
    // Window (0, 5000) has ratio 1.0 > 0.5, should be removed
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].start, 5000);
}

#[test]
fn apply_background_filter_keeps_low_ratio_windows() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.9999)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.990)]),
        ("E", "F", vec![make_record("chr1", 0, 5000, 0.985)]),
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);

    let mut records = vec![make_record("chr1", 0, 5000, 0.9999)];
    apply_background_filter(&mut records, &bg, 0.5);
    // Ratio = 1/3 = 0.33 <= 0.5, should be kept
    assert_eq!(records.len(), 1);
}

#[test]
fn apply_background_filter_unknown_window_kept() {
    let bg = WindowBackgroundStats {
        high_identity_counts: HashMap::new(),
        identity_sums: HashMap::new(),
    };

    let mut records = vec![make_record("chr1", 0, 5000, 0.999)];
    apply_background_filter(&mut records, &bg, 0.5);
    // Unknown window → default keep
    assert_eq!(records.len(), 1);
}

#[test]
fn apply_background_filter_exact_boundary() {
    // ratio = 0.5, threshold = 0.5 → ratio <= threshold → keep
    let mut counts = HashMap::new();
    counts.insert((0u64, 5000u64), (1usize, 2usize));
    let bg = WindowBackgroundStats {
        high_identity_counts: counts,
        identity_sums: HashMap::new(),
    };

    let mut records = vec![make_record("chr1", 0, 5000, 0.999)];
    apply_background_filter(&mut records, &bg, 0.5);
    assert_eq!(records.len(), 1); // ratio = 0.5 == threshold → kept
}

#[test]
fn apply_background_filter_empty_records() {
    let bg = WindowBackgroundStats {
        high_identity_counts: HashMap::new(),
        identity_sums: HashMap::new(),
    };
    let mut records: Vec<IbsRecord> = vec![];
    apply_background_filter(&mut records, &bg, 0.5);
    assert!(records.is_empty());
}

// =====================================================================
// compute_window_mean_var tests
// =====================================================================

#[test]
fn compute_mean_var_two_pairs() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.998)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 1.000)]),
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let stats = compute_window_mean_var(&bg, &pair_data, 0.0);

    let w = stats.get(&(0, 5000)).unwrap();
    assert!((w.mean - 0.999).abs() < 1e-10);
    // std = sqrt(((0.998-0.999)^2 + (1.0-0.999)^2) / 2) = sqrt(2e-6/2) = 0.001
    assert!((w.std - 0.001).abs() < 1e-6);
}

#[test]
fn compute_mean_var_single_observation_excluded() {
    // With count=1, the window should NOT appear in mean_var (needs count > 1)
    let pair_data = make_pair_data(vec![(
        "A",
        "B",
        vec![make_record("chr1", 0, 5000, 0.999)],
    )]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let stats = compute_window_mean_var(&bg, &pair_data, 0.0);
    assert!(stats.get(&(0, 5000)).is_none());
}

#[test]
fn compute_mean_var_identity_floor_applied() {
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.3)]),  // below floor
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.999)]), // above floor (but only 1)
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.5);
    let stats = compute_window_mean_var(&bg, &pair_data, 0.5);
    // Only 1 observation passes the floor, so count < 2 → excluded
    assert!(stats.get(&(0, 5000)).is_none());
}

#[test]
fn compute_mean_var_identical_values() {
    // All same identity → variance = 0 → std clamped to 1e-8
    let pair_data = make_pair_data(vec![
        ("A", "B", vec![make_record("chr1", 0, 5000, 0.999)]),
        ("C", "D", vec![make_record("chr1", 0, 5000, 0.999)]),
        ("E", "F", vec![make_record("chr1", 0, 5000, 0.999)]),
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let stats = compute_window_mean_var(&bg, &pair_data, 0.0);

    let w = stats.get(&(0, 5000)).unwrap();
    assert!((w.mean - 0.999).abs() < 1e-10);
    assert!((w.std - 1e-8).abs() < 1e-12); // clamped min
}

#[test]
fn compute_mean_var_multiple_windows() {
    let pair_data = make_pair_data(vec![
        (
            "A",
            "B",
            vec![
                make_record("chr1", 0, 5000, 0.998),
                make_record("chr1", 5000, 10000, 0.990),
            ],
        ),
        (
            "C",
            "D",
            vec![
                make_record("chr1", 0, 5000, 1.000),
                make_record("chr1", 5000, 10000, 0.980),
            ],
        ),
    ]);
    let bg = compute_window_background(&pair_data, 0.999, 0.0);
    let stats = compute_window_mean_var(&bg, &pair_data, 0.0);

    assert!(stats.contains_key(&(0, 5000)));
    assert!(stats.contains_key(&(5000, 10000)));
}

// =====================================================================
// normalize_identity_by_background tests
// =====================================================================

#[test]
fn normalize_identity_mean_maps_to_half() {
    let mut stats = HashMap::new();
    stats.insert(
        (0u64, 5000u64),
        WindowMeanVar {
            mean: 0.999,
            std: 0.001,
        },
    );

    let mut records = vec![make_record("chr1", 0, 5000, 0.999)]; // exactly at mean
    normalize_identity_by_background(&mut records, &stats);
    // z = 0 → sigmoid(0) = 0.5
    assert!((records[0].identity - 0.5).abs() < 1e-10);
}

#[test]
fn normalize_identity_above_mean_above_half() {
    let mut stats = HashMap::new();
    stats.insert(
        (0u64, 5000u64),
        WindowMeanVar {
            mean: 0.999,
            std: 0.001,
        },
    );

    let mut records = vec![make_record("chr1", 0, 5000, 1.000)]; // 1 std above mean
    normalize_identity_by_background(&mut records, &stats);
    // z = 1 → sigmoid(1) ≈ 0.731
    assert!(records[0].identity > 0.5);
    assert!((records[0].identity - 0.7310585786).abs() < 1e-6);
}

#[test]
fn normalize_identity_below_mean_below_half() {
    let mut stats = HashMap::new();
    stats.insert(
        (0u64, 5000u64),
        WindowMeanVar {
            mean: 0.999,
            std: 0.001,
        },
    );

    let mut records = vec![make_record("chr1", 0, 5000, 0.997)]; // 2 std below mean
    normalize_identity_by_background(&mut records, &stats);
    // z = -2 → sigmoid(-2) ≈ 0.119
    assert!(records[0].identity < 0.5);
    assert!((records[0].identity - 0.1192029220).abs() < 1e-6);
}

#[test]
fn normalize_identity_no_stats_unchanged() {
    let stats: HashMap<(u64, u64), WindowMeanVar> = HashMap::new();
    let mut records = vec![make_record("chr1", 0, 5000, 0.999)];
    normalize_identity_by_background(&mut records, &stats);
    // No stats for this window → identity unchanged
    assert!((records[0].identity - 0.999).abs() < 1e-10);
}

#[test]
fn normalize_identity_preserves_ordering() {
    let mut stats = HashMap::new();
    stats.insert(
        (0u64, 5000u64),
        WindowMeanVar {
            mean: 0.990,
            std: 0.005,
        },
    );

    let mut records = vec![
        make_record("chr1", 0, 5000, 0.985),
        make_record("chr1", 0, 5000, 0.990),
        make_record("chr1", 0, 5000, 0.999),
    ];
    // Clone original for ordering check
    let orig_order: Vec<f64> = records.iter().map(|r| r.identity).collect();

    normalize_identity_by_background(&mut records, &stats);

    // Original ordering preserved
    assert!(records[0].identity < records[1].identity);
    assert!(records[1].identity < records[2].identity);

    // Original values had same ordering
    assert!(orig_order[0] < orig_order[1]);
    assert!(orig_order[1] < orig_order[2]);
}

#[test]
fn normalize_identity_empty_records() {
    let mut stats = HashMap::new();
    stats.insert(
        (0u64, 5000u64),
        WindowMeanVar {
            mean: 0.999,
            std: 0.001,
        },
    );
    let mut records: Vec<IbsRecord> = vec![];
    normalize_identity_by_background(&mut records, &stats);
    assert!(records.is_empty());
}

// =====================================================================
// CLI binary integration tests (ibd-validate)
// =====================================================================

fn create_minimal_ibs_file() -> NamedTempFile {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    // 10 windows with high identity (should look IBD-like)
    for i in 0..10 {
        writeln!(
            f,
            "chr1\t{}\t{}\tHG00280#1#contig1\tHG00733#2#contig2\t0.9999",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    // 10 windows with lower identity for a different pair
    for i in 0..10 {
        writeln!(
            f,
            "chr1\t{}\t{}\tHG00280#1#contig1\tHG01928#1#contig3\t0.997",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn ibd_validate_missing_required_args() {
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.assert().failure();
}

#[test]
fn ibd_validate_nonexistent_input() {
    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg("/nonexistent/file.tsv")
        .arg("-o")
        .arg(output.path());
    cmd.assert().failure();
}

#[test]
fn ibd_validate_basic_run() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--min-len-bp")
        .arg("5000")
        .arg("--min-windows")
        .arg("3");
    cmd.assert().success();

    // Output should have a header line
    let content = std::fs::read_to_string(output.path()).unwrap();
    assert!(content.starts_with("chrom\tstart\tend\tgroup.a\tgroup.b"));
}

#[test]
fn ibd_validate_with_population() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--population")
        .arg("EUR");
    cmd.assert().success();
}

#[test]
fn ibd_validate_invalid_population() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--population")
        .arg("INVALID_POP");
    cmd.assert().failure();
}

#[test]
fn ibd_validate_with_logit_transform() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--logit-transform");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_identity_floor() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--identity-floor")
        .arg("0.5");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_distance_aware() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--distance-aware");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_background_filter() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--background-filter")
        .arg("--bg-identity-threshold")
        .arg("0.999")
        .arg("--bg-ratio-threshold")
        .arg("0.8");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_bg_normalize() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--bg-normalize");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_exclude_regions() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut exclude = NamedTempFile::new().unwrap();
    writeln!(exclude, "chr1\t0\t25000").unwrap();
    exclude.flush().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--exclude-regions")
        .arg(exclude.path());
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_states_output() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();
    let states_output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--states-output")
        .arg(states_output.path());
    cmd.assert().success();

    // States output should have header and data
    let content = std::fs::read_to_string(states_output.path()).unwrap();
    assert!(content.starts_with("group.a\tgroup.b\tstart\tend\tpredicted_state\tposterior"));
    // Should have 20 data lines (10 windows x 2 pairs)
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 20);
}

#[test]
fn ibd_validate_with_posterior_threshold() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--posterior-threshold")
        .arg("0.9");
    cmd.assert().success();
}

#[test]
fn ibd_validate_with_adaptive_transitions() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--adaptive-transitions");
    cmd.assert().success();
}

#[test]
fn ibd_validate_baum_welch_zero_iters() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--baum-welch-iters")
        .arg("0");
    cmd.assert().success();
}

#[test]
fn ibd_validate_coverage_feature_without_columns() {
    // Coverage feature enabled but IBS file has no length columns
    // Should produce a warning but not fail
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--coverage-feature");
    cmd.assert().success();
}

#[test]
fn ibd_validate_coverage_feature_with_columns() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity\tgroup.a.length\tgroup.b.length"
    )
    .unwrap();
    for i in 0..10 {
        writeln!(
            f,
            "chr1\t{}\t{}\tHG00280#1#c1\tHG00733#2#c2\t0.9999\t5000\t4900",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path())
        .arg("--coverage-feature");
    cmd.assert().success();
}

#[test]
fn ibd_validate_self_comparisons_skipped() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    // Self-comparison — should be skipped
    for i in 0..10 {
        writeln!(
            f,
            "chr1\t{}\t{}\tHG00280#1#c1\tHG00280#1#c1\t0.9999",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path());
    cmd.assert().success();

    // Output should have header only (no segments because no pairs after self-removal)
    let content = std::fs::read_to_string(output.path()).unwrap();
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 0);
}

#[test]
fn ibd_validate_too_few_windows_skipped() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    // Only 2 windows — below the 3-observation minimum
    writeln!(f, "chr1\t0\t5000\tHG00280#1#c1\tHG00733#2#c2\t0.9999").unwrap();
    writeln!(f, "chr1\t5000\t10000\tHG00280#1#c1\tHG00733#2#c2\t0.9999").unwrap();
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path())
        .arg("--min-windows")
        .arg("3");
    cmd.assert().success();

    // No segments — pair has < 3 records, skipped by process_pair
    let content = std::fs::read_to_string(output.path()).unwrap();
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 0);
}

#[test]
fn ibd_validate_missing_column_fails() {
    let mut f = NamedTempFile::new().unwrap();
    // Missing "estimated.identity" column
    writeln!(f, "chrom\tstart\tend\tgroup.a\tgroup.b").unwrap();
    writeln!(f, "chr1\t0\t5000\tA\tB").unwrap();
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path());
    cmd.assert().failure();
}

#[test]
fn ibd_validate_invalid_p_enter() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--p-enter-ibd")
        .arg("1.5"); // invalid — not in (0, 1)
    cmd.assert().failure();
}

#[test]
fn ibd_validate_combined_logit_adaptive_distance() {
    let input = create_minimal_ibs_file();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(input.path())
        .arg("-o")
        .arg(output.path())
        .arg("--logit-transform")
        .arg("--adaptive-transitions")
        .arg("--distance-aware");
    cmd.assert().success();
}

#[test]
fn ibd_validate_all_populations() {
    let input = create_minimal_ibs_file();
    for pop in &["AFR", "EUR", "EAS", "CSA", "AMR", "InterPop", "Generic"] {
        let output = NamedTempFile::new().unwrap();
        let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
        cmd.arg("-i")
            .arg(input.path())
            .arg("-o")
            .arg(output.path())
            .arg("--population")
            .arg(pop);
        cmd.assert().success();
    }
}

#[test]
fn ibd_validate_empty_file_fails() {
    let f = NamedTempFile::new().unwrap();
    let output = NamedTempFile::new().unwrap();

    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path());
    cmd.assert().failure();
}

#[test]
fn ibd_validate_header_only_produces_no_segments() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path());
    cmd.assert().success();

    let content = std::fs::read_to_string(output.path()).unwrap();
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 0);
}

#[test]
fn ibd_validate_coord_stripping_in_pair_key() {
    // Two records with different coordinate suffixes but same base haplotype
    // Should be treated as same pair
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    for i in 0..5 {
        writeln!(
            f,
            "chr1\t{}\t{}\tHG00280#1#c1:{}-{}\tHG00733#2#c2:{}-{}\t0.9999",
            i * 5000,
            (i + 1) * 5000,
            i * 5000,
            (i + 1) * 5000,
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let states_output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path())
        .arg("--states-output")
        .arg(states_output.path())
        .arg("--min-windows")
        .arg("3")
        .arg("--min-len-bp")
        .arg("1");
    cmd.assert().success();

    // States should use stripped IDs (no coordinate suffix)
    let content = std::fs::read_to_string(states_output.path()).unwrap();
    let first_data = content.lines().nth(1).unwrap();
    assert!(first_data.starts_with("HG00280#1#c1\tHG00733#2#c2"));
}

#[test]
fn ibd_validate_invalid_data_lines_skipped() {
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    writeln!(f, "chr1\tabc\t5000\tA#1\tB#2\t0.999").unwrap(); // invalid start
    writeln!(f, "chr1\t0\txyz\tA#1\tB#2\t0.999").unwrap(); // invalid end
    writeln!(f, "chr1\t0\t5000\tA#1\tB#2\tnot_a_number").unwrap(); // invalid identity
    // Too few fields:
    writeln!(f, "chr1\t0\t5000").unwrap();
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path());
    cmd.assert().success();

    // Should produce no segments (all lines invalid or skipped)
    let content = std::fs::read_to_string(output.path()).unwrap();
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 0);
}

#[test]
fn ibd_validate_pair_ordering_canonical() {
    // Test that pair (B, A) is treated the same as (A, B)
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity"
    )
    .unwrap();
    // First 5 windows: A→B order
    for i in 0..5 {
        writeln!(
            f,
            "chr1\t{}\t{}\tAAA#1\tBBB#2\t0.9999",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    // Next 5 windows: B→A order (should merge into same pair)
    for i in 5..10 {
        writeln!(
            f,
            "chr1\t{}\t{}\tBBB#2\tAAA#1\t0.9999",
            i * 5000,
            (i + 1) * 5000
        )
        .unwrap();
    }
    f.flush().unwrap();

    let output = NamedTempFile::new().unwrap();
    let states_output = NamedTempFile::new().unwrap();
    let mut cmd = Command::cargo_bin("ibd-validate").unwrap();
    cmd.arg("-i")
        .arg(f.path())
        .arg("-o")
        .arg(output.path())
        .arg("--states-output")
        .arg(states_output.path());
    cmd.assert().success();

    // Should have 10 windows for the single canonical pair
    let content = std::fs::read_to_string(states_output.path()).unwrap();
    let data_lines = content.lines().skip(1).count();
    assert_eq!(data_lines, 10);
}
