use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use clap::Parser;
use hprc_common::{ColumnIndices, Region};
use rayon::prelude::*;

fn validate_cutoff(val: &str) -> Result<f64, String> {
    let v: f64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("cutoff must be in [0.0, 1.0], got {}", v))
    }
}

fn validate_positive_u64(val: &str) -> Result<u64, String> {
    let v: u64 = val.parse().map_err(|_| format!("'{}' is not a valid number", val))?;
    if v > 0 {
        Ok(v)
    } else {
        Err("value must be > 0 (window_size=0 causes infinite loop)".to_string())
    }
}

/// ibs: wrapper around `impg similarity` to obtain IBS segments.
///
/// Pipeline:
///   1. Slide a window across a reference chromosome.
///   2. Run `impg similarity` in each window.
///   3. For each window, immediately:
///        - filter rows by estimated.identity >= cutoff
///        - drop self-self and ref-involving comparisons
///        - drop duplicated A–B / B–A (keep canonical order)
///        - reduce to: chrom, start, end, group.a, group.b, estimated.identity
///        - append to output (streaming)
#[derive(Parser, Debug)]
#[command(name = "ibs", version, about)]
struct Args {
    /// Sequence file(s) for impg (e.g. .agc)
    #[arg(long = "sequence-files", required = true)]
    sequence_files: String,

    /// Alignment file (.paf/.paf.gz/.1aln) [passed to impg as -p]
    #[arg(short = 'a', required = true)]
    align: String,

    /// Reference name (e.g. CHM13)
    #[arg(short = 'r', required = true)]
    ref_name: String,

    /// Region, e.g. chr1:1-248956422 or chr1
    #[arg(long = "region", required_unless_present = "bed")]
    region: Option<String>,

    /// BED file with regions to process (chrom, start, end, tab-separated)
    /// Enables batch processing of multiple regions in a single run
    #[arg(long = "bed", conflicts_with = "region")]
    bed: Option<String>,

    /// Window size in bp (must be > 0; window_size=0 causes infinite loop)
    #[arg(long = "size", required = true, value_parser = validate_positive_u64)]
    window_size: u64,

    /// Haplotypes to compare (e.g. ibs_example.txt)
    #[arg(long = "subset-sequence-list", required = true)]
    subset_list: String,

    /// Output file
    #[arg(long = "output", required = true)]
    output: String,

    /// Cutoff on estimated.identity (must be in [0.0, 1.0], default: 0.999 to account for sequencing errors)
    #[arg(short = 'c', default_value = "0.999", value_parser = validate_cutoff)]
    cutoff: f64,

    /// Total length of REGION if you use -region chr1 (without coordinates)
    #[arg(long = "region-length")]
    region_length: Option<u64>,

    /// Number of threads for parallel processing (default: auto-detect)
    #[arg(short = 't', long = "threads")]
    threads: Option<usize>,

    /// File with query sample names (one per line). When used with --ref-samples,
    /// only emits pairs where one haplotype is from a query sample and the other
    /// from a reference sample. This gives ~3-5x speedup for ancestry workflows.
    #[arg(long = "query-samples")]
    query_samples: Option<String>,

    /// File with reference sample names (one per line). Used together with
    /// --query-samples to filter output to only query-vs-reference pairs.
    #[arg(long = "ref-samples")]
    ref_samples: Option<String>,
}

/// Extract sample name from a haplotype ID like "HG01175#1" → "HG01175".
/// If no '#' is found, returns the entire string.
fn extract_sample_name(hap_id: &str) -> &str {
    hap_id.split('#').next().unwrap_or(hap_id)
}

/// Load sample names from a file (one per line, ignoring blank lines and comments).
fn load_sample_set(path: &str) -> Result<HashSet<String>> {
    let file = File::open(path)
        .context(format!("Failed to open sample file: {}", path))?;
    let reader = BufReader::new(file);
    let mut samples = HashSet::new();
    for line in reader.lines() {
        let line = line.context("Failed to read sample file")?;
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            samples.insert(trimmed.to_string());
        }
    }
    Ok(samples)
}

/// Check if a pair passes the query/ref filter.
/// Returns true if:
///   - No filter is active (both sets are None), OR
///   - Only query_set is provided: at least one haplotype is from a query sample, OR
///   - Only ref_set is provided: at least one haplotype is from a ref sample, OR
///   - Both are provided: one is query and the other is ref (cross-set pair)
fn passes_sample_filter(
    group_a: &str,
    group_b: &str,
    query_set: Option<&HashSet<String>>,
    ref_set: Option<&HashSet<String>>,
) -> bool {
    match (query_set, ref_set) {
        (None, None) => true,
        (Some(qs), None) => {
            let sa = extract_sample_name(group_a);
            let sb = extract_sample_name(group_b);
            qs.contains(sa) || qs.contains(sb)
        }
        (None, Some(rs)) => {
            let sa = extract_sample_name(group_a);
            let sb = extract_sample_name(group_b);
            rs.contains(sa) || rs.contains(sb)
        }
        (Some(qs), Some(rs)) => {
            let sa = extract_sample_name(group_a);
            let sb = extract_sample_name(group_b);
            // Cross-set: one in query AND other in ref
            (qs.contains(sa) && rs.contains(sb)) || (qs.contains(sb) && rs.contains(sa))
        }
    }
}

/// Filtered row from impg output
#[derive(Clone)]
struct FilteredRow {
    chrom: String,
    start: String,
    end: String,
    group_a: String,
    group_b: String,
    identity: f64,
}

/// Process a single window and return filtered results
fn process_window(
    args: &Args,
    region: &Region,
    window_start: u64,
    window_end: u64,
    query_set: Option<&HashSet<String>>,
    ref_set: Option<&HashSet<String>>,
) -> Result<Vec<FilteredRow>> {
    let ref_region = format!(
        "{}#0#{}:{}-{}",
        args.ref_name, region.chrom, window_start, window_end
    );

    eprintln!("Processing window {}", ref_region);

    // Run impg similarity
    let mut child = Command::new("impg")
        .arg("similarity")
        .arg("--sequence-files")
        .arg(&args.sequence_files)
        .arg("-a")
        .arg(&args.align)
        .arg("-r")
        .arg(&ref_region)
        .arg("--subset-sequence-list")
        .arg(&args.subset_list)
        .arg("--force-large-region")
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to spawn impg process. Is 'impg' in PATH?")?;

    let stdout = child.stdout.take()
        .context("Failed to capture stdout from impg")?;
    let reader = BufReader::new(stdout);

    let mut lines = reader.lines();

    // Parse header
    let header_line = lines.next()
        .context("impg produced no output")?
        .context("Failed to read header line")?;

    let cols = ColumnIndices::from_header(&header_line)
        .context("Failed to parse impg header")?;

    let ref_prefix = format!("{}#", args.ref_name);
    let mut results = Vec::new();

    // Process data rows
    for line_result in lines {
        let line = line_result.context("Failed to read line from impg output")?;
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() <= cols.max_index() {
            continue; // Skip malformed lines
        }

        // Parse estimated identity and apply cutoff
        let identity: f64 = match fields[cols.estimated_identity].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        if identity < args.cutoff {
            continue;
        }

        let group_a = fields[cols.group_a];
        let group_b = fields[cols.group_b];

        // Skip self-self comparisons
        if group_a == group_b {
            continue;
        }

        // Skip comparisons involving the reference
        if group_a.starts_with(&ref_prefix) || group_b.starts_with(&ref_prefix) {
            continue;
        }

        // Keep only canonical order (A < B lexicographically)
        if group_a > group_b {
            continue;
        }

        // Apply query/ref sample filter
        if !passes_sample_filter(group_a, group_b, query_set, ref_set) {
            continue;
        }

        // Collect the filtered row
        results.push(FilteredRow {
            chrom: fields[cols.chrom].to_string(),
            start: fields[cols.start].to_string(),
            end: fields[cols.end].to_string(),
            group_a: group_a.to_string(),
            group_b: group_b.to_string(),
            identity,
        });
    }

    // Wait for child process to complete
    let status = child.wait().context("Failed to wait for impg process")?;
    if !status.success() {
        bail!("impg process exited with status: {}", status);
    }

    Ok(results)
}

/// Parse regions from a BED file (chrom, start, end, tab-separated).
fn parse_bed_regions(bed_path: &str) -> Result<Vec<Region>> {
    let file = File::open(bed_path)
        .context(format!("Failed to open BED file: {}", bed_path))?;
    let reader = BufReader::new(file);
    let mut regions = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.context("Failed to read BED file")?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            bail!("BED line {} has fewer than 3 fields: {}", line_num + 1, line);
        }
        let chrom = fields[0].to_string();
        let start: u64 = fields[1].parse()
            .context(format!("Invalid start coordinate on BED line {}", line_num + 1))?;
        let end: u64 = fields[2].parse()
            .context(format!("Invalid end coordinate on BED line {}", line_num + 1))?;

        // BED is 0-based half-open; Region is 1-based inclusive
        regions.push(Region { chrom, start: start + 1, end });
    }

    Ok(regions)
}

/// Process a single region and return all window results.
fn process_region(
    args: &Args,
    region: &Region,
    query_set: Option<&HashSet<String>>,
    ref_set: Option<&HashSet<String>>,
) -> Result<Vec<FilteredRow>> {
    let mut windows = Vec::new();
    let mut start_pos = region.start;
    while start_pos <= region.end {
        let end_pos = (start_pos + args.window_size - 1).min(region.end);
        windows.push((start_pos, end_pos));
        start_pos = end_pos + 1;
    }

    let error_count = std::sync::atomic::AtomicUsize::new(0);
    let total_windows = windows.len();

    let all_results: Vec<Vec<FilteredRow>> = windows
        .par_iter()
        .filter_map(|(start, end)| {
            match process_window(args, region, *start, *end, query_set, ref_set) {
                Ok(rows) => Some(rows),
                Err(e) => {
                    eprintln!("WARNING: skipping window {}:{}-{}: {}", region.chrom, start, end, e);
                    error_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        })
        .collect();

    let errors = error_count.load(std::sync::atomic::Ordering::Relaxed);
    if errors > 0 {
        eprintln!("WARNING: {} of {} windows failed processing in region {}",
            errors, total_windows, region.chrom);
        if total_windows > 0 && errors * 2 > total_windows {
            bail!("Too many window failures ({}/{}) in region {}", errors, total_windows, region.chrom);
        }
    }

    Ok(all_results.into_iter().flatten().collect())
}

fn run() -> Result<()> {
    let args = Args::parse();

    // Validate input files exist
    if !Path::new(&args.sequence_files).exists() {
        bail!("sequence-files does not exist: {}", args.sequence_files);
    }
    if !Path::new(&args.align).exists() {
        bail!("alignment file does not exist: {}", args.align);
    }
    if !Path::new(&args.subset_list).exists() {
        bail!("subset-sequence-list does not exist: {}", args.subset_list);
    }

    // Configure thread pool if --threads is specified
    if let Some(num_threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .context("Failed to configure thread pool")?;
        eprintln!("Using {} threads for parallel processing", num_threads);
    }

    // Load query/ref sample sets if provided
    let query_set = match &args.query_samples {
        Some(path) => {
            if !Path::new(path).exists() {
                bail!("query-samples file does not exist: {}", path);
            }
            let set = load_sample_set(path)?;
            eprintln!("Loaded {} query samples from {}", set.len(), path);
            Some(set)
        }
        None => None,
    };
    let ref_set = match &args.ref_samples {
        Some(path) => {
            if !Path::new(path).exists() {
                bail!("ref-samples file does not exist: {}", path);
            }
            let set = load_sample_set(path)?;
            eprintln!("Loaded {} reference samples from {}", set.len(), path);
            Some(set)
        }
        None => None,
    };

    if let (Some(qs), Some(rs)) = (&query_set, &ref_set) {
        let nq = qs.len();
        let nr = rs.len();
        // Each sample has 2 haplotypes, so max cross-pairs = 2*nq * 2*nr
        eprintln!(
            "Query-ref mode: {} query × {} ref samples → max {} cross-haplotype pairs (vs {} all-pairs)",
            nq, nr, 2 * nq * 2 * nr,
            {
                let total = 2 * (nq + nr);
                total * (total - 1) / 2
            }
        );
    }

    // Check that impg is available
    if Command::new("impg").arg("--version").output().is_err() {
        bail!("'impg' is not in PATH");
    }

    // Collect regions: either from --region or --bed
    let regions = if let Some(ref bed_path) = args.bed {
        if !Path::new(bed_path).exists() {
            bail!("BED file does not exist: {}", bed_path);
        }
        parse_bed_regions(bed_path)?
    } else if let Some(ref region_str) = args.region {
        vec![Region::parse(region_str, args.region_length)?]
    } else {
        bail!("Either --region or --bed must be specified");
    };

    eprintln!("Processing {} region(s)...", regions.len());

    // Open output file and write header
    let output_file = File::create(&args.output)
        .context(format!("Failed to create output file: {}", args.output))?;
    let mut output = BufWriter::new(output_file);
    writeln!(output, "chrom\tstart\tend\tgroup.a\tgroup.b\testimated.identity")?;

    // Process each region
    for (i, region) in regions.iter().enumerate() {
        eprintln!("[{}/{}] Processing {}:{}-{}", i + 1, regions.len(),
            region.chrom, region.start, region.end);

        let region_results = process_region(
            &args,
            region,
            query_set.as_ref(),
            ref_set.as_ref(),
        )?;

        for row in &region_results {
            writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}",
                row.chrom, row.start, row.end, row.group_a, row.group_b, row.identity
            )?;
        }
    }

    output.flush()?;
    eprintln!("IBS written to: {}", args.output);

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("ERROR: {:#}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // validate_cutoff
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_cutoff_zero() {
        assert_eq!(validate_cutoff("0.0").unwrap(), 0.0);
    }

    #[test]
    fn test_validate_cutoff_one() {
        assert_eq!(validate_cutoff("1.0").unwrap(), 1.0);
    }

    #[test]
    fn test_validate_cutoff_mid() {
        assert_eq!(validate_cutoff("0.5").unwrap(), 0.5);
    }

    #[test]
    fn test_validate_cutoff_negative() {
        assert!(validate_cutoff("-0.1").is_err());
    }

    #[test]
    fn test_validate_cutoff_above_one() {
        assert!(validate_cutoff("1.1").is_err());
    }

    #[test]
    fn test_validate_cutoff_not_a_number() {
        assert!(validate_cutoff("abc").is_err());
    }

    #[test]
    fn test_validate_cutoff_empty() {
        assert!(validate_cutoff("").is_err());
    }

    #[test]
    fn test_validate_cutoff_scientific_notation() {
        assert_eq!(validate_cutoff("1e-5").unwrap(), 1e-5);
    }

    // -----------------------------------------------------------------------
    // validate_positive_u64
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_positive_u64_one() {
        assert_eq!(validate_positive_u64("1").unwrap(), 1);
    }

    #[test]
    fn test_validate_positive_u64_large() {
        assert_eq!(validate_positive_u64("10000").unwrap(), 10000);
    }

    #[test]
    fn test_validate_positive_u64_zero() {
        assert!(validate_positive_u64("0").is_err());
    }

    #[test]
    fn test_validate_positive_u64_negative() {
        assert!(validate_positive_u64("-1").is_err());
    }

    #[test]
    fn test_validate_positive_u64_not_a_number() {
        assert!(validate_positive_u64("abc").is_err());
    }

    #[test]
    fn test_validate_positive_u64_float() {
        assert!(validate_positive_u64("1.5").is_err());
    }

    #[test]
    fn test_validate_positive_u64_empty() {
        assert!(validate_positive_u64("").is_err());
    }

    // -----------------------------------------------------------------------
    // parse_bed_regions
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_bed_basic() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "chr1\t100\t200\nchr2\t300\t400\n").unwrap();

        let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].chrom, "chr1");
        assert_eq!(regions[0].start, 101); // BED 0-based → 1-based
        assert_eq!(regions[0].end, 200);
        assert_eq!(regions[1].chrom, "chr2");
        assert_eq!(regions[1].start, 301);
        assert_eq!(regions[1].end, 400);
    }

    #[test]
    fn test_parse_bed_with_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "# header\nchr1\t0\t1000\n\nchr2\t500\t600\n").unwrap();

        let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_parse_bed_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "").unwrap();

        let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn test_parse_bed_too_few_fields() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "chr1\t100\n").unwrap();

        assert!(parse_bed_regions(bed.to_str().unwrap()).is_err());
    }

    #[test]
    fn test_parse_bed_nonexistent_file() {
        assert!(parse_bed_regions("/nonexistent/path/file.bed").is_err());
    }

    #[test]
    fn test_parse_bed_extra_columns() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "chr1\t100\t200\tname\t0\t+\n").unwrap();

        let regions = parse_bed_regions(bed.to_str().unwrap()).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start, 101);
        assert_eq!(regions[0].end, 200);
    }

    #[test]
    fn test_parse_bed_invalid_start_coordinate() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "chr1\tabc\t200\n").unwrap();

        assert!(parse_bed_regions(bed.to_str().unwrap()).is_err());
    }

    #[test]
    fn test_parse_bed_invalid_end_coordinate() {
        let dir = tempfile::tempdir().unwrap();
        let bed = dir.path().join("test.bed");
        std::fs::write(&bed, "chr1\t100\txyz\n").unwrap();

        assert!(parse_bed_regions(bed.to_str().unwrap()).is_err());
    }

    // -----------------------------------------------------------------------
    // extract_sample_name
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_sample_name_with_haplotype() {
        assert_eq!(extract_sample_name("HG01175#1"), "HG01175");
    }

    #[test]
    fn test_extract_sample_name_with_hap2() {
        assert_eq!(extract_sample_name("HG01175#2"), "HG01175");
    }

    #[test]
    fn test_extract_sample_name_no_hash() {
        assert_eq!(extract_sample_name("HG01175"), "HG01175");
    }

    #[test]
    fn test_extract_sample_name_with_coords() {
        // impg can output "HG01175#1#chr1:100-200"
        assert_eq!(extract_sample_name("HG01175#1#chr1:100-200"), "HG01175");
    }

    #[test]
    fn test_extract_sample_name_empty() {
        assert_eq!(extract_sample_name(""), "");
    }

    // -----------------------------------------------------------------------
    // load_sample_set
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_sample_set_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "HG01175\nHG02257\nNA19239\n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 3);
        assert!(set.contains("HG01175"));
        assert!(set.contains("HG02257"));
        assert!(set.contains("NA19239"));
    }

    #[test]
    fn test_load_sample_set_with_blanks_and_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "# header\nHG01175\n\n# another comment\nNA19239\n  \n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains("HG01175"));
        assert!(set.contains("NA19239"));
    }

    #[test]
    fn test_load_sample_set_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_load_sample_set_nonexistent() {
        assert!(load_sample_set("/nonexistent/samples.txt").is_err());
    }

    #[test]
    fn test_load_sample_set_trims_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "  HG01175  \n\tNA19239\t\n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 2);
        assert!(set.contains("HG01175"));
        assert!(set.contains("NA19239"));
    }

    // -----------------------------------------------------------------------
    // passes_sample_filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_no_sets_passes() {
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", None, None));
    }

    #[test]
    fn test_filter_query_only_passes_when_a_in_query() {
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_query_only_passes_when_b_in_query() {
        let qs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_query_only_rejects_neither() {
        let qs: HashSet<String> = ["HG00099"].iter().map(|s| s.to_string()).collect();
        assert!(!passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_ref_only_passes_when_in_ref() {
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", None, Some(&rs)));
    }

    #[test]
    fn test_filter_ref_only_rejects_neither() {
        let rs: HashSet<String> = ["HG00099"].iter().map(|s| s.to_string()).collect();
        assert!(!passes_sample_filter("HG01175#1", "NA19239#2", None, Some(&rs)));
    }

    #[test]
    fn test_filter_both_cross_pair_passes() {
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        // a in query, b in ref → pass
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_both_cross_pair_reversed() {
        let qs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        // a in ref, b in query → also pass
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_both_rejects_same_set_pair() {
        let qs: HashSet<String> = ["HG01175", "HG02257"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        // Both a and b in query → reject (not a cross-set pair)
        assert!(!passes_sample_filter("HG01175#1", "HG02257#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_both_rejects_unknown_samples() {
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        // Neither in any set → reject
        assert!(!passes_sample_filter("HG00099#1", "HG00100#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_both_rejects_both_in_ref() {
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239", "HG02257"].iter().map(|s| s.to_string()).collect();
        // Both in ref set → reject
        assert!(!passes_sample_filter("NA19239#1", "HG02257#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_sample_in_both_sets() {
        // Edge case: sample appears in both query and ref
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["HG01175", "NA19239"].iter().map(|s| s.to_string()).collect();
        // a=HG01175 (in both), b=NA19239 (in ref) → passes because a∈query, b∈ref
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_with_coord_suffix() {
        // Haplotype IDs might have coordinate suffixes stripped already,
        // but the sample name extraction should still work
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1#chr1:100-200", "NA19239#2", Some(&qs), Some(&rs)));
    }

    // -----------------------------------------------------------------------
    // Additional edge cases for passes_sample_filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_empty_query_set_rejects_all() {
        let qs: HashSet<String> = HashSet::new();
        // Empty query set → no sample matches → reject
        assert!(!passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_empty_ref_set_rejects_all() {
        let rs: HashSet<String> = HashSet::new();
        assert!(!passes_sample_filter("HG01175#1", "NA19239#2", None, Some(&rs)));
    }

    #[test]
    fn test_filter_empty_both_sets_rejects_all() {
        let qs: HashSet<String> = HashSet::new();
        let rs: HashSet<String> = HashSet::new();
        assert!(!passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_same_sample_both_haplotypes_query_only() {
        // Both haplotypes from same sample, sample in query → passes
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "HG01175#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_same_sample_both_haplotypes_cross_set() {
        // Both haplotypes from same sample: if in query set, need other in ref set
        // Here both haps are from same sample (HG01175, in query), no one in ref
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        // HG01175#1 in query, HG01175#2 NOT in ref → reject (not cross-set)
        assert!(!passes_sample_filter("HG01175#1", "HG01175#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_same_sample_in_both_sets_self_pair() {
        // Sample in both query AND ref: self-pairing should pass (a∈query, b∈ref)
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "HG01175#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_query_only_both_in_query() {
        // Both samples in query set → at least one matches → pass
        let qs: HashSet<String> = ["HG01175", "NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", Some(&qs), None));
    }

    #[test]
    fn test_filter_ref_only_both_in_ref() {
        // Both samples in ref set → at least one matches → pass
        let rs: HashSet<String> = ["HG01175", "NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(passes_sample_filter("HG01175#1", "NA19239#2", None, Some(&rs)));
    }

    #[test]
    fn test_filter_one_query_one_unknown_cross_set() {
        // a in query, b unknown → neither b∈ref → cross-set fails
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(!passes_sample_filter("HG01175#1", "HG00099#2", Some(&qs), Some(&rs)));
    }

    #[test]
    fn test_filter_one_ref_one_unknown_cross_set() {
        // a unknown, b in ref → a not in query → cross-set fails
        let qs: HashSet<String> = ["HG01175"].iter().map(|s| s.to_string()).collect();
        let rs: HashSet<String> = ["NA19239"].iter().map(|s| s.to_string()).collect();
        assert!(!passes_sample_filter("HG00099#1", "NA19239#2", Some(&qs), Some(&rs)));
    }

    // -----------------------------------------------------------------------
    // load_sample_set: additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_sample_set_duplicates_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "HG01175\nHG01175\nNA19239\nNA19239\nNA19239\n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 2, "Duplicates should be deduplicated");
        assert!(set.contains("HG01175"));
        assert!(set.contains("NA19239"));
    }

    #[test]
    fn test_load_sample_set_comment_after_hash() {
        // Lines starting with # are comments; sample names like HG01175 shouldn't be confused
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "#HG01175\nNA19239\n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 1);
        assert!(!set.contains("HG01175"), "#HG01175 is a comment line");
        assert!(set.contains("NA19239"));
    }

    #[test]
    fn test_load_sample_set_only_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.txt");
        std::fs::write(&path, "\n\n  \n\t\n").unwrap();

        let set = load_sample_set(path.to_str().unwrap()).unwrap();
        assert_eq!(set.len(), 0);
    }

    // -----------------------------------------------------------------------
    // extract_sample_name: additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_sample_name_just_hash() {
        // Edge case: string is just "#"
        assert_eq!(extract_sample_name("#"), "");
    }

    #[test]
    fn test_extract_sample_name_hash_at_end() {
        assert_eq!(extract_sample_name("HG01175#"), "HG01175");
    }

    #[test]
    fn test_extract_sample_name_multiple_hashes() {
        // "NA12878#1#chr20:1000-2000#extra" → sample name is "NA12878"
        assert_eq!(extract_sample_name("NA12878#1#chr20:1000-2000#extra"), "NA12878");
    }
}
