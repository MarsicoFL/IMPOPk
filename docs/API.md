# HPRC-IBD Library API Documentation

This document describes the public API of the `hprc-ibd` Rust library, which provides tools for Identity-By-State (IBS) and Identity-By-Descent (IBD) analysis in pangenome data.

## Overview

The library is organized into the following modules:

| Module | Purpose |
|--------|---------|
| `hprc_ibd` (root) | Core types: `Region`, `Window`, `WindowIterator`, error handling |
| `hprc_ibd::hmm` | Hidden Markov Model for IBD state inference |
| `hprc_ibd::segment` | Segment detection and merging algorithms |
| `hprc_ibd::stats` | Statistical utilities (Gaussian, k-means, online stats) |

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hprc-ibd = { path = "path/to/production/ibs-cli" }
```

Or if published to crates.io:

```toml
[dependencies]
hprc-ibd = "0.1"
```

---

## Core Module (`hprc_ibd`)

### Error Types

```rust
use hprc_ibd::{IbdError, Result};

/// Custom error types for the library
pub enum IbdError {
    /// IO error wrapper
    Io(std::io::Error),

    /// Parse error with message
    Parse(String),

    /// Invalid region format
    InvalidRegion(String),

    /// Missing required column in data
    MissingColumn(String),

    /// External tool (e.g., impg) error
    ExternalTool(String),

    /// Invalid parameter value
    InvalidParameter(String),
}

/// Convenience type alias
pub type Result<T> = std::result::Result<T, IbdError>;
```

### Region

Represents a genomic region for analysis.

```rust
use hprc_ibd::Region;

/// Genomic region specification
pub struct Region {
    pub chrom: String,  // Chromosome name (e.g., "chr1")
    pub start: u64,     // Start position (1-based)
    pub end: u64,       // End position (inclusive)
}
```

#### Methods

```rust
impl Region {
    /// Parse a region string like "chr1:1-1000000" or "chr1"
    ///
    /// # Arguments
    /// * `region_str` - Region string to parse
    /// * `region_length` - Required if region_str omits coordinates
    ///
    /// # Examples
    /// ```rust
    /// use hprc_ibd::Region;
    ///
    /// // With coordinates
    /// let region = Region::parse("chr1:1000-2000", None).unwrap();
    /// assert_eq!(region.chrom, "chr1");
    /// assert_eq!(region.start, 1000);
    /// assert_eq!(region.end, 2000);
    ///
    /// // Without coordinates (requires length)
    /// let region = Region::parse("chr1", Some(248956422)).unwrap();
    /// assert_eq!(region.start, 1);
    /// assert_eq!(region.end, 248956422);
    /// ```
    pub fn parse(region_str: &str, region_length: Option<u64>) -> Result<Self>;

    /// Format region for impg reference specification
    ///
    /// # Arguments
    /// * `ref_name` - Reference name (e.g., "CHM13")
    ///
    /// # Returns
    /// String in format "REF#0#chrom:start-end"
    ///
    /// # Example
    /// ```rust
    /// use hprc_ibd::Region;
    ///
    /// let region = Region::parse("chr1:1000-2000", None).unwrap();
    /// let impg_ref = region.to_impg_ref("CHM13");
    /// assert_eq!(impg_ref, "CHM13#0#chr1:1000-2000");
    /// ```
    pub fn to_impg_ref(&self, ref_name: &str) -> String;
}
```

### Window

Represents a single analysis window within a region.

```rust
use hprc_ibd::Window;

/// Window specification for sliding window analysis
pub struct Window {
    pub start: u64,  // Window start position
    pub end: u64,    // Window end position
}

impl Window {
    /// Create a new window
    pub fn new(start: u64, end: u64) -> Self;

    /// Calculate window length in base pairs
    /// Returns end - start + 1
    pub fn length(&self) -> u64;
}
```

### WindowIterator

Iterator that generates windows across a genomic region.

```rust
use hprc_ibd::{Region, WindowIterator};

/// Iterator over windows in a region
pub struct WindowIterator {
    // Internal fields...
}

impl WindowIterator {
    /// Create a new window iterator
    ///
    /// # Arguments
    /// * `region` - The genomic region to iterate over
    /// * `window_size` - Size of each window in base pairs
    ///
    /// # Panics
    /// Panics if `window_size` is 0 (would cause infinite loop)
    ///
    /// # Example
    /// ```rust
    /// use hprc_ibd::{Region, WindowIterator};
    ///
    /// let region = Region { chrom: "chr1".to_string(), start: 1, end: 10000 };
    /// let windows: Vec<_> = WindowIterator::new(&region, 5000).collect();
    /// assert_eq!(windows.len(), 2);
    /// ```
    pub fn new(region: &Region, window_size: u64) -> Self;
}

impl Iterator for WindowIterator {
    type Item = Window;
    fn next(&mut self) -> Option<Self::Item>;
}
```

---

## HMM Module (`hprc_ibd::hmm`)

### HmmParams

Parameters for the two-state Hidden Markov Model.

```rust
use hprc_ibd::hmm::HmmParams;
use hprc_ibd::stats::GaussianParams;

/// HMM parameters for IBD calling
pub struct HmmParams {
    /// Initial state probabilities [P(non-IBD), P(IBD)]
    pub initial: [f64; 2],

    /// Transition matrix
    /// transition[from][to] = P(state_to | state_from)
    /// transition[0][0] = P(stay in non-IBD)
    /// transition[0][1] = P(enter IBD)
    /// transition[1][0] = P(exit IBD)
    /// transition[1][1] = P(stay in IBD)
    pub transition: [[f64; 2]; 2],

    /// Emission distributions for each state
    /// emission[0] = non-IBD Gaussian
    /// emission[1] = IBD Gaussian
    pub emission: [GaussianParams; 2],
}
```

#### Methods

```rust
impl HmmParams {
    /// Create HMM parameters from expected IBD segment length
    ///
    /// # Arguments
    /// * `expected_ibd_windows` - Expected number of windows per IBD segment
    /// * `p_enter_ibd` - Probability of transitioning into IBD state
    ///
    /// # Panics
    /// Panics if `p_enter_ibd` is not in range (0, 1)
    ///
    /// # Example
    /// ```rust
    /// use hprc_ibd::hmm::HmmParams;
    ///
    /// // Expect IBD segments of ~50 windows, rare IBD events
    /// let params = HmmParams::from_expected_length(50.0, 0.0001);
    /// ```
    pub fn from_expected_length(expected_ibd_windows: f64, p_enter_ibd: f64) -> Self;

    /// Estimate emission parameters from observed data using k-means
    ///
    /// # Arguments
    /// * `observations` - Sequence identity values
    ///
    /// # Notes
    /// - Requires at least 3 observations
    /// - Automatically handles low-variance data
    /// - Updates emission[0] and emission[1] in place
    ///
    /// # Example
    /// ```rust
    /// use hprc_ibd::hmm::HmmParams;
    ///
    /// let mut params = HmmParams::from_expected_length(50.0, 0.0001);
    /// let observations = vec![0.5, 0.6, 0.4, 0.999, 0.998, 0.995];
    /// params.estimate_emissions(&observations);
    /// ```
    pub fn estimate_emissions(&mut self, observations: &[f64]);
}
```

### Viterbi Algorithm

```rust
use hprc_ibd::hmm::{viterbi, HmmParams};

/// Run Viterbi algorithm to find most likely state sequence
///
/// # Arguments
/// * `observations` - Sequence of identity observations
/// * `params` - HMM parameters
///
/// # Returns
/// Vector of states (0 = non-IBD, 1 = IBD) for each observation
///
/// # Example
/// ```rust
/// use hprc_ibd::hmm::{viterbi, HmmParams};
///
/// let params = HmmParams::from_expected_length(10.0, 0.001);
/// let observations = vec![0.5, 0.6, 0.99, 0.995, 0.998, 0.5, 0.4];
/// let states = viterbi(&observations, &params);
/// // states might be: [0, 0, 1, 1, 1, 0, 0]
/// ```
pub fn viterbi(observations: &[f64], params: &HmmParams) -> Vec<usize>;
```

### Segment Extraction

```rust
use hprc_ibd::hmm::extract_ibd_segments;

/// Extract IBD segments from state sequence
///
/// # Arguments
/// * `states` - State sequence from viterbi()
///
/// # Returns
/// Vector of tuples: (start_index, end_index, n_windows)
///
/// # Example
/// ```rust
/// use hprc_ibd::hmm::extract_ibd_segments;
///
/// let states = vec![0, 0, 1, 1, 1, 0, 0, 1, 1, 0];
/// let segments = extract_ibd_segments(&states);
/// // segments = [(2, 4, 3), (7, 8, 2)]
/// // Two IBD segments: windows 2-4 (3 windows) and 7-8 (2 windows)
/// ```
pub fn extract_ibd_segments(states: &[usize]) -> Vec<(usize, usize, usize)>;
```

---

## Segment Module (`hprc_ibd::segment`)

### Segment

Represents a detected IBS/IBD segment.

```rust
use hprc_ibd::segment::Segment;

/// Represents an IBS/IBD segment
pub struct Segment {
    pub chrom: String,          // Chromosome
    pub start: u64,             // Start position (bp)
    pub end: u64,               // End position (bp)
    pub hap_a: String,          // First haplotype ID
    pub hap_b: String,          // Second haplotype ID
    pub n_windows: usize,       // Number of windows in segment
    pub mean_identity: f64,     // Average identity across segment
    pub min_identity: f64,      // Minimum identity observed
    pub identity_sum: f64,      // Sum of identities (for averaging)
    pub n_called: usize,        // Number of windows with data
}

impl Segment {
    /// Calculate segment length in base pairs
    pub fn length_bp(&self) -> u64;

    /// Calculate fraction of windows with identity calls
    pub fn fraction_called(&self) -> f64;
}
```

### RleParams

Parameters for RLE-based segment detection.

```rust
use hprc_ibd::segment::RleParams;

/// Parameters for RLE-based segment calling
pub struct RleParams {
    /// Minimum identity threshold (default: 0.9995)
    pub min_identity: f64,

    /// Maximum gap (missing windows) to bridge (default: 1)
    pub max_gap: usize,

    /// Minimum windows per segment (default: 3)
    pub min_windows: usize,

    /// Minimum segment length in bp (default: 5000)
    pub min_length_bp: u64,

    /// Tolerance for identity drops (default: 0.0)
    pub drop_tolerance: f64,
}

impl Default for RleParams {
    fn default() -> Self;
}
```

### Segment Detection

```rust
use hprc_ibd::segment::{detect_segments_rle, IdentityTrack, RleParams, Segment};

/// Track of per-window identities for a haplotype pair
pub struct IdentityTrack {
    /// Vector of (window_index, identity) tuples
    pub windows: Vec<(usize, f64)>,
    /// Total number of windows in region
    pub n_total_windows: usize,
}

impl IdentityTrack {
    /// Get identity for a specific window index
    pub fn get(&self, idx: usize) -> Option<f64>;

    /// Convert to HashMap for efficient lookup
    pub fn to_map(&self) -> std::collections::HashMap<usize, f64>;
}

/// Detect segments using run-length encoding approach
///
/// # Arguments
/// * `track` - Identity track for haplotype pair
/// * `window_positions` - Vector of (start, end) for each window
/// * `params` - Detection parameters
/// * `chrom` - Chromosome name
/// * `hap_a` - First haplotype ID
/// * `hap_b` - Second haplotype ID
///
/// # Returns
/// Vector of detected segments meeting all criteria
pub fn detect_segments_rle(
    track: &IdentityTrack,
    window_positions: &[(u64, u64)],
    params: &RleParams,
    chrom: &str,
    hap_a: &str,
    hap_b: &str,
) -> Vec<Segment>;
```

### Segment Merging

```rust
use hprc_ibd::segment::{merge_segments, Segment};

/// Merge overlapping segments (same haplotype pair only)
///
/// # Arguments
/// * `segments` - Mutable vector of segments to merge in place
///
/// # Notes
/// - Only merges segments with same chromosome and haplotype pair
/// - Updates statistics (n_windows, mean_identity, etc.) correctly
///
/// # Example
/// ```rust
/// use hprc_ibd::segment::{merge_segments, Segment};
///
/// let mut segments = vec![
///     // ... overlapping segments ...
/// ];
/// merge_segments(&mut segments);
/// // segments now contains merged results
/// ```
pub fn merge_segments(segments: &mut Vec<Segment>);
```

---

## Stats Module (`hprc_ibd::stats`)

### GaussianParams

```rust
use hprc_ibd::stats::GaussianParams;

/// Gaussian distribution parameters
pub struct GaussianParams {
    pub mean: f64,  // Distribution mean
    pub std: f64,   // Standard deviation
}

impl GaussianParams {
    /// Create new Gaussian parameters
    pub fn new(mean: f64, std: f64) -> Self;

    /// Calculate probability density at x
    pub fn pdf(&self, x: f64) -> f64;

    /// Calculate log probability density at x
    /// More numerically stable for HMM computations
    pub fn log_pdf(&self, x: f64) -> f64;
}
```

### K-means Clustering

```rust
use hprc_ibd::stats::kmeans_1d;

/// Simple 1D k-means clustering
///
/// # Arguments
/// * `data` - Data points to cluster
/// * `k` - Number of clusters
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Some((centers, assignments)) or None if clustering fails
///
/// # Example
/// ```rust
/// use hprc_ibd::stats::kmeans_1d;
///
/// let data = vec![0.1, 0.2, 0.15, 0.9, 0.95, 0.88];
/// if let Some((centers, assignments)) = kmeans_1d(&data, 2, 20) {
///     // centers[0] ~ 0.15, centers[1] ~ 0.91
///     // assignments indicates cluster for each data point
/// }
/// ```
pub fn kmeans_1d(data: &[f64], k: usize, max_iter: usize) -> Option<(Vec<f64>, Vec<usize>)>;
```

### OnlineStats

```rust
use hprc_ibd::stats::OnlineStats;

/// Welford's online algorithm for mean and variance
///
/// Computes running statistics without storing all values.
pub struct OnlineStats {
    // Internal fields...
}

impl OnlineStats {
    /// Create new statistics tracker
    pub fn new() -> Self;

    /// Add a value to the running statistics
    pub fn add(&mut self, x: f64);

    /// Get count of values added
    pub fn count(&self) -> usize;

    /// Get current mean
    pub fn mean(&self) -> f64;

    /// Get sample variance
    pub fn variance(&self) -> f64;

    /// Get sample standard deviation
    pub fn std(&self) -> f64;
}
```

---

## Usage Examples

### Complete IBD Detection Pipeline

```rust
use hprc_ibd::{Region, WindowIterator};
use hprc_ibd::hmm::{HmmParams, viterbi, extract_ibd_segments};

fn detect_ibd_for_pair(
    observations: &[f64],
    expected_seg_windows: f64,
) -> Vec<(usize, usize, usize)> {
    // Create HMM parameters
    let mut params = HmmParams::from_expected_length(expected_seg_windows, 0.0001);

    // Estimate emission distributions from data
    params.estimate_emissions(observations);

    // Run Viterbi algorithm
    let states = viterbi(observations, &params);

    // Extract IBD segments
    extract_ibd_segments(&states)
}

fn main() {
    // Example usage
    let region = Region::parse("chr20:1-1000000", None).unwrap();
    let window_size = 5000;

    // Iterate over windows
    for window in WindowIterator::new(&region, window_size) {
        println!("Window: {}-{}", window.start, window.end);
    }

    // Example identity data
    let observations = vec![
        0.5, 0.6, 0.55,  // Non-IBD region
        0.999, 0.998, 0.9995, 0.999, 0.9985,  // IBD region
        0.5, 0.4, 0.6,  // Non-IBD region
    ];

    let segments = detect_ibd_for_pair(&observations, 50.0);
    for (start, end, n_windows) in segments {
        println!("IBD segment: windows {}-{} ({} windows)", start, end, n_windows);
    }
}
```

### Using Statistics Module

```rust
use hprc_ibd::stats::{GaussianParams, OnlineStats, kmeans_1d};

fn main() {
    // Gaussian distributions
    let ibd_dist = GaussianParams::new(0.999, 0.005);
    let non_ibd_dist = GaussianParams::new(0.5, 0.2);

    let identity = 0.998;
    println!("P(IBD|{}) = {:.6}", identity, ibd_dist.pdf(identity));
    println!("P(non-IBD|{}) = {:.6}", identity, non_ibd_dist.pdf(identity));

    // Online statistics
    let mut stats = OnlineStats::new();
    for x in [0.999, 0.998, 0.9995, 0.997, 0.9985] {
        stats.add(x);
    }
    println!("Mean: {:.6}, Std: {:.6}", stats.mean(), stats.std());

    // K-means clustering
    let data = vec![0.5, 0.6, 0.55, 0.999, 0.998, 0.995];
    if let Some((centers, _)) = kmeans_1d(&data, 2, 20) {
        println!("Cluster centers: {:?}", centers);
    }
}
```

---

## Binary Executables

The crate also provides three binaries:

| Binary | Purpose | Main Dependencies |
|--------|---------|-------------------|
| `ibs` | Generate IBS windows from impg | External: `impg` |
| `jacquard` | Compute Jacquard identity states | Internal only |
| `ibd` | HMM-based IBD detection | External: `impg`, Internal: `hprc_ibd::hmm` |

See the [tutorials](tutorials/) for detailed usage of each binary.

---

## Version History

See [CHANGELOG.md](../CHANGELOG.md) for version history and changes.
