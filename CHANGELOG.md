# Changelog

All notable changes to the HPRCv2-IBD project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive API documentation (`docs/API.md`)
- Expanded tutorials with examples, troubleshooting, and output interpretation
- Biological interpretation guide for Jacquard Delta states
- HMM parameter tuning guidance in IBD tutorial
- Visualization suggestions for Jacquard coefficients

### Changed
- Improved inline rustdoc documentation in all Rust modules
- Enhanced tutorial documentation with more usage examples

## [0.2.0] - 2026-01-13

### Added
- **HMM Module** (`src/hmm.rs`): Complete Hidden Markov Model implementation for IBD detection
  - Two-state HMM (IBD/non-IBD) with Gaussian emissions
  - Viterbi algorithm for optimal state sequence decoding
  - Automatic emission parameter estimation via k-means clustering
  - Configurable transition probabilities based on expected segment length

- **Statistics Module** (`src/stats.rs`): Statistical utilities
  - `GaussianParams` struct with PDF and log-PDF computation
  - `kmeans_1d` function for 1D clustering
  - `OnlineStats` for streaming mean/variance computation (Welford's algorithm)

- **Segment Module** (`src/segment.rs`): Segment detection and management
  - `Segment` struct for representing IBS/IBD segments
  - `RleParams` for configurable segment detection parameters
  - `detect_segments_rle` for run-length encoding based detection
  - `merge_segments` for combining overlapping segments

- **IBD Binary** (`src/bin/ibd.rs`): Command-line IBD detection tool
  - Integration with impg for identity collection
  - HMM-based segment calling
  - Configurable parameters for sensitivity/specificity trade-offs

- **Library Structure**: Reorganized as proper Rust library (`src/lib.rs`)
  - Public API with documented modules
  - Custom error types (`IbdError`)
  - Core types: `Region`, `Window`, `WindowIterator`

- **Testing Guide** (`TESTING_GUIDE.md`): Comprehensive testing documentation
  - Unit test instructions
  - Integration test procedures
  - Test fixture documentation

### Fixed
- **Infinite loop prevention**: Added validation to prevent zero window size in `WindowIterator`
- **NaN handling in k-means**: Changed from `partial_cmp` to `total_cmp` for safe NaN sorting
- **Invalid probability handling**: Added validation for `p_enter_ibd` parameter (must be in (0, 1))
- **Segment merging bug**: Fixed merge logic to only combine segments from same haplotype pair
- **Floating-point precision**: Improved numerical stability in log-space HMM computations
- **Memory safety**: Ensured all array accesses are bounds-checked

### Changed
- Shell scripts cleaned up: removed verbose/Spanish comments, improved error handling
- README updated with data download instructions and corrected grammar

## [0.1.0] - 2025-12-01

### Added
- **IBS Binary** (`src/main.rs`): Initial IBS detection pipeline
  - Sliding window analysis over genomic regions
  - Integration with impg for sequence similarity
  - Identity filtering and deduplication

- **Jacquard Binary** (`src/bin/jacquard.rs`): Jacquard coefficient calculator
  - Nine-state identity classification
  - Union-find algorithm for haplotype clustering
  - Support for diploid sample comparisons

- **Shell Scripts** (legacy interface):
  - `scripts/ibs.sh`: Bash IBS detection
  - `scripts/ibd.sh`: R-based HMM IBD calling
  - `scripts/jacquard_coeffs.sh`: Shell Jacquard calculator
  - `scripts/run_full.sh`: Parallel pipeline launcher

- **Parity Testing Framework**: Script-vs-Rust comparison testing
  - Spec files in `tests/parity/*.toml`
  - Fixtures in `tests/data/`
  - Automated comparison in `tests/parity.rs`

- **Documentation**:
  - Basic tutorials for each script
  - Conceptual framework document
  - Porting guide for Bash-to-Rust migration

### Notes
- Initial release focused on IBS detection and Jacquard coefficients
- IBD detection via shell script (R-based HMM) only

---

## Bug Fix Details (v0.2.0)

### 1. WindowIterator Zero Size Panic

**Problem**: Creating a `WindowIterator` with `window_size = 0` would cause an infinite loop.

**Fix**: Added explicit assertion with clear error message:
```rust
assert!(window_size > 0, "window_size must be greater than 0 to avoid infinite loop");
```

### 2. NaN Values in K-means Sorting

**Problem**: Using `partial_cmp` for sorting could panic on NaN values.

**Fix**: Changed to `total_cmp` which defines a total ordering including NaN:
```rust
sorted.sort_by(|a, b| a.total_cmp(b));
```

### 3. Invalid p_enter_ibd Parameter

**Problem**: Values of 0 or 1 for `p_enter_ibd` would produce invalid transition matrices.

**Fix**: Added parameter validation:
```rust
assert!(p_enter_ibd > 0.0 && p_enter_ibd < 1.0,
        "p_enter_ibd must be in range (0, 1), got {}", p_enter_ibd);
```

### 4. Cross-Haplotype Segment Merging

**Problem**: `merge_segments` could incorrectly merge overlapping segments from different haplotype pairs.

**Fix**: Added haplotype pair check before merging:
```rust
let same_haplotypes = seg.hap_a == last.hap_a && seg.hap_b == last.hap_b;
if seg.chrom == last.chrom && same_haplotypes && seg.start <= last.end {
    // Merge only if same haplotype pair
}
```

### 5. Numerical Precision in HMM

**Problem**: Direct probability multiplication in Viterbi could underflow to zero.

**Fix**: Implemented all computations in log-space:
```rust
let log_initial: [f64; 2] = [params.initial[0].ln(), params.initial[1].ln()];
// All subsequent computations use log probabilities
```

### 6. Bounds Checking in Segment Detection

**Problem**: Potential out-of-bounds access when finalizing segments.

**Fix**: Added explicit bounds checks with Option return:
```rust
let start_bp = window_positions.get(start_idx)?.0;
let end_bp = window_positions.get(end_idx)?.1;
```

---

## Migration Guide

### From v0.1.0 to v0.2.0

1. **Library Usage**: Import from `hprc_ibd` instead of individual files
   ```rust
   use hprc_ibd::{Region, WindowIterator};
   use hprc_ibd::hmm::{HmmParams, viterbi};
   ```

2. **IBD Detection**: Use the new `ibd` binary instead of shell script
   ```bash
   # Old (v0.1.0)
   ./scripts/ibd.sh --region chr20:1-1000000 ...

   # New (v0.2.0)
   ./target/release/ibd --region chr20:1-1000000 ...
   ```

3. **Parameter Names**: Some CLI flags have changed
   - `--expected-seg-windows` replaces `--expected-ibd-length`
   - `--p-enter-ibd` is a new parameter for HMM tuning

---

## Contributors

- Franco Marsico ([@MarsicoFL](https://github.com/MarsicoFL))
- HPRC Analysis Team

---

## Links

- [GitHub Repository](https://github.com/MarsicoFL/HPRCv2-IBD)
- [HPRC Data Portal](https://humanpangenome.org/)
- [impg Tool](https://github.com/ekg/impg)
