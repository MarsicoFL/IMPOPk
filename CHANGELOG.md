# Changelog

All notable changes to HPRCv2-IBD will be documented in this file.

## [0.2.0] - 2026-01-26

### Added
- Cargo workspace for unified builds
- Parameter validation in all CLIs
- GaussianParams validation (std > 0)
- ColumnIndices moved to hprc-common (shared code)
- Haplotype normalization in segment merging
- LICENSE and CHANGELOG files

### Changed
- window_size is now a parameter in HMM emissions (was hardcoded 5000)
- Scripts moved from src/*/scripts/ to bin/
- Analysis scripts moved to scripts/analysis/

### Fixed
- Segment merge bug: n_windows double-counting for overlapping segments
- CLI validation: prevent infinite loops with window_size=0
- Parity test path after script relocation

### Removed
- archive/ directory (old reports)
- experiments/_deprecated/ (failed cutoff approach)
- Obsolete JSON files in chr1_full
- Individual Cargo.lock files (workspace manages this)

## [0.1.0] - 2026-01-15

### Added
- Initial release with ibd-cli, ibs-cli, jacquard-cli
- HMM-based IBD detection
- IBS window detection
- Jacquard coefficient estimation
- Phase 1 experiments: chr1_full, chr2_50Mb_full, selection_scan, full_population, benchmarks
