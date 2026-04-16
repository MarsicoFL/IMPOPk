# impopk Source Code

This directory contains the core Rust CLI tools for IBD detection and local ancestry inference from pangenome data.

## Crates

### common
Shared types and utilities: `Region`, `Window`, `WindowIterator`, `ColumnIndices`, `HprcError`.

### ibs-cli
Identity-by-State window detection from pangenome alignments.
- **`ibs`**: Wrapper around `impg similarity` (requires impg + AGC). Produces the windowed pairwise identity TSV consumed by all downstream HMMs.

### ibd-cli
IBD segment detection using Hidden Markov Models.
- **`ibd`**: 2-state HMM with Viterbi, forward-backward, and Baum-Welch training
- **`ibd-validate`**: Validation tool for comparing IBD results against gold standards

### ancestry-cli
Local ancestry inference using N-state HMM.
- **`ancestry`**: Supports auto-configuration, pairwise contrast emissions, eGRM output, demographic inference, and 40+ configurable parameters

### jacquard-cli
Jacquard delta coefficient estimation for relatedness analysis.
- **`jacquard`**: Computes 9 condensed delta coefficients from 4-haplotype IBS patterns

## Building

From the workspace root:

```bash
cargo build --release
```

Binaries are placed in `target/release/`.
