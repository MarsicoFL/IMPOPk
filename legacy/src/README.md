# HPRCv2-IBD Source Code

This directory contains the core Rust CLI tools for IBD detection from pangenome data.

## Tools

### ibd-cli
Identity-by-Descent detection using Hidden Markov Models.

```bash
cd ibd-cli && cargo build --release
./target/release/ibd-hmm --help
```

**Key features:**
- HMM-based IBD segment detection
- Viterbi decoding for optimal state sequence
- Emission parameter estimation with quality filtering
- JSON output with segment coordinates and statistics

### ibs-cli
Identity-by-State window detection from pangenome alignments.

```bash
cd ibs-cli && cargo build --release
./target/release/ibs --help
```

**Key features:**
- Fast window-based IBS computation
- Cosine similarity metric
- Support for AGC (compressed assembly) format
- Parallel processing for large datasets

### jacquard-cli
Jacquard delta coefficient estimation for relatedness analysis.

```bash
cd jacquard-cli && cargo build --release
./target/release/jacquard --help
```

**Key features:**
- Union-Find based delta classification
- Multiple coefficient estimation
- Parity testing framework

## Building All Tools

```bash
# Build all tools in release mode
for tool in ibd-cli ibs-cli jacquard-cli; do
    cd $tool && cargo build --release && cd ..
done
```

## Dependencies

- Rust 1.70+ (install via rustup)
- Standard Rust crates (see individual Cargo.toml files)

## Code Statistics

| Tool | Lines of Code | Main Algorithm |
|------|---------------|----------------|
| ibd-cli | ~1,700 | HMM Viterbi |
| ibs-cli | ~800 | Window similarity |
| jacquard-cli | ~500 | Union-Find |
