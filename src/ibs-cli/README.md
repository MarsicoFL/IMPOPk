# IBS CLI

`src/ibs-cli/` hosts the IBS (Identity-by-State) detection tool for computing
pairwise identity in sliding windows from pangenome alignments.

## Structure

- `src/` and `Cargo.toml`: the Rust crate implementing the IBS detection algorithm.
  Build with `cargo build --release` and run with `cargo run --bin ibs -- --help`.
- `examples/`: example IBS outputs that serve as reference fixtures.
- `tests/`: unit tests for the IBS implementation.

## Usage

```bash
ibs \
    --sequence-files assemblies.agc \
    -a alignments.paf.gz \
    --subset-sequence-list samples.txt \
    --region chr1:1-10000000 \
    --size 5000 \
    -c 0.999 \
    -m cosine \
    --output ibs_results.tsv
```

See the main [README](../../README.md) for full documentation.
