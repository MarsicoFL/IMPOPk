# HPRCv2-IBD

Identity-by-Descent (IBD) detection from pangenome assemblies using haplotype-level identity analysis.

## Overview

A suite of Rust CLI tools for detecting IBD segments from whole-genome assemblies:

- **IBS Detection**: Sliding window identity-by-state computation from pangenome alignments
- **IBD Inference**: Hidden Markov Model (Viterbi) to distinguish true IBD from background IBS
- **Jacquard Coefficients**: Delta coefficient estimation for relatedness analysis

## Tools

| Tool | Description | Documentation |
|------|-------------|---------------|
| [ibs-cli](src/ibs-cli/) | Window-based IBS detection | [README](src/ibs-cli/README.md) |
| [ibd-cli](src/ibd-cli/) | HMM-based IBD inference | [README](src/ibd-cli/README.md) |
| [jacquard-cli](src/jacquard-cli/) | Jacquard delta coefficients | [README](src/jacquard-cli/README.md) |

## Installation

### Requirements

- **Rust** 1.70+ ([rustup.rs](https://rustup.rs/))

### Build

```bash
# Build all tools
cd src/ibs-cli && cargo build --release
cd ../ibd-cli && cargo build --release
cd ../jacquard-cli && cargo build --release
```

Binaries will be in `src/*/target/release/`.

## Usage

### 1. IBS Detection

Compute pairwise identity in sliding windows:

```bash
./src/ibs-cli/target/release/ibs \
    --sequence-files assemblies.agc \
    -a alignments.paf.gz \
    --subset-sequence-list samples.txt \
    --region chr1:1-10000000 \
    --size 5000 \
    -t 0.999 \
    -m cosine \
    --output ibs_results.tsv
```

**Parameters:**
- `--sequence-files`: AGC archive with assemblies
- `-a`: PAF alignments to reference
- `--subset-sequence-list`: File with haplotype IDs (one per line)
- `--region`: Genomic region (chr:start-end)
- `--size`: Window size in bp
- `-t`: Identity threshold
- `-m`: Similarity metric (cosine, jaccard)

### 2. IBD Inference

Infer IBD segments from IBS data using HMM:

```bash
./src/ibd-cli/target/release/ibd-hmm inference \
    --input ibs_results.tsv \
    --output ibd_segments.json
```

**Output:** JSON file with IBD segments including coordinates, identity, and posterior probabilities.

### 3. Jacquard Coefficients

Compute Jacquard delta coefficients:

```bash
./src/jacquard-cli/target/release/jacquard \
    --input identity_data.tsv \
    --output coefficients.json
```

## Input Data

The tools require:

1. **Assemblies**: AGC-compressed genome assemblies
2. **Alignments**: PAF alignments to a reference genome
3. **Sample list**: Text file with haplotype identifiers

### Sample List Format

```
HG00096#1
HG00096#2
HG00097#1
HG00097#2
```

Format: `{sample_id}#{haplotype}` (1 = hap1/maternal, 2 = hap2/paternal)

## Data Sources

Required external files:

| File | Size | Download |
|------|------|----------|
| HPRC_r2_assemblies_0.6.1.agc | 3.1 GB | [Download](https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/) |
| hprc465vschm13.aln.paf.gz | 5.3 GB | [Download](https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz) |
| hprc465vschm13.aln.paf.gz.impg | 315 MB | [Download](https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg) |

## Documentation

- [IBS Tutorial](docs/tutorials/ibs.md)
- [IBD Tutorial](docs/tutorials/ibd.md)
- [API Reference](docs/API.md)

## Repository Structure

```
HPRCv2-IBD/
├── src/                    # Source code (Rust CLI tools)
│   ├── ibd-cli/
│   ├── ibs-cli/
│   └── jacquard-cli/
├── data/                   # Input data and sample lists
├── docs/                   # Documentation and tutorials
└── experiments/            # Analysis scripts and examples
```

## License

MIT License

## Citation

If using these tools, please cite this repository.
