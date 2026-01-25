# HPRCv2-IBD

Identity-by-Descent (IBD) detection from Human Pangenome Reference Consortium (HPRC) assemblies using haplotype-level identity analysis.

## Overview

This project provides tools and analysis pipelines for:

- **IBS Detection**: Sliding window identity-by-state analysis using pangenome alignments
- **IBD Inference**: Hidden Markov Model (Viterbi) to distinguish true IBD from background IBS
- **Selection Scans**: Detection of positive selection signatures through IBS enrichment
- **Population Genetics**: Comparative analysis across 5 continental populations

## Repository Structure

```
HPRCv2-IBD/
├── src/                    # PACKAGE SOURCE CODE (Rust CLI tools)
│   ├── ibd-cli/            # IBD detection with HMM
│   ├── ibs-cli/            # IBS window detection
│   └── jacquard-cli/       # Jacquard delta coefficients
│
├── data/                   # INPUT DATA
│   ├── assemblies/         # HPRC genome assemblies (symlinks)
│   ├── alignments/         # CHM13 reference alignments (symlinks)
│   └── samples/            # Population sample lists
│
├── experiments/            # ANALYSIS EXPERIMENTS
│   ├── chr1_full/          # Full chromosome 1 IBD analysis
│   ├── selection_scan/     # Selection signature detection
│   ├── full_population/    # 5-population IBS matrices
│   └── benchmarks/         # Performance benchmarks
│
├── reports/                # FINAL REPORTS
│   ├── main/               # Main analysis report (PDF, LaTeX, figures)
│   └── supplementary/      # Additional analysis notes
│
├── docs/                   # DOCUMENTATION
│   ├── tutorials/          # How-to guides
│   └── methods/            # Scientific methodology
│
└── archive/                # Old versions (for reference)
```

## Quick Start

### 1. Build Tools

```bash
cd src/ibs-cli && cargo build --release
cd ../ibd-cli && cargo build --release
```

### 2. Run IBS Detection

```bash
./src/ibs-cli/target/release/ibs \
    --sequence-files data/assemblies/HPRC_r2_assemblies_0.6.1.agc \
    -a data/alignments/hprc465vschm13.aln.paf.gz \
    --subset-sequence-list data/samples/EUR.txt \
    --region chr1:1-10000000 \
    --size 5000 \
    -t 0.999 -m cosine \
    --output results.tsv
```

### 3. Run IBD Inference

```bash
./src/ibd-cli/target/release/ibd-hmm inference \
    --input results.tsv \
    --output ibd_segments.json
```

## Requirements

- **Rust** 1.70+ (install via [rustup](https://rustup.rs/))
- **Python** 3.8+ (for analysis scripts)
- **LaTeX** (for report compilation, optional)

Python dependencies:
```bash
pip install numpy matplotlib scipy pandas
```

## Data Sources

External data files (symlinked in `data/`):

| File | Size | Source |
|------|------|--------|
| HPRC_r2_assemblies_0.6.1.agc | 3.1 GB | [HPRC](https://humanpangenome.org/) |
| hprc465vschm13.aln.paf.gz | 5.3 GB | [GarrisonLab](https://garrisonlab.s3.amazonaws.com/) |
| hprc465vschm13.aln.paf.gz.impg | 315 MB | [GarrisonLab](https://garrisonlab.s3.amazonaws.com/) |

## Population Data

| Population | Individuals | Haplotypes | Description |
|------------|-------------|------------|-------------|
| AFR | 67 | 134 | African ancestry |
| EUR | 30 | 60 | European ancestry |
| EAS | 50 | 100 | East Asian ancestry |
| CSA | 36 | 72 | Central/South Asian ancestry |
| AMR | 44 | 88 | Admixed American ancestry |
| **Total** | **227** | **454** | |

## Main Results

### Validated Findings

1. **HMM Model Performance**: d' > 5 for both EUR and AFR (excellent state separation)
2. **EUR IBD Detection**: Valid segments with ~99.99% identity
3. **Selection Signatures**: 2.5-2.7x IBS enrichment at LCT, EDAR, SLC24A5

### Data Quality Notes

**Important**: See `reports/main/DATA_QUALITY_NOTES.md` for critical information about:
- AFR IBD segment validity issues (require re-inference)
- Centromeric region artifacts
- v2 corrected emission parameters

## Documentation

- **Main Report**: `reports/main/HPRCv2_IBD_Report.pdf`
- **Tutorials**: `docs/tutorials/`
- **API Reference**: `docs/api/`
- **Methods**: `docs/methods/`

## Experiments

| Experiment | Description | Data Size |
|------------|-------------|-----------|
| `chr1_full` | Full chromosome 1 IBD analysis | 45 GB |
| `selection_scan` | 5 known selection loci | 2 GB |
| `full_population` | 5-population IBS matrices | 6.8 GB |
| `benchmarks` | Performance scaling tests | 500 MB |

## License

MIT License - See LICENSE file for details.

## Citation

If using this pipeline or data, please cite:
- Human Pangenome Reference Consortium (2023)
- This repository
