# HPRCv2-IBD

IBS detection and IBD inference from HPRC pangenome assemblies using haplotype-level identity analysis.

## What it does

- **IBS detection**: Sliding window analysis over chromosomes using `impg` similarity
- **IBD inference**: 2-state HMM (Viterbi) to distinguish true IBD from sporadic IBS
- **Identity states**: Jacquard-style coefficients for diploid pairs

## Repository Structure

```
HPRCv2-IBD/
├── experiments/           # Completed experiments with data
│   ├── chr1_full/        # Full chr1 analysis (EUR: 90M, AFR: 284M records)
│   ├── chr2_50Mb_full/   # Chr2 50Mb reference experiment
│   ├── chr2_50Mb_filtered/
│   ├── selection_scan/   # Selection scan (LCT, DARC, EDAR, HBB, SLC24A5)
│   ├── full_population/  # Full population IBS matrices
│   └── benchmarks/       # Performance benchmarks
├── tools/                # Rust CLI tools
│   ├── ibd-cli/         # IBD detection with HMM
│   ├── ibs-cli/         # IBS window detection
│   └── jacquard-cli/    # Jacquard coefficients
├── sample_lists/         # Population sample lists (EUR, AFR, EAS, CSA, AMR)
├── data/                 # Symlinks to AGC/PAF files
└── docs/                 # Documentation and tutorials
```

## Requirements

- Rust 1.70+
- [impg](https://github.com/pangenome/impg)
- GNU coreutils, parallel

## Quick Start

```bash
# Build tools
cd tools/ibs-cli && cargo build --release
cd ../ibd-cli && cargo build --release

# Run IBS detection
./target/release/ibs \
  --sequence-files ../../data/HPRC_r2_assemblies_0.6.1.agc \
  -a ../../data/hprc465vschm13.aln.paf.gz \
  --region "CHM13#0#chr2:1-50000000" \
  --window-size 5000 \
  --sample-list ../../sample_lists/HPRCv2_EUR_full.txt \
  --output ibs_results.tsv
```

## Data Sources

Required external files (symlinked in `data/`):

```bash
# Sequence archive (3.1 GB)
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc

# Alignments and index (5.3 GB + 315 MB)
wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz
wget https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg
```

## Main Experiments

| Experiment | Data Size | Description |
|------------|-----------|-------------|
| `chr1_full` | 45 GB | Full chromosome 1: EUR (30 ind), AFR (67 ind) |
| `selection_scan` | 495 MB | Known selection regions across populations |
| `full_population` | 6.8 GB | Inter/intra-population IBS matrices |
