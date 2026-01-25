# HPRCv2-IBD Repository Structure

This document describes the organization of the repository.

## Design Principles

1. **Separation of concerns**: Source code, data, experiments, and reports are clearly separated
2. **Reproducibility**: Each experiment is self-contained with scripts and results
3. **Documentation**: Every directory has a README explaining its contents
4. **Cleanliness**: Intermediate files are kept separate from final outputs

## Directory Overview

```
HPRCv2-IBD/
├── README.md               # Project overview and quick start
├── STRUCTURE.md            # This file - repository organization
├── LICENSE                 # License information
├── .gitignore              # Git ignore patterns
│
├── src/                    # SOURCE CODE
├── data/                   # INPUT DATA
├── experiments/            # ANALYSIS EXPERIMENTS
├── reports/                # FINAL REPORTS
├── docs/                   # DOCUMENTATION
└── archive/                # ARCHIVED FILES
```

## Detailed Structure

### src/ - Package Source Code

Contains the Rust CLI tools being developed:

```
src/
├── README.md               # Package overview
├── ibd-cli/                # IBD detection tool
│   ├── Cargo.toml          # Rust package manifest
│   ├── README.md           # Tool documentation
│   ├── src/                # Source code
│   │   ├── main.rs         # CLI entry point
│   │   ├── lib.rs          # Core types
│   │   ├── hmm.rs          # HMM Viterbi algorithm
│   │   ├── stats.rs        # Statistical distributions
│   │   └── segment.rs      # Segment detection
│   └── target/             # Build artifacts
├── ibs-cli/                # IBS detection tool
│   ├── Cargo.toml
│   ├── README.md
│   └── src/
└── jacquard-cli/           # Jacquard coefficients tool
    ├── Cargo.toml
    ├── README.md
    └── src/
```

### data/ - Input Data

Contains symlinks to external data and population sample lists:

```
data/
├── README.md               # Data documentation
├── assemblies/             # HPRC genome assemblies
│   └── HPRC_r2_assemblies_0.6.1.agc -> (symlink, 3.1 GB)
├── alignments/             # Reference alignments
│   ├── hprc465vschm13.aln.paf.gz -> (symlink, 5.3 GB)
│   └── hprc465vschm13.aln.paf.gz.impg -> (symlink, 315 MB)
└── samples/                # Population sample lists
    ├── AFR.txt             # African (134 haplotypes)
    ├── EUR.txt             # European (60 haplotypes)
    ├── EAS.txt             # East Asian (100 haplotypes)
    ├── CSA.txt             # Central/South Asian (72 haplotypes)
    └── AMR.txt             # Admixed American (88 haplotypes)
```

### experiments/ - Analysis Experiments

Each experiment follows a consistent structure:

```
experiments/
├── README.md               # Experiments overview
├── chr1_full/              # Full chromosome 1 analysis
│   ├── README.md           # Experiment description
│   ├── scripts/            # Analysis scripts
│   │   ├── 01_generate_ibs.sh
│   │   ├── 02_ibd_inference.py
│   │   └── 03_visualize.py
│   ├── results/            # Final outputs
│   │   ├── json/           # Statistical summaries
│   │   └── figures/        # Visualizations
│   └── data/               # Intermediate data (large)
├── selection_scan/         # Selection signature detection
│   ├── README.md
│   ├── scripts/
│   ├── results/
│   └── regions/            # Per-locus data
├── full_population/        # 5-population analysis
│   ├── README.md
│   ├── scripts/
│   ├── results/
│   ├── intra/              # Within-population
│   └── inter/              # Between-population
└── benchmarks/             # Performance benchmarks
    ├── README.md
    ├── scripts/
    └── results/
```

### reports/ - Final Reports

Publication-ready reports and visualizations:

```
reports/
├── README.md               # Reports overview
├── main/                   # Main analysis report
│   ├── HPRCv2_IBD_Report.pdf    # Final PDF
│   ├── HPRCv2_IBD_Report.tex    # LaTeX source
│   ├── DATA_QUALITY_NOTES.md    # Important data quality info
│   ├── generate_figures.py      # Figure generation
│   └── figures/                 # Publication figures
│       ├── fig1_emission_distributions.png
│       ├── fig2_eur_ibd_landscape.png
│       ├── fig3_data_quality_comparison.png
│       ├── fig4_selection_scan.png
│       ├── fig5_population_summary.png
│       ├── fig6_ibd_tracks_detailed.png
│       ├── fig7_posterior_quality.png
│       └── fig8_selection_bars.png
└── supplementary/          # Additional materials
    ├── extended_figures/
    └── *.md                # Analysis notes
```

### docs/ - Documentation

Tutorials, API reference, and scientific methodology:

```
docs/
├── README.md               # Documentation index
├── tutorials/              # How-to guides
│   ├── quickstart.md
│   ├── ibs_analysis.md
│   ├── ibd_detection.md
│   └── selection_scan.md
├── api/                    # API documentation
│   └── CLI_REFERENCE.md
└── methods/                # Scientific methods
    ├── hmm_framework.md
    ├── emission_estimation.md
    └── validation.md
```

### archive/ - Archived Files

Old versions kept for reference:

```
archive/
├── sample_lists/           # Original sample list files
└── old_reports/            # Previous report versions
    ├── figures_v1/
    ├── figures_science/
    ├── latex/
    └── scripts/
```

## File Naming Conventions

- **Scripts**: Numbered prefix for execution order (01_, 02_, 03_)
- **Results**: Descriptive names with population/region identifiers
- **Figures**: Numbered prefix for report ordering (fig1_, fig2_)
- **Versions**: Use v2, v3 suffix for major revisions

## Git Ignore Patterns

Large files and build artifacts are excluded from git:

```
# Build artifacts
target/
__pycache__/

# Large data files
*.tsv
*.agc
*.paf.gz

# LaTeX artifacts
*.aux
*.log
*.out
```

## Disk Usage Summary

| Directory | Size | Contents |
|-----------|------|----------|
| experiments/ | ~55 GB | Analysis data |
| src/ | ~1.2 GB | Tools + build artifacts |
| reports/ | ~35 MB | PDFs, figures |
| data/ | symlinks | Points to external data |
| docs/ | ~4 MB | Documentation |

## Maintenance

### Adding New Experiments

1. Create directory under `experiments/`
2. Add `README.md` describing the experiment
3. Create `scripts/` and `results/` subdirectories
4. Follow naming conventions

### Updating Reports

1. Edit `reports/main/HPRCv2_IBD_Report.tex`
2. Regenerate figures with `python generate_figures.py`
3. Compile PDF: `pdflatex HPRCv2_IBD_Report.tex`

### Archiving Old Versions

Move to `archive/` with date suffix:
```bash
mv old_file archive/old_file_20260125
```
