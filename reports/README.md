# HPRCv2-IBD Reports

Analysis reports and visualizations for the IBD detection pipeline.

## Directory Structure

```
reports/
├── README.md                    # This file
├── main/                        # Main analysis report
│   ├── HPRCv2_IBD_Report.pdf    # Final report (PDF)
│   ├── HPRCv2_IBD_Report.tex    # LaTeX source
│   ├── DATA_QUALITY_NOTES.md    # Important data quality information
│   ├── generate_figures.py      # Figure generation script
│   └── figures/                 # Publication-ready figures
└── supplementary/               # Supplementary materials
    ├── extended_figures/        # Additional visualizations
    └── *.md                     # Analysis notes and validation reports
```

## Main Report

**HPRCv2_IBD_Report.pdf** - Complete analysis of IBD detection from HPRC pangenome data.

Key sections:
1. Data and experimental setup
2. Data quality assessment (important!)
3. HMM methodology
4. EUR IBD detection results (validated)
5. Selection scan at known loci
6. Limitations and next steps

### Important: Data Quality

Read `main/DATA_QUALITY_NOTES.md` before interpreting results:
- **EUR IBD segments:** VALID (identity ~99.99%)
- **AFR IBD segments:** INVALID (require re-inference)
- **Selection scan IBS rates:** VALID
- **v2 emission parameters:** VALID for both populations

## Figures

All figures in `main/figures/` are publication-ready:

| Figure | Description |
|--------|-------------|
| `fig1_emission_distributions.png` | HMM emission parameters (v2 corrected) |
| `fig2_eur_ibd_landscape.png` | EUR IBD along chromosome 1 |
| `fig3_data_quality_comparison.png` | EUR vs AFR data quality |
| `fig4_selection_scan.png` | IBS rates at selection loci |
| `fig5_population_summary.png` | Population comparison |
| `fig6_ibd_tracks_detailed.png` | Individual IBD tracks (EUR) |
| `fig7_posterior_quality.png` | HMM posterior probabilities |
| `fig8_selection_bars.png` | Selection loci bar chart |

## Regenerating Figures

```bash
cd main
python generate_figures.py
```

Requires: numpy, matplotlib, scipy

## Compiling LaTeX

```bash
cd main
pdflatex HPRCv2_IBD_Report.tex
pdflatex HPRCv2_IBD_Report.tex  # Run twice for cross-references
```

## Archived Reports

Old report versions are in `../archive/old_reports/` for reference.
