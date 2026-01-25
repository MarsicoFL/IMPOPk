# HPRCv2-IBD Experiments

This directory contains analysis experiments using the IBD detection pipeline.

## Experiments Overview

| Experiment | Description | Status |
|------------|-------------|--------|
| `chr1_full/` | Full chromosome 1 IBD analysis (EUR/AFR) | Complete |
| `selection_scan/` | Selection signature detection at 5 known loci | Complete |
| `full_population/` | 5-population IBS matrix analysis | Complete |
| `chr2_50Mb_full/` | Chromosome 2 50Mb pilot analysis | Complete |
| `chr2_50Mb_filtered/` | Filtered version of chr2 analysis | Complete |
| `benchmarks/` | Performance scaling benchmarks | Partial |

## Directory Structure

Each experiment follows this structure:
```
experiment_name/
├── README.md           # Experiment description and results
├── scripts/            # Analysis scripts (Python, Bash)
├── results/            # Final results (JSON, figures)
│   ├── json/           # Statistical summaries
│   └── figures/        # Visualizations
└── data/               # Intermediate data (if needed)
```

## Experiment Details

### chr1_full/ - Chromosome 1 Full Analysis
- **Goal:** Detect IBD segments across full chromosome 1 (248.9 Mb)
- **Populations:** EUR (100 pairs), AFR (100 pairs)
- **Key Results:**
  - EUR: 46 valid IBD segments (identity ~99.99%)
  - AFR: Requires re-inference (see data quality notes)
- **Data Size:** ~45 GB (EUR: 11 GB, AFR: 34 GB)

### selection_scan/ - Selection Signature Analysis
- **Goal:** Detect IBS enrichment at known selection loci
- **Loci:** LCT, SLC24A5, EDAR, HBB, DARC
- **Populations:** All 5 (AFR, EUR, EAS, CSA, AMR)
- **Key Results:** 2.5-2.7x enrichment at expected targets

### full_population/ - Population IBS Matrix
- **Goal:** Compute pairwise IBS within and between populations
- **Analyses:**
  - Intra-population (5 comparisons)
  - Inter-population (10 comparisons)
- **Data Size:** ~6.8 GB

### benchmarks/ - Performance Analysis
- **Goal:** Measure scaling behavior of IBS/IBD tools
- **Tests:** Haplotype scaling, window scaling
- **Status:** Partially complete

## Running Experiments

Each experiment has its own scripts. General pattern:

```bash
# Generate IBS data
cd experiment_name/scripts
./01_generate_ibs.sh

# Run IBD inference
python 02_ibd_inference.py

# Generate visualizations
python 03_visualize.py
```

## Data Quality Notes

See `/reports/DATA_QUALITY_NOTES.md` for important information about:
- AFR IBD segment validity issues
- Corrected v2 emission parameters
- Centromeric region artifacts

## Disk Usage

```
chr1_full/        ~45 GB (largest - full chromosome data)
full_population/  ~6.8 GB
selection_scan/   ~2 GB
chr2_50Mb_*/      ~1 GB each
benchmarks/       ~500 MB
```

Total: ~55 GB of analysis data
