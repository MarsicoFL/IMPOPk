# LCT Region IBS Experiments

Pilot analysis of IBS patterns in the Lactase (LCT) gene region using HPRC v2 data.

## Region

- **Location**: chr2:130,787,850-140,837,183 (~10 Mb)
- **Gene**: LCT (Lactase) - canonical example of recent positive selection
- **Window size**: 5,000 bp
- **Threshold**: T = 0.999

## Populations

| Population | Individuals | Haplotypes | Pairs |
|------------|-------------|------------|-------|
| AFR | 4 | 8 | 28 |
| EUR | 4 | 8 | 28 |
| EAS | 4 | 8 | 28 |
| CSA | 4 | 8 | 28 |
| AMR | 3 | 6 | 15 |

## Directory Structure

```
experiments/
├── README.md
├── data/                    # Sample lists per population
│   ├── AFR_4samples.txt
│   ├── EUR_4samples.txt
│   ├── EAS_4samples.txt
│   ├── CSA_4samples.txt
│   └── AMR_3samples.txt
├── scripts/                 # Experiment execution scripts
│   ├── run_experiment.sh    # Run single experiment
│   ├── run_all_inter.sh     # Run all inter-population comparisons
│   ├── analyze_ibs.py       # Basic analysis script
│   └── ...
├── results/                 # IBS output TSV files (NOT in git)
│   ├── AFR_intra/
│   ├── EUR_intra/
│   ├── ...
│   └── *_vs_*_inter/
└── analysis/                # Analysis code and outputs
    ├── ibs_analysis.py      # Main analysis script
    └── output/              # Generated figures and tables
        ├── ibs_enrichment_rates.png
        ├── ibs_enrichment_rates.pdf
        └── ibs_metrics.tsv
```

## Results

**Note**: The `results/` directory contains ~150 MB of TSV files and is excluded from git.
To reproduce, run the experiment scripts or contact the authors for data access.

### Key Findings (Preliminary)

IBS rate normalized by sample size, using AFR as baseline:

| Population | IBS Rate | Fold Enrichment |
|------------|----------|-----------------|
| EUR | 65.46% | 1.71× |
| EAS | 62.76% | 1.64× |
| AMR | 57.69% | 1.50× |
| CSA | 54.18% | 1.41× |
| AFR | 38.36% | 1.00× |

## Usage

### Run Analysis

```bash
cd experiments
python3 analysis/ibs_analysis.py
```

### Run Experiments (requires impg and data)

```bash
cd experiments
./scripts/run_experiment.sh intra AFR chr2:130787850-140837183 results/AFR_intra
```

## Runtime

- 15 experiments (5 intra + 10 inter) completed in ~80 minutes
- Single node, 20 CPU cores
- ~1.28 million IBS records generated
