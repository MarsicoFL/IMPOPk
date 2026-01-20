# exp02: Full Distribution IBD Analysis

## Objective

Address the critical issues identified in exp01 by using the **complete identity distribution** without cutoff filtering.

## Key Improvements over exp01

| Issue (exp01) | Solution (exp02) |
|---------------|------------------|
| Cutoff ≥0.99 truncates distribution | No cutoff - capture ALL values |
| σ_non-IBD underestimated 3x | Estimate σ from empirical data |
| Poor d' separability (~0.4) | Better calibration → d' ~1.5-2 |
| IBD overdetection (11-34%) | Properly calibrated thresholds |

## Data Generation

```bash
# Generate FULL IBS data using ibs-bed-full.sh (NO CUTOFF)
cd /path/to/ibs-cli/scripts

for POP in EUR AFR EAS; do
  ./ibs-bed-full.sh \
    --sequence-files ../data/HPRC_r2_assemblies_0.6.1.agc \
    -a ../data/hprc465vschm13.aln.paf.gz \
    -r CHM13 -region chr2:1-50000000 -size 5000 \
    --subset-sequence-list ../sample_lists/HPRCv2_${POP}subset.txt \
    --output data/${POP}_chr2_50Mb_full.tsv \
    -j 6
done
```

**Warning**: Output files will be ~10-50x larger than filtered version (~20-100 GB total).

## Analysis Pipeline

1. **Estimate empirical parameters**:
   ```bash
   python3 scripts/estimate_emissions.py
   ```

2. **Run IBD inference with calibrated HMM**:
   ```bash
   python3 scripts/run_analysis.py --use-empirical-params
   ```

3. **Generate report**:
   ```bash
   python3 scripts/generate_report.py
   ```

## Expected Output

```
exp02_chr2_50Mb_full/
├── data/
│   ├── EUR_chr2_50Mb_full.tsv    # ~20-50 GB
│   ├── AFR_chr2_50Mb_full.tsv
│   └── EAS_chr2_50Mb_full.tsv
├── results/
│   ├── empirical_params.json     # Calibrated emission parameters
│   ├── figures/                  # Publication-ready plots
│   └── json/                     # Structured results
└── REPORT.md
```

## Success Criteria

- [ ] d' > 1.5 for all populations
- [ ] IBD fraction < 5% for random pairs
- [ ] Segment lengths follow expected distribution
- [ ] Posteriors well-calibrated (coverage)
