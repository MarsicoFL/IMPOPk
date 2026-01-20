# Full Population LCT Analysis

IBS analysis of the LCT region using ALL available haplotypes per ancestry.

## Sample Counts

| Population | Individuals | Haplotypes | Intra Pairs |
|------------|-------------|------------|-------------|
| AFR | 32 | 64 | 2,016 |
| EUR | 30 | 60 | 1,770 |
| EAS | 27 | 54 | 1,431 |
| CSA | 11 | 22 | 231 |
| AMR | 3 | 6 | 15 |

**Total**: 103 individuals, 206 haplotypes

## Region

- **Location**: chr2:130,787,850-140,837,183 (~10 Mb)
- **Gene**: LCT (Lactase)
- **Window size**: 5,000 bp

## Experiments

### Intra-Population (5 experiments)

Compare all haplotype pairs within each population.

```bash
./scripts/run_all_full_intra.sh           # Run all
./scripts/run_all_full_intra.sh EUR       # Run single population
```

### Inter-Population (10 experiments)

Compare all cross-population haplotype pairs.

| Comparison | Hap1 × Hap2 | Cross-Pairs |
|------------|-------------|-------------|
| AFR-EUR | 64 × 60 | 3,840 |
| AFR-EAS | 64 × 54 | 3,456 |
| AFR-CSA | 64 × 22 | 1,408 |
| AFR-AMR | 64 × 6 | 384 |
| EUR-EAS | 60 × 54 | 3,240 |
| EUR-CSA | 60 × 22 | 1,320 |
| EUR-AMR | 60 × 6 | 360 |
| EAS-CSA | 54 × 22 | 1,188 |
| EAS-AMR | 54 × 6 | 324 |
| CSA-AMR | 22 × 6 | 132 |

```bash
./scripts/run_all_full_inter.sh           # Run all
./scripts/run_all_full_inter.sh AFR EUR   # Run single pair
```

## Expected Results

Based on pilot analysis:
- EUR IBS rate ~65% (1.71× vs AFR)
- EAS IBS rate ~63% (1.64× vs AFR)
- AMR IBS rate ~58% (1.50× vs AFR)
- CSA IBS rate ~54% (1.41× vs AFR)
- AFR IBS rate ~38% (baseline)
