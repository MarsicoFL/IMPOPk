# exp01: chr2 50Mb with Cutoff Filtering

## Status: COMPLETED (with critical issues identified)

## Objective
Full-scale IBD detection on 50 Mb region using cutoff-filtered IBS data.

## Region
- **Chromosome**: chr2
- **Coordinates**: 1-50 Mb
- **Window size**: 5000 bp
- **Cutoff**: identity >= 0.99

## Populations
- EUR (European)
- AFR (African)
- EAS (East Asian)

## Critical Issues Found

See `CRITICAL_ANALYSIS.md` for full details.

| Problem | Details |
|---------|---------|
| Variance underestimated | Model σ=0.0007, empirical σ=0.002 (3x error) |
| Poor separability | d' = 0.3-0.7 (should be >2) |
| Truncation bias | Cutoff >=0.99 removes most non-IBD distribution |
| IBD overdetection | 11-34% IBD vs expected ~1% |

## Conclusion
**Cutoff filtering prevents proper emission parameter estimation.**

Solution: exp02 with full distribution (no cutoff).

## Data Files
- `data/EUR_chr2_50Mb_ibs.tsv` (~2 GB)
- `data/AFR_chr2_50Mb_ibs.tsv` (~2 GB)
- `data/EAS_chr2_50Mb_ibs.tsv` (~1.6 GB)
