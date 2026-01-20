# exp02: Full Distribution IBD Analysis Report

**Generated**: 2026-01-17 10:02:35
**Data Source**: HPRC v2 Pangenome
**Region**: chr2:1-50Mb
**Window Size**: 5 kb
**Key Improvement**: Full identity distribution (no cutoff)

## 1. Empirical Emission Parameters

Unlike exp01 (which used cutoff >= 0.99), this analysis uses the
complete identity distribution for proper parameter estimation.

### 1.1 Non-IBD Distribution

| Population | Empirical Mean | Empirical Std | Theoretical Mean | d' |
|------------|----------------|---------------|------------------|-----|
| EUR | 0.999050 | 0.000974 | 0.999150 | 1.17 |
| AFR | 0.998445 | 0.001025 | 0.998750 | 1.86 |

### 1.2 Quality Assessment (d' separability)

d' measures how well the IBD and non-IBD distributions are separated:
- d' < 1: Poor separation (high overlap)
- d' = 1-2: Moderate separation
- d' > 2: Good separation (low overlap)

- **EUR**: d' = 1.17 (Moderate)
- **AFR**: d' = 1.86 (Moderate)

## 2. IBD Detection Results

| Population | Pairs | Segments | Mean IBD (Mb) | Mean Fraction | Mean Length (kb) |
|------------|-------|----------|---------------|---------------|------------------|
| EUR | 25 | 2742 | 12.02 | 0.240 | 109.6 |
| AFR | 25 | 1184 | 4.42 | 0.088 | 93.3 |

## 3. Comparison with exp01 (Filtered)

| Metric | exp01 (cutoff >= 0.99) | exp02 (full) |
|--------|------------------------|--------------|
| Identity range | 0.99 - 1.0 | 0.1 - 1.0 |
| Non-IBD std | ~0.0007 (underestimated) | Empirical |
| d' expected | 0.3-0.7 | >1.5 |

## 4. Conclusions

1. **Improved separability**: Mean d' = 1.52 (vs ~0.5 in exp01)
2. Full distribution enables proper emission parameter estimation
3. IBD detection calibration is more reliable
