# chr1_full: Full Chromosome 1 IBD Analysis Report

**Generated**: 2026-01-19 10:57
**Data Source**: HPRC v2 Pangenome
**Region**: chr1:1-248,956,422 (249 Mb)
**Window Size**: 5 kb
**Total Windows**: 49,792

## 1. Empirical Emission Parameters

### 1.1 Non-IBD Distribution

| Population | Empirical Mean | Empirical Std | Theoretical Mean | d' |
|------------|----------------|---------------|------------------|-----|
| AFR | 0.931162 | 0.130103 | 0.998750 | 0.00 |

### 1.2 Quality Assessment (d' separability)

d' measures how well the IBD and non-IBD distributions are separated:
- d' < 1: Poor separation (high overlap)
- d' = 1-2: Moderate separation
- d' > 2: Good separation (low overlap)

- **AFR**: d' = 0.00 (Poor)

## 2. IBD Detection Results

| Population | Pairs | Segments | Mean IBD (Mb) | Mean Fraction | Mean Length (kb) |
|------------|-------|----------|---------------|---------------|------------------|
| AFR | 50 | 112 | 0.19 | 0.001 | 84.8 |

## 3. Comparison with exp02 (chr2:1-50Mb)

| Metric | exp02 (chr2 50Mb) | chr1_full (249 Mb) |
|--------|-------------------|---------------------|
| Region size | 50 Mb | 249 Mb |
| Windows | 10,000 | 49,792 |
| Scale factor | 1x | ~5x |

## 4. Conclusions

1. d' = 0.00 indicates room for model improvement
