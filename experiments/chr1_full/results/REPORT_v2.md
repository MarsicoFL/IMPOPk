# chr1_full: Full Chromosome 1 IBD Analysis Report (CORRECTED)

**Generated**: 2026-01-20 18:35
**Version**: v2 (corrected emission estimation)
**Quality Filter**: identity >= 0.9
**Region**: chr1:1-248,956,422 (249 Mb)
**Window Size**: 5 kb

## Key Corrections from v1

1. **Quality filter**: Exclude windows with identity < 0.90
   - These represent gaps, poor alignments, or structural variants
   - Previously biased the non-IBD mean severely downward

2. **Proper non-IBD estimation**: Use windows with 0.90 <= identity < 0.999
   - This is the true non-IBD bulk distribution

3. **Proper IBD estimation**: Use windows with identity >= 0.9999
   - True IBD should be nearly identical

## Empirical Emission Parameters

| Population | Quality % | Non-IBD Mean | Non-IBD Std | IBD Mean | IBD Std | d' |
|------------|-----------|--------------|-------------|----------|---------|-----|
| EUR | 91.2% | 0.997206 | 0.000917 | 0.999988 | 0.000100 | 4.27 |
| AFR | 86.1% | 0.996493 | 0.001322 | 0.999986 | 0.000100 | 3.73 |

## Quality Assessment

d' separability (higher is better):
- d' > 2: Good separation
- d' 1-2: Moderate separation
- d' < 1: Poor separation

- **EUR**: d' = 4.27 (Good)
- **AFR**: d' = 3.73 (Good)

## IBD Detection Results

| Population | Pairs | Segments | Mean IBD (Mb) | Mean Fraction | Mean Length (kb) |
|------------|-------|----------|---------------|---------------|------------------|
| EUR | 20 | 437 | 1.97 | 0.1561 | 90.0 |
| AFR | 20 | 13 | 0.06 | 0.0037 | 91.5 |
