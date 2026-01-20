# exp02: Full IBD Analysis - Segments >= 2 Mb

**Generated**: 2026-01-17 10:41
**Data Source**: HPRC v2 Pangenome
**Region**: chr2:1-50Mb
**Window Size**: 5 kb
**Minimum Segment Length**: 2 Mb

## 1. Analysis Overview

This analysis detects IBD segments of at least 2 Mb, which are biologically significant
as they indicate recent common ancestry (within ~25-50 generations for 2Mb segments).

### 1.1 HMM Parameters for Long Segments

| Parameter | Value | Description |
|-----------|-------|-------------|
| p_exit_ibd | 0.001 | Expected segment ~1000 windows (5 Mb) |
| p_enter_ibd | 0.00005 | Low entry to reduce false positives |
| min_segment | 2 Mb | 400 windows at 5kb/window |
| segment_merge | 100 windows | Merge segments within 500 kb |

## 2. Population Comparison

| Population | Total Pairs | Pairs with IBD | % with IBD | Total Segments |
|------------|-------------|----------------|------------|----------------|
| AFR | 1,953 | 545 | 27.9% | 625 |
| EUR | 1,830 | 1,743 | 95.2% | 5,703 |

**EUR/AFR ratio**: 9.1x more segments in EUR

### 2.1 Segment Length Statistics

| Population | Mean Length | Max Length | Min Length |
|------------|-------------|------------|------------|
| AFR | 2.54 Mb | 5.87 Mb | 2.00 Mb |
| EUR | 3.03 Mb | 10.70 Mb | 2.00 Mb |

## 3. Emission Parameters (d' Separability)

| Population | Non-IBD Mean | Non-IBD Std | IBD Mean | d' |
|------------|--------------|-------------|----------|-----|
| AFR | 0.9984 | 0.00102 | 0.9998 | 1.86 |
| EUR | 0.9991 | 0.00097 | 0.9999 | 1.17 |

## 4. Top 5 Longest Segments per Population

### AFR
| Rank | Pair | Length | Position | Posterior |
|------|------|--------|----------|-----------|
| 1 | NA18505#1 - NA20346#2 | 5.87 Mb | 23.7-29.6 Mb | 0.979 |
| 2 | NA19700#2 - NA19835#1 | 5.32 Mb | 11.8-17.1 Mb | 0.977 |
| 3 | HG02583#2 - HG03225#2 | 5.07 Mb | 21.1-26.2 Mb | 0.987 |
| 4 | HG03139#2 - HG03369#2 | 5.02 Mb | 20.3-25.3 Mb | 0.978 |
| 5 | HG02583#2 - HG03139#2 | 4.68 Mb | 24.5-29.2 Mb | 0.991 |

### EUR
| Rank | Pair | Length | Position | Posterior |
|------|------|--------|----------|-----------|
| 1 | HG00253#2 - NA20503#2 | 10.70 Mb | 30.9-41.6 Mb | 0.961 |
| 2 | HG00128#2 - HG00232#2 | 10.24 Mb | 12.3-22.5 Mb | 0.966 |
| 3 | HG00097#2 - HG00232#2 | 10.23 Mb | 13.1-23.4 Mb | 0.956 |
| 4 | HG00253#1 - HG01784#1 | 9.69 Mb | 25.3-35.0 Mb | 0.950 |
| 5 | HG00126#1 - NA20806#1 | 9.49 Mb | 12.8-22.2 Mb | 0.951 |

## 5. Biological Interpretation

### 5.1 Population Differences

The dramatic difference between EUR and AFR IBD sharing reflects their demographic histories:

1. **EUR (95% of pairs have >= 2Mb IBD)**
   - Recent population bottleneck during Out-of-Africa migration (~50-70 kya)
   - Founder effects in European populations
   - Smaller effective population size (Ne ~ 10,000-20,000)

2. **AFR (28% of pairs have >= 2Mb IBD)**
   - Larger, more diverse population with deeper coalescence times
   - Higher effective population size (Ne ~ 30,000-50,000)
   - Greater genetic diversity (pi = 0.125% vs 0.085% for EUR)

### 5.2 Segment Length Distribution

The longer average segment length in EUR (3.03 Mb vs 2.54 Mb) is consistent with:
- More recent common ancestry on average
- Less time for recombination to break up shared haplotypes

### 5.3 Potential Applications

These long IBD segments can be used for:
- Cryptic relatedness detection
- Phasing refinement
- Founder effect analysis
- Selection scan validation (regions with unusually high/low IBD)

## 6. Figures

1. `fig1_2mb_main_analysis.png` - Overview of segment counts and length distribution
2. `fig2_2mb_population_comparison.png` - IBD fraction comparison with summary table
3. `fig3_2mb_top_segments.png` - Top 15 longest segments per population
4. `fig4_2mb_genomic_distribution.png` - Genomic distribution of IBD segments

## 7. Data Files

- `json/AFR_2mb_full_results.json` - Complete AFR results with all segments
- `json/EUR_2mb_full_results.json` - Complete EUR results with all segments
