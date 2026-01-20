# Full Chromosome 2 IBD Analysis Report

**Generated**: 2026-01-16 14:47:26
**Data Source**: HPRC v2 Pangenome
**Chromosome**: chr2 (243.2 Mb)
**Window Size**: 5 kb
**Method**: HMM with Forward-Backward Algorithm

## 1. Methods

### 1.1 Data Generation

IBS data was generated using `ibs-cli` with the following parameters:
- Reference: CHM13 (T2T)
- Region: chr2:1-243,199,373 (full chromosome)
- Window size: 5,000 bp
- Identity cutoff: 0.999

### 1.2 IBD Detection Model

| Population | Non-IBD Mean | Non-IBD Std | IBD Mean | IBD Std |
|------------|--------------|-------------|----------|---------|
| AFR | 0.99875 | 0.00087 | 0.9997 | 0.0005 |
| EUR | 0.99915 | 0.00071 | 0.9997 | 0.0005 |
| EAS | 0.99920 | 0.00069 | 0.9997 | 0.0005 |

## 2. Results

### 2.1 Summary Statistics

| Population | Pairs | Mean IBD (Mb) | Mean Fraction | Segments | Mean Length (kb) |
|------------|-------|---------------|---------------|----------|------------------|
| AFR | 25 | 5.84 | 0.117 | 689 | 212.0 |
| EUR | 25 | 12.98 | 0.260 | 1291 | 251.5 |
| EAS | 25 | 16.91 | 0.338 | 1524 | 277.4 |

### 2.2 Longest IBD Segments

**AFR**:
- HG02922 - HG02965: 3.27 Mb (posterior: 0.997, pos: 3.7-7.0 Mb)
- HG02976 - HG03270: 1.14 Mb (posterior: 0.996, pos: 47.6-48.8 Mb)
- HG03225 - NA20346: 1.13 Mb (posterior: 0.996, pos: 31.6-32.8 Mb)
- HG03521 - NA19036: 0.86 Mb (posterior: 0.994, pos: 26.7-27.6 Mb)
- NA19391 - NA19468: 0.80 Mb (posterior: 0.988, pos: 35.1-35.9 Mb)

**EUR**:
- HG00272 - HG00272: 3.76 Mb (posterior: 0.998, pos: 26.5-30.3 Mb)
- HG00126 - HG00140: 1.76 Mb (posterior: 0.996, pos: 0.9-2.7 Mb)
- HG00128 - HG00146: 1.71 Mb (posterior: 0.994, pos: 1.0-2.7 Mb)
- HG01784 - NA20806: 1.67 Mb (posterior: 0.991, pos: 26.8-28.4 Mb)
- HG00232 - HG00253: 1.61 Mb (posterior: 0.991, pos: 26.7-28.3 Mb)

**EAS**:
- NA18940 - NA18940: 4.26 Mb (posterior: 0.999, pos: 0.0-4.3 Mb)
- NA18967 - NA18970: 1.57 Mb (posterior: 0.993, pos: 31.2-32.8 Mb)
- NA18565 - NA18952: 1.38 Mb (posterior: 0.993, pos: 31.6-33.0 Mb)
- HG02178 - NA18974: 1.34 Mb (posterior: 0.987, pos: 27.0-28.3 Mb)
- NA18570 - NA18952: 1.33 Mb (posterior: 0.990, pos: 27.0-28.3 Mb)

### 2.3 Quality Metrics

- Total segments detected: 3504
- Mean posterior probability: 0.938
- Segments with P(IBD) > 0.9: 84.1%
- Segments with P(IBD) > 0.95: 50.5%

## 3. Key Findings

1. **Longest mean IBD segments**: EAS (277.4 kb)
   - Consistent with population bottleneck history

## 4. Conclusion

This full chromosome analysis provides comprehensive IBD patterns across
the entire chr2, enabling detection of both short and long IBD segments
with rigorous posterior probability assessment.
