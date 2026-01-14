# Results Template: Pangenome-Based IBS/IBD Analysis

This document provides standardized templates for reporting IBS/IBD results from the HPRCv2-IBD pipeline. Consistent reporting ensures reproducibility and facilitates cross-study comparisons.

---

## 1. IBS Window Analysis Results

### 1.1 Summary Statistics Template

**Table: IBS Detection Summary**

| Metric | Value | Unit |
|--------|-------|------|
| Reference genome | [e.g., CHM13] | - |
| Chromosome | [e.g., chr20] | - |
| Region analyzed | [start]-[end] | bp |
| Total region length | [value] | Mb |
| Window size | [value] | bp |
| Total windows | [value] | count |
| Identity threshold | [value] | fraction |
| Samples analyzed | [value] | count |
| Haplotypes analyzed | [value] | count |
| Total haplotype pairs | [value] | count |
| IBS-positive windows (total) | [value] | count |
| Mean IBS pairs per window | [value] | count |
| Median IBS pairs per window | [value] | count |
| Processing time | [value] | hours |

### 1.2 Per-Chromosome Summary Template

**Table: Chromosome-Level IBS Statistics**

| Chromosome | Length (Mb) | Windows | IBS Events | Mean Identity | Coverage (%) |
|------------|-------------|---------|------------|---------------|--------------|
| chr1 | | | | | |
| chr2 | | | | | |
| ... | | | | | |
| chr22 | | | | | |
| chrX | | | | | |
| **Total** | | | | | |

### 1.3 Population-Level Statistics Template

**Table: IBS Sharing by Population**

| Population | N Samples | Mean Pairwise IBS (%) | Median IBS Segment Length (kb) | IBS Density |
|------------|-----------|----------------------|-------------------------------|-------------|
| AFR | | | | |
| AMR | | | | |
| EAS | | | | |
| EUR | | | | |
| SAS | | | | |

---

## 2. Identity State Distribution Results

### 2.1 Jacquard-Style Coefficients Template

**Table: Identity State Frequencies for Sample Pair [A] vs [B]**

| State | Configuration | Frequency | Count | Interpretation |
|-------|---------------|-----------|-------|----------------|
| S1 | All four identical | [0.xxxx] | [n] | Complete identity |
| S2 | A autozygous, B autozygous, no sharing | [0.xxxx] | [n] | Independent homozygosity |
| S3 | Three-way (2A + 1B) | [0.xxxx] | [n] | Partial A-dominant sharing |
| S4 | A autozygous only | [0.xxxx] | [n] | A homozygous, no B sharing |
| S5 | Three-way (1A + 2B) | [0.xxxx] | [n] | Partial B-dominant sharing |
| S6 | B autozygous only | [0.xxxx] | [n] | B homozygous, no A sharing |
| S7 | Cross-pairs identical | [0.xxxx] | [n] | Complementary sharing |
| S8 | Single cross-pair | [0.xxxx] | [n] | Single haplotype sharing |
| S9 | All four different | [0.xxxx] | [n] | No observed IBS |

**Metadata:**
- Chromosome: [value]
- Region: [start]-[end]
- Window size: [value] bp
- Total windows: [value]
- Windows with data: [value]
- Unclassified windows: [value]

### 2.2 Population Matrix Template

**Table: Mean S7 (Cross-Pair Identity) by Population Pair**

|     | AFR | AMR | EAS | EUR | SAS |
|-----|-----|-----|-----|-----|-----|
| AFR | | | | | |
| AMR | | | | | |
| EAS | | | | | |
| EUR | | | | | |
| SAS | | | | | |

---

## 3. IBD Segment Results

### 3.1 Segment Detection Summary

**Table: IBD Segment Summary Statistics**

| Metric | Value | Unit |
|--------|-------|------|
| Total IBD segments detected | [value] | count |
| Total IBD length (genome-wide) | [value] | Mb |
| Mean segment length | [value] | kb |
| Median segment length | [value] | kb |
| Segment length SD | [value] | kb |
| Minimum segment length | [value] | kb |
| Maximum segment length | [value] | kb |
| Mean sequence identity in segments | [value] | fraction |
| Segments per sample pair (mean) | [value] | count |

### 3.2 IBD Segment Length Distribution Template

**Table: IBD Segment Length Distribution**

| Length Range (kb) | Count | Percentage | Cumulative (%) |
|-------------------|-------|------------|----------------|
| 10-50 | | | |
| 50-100 | | | |
| 100-500 | | | |
| 500-1000 | | | |
| 1000-5000 | | | |
| > 5000 | | | |

### 3.3 Individual Segment Output Format

**Output File Header:**

```
chrom	start	end	hap_a	hap_b	n_windows	mean_identity	min_identity	length_bp	fraction_called
```

**Example Rows:**

```
chr20	1000000	1500000	HG00096#1	HG00097#1	100	0.9987	0.9954	500001	0.98
chr20	2500000	3200000	HG00096#1	HG00097#2	140	0.9992	0.9978	700001	1.00
```

---

## 4. HMM Inference Results

### 4.1 Model Parameters Template

**Table: HMM Parameters Used**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial P(non-IBD) | [value] | Prior probability of non-IBD state |
| Initial P(IBD) | [value] | Prior probability of IBD state |
| P(enter IBD) | [value] | Transition: non-IBD to IBD |
| P(exit IBD) | [value] | Transition: IBD to non-IBD |
| Expected IBD length | [value] windows | Used to derive transitions |
| Non-IBD emission mean | [value] | Gaussian mean for state 0 |
| Non-IBD emission SD | [value] | Gaussian SD for state 0 |
| IBD emission mean | [value] | Gaussian mean for state 1 |
| IBD emission SD | [value] | Gaussian SD for state 1 |
| Emission estimation method | [k-means/default] | How parameters were set |

### 4.2 State Assignment Summary

**Table: HMM State Assignment Summary**

| Metric | Value | Percentage |
|--------|-------|------------|
| Windows assigned to non-IBD | [value] | [%] |
| Windows assigned to IBD | [value] | [%] |
| State transitions detected | [value] | - |
| Mean IBD run length | [value] windows | - |
| Mean non-IBD run length | [value] windows | - |

---

## 5. Comparative Analysis Results

### 5.1 Tool Comparison Template

**Table: IBD Detection Comparison with Other Methods**

| Metric | HPRCv2-IBD | GERMLINE2 | RefinedIBD | IBIS | Hap-IBD |
|--------|------------|-----------|------------|------|---------|
| Input data type | Pangenome | VCF | VCF | VCF | VCF |
| Total segments | | | | | |
| Mean segment length (kb) | | | | | |
| Sensitivity (%) | | | | | |
| Specificity (%) | | | | | |
| Concordance with pedigree (%) | | | | | |
| Runtime (hours) | | | | | |

### 5.2 Concordance Matrix Template

**Table: Pairwise Method Concordance (Jaccard Index)**

|             | HPRCv2-IBD | GERMLINE2 | RefinedIBD | IBIS |
|-------------|------------|-----------|------------|------|
| HPRCv2-IBD | 1.000 | | | |
| GERMLINE2 | | 1.000 | | |
| RefinedIBD | | | 1.000 | |
| IBIS | | | | 1.000 |

### 5.3 Overlap Analysis Template

**Table: Segment Overlap with [Comparison Tool]**

| Category | Count | Percentage | Total Length (Mb) |
|----------|-------|------------|-------------------|
| Exact match | | | |
| Partial overlap (>50%) | | | |
| Partial overlap (10-50%) | | | |
| Minimal overlap (<10%) | | | |
| HPRCv2-IBD only | | | |
| Comparison tool only | | | |

---

## 6. Required Figures

### 6.1 Essential Figures Checklist

**Main Text Figures:**

1. **Figure 1: Pipeline Overview**
   - Flowchart showing data flow from pangenome to IBD calls
   - Include impg, IBS detection, state classification, HMM stages

2. **Figure 2: IBS Density Heatmap**
   - Chromosome-level heatmap of IBS sharing
   - Rows: samples (ordered by population)
   - Columns: genomic windows
   - Color: IBS density

3. **Figure 3: IBD Segment Length Distribution**
   - Histogram or density plot
   - Compare across populations
   - Log-scale x-axis recommended

4. **Figure 4: Identity State Distribution**
   - Bar chart of S1-S9 frequencies
   - Stratified by relationship type (unrelated, cousins, siblings)

5. **Figure 5: Population Structure from IBD**
   - PCA or MDS based on IBD sharing
   - Color by superpopulation
   - Include 95% confidence ellipses

**Supplementary Figures:**

- S1: Window size sensitivity analysis
- S2: Identity threshold ROC curves
- S3: HMM parameter sensitivity
- S4: Method comparison scatter plots
- S5: Chromosome-level IBD density profiles
- S6: Per-sample IBD burden distribution

### 6.2 Figure Specifications

| Figure | Dimensions | Format | Resolution |
|--------|------------|--------|------------|
| Main text | 180mm width | PDF/SVG | 300+ dpi |
| Supplementary | 180mm width | PDF/SVG | 300+ dpi |
| Heatmaps | Variable | PNG/PDF | 300+ dpi |

---

## 7. Statistical Tests

### 7.1 Recommended Statistical Analyses

**Population Comparisons:**
- Kruskal-Wallis test for IBD segment length across populations
- Post-hoc Dunn's test with Bonferroni correction
- Report: test statistic, p-value, effect size

**Method Concordance:**
- Cohen's kappa for segment classification agreement
- Pearson/Spearman correlation for length estimates
- Bland-Altman analysis for systematic bias

**Sensitivity Analysis:**
- Bootstrap confidence intervals (1000 replicates)
- Jackknife standard errors for population estimates

### 7.2 Statistical Summary Template

```
Statistical Test: [Test name]
Comparison: [Groups compared]
Test statistic: [value]
Degrees of freedom: [value]
P-value: [value]
Effect size: [value] ([measure name])
Interpretation: [Brief statement]
```

---

## 8. Data Availability Statement Template

```
Data Availability

The HPRC pangenome assemblies used in this study are available from:
- AGC archive: [URL]
- Alignment files: [URL]
- Implicit graph: [URL]

Raw IBS/IBD results are available at: [Repository URL]

The HPRCv2-IBD pipeline is available at: https://github.com/[repo]
Version used: [tag/commit]
```

---

## 9. Reporting Checklist

Before submission, ensure the following are reported:

**Methods:**
- [ ] Window size and rationale
- [ ] Identity threshold and rationale
- [ ] HMM parameters (or estimation method)
- [ ] Minimum segment length filter
- [ ] Software versions (impg, HPRCv2-IBD)

**Results:**
- [ ] Total samples and haplotypes analyzed
- [ ] Number of IBD segments detected
- [ ] Segment length distribution statistics
- [ ] Population-stratified results
- [ ] Comparison with at least one alternative method

**Data:**
- [ ] Data availability statement
- [ ] Code availability statement
- [ ] Reproducibility information (parameters, random seeds)
