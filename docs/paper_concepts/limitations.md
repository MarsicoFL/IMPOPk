# Limitations and Caveats

This document describes known limitations, edge cases, and important considerations when using the HPRCv2-IBD pipeline for pangenome-based IBS/IBD detection. Understanding these limitations is essential for proper interpretation of results.

---

## 1. Methodological Limitations

### 1.1 IBS vs. IBD Distinction

**Fundamental Limitation:**
The pipeline detects Identity-By-State (IBS), which is a necessary but not sufficient condition for Identity-By-Descent (IBD). Two haplotypes may show high sequence identity for reasons other than recent shared ancestry:

| Scenario | Description | Impact |
|----------|-------------|--------|
| **Identity by Chance (IBC)** | Common variants shared due to high population frequency | False positive IBD calls |
| **Coalescent Identity** | Deep coalescence in regions of low diversity | Overestimation of recent IBD |
| **Selective Sweeps** | Reduced diversity due to positive selection | Region-specific false positives |
| **Balancing Selection** | Ancient haplotypes maintained by selection | May mask true IBD patterns |

**Mitigation:**
- The HMM approach helps distinguish sustained IBD from sporadic IBS
- Population-specific allele frequencies can inform prior expectations
- Comparison with pedigree data for validation when available

### 1.2 Binary IBD Model

**Limitation:** The pipeline uses a binary (IBD = 0 or 1) model when comparing two haplotypes, rather than probabilistic IBD scores.

**Implications:**
- No uncertainty quantification at the segment level
- Boundary positions are point estimates without confidence intervals
- Cannot express partial or uncertain IBD states

**When This Matters:**
- Near segment boundaries where IBD status is ambiguous
- In regions with intermediate sequence identity
- For downstream analyses requiring posterior probabilities

### 1.3 Window-Based Discretization

**Limitation:** Continuous genomic coordinates are discretized into fixed-size windows.

**Implications:**

| Window Size | Trade-off |
|-------------|-----------|
| Small (1-2 kb) | Higher resolution but increased noise, more impg queries |
| Medium (5 kb) | Balanced resolution and stability (recommended) |
| Large (10+ kb) | Stable estimates but may miss short IBD segments |

**Edge Cases:**
- IBD segment boundaries do not align with window boundaries
- Very short IBD segments (< window size) cannot be detected
- Reported segment boundaries have resolution limited to window size

---

## 2. Technical Limitations

### 2.1 Window Size Constraints

**Minimum Window Size:**
- Practical minimum: ~1,000 bp
- Below this, impg query overhead dominates runtime
- Sequence identity becomes unreliable in very short regions

**Maximum Window Size:**
- No hard limit, but >50 kb windows lose resolution
- May miss recombination breakpoints within windows
- Reduces power to detect short IBD segments

**Recommended Range:** 2,000 - 10,000 bp

### 2.2 Identity Threshold Selection

**The threshold paradox:**
- High threshold (>0.999): Reduces false positives but misses true IBD with sequencing errors
- Low threshold (<0.99): Captures more true IBD but increases false positive rate

**Threshold Sensitivity:**

| Threshold | Typical Use Case | Caveat |
|-----------|------------------|--------|
| 1.0 (exact match) | Identical twins, very recent IBD | Misses any sequencing variation |
| 0.9999 | Close relatives | May miss IBD with assembly errors |
| 0.999 | General IBD detection | Recommended default |
| 0.99 | Ancient IBD, diverse populations | Higher false positive rate |

### 2.3 HMM Parameter Sensitivity

**Transition Probabilities:**
- Expected segment length ($L$) directly affects $P_{exit} = 1/L$
- Misspecified $L$ leads to over/under-segmentation

| If L is... | Effect |
|------------|--------|
| Too short | Over-segments true IBD regions |
| Too long | Merges distinct IBD segments, misses boundaries |

**Emission Parameters:**
- Default parameters assume bimodal identity distribution
- Data-driven estimation (k-means) may fail with:
  - Highly related populations (unimodal high identity)
  - Highly diverse populations (unimodal low identity)
  - Small sample sizes

### 2.4 Minimum Segment Length Filter

**Default: 5,000 - 10,000 bp**

**Trade-offs:**

| Filter Setting | Consequence |
|----------------|-------------|
| Too low | Retains spurious short segments from noise |
| Too high | Misses genuine short IBD (e.g., from many generations ago) |

**Population-Specific Considerations:**
- African populations: Shorter IBD expected due to larger effective population size
- Bottlenecked populations: Longer IBD segments common
- Consider adjusting filter based on population history

---

## 3. Population-Specific Considerations

### 3.1 African Populations

**Challenges:**
- Highest genetic diversity among human populations
- Shorter average IBD segments due to larger $N_e$
- More ancient population structure

**Implications:**
- May require lower identity thresholds
- Shorter minimum segment filters may be appropriate
- Higher false positive rate relative to other populations

### 3.2 Bottlenecked Populations

**Examples:** Ashkenazi Jewish, Finnish, founder populations

**Characteristics:**
- Extensive long-range IBD sharing
- Higher background IBS rates
- May show elevated S1/S2/S7 states even for unrelated individuals

**Implications:**
- IBD sharing may be detected between most pairs
- Kinship inference requires population-specific calibration
- Consider using population-matched controls

### 3.3 Admixed Populations

**Challenge:** Ancestry-specific IBD patterns

**Considerations:**
- IBD segments may reflect different ancestral populations
- Local ancestry affects expected IBD sharing
- Population structure can confound IBD-based analyses

**Recommendations:**
- Consider local ancestry inference alongside IBD
- Stratify analyses by ancestral background when possible
- Use appropriate reference panels for comparison

### 3.4 Isolated and Underrepresented Populations

**Limitations:**
- Reference assemblies may not represent all populations equally
- Alignment quality may vary for underrepresented groups
- Population-specific variants may affect identity calculations

---

## 4. Data and Input Limitations

### 4.1 Assembly Quality Dependencies

**Requirements for accurate IBD detection:**
- High-quality, contiguous assemblies
- Consistent assembly methodology across samples
- Accurate sequence representation

**Potential Issues:**

| Assembly Problem | Impact on IBD Detection |
|------------------|------------------------|
| Misassembly | False IBD calls or missed true IBD |
| Collapsed repeats | Artificial IBS in repetitive regions |
| Haplotype switching | Spurious recombination signals |
| Low coverage regions | Missing data, gaps in IBD tracks |
| Contamination | False sharing with contaminant |

### 4.2 Reference Alignment Limitations

**The impg approach depends on:**
- Accurate alignment to reference (CHM13)
- Consistent alignment parameters across haplotypes
- Correct handling of structural variants

**Known Issues:**
- Regions not well-represented in reference may have poor alignments
- Large structural variants may cause alignment artifacts
- Reference bias may affect identity calculations

### 4.3 Missing Data Handling

**Current Approach:**
- Windows without impg output treated as missing
- Gap tolerance parameter allows bridging over missing windows
- Missing windows counted as S9 (no IBS) for Jacquard calculations

**Limitations:**
- Cannot distinguish "no IBS" from "no data"
- High missing rate may bias results
- Consider excluding regions with >X% missing data

---

## 5. Computational Limitations

### 5.1 Scalability

**Current Constraints:**

| Operation | Scaling | Practical Limit |
|-----------|---------|-----------------|
| impg queries | O(N windows) | ~100,000 windows/chromosome |
| Pairwise comparisons | O(P pairs) | ~200,000 pairs practical |
| Memory (IBS table) | O(N x P_IBS) | Depends on IBS density |

**For HPRC-scale data (465 haplotypes):**
- ~100,000 haplotype pairs
- Chromosome-level analysis feasible
- Genome-wide requires parallelization

### 5.2 Runtime Considerations

**Typical Processing Times (single chromosome, ~60 Mb):**

| Configuration | Approximate Time |
|---------------|-----------------|
| Single thread | 6-12 hours |
| 10 parallel workers | 1-2 hours |
| 50 parallel workers | 15-30 minutes |

**Bottlenecks:**
- impg similarity queries (disk I/O bound)
- Large output file writing
- HMM inference for many pairs

### 5.3 Storage Requirements

**Approximate storage per chromosome:**

| Output | Size Estimate |
|--------|---------------|
| Raw IBS windows | 100 MB - 1 GB |
| Jacquard coefficients | ~10 KB per pair |
| IBD segments | 1-10 MB |
| Total (compressed) | ~500 MB |

---

## 6. Interpretation Caveats

### 6.1 Identity States are Observational

**Key Point:** The nine identity states (S1-S9) describe observed IBS patterns, not true IBD configurations.

**Implications:**
- S7 (cross-pair matching) does not prove common ancestry
- S9 (no sharing) may reflect missing data, not true non-IBD
- State frequencies are influenced by population history

### 6.2 IBD Segment Boundaries are Approximate

**Sources of Boundary Uncertainty:**
1. Window size discretization (~5 kb resolution)
2. HMM state assignment (soft boundaries)
3. True recombination points not directly observed

**Recommendation:** Report boundaries with appropriate precision (round to nearest kb).

### 6.3 Relatedness Inference Requires Calibration

**IBD sharing alone cannot determine:**
- Exact relationship type without additional information
- Direction of relationship (who is the ancestor)
- Shared environment vs. shared genetics effects

**For kinship inference:**
- Use population-matched empirical distributions
- Consider total IBD sharing, not just segment count
- Validate against known pedigrees when possible

---

## 7. Comparison with Other Methods

### 7.1 Differences from VCF-Based Methods

| Aspect | Pangenome (HPRCv2-IBD) | VCF-Based (GERMLINE, etc.) |
|--------|------------------------|----------------------------|
| Input | Complete assemblies | Genotype calls |
| Phase information | Inherent | Statistical/imputed |
| Structural variants | Included | Often excluded |
| Rare variants | Captured | May be filtered |
| Reference bias | Minimal | Present |
| Scalability | Moderate | High |

### 7.2 Expected Differences in Results

**Compared to VCF-based methods:**
- More IBD detected in SV-rich regions
- Potentially different segment boundaries
- May detect IBD missed due to phasing errors in VCF approaches
- May miss IBD detectable only through rare SNVs

### 7.3 Validation Recommendations

When comparing to other methods:
1. Use consistent genomic regions
2. Account for different boundary definitions
3. Consider segment length distributions
4. Evaluate on samples with known relationships

---

## 8. Known Edge Cases

### 8.1 Centromeric and Telomeric Regions

**Issue:** Repetitive sequences cause unreliable alignments

**Recommendation:** Consider excluding or flagging:
- Centromeres (typically 1-5 Mb per chromosome)
- Telomeric regions (terminal ~500 kb)
- Known segmental duplications

### 8.2 Sex Chromosomes

**X Chromosome:**
- Males are hemizygous (only one haplotype)
- Requires special handling for male-female comparisons
- PAR regions behave like autosomes

**Y Chromosome:**
- Very low diversity
- Most pairs will show high IBS regardless of IBD
- Consider excluding from standard analyses

### 8.3 Mitochondrial DNA

**Not recommended for this pipeline:**
- Very high copy number
- Non-Mendelian inheritance
- Use specialized tools for mtDNA analysis

### 8.4 Highly Homozygous Individuals

**Issue:** Individuals with high autozygosity (e.g., from consanguinity) may show:
- Elevated S1, S2, S4, S6 states
- Long runs of homozygosity affecting pair comparisons

**Recommendation:**
- Calculate individual-level homozygosity statistics
- Flag highly homozygous samples
- Consider separate analysis or interpretation

---

## 9. Recommendations for Robust Analysis

### 9.1 Pre-Analysis Checks

1. Verify assembly quality metrics for all samples
2. Check for population stratification
3. Identify and handle related individuals in reference panel
4. Assess missing data rates per sample and region

### 9.2 Parameter Selection

1. Test window size sensitivity on subset of data
2. Use population-appropriate identity thresholds
3. Validate HMM parameters against known relationships
4. Document all parameter choices

### 9.3 Result Validation

1. Compare with at least one alternative method
2. Check consistency across chromosomes
3. Validate against known pedigree relationships (if available)
4. Assess population-level patterns for biological plausibility

### 9.4 Reporting Standards

1. Report all parameters used
2. Acknowledge relevant limitations
3. Provide population-stratified results when applicable
4. Include sensitivity analyses for key parameters

---

## 10. Future Improvements

Areas for potential methodological enhancement:

1. **Probabilistic IBD scoring** - Replace binary calls with posterior probabilities
2. **Population-aware priors** - Incorporate allele frequency information
3. **Adaptive windowing** - Variable window sizes based on local diversity
4. **Multi-way IBD** - Extend beyond pairwise to detect segments shared among multiple individuals
5. **Integration with local ancestry** - Account for ancestry-specific IBD patterns
6. **Improved boundary detection** - Higher resolution segment endpoints

---

## Summary

The HPRCv2-IBD pipeline provides a powerful approach to IBD detection using pangenome assemblies, but users should be aware of:

1. **The IBS/IBD distinction** - We detect IBS and infer IBD
2. **Parameter sensitivity** - Results depend on threshold and window choices
3. **Population effects** - Different populations require different considerations
4. **Technical constraints** - Assembly quality and computational resources matter
5. **Interpretation limits** - Biological conclusions require appropriate caveats

Proper use of this pipeline requires understanding these limitations and applying appropriate quality control and validation strategies.
