# Conceptualization: IBD Detection from Implicit Pangenome Graph Output

## 1. Introduction and Motivation

### 1.1 The Problem

Identity-by-Descent (IBD) refers to segments of DNA shared between individuals due to recent common ancestry. Traditional IBD detection methods (hap-ibd, GERMLINE, IBDseq) operate on phased SNP genotype data, modeling discrete allelic concordance/discordance events. Our challenge is fundamentally different: we must adapt IBD detection for **implicit pangenome graph (impg)** output, which provides continuous sequence identity values over alignment tracts rather than discrete SNP calls.

### 1.2 Key Distinctions from Traditional Approaches

| Aspect | Traditional (SNP-based) | Our Approach (impg-based) |
|--------|------------------------|---------------------------|
| **Data type** | Discrete genotypes (0/1/2) | Continuous identity [0,1] |
| **Error model** | Allelic discordance rate ε | Identity distribution |
| **Units** | cM (genetic distance) | bp (physical distance) |
| **Phasing** | Required externally | Implicit in alignments |
| **Resolution** | SNP positions | Window-based (~5kb) |

### 1.3 Scope: Haplotype IBD

We work strictly with **haplotype pairs**, not diploid individuals. Therefore:

- **IBD = 0**: No shared recent ancestry between two haplotypes
- **IBD = 1**: Shared recent ancestry (identical by descent)

This is NOT the diploid framework where IBD0/IBD1/IBD2 describes sharing 0, 1, or 2 haplotypes between two individuals.

---

## 2. Review of Existing Approaches

### 2.1 hap-ibd (Browning Lab)

**Methodology**: Seed-and-extend algorithm on phased haplotypes.

**Key characteristics**:
- Uses IBS (Identity-by-State) segments as seeds
- Extends seeds across small gaps (default: 1000 bp)
- Filters by genetic length (default min: 2.0 cM)
- No explicit statistical model for IBD probability

**Parameters**:
```
min-seed: 2.0 cM    (minimum seed length)
max-gap: 1000 bp    (maximum extension gap)
min-markers: 100    (minimum markers per segment)
```

**Relevance to our approach**: hap-ibd's seed-and-extend is algorithmically simple but lacks a probabilistic framework. It cannot quantify uncertainty in segment calls or handle the continuous identity values from impg.

### 2.2 ibd-ends (Browning Lab)

**Methodology**: Probabilistic refinement of IBD segment endpoints.

**Key innovation**: Models the probability distribution of segment endpoints rather than point estimates.

**Error model** (from Browning & Browning, 2020):
```
P(discordance | IBD) = ε ≈ 0.0005
```

The model incorporates:
1. **Base-level discordances** (sequencing/phasing errors)
2. **Gene conversion tracts** (local non-IBD within IBD)

**Key parameters**:
```
err: 0.0005        (discordance rate within IBD)
gc-err: 0.1        (gene conversion tract discordance rate)
gc-bp: 1000        (maximum gene conversion tract length)
```

**Relevance to our approach**: The error rate ε provides a theoretical foundation for our emission model. In the Browning framework:

$$P(\text{discordance} | \text{IBD}) = \epsilon \approx 0.0005$$

Translating to our continuous framework:
$$E[\text{identity} | \text{IBD}] = 1 - \epsilon \approx 0.9995$$

### 2.3 IBDrecomb

**Methodology**: Detects recombination events within IBD segments.

**Relevance**: Limited direct applicability, but highlights that IBD segments are not monolithic—internal structure (recombination, gene conversion) matters.

---

## 3. Theoretical Framework

### 3.1 Observable and Latent Variables

Following the IBS conceptualization framework:

**Observable**: Sequence identity $S_i \in [0, 1]$ at window $i$

**Latent**: IBD state $Z_i \in \{0, 1\}$ at window $i$

The fundamental inference problem is:
$$P(Z_1, \ldots, Z_n | S_1, \ldots, S_n)$$

### 3.2 Hidden Markov Model Formulation

#### States
- **State 0 (Non-IBD)**: Haplotypes do not share recent common ancestry
- **State 1 (IBD)**: Haplotypes share recent common ancestry

#### Transition Model

The transition matrix captures the tendency of IBD to form contiguous tracts:

$$
A = \begin{pmatrix}
1 - \alpha & \alpha \\
\beta & 1 - \beta
\end{pmatrix}
$$

Where:
- $\alpha$ = P(enter IBD | currently non-IBD)
- $\beta$ = P(exit IBD | currently IBD)

**Relationship to expected segment length**:
$$E[\text{IBD segment length}] = \frac{1}{\beta}$$

For expected IBD segments of 50 windows:
$$\beta = \frac{1}{50} = 0.02$$

#### Emission Model: Two Alternatives

##### Alternative A: Gaussian Emissions (Current Implementation)

$$P(S_i | Z_i = 0) = \mathcal{N}(S_i; \mu_0, \sigma_0)$$
$$P(S_i | Z_i = 1) = \mathcal{N}(S_i; \mu_1, \sigma_1)$$

**Parameters** (theoretical estimates from population genetics):

| State | Mean | Std | Interpretation |
|-------|------|-----|----------------|
| Non-IBD | ~0.999 | ~0.001 | 1 - π (nucleotide diversity) |
| IBD | ~0.9995 | ~0.001 | 1 - ε (error rate) |

**Critical issue with current implementation**:
```rust
// CURRENT (INCORRECT):
emission: [
    GaussianParams { mean: 0.5, std: 0.2 },  // Non-IBD
    GaussianParams { mean: 0.99, std: 0.01 }, // IBD
]
```

The value 0.5 for non-IBD is biologically implausible. Human haplotypes share ~99.9% identity even without recent common ancestry.

##### Alternative B: Bernoulli/Binomial Emissions (Browning-style)

For each window of $n$ positions:
$$P(S_i | Z_i = 0) = \text{Binomial}(n \cdot (1-S_i); n, p_0)$$
$$P(S_i | Z_i = 1) = \text{Binomial}(n \cdot (1-S_i); n, \epsilon)$$

Where:
- $p_0$ = expected discordance rate under no IBD (population-specific)
- $\epsilon$ = expected discordance rate under IBD (~0.0005)

**Conversion between frameworks**:

If $d$ = number of discordances in window of size $W$:
$$S = 1 - \frac{d}{W}$$

Expected values:
$$E[S | \text{Non-IBD}] = 1 - p_0 = 1 - \pi$$
$$E[S | \text{IBD}] = 1 - \epsilon \approx 0.9995$$

### 3.3 Population-Specific Background

The non-IBD emission distribution depends on population-specific nucleotide diversity (π):

| Population | π | E[S|Non-IBD] | σ (theoretical) |
|------------|---|--------------|-----------------|
| AFR | 0.00125 | 0.99875 | 0.00087 |
| EUR | 0.00085 | 0.99915 | 0.00071 |
| EAS | 0.00080 | 0.99920 | 0.00069 |
| CSA | 0.00095 | 0.99905 | 0.00076 |
| AMR | 0.00100 | 0.99900 | 0.00078 |

**Inter-population comparisons** (using Fst):

| Comparison | Fst | E[S|Non-IBD] | σ |
|------------|-----|--------------|---|
| AFR-EAS | 0.192 | 0.99873 | 0.00101 |
| AFR-EUR | 0.153 | 0.99876 | 0.00100 |
| EUR-EAS | 0.111 | 0.99907 | 0.00086 |

### 3.4 The Discrimination Challenge

The fundamental challenge is distinguishing:
- **Non-IBD**: identity ≈ 0.999 (1 - π)
- **IBD**: identity ≈ 0.9995 (1 - ε)

This is a **0.05% difference** in identity, not the 50% difference implied by the current (incorrect) default of 0.5.

**Implications**:
1. The Gaussian emissions must have very small standard deviations
2. Multiple windows are needed for confident calls
3. Filtering thresholds must be carefully calibrated

---

## 4. Adaptation for impg Output

### 4.1 impg Similarity as Input

The implicit pangenome graph provides:
- Alignment tracts between haplotype pairs
- Sequence identity per tract

**Window construction**:
```
Input: impg alignments (target_chrom, target_start, target_end, identity)
Output: Windowed identity values (fixed 5kb windows)
```

For each window, we aggregate identity from overlapping alignments:
$$S_w = \frac{\sum_i |A_i \cap W| \cdot s_i}{\sum_i |A_i \cap W|}$$

Where $A_i$ is alignment $i$ with identity $s_i$ and $W$ is the window.

### 4.2 Handling Missing Data

Unlike SNP data, impg may have regions without alignments (structural variants, highly divergent regions). Options:

1. **Skip windows**: Reduce resolution but maintain accuracy
2. **Impute as non-IBD**: Conservative, may miss IBD
3. **Model explicitly**: Add a "missing" emission distribution

**Recommendation**: Skip windows without sufficient alignment coverage (< 50% of window).

### 4.3 Window Size Considerations

| Window Size | Pros | Cons |
|-------------|------|------|
| 1 kb | High resolution | Noisy identity estimates |
| 5 kb | Balance | Standard choice |
| 10 kb | Stable estimates | May miss short IBD |
| 20 kb | Very stable | Low resolution |

For IBD detection, 5 kb windows provide a good balance. Shorter windows increase noise; longer windows may merge distinct IBD segments.

---

## 5. Alternative Modeling Approaches

### 5.1 Approach A: Gaussian HMM (Current)

**Model**:
$$P(S | Z=0) = \mathcal{N}(\mu_0, \sigma_0)$$
$$P(S | Z=1) = \mathcal{N}(\mu_1, \sigma_1)$$

**Pros**:
- Simple, well-understood
- Continuous emissions natural for identity data
- Fast inference (Viterbi)

**Cons**:
- Requires correct parameterization (current defaults wrong)
- Gaussian may not capture true distribution shape
- Sensitive to outliers

**Corrected parameters**:
```rust
emission: [
    GaussianParams { mean: 0.998, std: 0.002 },  // Non-IBD (conservative)
    GaussianParams { mean: 0.9995, std: 0.001 }, // IBD
]
```

### 5.2 Approach B: Browning-style Discordance Model

**Model**:
$$P(d | Z=1) = \text{Binomial}(d; n, \epsilon)$$
$$P(d | Z=0) = \text{Binomial}(d; n, p_0)$$

Where $d$ = discordances in window, $n$ = window size.

**Pros**:
- Theoretically grounded (mutation/error rates)
- Naturally handles different window sizes
- Better interpretability

**Cons**:
- Requires converting identity to discordance counts
- Integer approximation introduces discretization error
- May not capture alignment artifacts

**Implementation**:
```rust
fn discordance_count(identity: f64, window_size: usize) -> usize {
    ((1.0 - identity) * window_size as f64).round() as usize
}

fn binomial_pmf(k: usize, n: usize, p: f64) -> f64 {
    let log_binom = ln_gamma(n+1) - ln_gamma(k+1) - ln_gamma(n-k+1);
    (log_binom + k as f64 * p.ln() + (n-k) as f64 * (1.0-p).ln()).exp()
}
```

### 5.3 Approach C: Beta Emissions

**Model**:
$$P(S | Z=0) = \text{Beta}(\alpha_0, \beta_0)$$
$$P(S | Z=1) = \text{Beta}(\alpha_1, \beta_1)$$

**Pros**:
- Natural support on [0, 1]
- Flexible shape (can be skewed, bimodal edges)
- Better for proportions/identities

**Cons**:
- More complex parameter estimation
- Less interpretable parameters
- May overfit to noise

**Parameter estimation from moments**:
$$\alpha = \mu \left( \frac{\mu(1-\mu)}{\sigma^2} - 1 \right)$$
$$\beta = (1-\mu) \left( \frac{\mu(1-\mu)}{\sigma^2} - 1 \right)$$

### 5.4 Approach D: Mixture Model Detection

**Idea**: First cluster identity values, then run HMM.

**Algorithm**:
1. Run Gaussian Mixture Model on all identity values
2. Identify clusters as Non-IBD/IBD based on means
3. Use cluster parameters as HMM emissions

**Pros**:
- Data-driven parameter estimation
- Adapts to specific datasets
- Current implementation partially does this

**Cons**:
- Requires sufficient IBD signal for clustering
- May fail on datasets with mostly non-IBD
- Two-stage inference loses joint optimization

### 5.5 Recommendation

**Primary**: Use Gaussian HMM with corrected population-specific parameters.

**Fallback**: Data-driven estimation via k-means (current `estimate_emissions`).

**Rationale**: The Gaussian model is simple, fast, and sufficient given the small difference between IBD and non-IBD distributions. The key improvement is using biologically correct default parameters.

---

## 6. IBD Prior from Population Genetics

### 6.1 Expected IBD Sharing

For two haplotypes from the same population, the probability of IBD at any locus depends on:
- Population size (Ne)
- Time to most recent common ancestor (TMRCA)
- Recombination rate

**Wright-Fisher approximation**:
$$P(\text{IBD at locus}) \approx \frac{1}{2N_e}$$

For Ne ≈ 10,000 (human effective population size):
$$P(\text{IBD}) \approx 0.00005$$

This is the prior probability of IBD at any random locus.

### 6.2 IBD Segment Length Distribution

The length of IBD segments follows an exponential distribution with rate dependent on the number of meioses separating two haplotypes:

$$L \sim \text{Exponential}(\frac{k}{100})$$

Where $k$ = number of meioses and $L$ is in Morgans.

For recent IBD (k ≈ 10 generations):
$$E[L] = \frac{100}{10} = 10 \text{ cM} \approx 10 \text{ Mb}$$

### 6.3 Transition Probability Calibration

Given:
- Window size: 5 kb
- Recombination rate: ~1 cM/Mb
- Expected IBD length: 10 cM (recent) to 1 cM (ancient)

For 10 cM expected IBD (~10 Mb):
$$\beta = \frac{5000 \text{ bp}}{10,000,000 \text{ bp}} = 0.0005$$

For 1 cM expected IBD (~1 Mb):
$$\beta = \frac{5000}{1,000,000} = 0.005$$

---

## 7. Validation Strategy

### 7.1 Synthetic Data Validation

**Generate synthetic haplotypes with known IBD**:
1. Create founder haplotypes with realistic diversity (π)
2. Simulate IBD sharing by copying segments
3. Add noise (sequencing error, alignment artifacts)
4. Run IBD detection
5. Compare to ground truth

**Metrics**:
- Precision: fraction of called IBD that is true IBD
- Recall: fraction of true IBD that is called
- Boundary accuracy: mean error in segment endpoints

### 7.2 Known IBD Validation (Pedigree Data)

Use samples with known pedigree relationships:
- Parent-child: 50% genome IBD
- Siblings: ~50% genome IBD (variable)
- First cousins: ~12.5% genome IBD

Compare detected IBD to expected from pedigree.

### 7.3 Cross-Method Validation

Run same samples through:
- Our impg-based method
- hap-ibd on VCF data
- IBDseq / GERMLINE

Compare segment calls for consistency.

### 7.4 Population Structure Validation

**LCT region test** (selection signal):
- High IBD in EUR (positive selection for lactase persistence)
- Background IBD in other populations

**EDAR region test** (selection in EAS)

---

## 8. Implementation Recommendations

### 8.1 Corrected Default Parameters

```rust
impl HmmParams {
    /// Population-specific parameters
    pub fn from_population(
        population: Population,
        expected_ibd_windows: f64,
        p_enter_ibd: f64,
    ) -> Self {
        let (bg_mean, bg_std) = match population {
            Population::AFR => (0.99875, 0.00087),
            Population::EUR => (0.99915, 0.00071),
            Population::EAS => (0.99920, 0.00069),
            Population::CSA => (0.99905, 0.00076),
            Population::AMR => (0.99900, 0.00078),
            Population::InterPop => (0.99890, 0.00100),
        };

        let p_stay_ibd = 1.0 - 1.0 / expected_ibd_windows;

        HmmParams {
            initial: [1.0 - p_enter_ibd, p_enter_ibd],
            transition: [
                [1.0 - p_enter_ibd, p_enter_ibd],
                [1.0 - p_stay_ibd, p_stay_ibd],
            ],
            emission: [
                GaussianParams { mean: bg_mean, std: bg_std },
                GaussianParams { mean: 0.9995, std: 0.001 },
            ],
        }
    }
}
```

### 8.2 Robust Emission Estimation

Improve `estimate_emissions` to:
1. Use population-specific priors
2. Handle unimodal data (mostly non-IBD or mostly IBD)
3. Apply regularization to prevent extreme estimates

```rust
pub fn estimate_emissions_robust(
    &mut self,
    observations: &[f64],
    population_prior: Option<Population>,
) {
    // Use prior as regularization
    let (prior_mean, prior_std) = match population_prior {
        Some(pop) => get_population_params(pop),
        None => (0.999, 0.001),  // Generic prior
    };

    // Only update if clear bimodal structure
    if let Some((clusters, _)) = kmeans_1d(observations, 2, 20) {
        let separation = (clusters[0] - clusters[1]).abs();
        if separation > 0.001 {  // Sufficient separation
            // Update emissions
        }
    }
    // Otherwise keep prior-based defaults
}
```

### 8.3 Filtering and Post-Processing

```rust
/// Filter IBD segments by length
pub fn filter_ibd_segments(
    segments: Vec<IbdSegment>,
    min_windows: usize,      // Minimum 10 windows (50 kb)
    min_mean_identity: f64,  // Minimum 0.999 mean identity
) -> Vec<IbdSegment> {
    segments.into_iter()
        .filter(|s| s.n_windows >= min_windows)
        .filter(|s| s.mean_identity >= min_mean_identity)
        .collect()
}
```

---

## 9. Open Questions and Future Directions

### 9.1 Structural Variant Handling

How should large insertions/deletions affect IBD calls?
- Option 1: Treat as missing data
- Option 2: Model separately (SV-specific emission)
- Option 3: Use graph coordinates instead of linear reference

### 9.2 Genotyping Error vs. True Variation

The current model conflates:
- Sequencing/assembly errors
- True mutations since MRCA
- Alignment artifacts

A more sophisticated model could separate these.

### 9.3 Multi-Way IBD

Currently: pairwise IBD between two haplotypes.
Extension: multi-way IBD among >2 haplotypes (population-level IBD clusters).

### 9.4 IBD Timing

Current model: binary IBD (yes/no).
Extension: estimate TMRCA or number of meioses for IBD segments.

---

## 10. Summary

### Key Contributions

1. **Adaptation for impg**: Framework for IBD detection from continuous identity values rather than discrete SNPs.

2. **Population-specific parameters**: Recognition that non-IBD background varies by population (0.999 not 0.5).

3. **Theoretical grounding**: Connection between population genetics (π, ε) and HMM emission parameters.

### Critical Fixes Needed

1. **Default emission parameters**: Change non-IBD mean from 0.5 to ~0.999.

2. **Population awareness**: Add population-specific parameter sets.

3. **Robust estimation**: Improve `estimate_emissions` to handle edge cases.

### Recommended Approach

Use **Gaussian HMM** with:
- Population-specific non-IBD emission (mean = 1 - π)
- Fixed IBD emission (mean = 0.9995, std = 0.001)
- Data-driven refinement when bimodal structure exists

This provides a principled, biologically grounded framework for IBD detection from implicit pangenome graph output.

---

## References

1. Browning, B. L., & Browning, S. R. (2020). Statistical phasing of 150,119 sequenced genomes. American Journal of Human Genetics, 107(5), 936-946.

2. Browning, S. R., & Browning, B. L. (2012). Identity by descent between distant relatives: detection and applications. Annual Review of Genetics, 46, 617-633.

3. Zhou, Y., et al. (2020). IBDrecomb: Fine mapping of IBD recombination events. American Journal of Human Genetics.

4. HPRC (2024). Human Pangenome Reference Consortium assemblies v2.

5. 1000 Genomes Project Consortium (2015). A global reference for human genetic variation. Nature, 526(7571), 68-74.
