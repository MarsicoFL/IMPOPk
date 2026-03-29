# Methodology

Scientific overview of the impopₖ approach to IBD detection, local ancestry inference, and relatedness estimation from pangenome alignments.

## 1. Overview: Assembly Identity as a TMRCA Estimator

Under the coalescent, pairwise sequence identity between two haplotypes directly estimates the time to their most recent common ancestor (TMRCA):

$$I = 1 - 2\mu t$$

where $\mu$ is the per-base mutation rate and $t$ is the TMRCA in generations. This relationship is accurate to first order under the infinite-sites model (the exact form is $I = e^{-2\mu t}$) and makes pairwise identity a sufficient statistic for TMRCA at each genomic locus.

Four independent research programs have converged on the conclusion that TMRCA-based relatedness is superior to genotype-based measures: the expected GRM (eGRM) from ancestral recombination graphs outperforms the canonical GRM in 97.5% of simulated association settings (Fan et al. 2022); the Branch GRM from genealogical trees provides a theoretically optimal measure (Lehmann et al. 2026); GeSi classifies TMRCA-based measures as "full" versus the "shallow" signal captured by standard GRM or KING (GeSi 2025); and Super Admixture connects admixture proportions directly to the coancestry matrix via per-population identity means (Chen et al. 2025).

Haplotype-resolved assemblies from the Human Pangenome Reference Consortium (HPRC) provide direct access to this signal. Pairwise identity from assemblies captures all variation simultaneously---SNPs, indels, and structural variants---without ascertainment bias or phasing error, and with reduced reference bias. The per-window identity matrix across $n$ haplotypes is, in effect, a local expected GRM estimated directly from assemblies, without requiring VCF generation, statistical phasing, or ARG inference.

## 2. IBS Computation: From PAF Alignments to Pairwise Identity Windows

Identity-by-state (IBS) values serve as the primary observations for all downstream HMM inference. Given a target genomic region and a window size $W$ (default 10 kb), we partition the region into non-overlapping windows and compute pairwise sequence identity between all haplotype pairs in a specified sample subset.

### Two computation modes

**impg-based mode** (`ibs`): Per-window queries via impg, which projects alignments through the pangenome graph structure using the reference coordinate system. This mode requires impg and AGC as external dependencies.

**PAF-direct mode** (`ibs-from-paf`): Parses CIGAR strings from the full PAF alignment file in a single streaming pass. Self-contained with no external dependencies, and empirically >1,000x faster than the impg-based mode.

### Window identity aggregation

For each window, identity from overlapping alignments is aggregated as a coverage-weighted mean:

$$S_w = \frac{\sum_i |A_i \cap W| \cdot s_i}{\sum_i |A_i \cap W|}$$

where $A_i$ is alignment $i$ with identity $s_i$ and $W$ is the window.

### Filtering

Raw output is filtered by: identity cutoff (>= 0.999), self-pair removal, reference removal, and canonical pair ordering to avoid duplicates. The output is a tab-separated file of per-window, per-pair identity values forming the input to IBD and ancestry HMMs.

## 3. IBD Detection: 2-State HMM

### Model formulation

IBD detection is formulated as a two-state Hidden Markov Model operating on windowed sequence identity values. The model distinguishes regions of shared recent ancestry (IBD, state 1) from background similarity (non-IBD, state 0).

### Emissions

Each state emits identity values $o_t \in [0,1]$ according to a Gaussian distribution:

$$P(o_t \mid s_t = s) = \mathcal{N}(o_t; \mu_s, \sigma_s^2)$$

The non-IBD mean derives from coalescent theory: $\mu_0 = 1 - \theta$, where $\pi \approx 4\mu N_e$ is the population-scaled nucleotide diversity. Population-specific $\pi$ values range from 0.00080 (EAS) to 0.00125 (AFR), yielding expected non-IBD identity of 0.99875 to 0.99920. The IBD mean is $\mu_1 = 0.9997$ with $\sigma_1 = 0.0005$, reflecting only sequencing and assembly errors.

| Population | $\pi$ (SNPs/bp) | E[identity | non-IBD] | SNR |
|------------|------------------|-------------------------|-----|
| AFR | 0.00125 | 0.99875 | 1.06 |
| EUR | 0.00085 | 0.99915 | 0.74 |
| EAS | 0.00080 | 0.99920 | 0.69 |
| CSA | 0.00095 | 0.99905 | 0.81 |
| AMR | 0.00100 | 0.99900 | 0.85 |

The per-window signal-to-noise ratio ranges from 0.69 (EAS) to 1.06 (AFR). Reliable detection requires accumulating evidence over multiple consecutive windows, corresponding to minimum detectable segments of approximately 90 kb (AFR) to 190 kb (EAS). African populations have the best detection power because the identity gap ($\Delta\mu \approx \theta$) is proportionally larger.

### Transitions

The transition matrix is parameterized by expected IBD segment length $L$ (in windows) and entry probability $p_{\text{enter}}$: $P(\text{stay IBD}) = 1 - 1/L$, with population-adaptive scaling reflecting coalescent depth. Distance-dependent transitions and recombination-rate-aware transitions via genetic maps improve biological realism.

### Parameter estimation

Emission parameters are estimated hierarchically:

1. **k-means clustering** into two groups
2. **EM fallback** with MAP regularization if clusters are degenerate ($\Delta\mu \leq 0.0005$)
3. **BIC model selection** to avoid overfitting a two-component model when the data does not support it

All parameters are then refined via Baum-Welch, which re-estimates emissions and transitions jointly over 20 iterations. Biological bounds are enforced after each M-step to maintain identifiability ($\mu_0 < \mu_1$) and numerical stability.

### Inference and segment extraction

The most likely state sequence is computed via the Viterbi algorithm in log-space; posterior state probabilities $\gamma_t(s)$ are computed via forward-backward. IBD segments are extracted by run-length encoding of the Viterbi path, then refined through a 5-step pipeline:

1. **Extract** -- run-length encoding of the Viterbi path
2. **Bridge** -- merge segments separated by short gaps
3. **Merge** -- combine overlapping or adjacent segments
4. **Refine boundaries** -- adjust endpoints using posterior probabilities (extend at $\gamma > 0.5$, trim at $\gamma < 0.2$)
5. **Smooth** -- remove spurious short segments below minimum length

Each segment receives a LOD score:

$$\text{LOD} = \sum_{t} \log_{10} \frac{P(o_t \mid \text{IBD})}{P(o_t \mid \text{non-IBD})}$$

following the hap-ibd convention, and a composite quality score $Q \in [0, 100]$ combining posterior strength, consistency, LOD evidence, and segment length.

An identity floor parameter (default 0.9) filters out alignment-gap windows before HMM inference, treating them as missing data rather than evidence of non-IBD. This reduces variance from $\sigma \approx 0.29$ (bimodal raw distribution) to $\sigma \approx 0.001$ (aligned regions only).

## 4. Local Ancestry Inference: N-State HMM

### Model formulation

The HMM framework extends to an $N$-state model where each hidden state corresponds to one of $N$ reference populations. Given pairwise similarity between a query haplotype and reference panel haplotypes, the model infers the most likely ancestral origin at each genomic window.

This performs a dimensional reduction relative to the Li & Stephens copying model underlying RFMix: instead of tracking which individual reference haplotype is copied at each SNP (requiring >500 haplotypes), we ask which population shows highest aggregate identity per window. This collapses the state space from $m$ haplotypes to $k$ populations; empirically, approximately 50 haplotypes suffice versus >1,400 for haplotype-matching methods.

### Emission model

For each population $k$, per-haplotype similarities are aggregated into a single score $f_k(o_t)$ using one of four functions: max (default), mean, median, or top-$j$ averaging. The emission probability uses a softmax parameterized by temperature $\tau$:

$$P(o_t \mid s_t = k) = \frac{\exp(f_k(o_t) / \tau)}{\sum_{j : f_j > 0} \exp(f_j(o_t) / \tau)}$$

The softmax is the Bayes-optimal posterior under Gaussian discriminant analysis with shared variance. The temperature $\tau$ is estimated adaptively from the median per-window spread in population scores, with theoretical scaling $\tau_{\text{opt}} \propto 1/\sqrt{\ln n}$ by extreme value theory.

### Pairwise contrast emissions

As an alternative to joint softmax, each population pair $(i, j)$ is evaluated independently via a Bradley-Terry comparison:

$$P_{ij}(k = i \mid o_t) = \frac{\exp(f_i / \tau_{ij})}{\exp(f_i / \tau_{ij}) + \exp(f_j / \tau_{ij})}$$

with per-pair adaptive temperature $\hat{\tau}_{ij} = \text{median}_t |f_i(o_t) - f_j(o_t)|$. The per-population emission is obtained by aggregating across all pairwise comparisons:

$$P(o_t \mid k) \propto \prod_{j \neq k} P_{kj}(k \mid o_t)$$

This resolves the problem that a single temperature cannot simultaneously be sharp for strong contrasts (AFR-EUR, $F_{ST} \approx 0.1$) and gentle for weak ones (EUR-AMR, $F_{ST} \approx 0.02$). Empirically, pairwise contrast improves simulated ancestry from 95.1% (joint softmax) to 97.95%.

### Automatic configuration

Optimal parameters differ substantially between datasets. We introduce automatic configuration based on $D_{\min}$, the minimum Cohen's $d$ across all reference population pairs:

$$D_{\min} = \min_{i < j} \frac{|\bar{f}_i - \bar{f}_j|}{\sqrt{(\hat{\sigma}_i^2 + \hat{\sigma}_j^2)/2}}$$

The pairwise weight scales with the coefficient of variation of pairwise Cohen's $d$ values, while the emission context scales inversely with $D_{\min}$: $ec^* = \text{round}(0.1/D_{\min})$, clamped to $[1, 15]$. This eliminates manual parameter tuning across different data regimes.

### Transitions and inference

The transition matrix is uniform across populations, parameterized by a single switch probability $p_{\text{switch}}$. When a genetic map is provided, $p_{\text{switch}}$ is modulated by local recombination rate via the Haldane map function. The most likely ancestry sequence is computed via Viterbi decoding; posterior ancestry probabilities via forward-backward. Posterior decoding ($\hat{a}_t = \arg\max_k \gamma_t(k)$) is available as an alternative that is advantageous for detecting short minority ancestry tracts where Viterbi's path constraint suppresses isolated switches.

The ancestry tool supports over 40 optional emission transforms and flags for fine-tuning. The `--auto-configure` flag is recommended for most use cases.

## 5. Jacquard Coefficients

For diploid relatedness estimation, the four-haplotype IBS sharing pattern between two individuals is analyzed using a Union-Find algorithm to compute the 9 condensed Jacquard delta coefficients. These coefficients describe the probability distribution over identity states at a locus and provide a complete characterization of relatedness between a pair of diploid individuals.

The input is the same per-window IBS matrix produced by `ibs` or `ibs-from-paf`, evaluated across the four haplotype comparisons (a1-b1, a1-b2, a2-b1, a2-b2) for each diploid pair.

## 6. Comparison with VCF-Based Approaches

| Aspect | impopₖ | hap-ibd / RFMix |
|--------|--------|-----------------|
| **Input** | PAF alignments + AGC assemblies | Phased VCF |
| **Phasing** | Assembly-resolved (no errors) | Statistical phasing (switch errors) |
| **Variation captured** | SNPs + indels + SVs simultaneously | SNPs only (standard) |
| **Ascertainment bias** | None (assembly-complete) | Marker-dependent (55-85% of variation) |
| **Resolution** | Window-based (~10 kb) | SNP-based (variable density) |
| **Reference panel** | ~50 haplotypes sufficient | >1,400 haplotypes recommended |
| **Speed (full genome)** | 195 seconds (PAF-direct) | Variable (minutes to hours) |
| **RAM** | ~100 MB (PAF-direct), ~2 GB (impg) | Depends on panel size |
| **IBD model** | 2-state Gaussian HMM | Seed-and-extend (hap-ibd) |
| **Ancestry model** | N-state softmax HMM | Li & Stephens copying model (RFMix) |
| **Theoretical basis** | Direct TMRCA estimator ($1 - 2\mu t$) | Allelic concordance |

The key advantage of the pangenome approach is that pairwise identity from assemblies is a more direct estimator of TMRCA than any VCF-derived quantity. The key limitation is that haplotype-resolved assemblies are currently available only for cohorts sequenced with long-read technologies, whereas VCF-based methods can leverage the much larger existing corpus of short-read and array genotyping data.

## 7. References

For full mathematical details, supplementary analyses, and validation results, see the manuscript in `paper/`:

- `paper/HPRCv2_IBD_paper.pdf` -- main text
- `paper/supplementary.pdf` -- supplementary material with detailed derivations

Key references:

- Browning & Browning (2020). *A fast, powerful method for detecting identity by descent.* AJHG.
- Maples et al. (2013). *RFMix: A discriminative modeling approach for rapid and robust local-ancestry inference.* AJHG.
- Fan et al. (2022). *The expected GRM from ARGs outperforms the canonical GRM.* Genetics.
- Lehmann et al. (2026). *Branch GRM: Optimal genetic relatedness from genealogical trees.* Genetics.
- Liao et al. (2023). *A draft human pangenome reference.* Nature.
- Hickey et al. (2024). *Pangenome graph construction with minigraph-cactus.* Nat Biotechnol.
