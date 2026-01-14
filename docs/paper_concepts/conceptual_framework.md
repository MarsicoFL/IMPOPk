# Conceptual Framework for Pangenome-Based IBD Detection

## Executive Summary

This document clarifies the theoretical foundation for Identity-By-Descent (IBD) detection in pangenome assemblies, contrasting it with traditional VCF-based approaches. The key insight is that **pangenome assemblies enable binary IBD classification at the haplotype level**, fundamentally simplifying the IBD inference problem.

---

## 1. Why Binary IBD? The Haplotype-Level Model

### The Traditional Problem: Diploid Genotypes and IBD States

In classical population genetics using VCF-based data, IBD analysis confronts a fundamental challenge: we observe diploid *genotypes* (combinations of two alleles at each site), not individual haplotypes. When comparing two diploid individuals, we must consider which of their four haplotypes (two per individual) share ancestry. This leads to the well-known IBD0/IBD1/IBD2 classification:

- **IBD0**: Neither pair of haplotypes is IBD (0 haplotypes shared)
- **IBD1**: Exactly one haplotype pair is IBD (1 haplotype shared)
- **IBD2**: Both haplotype pairs are IBD (2 haplotypes shared, one from each parent)

This three-state model exists because phasing information is typically unavailable or unreliable in short-read VCF data. We must integrate over all possible phase configurations, making IBD a probabilistic inference problem.

### The Pangenome Solution: Direct Haplotype Comparison

Pangenome assemblies fundamentally change this landscape. Each assembly represents a single, complete haplotype derived from long-read sequencing and assembly. When we compare two assemblies, we are comparing **exactly two haplotypes** - one from each individual. At any given locus, only two possibilities exist:

- **IBD = 0**: The two haplotypes do NOT share recent common ancestry at this locus
- **IBD = 1**: The two haplotypes DO share recent common ancestry at this locus

This binary model is not a simplification or approximation - it is the correct model for haplotype-to-haplotype comparison. The complexity of IBD0/IBD1/IBD2 arises specifically from diploid genotype ambiguity, which does not exist when comparing resolved haplotypes.

### Mathematical Formulation

For two haplotypes h_A and h_B at genomic region R:

```
IBD(h_A, h_B, R) in {0, 1}
```

This contrasts with the diploid case for individuals A and B with haplotypes {h_A1, h_A2} and {h_B1, h_B2}:

```
IBD(A, B, R) in {0, 1, 2}
where:
  IBD = sum over i,j of IBD(h_Ai, h_Bj, R) for the transmitted haplotypes
```

The pangenome approach separates what traditional methods conflate: we first resolve haplotypes through assembly, then perform IBD analysis on each haplotype pair independently.

---

## 2. Terminology: From Jacquard Coefficients to Identity State Frequencies

### The Challenge of Existing Terminology

The codebase currently uses "Jacquard Delta coefficients" to describe the nine possible identity configurations between two diploid individuals. However, this terminology requires clarification:

**Jacquard's condensed identity coefficients (Delta_1 through Delta_9)** are theoretical constructs defined in terms of *true* Identity-By-Descent. They describe the probability that four haplotypes (two from each of two individuals) fall into specific IBD configurations based on genealogical relationships.

**What we compute** from IBS (Identity-By-State) observations in pangenome data are *empirical frequencies* of sequence identity patterns - not true IBD probabilities. IBS is observable; IBD must be inferred.

### Recommended Terminology

We propose the following terminology to maintain scientific precision:

| Current Term | Recommended Term | Rationale |
|-------------|------------------|-----------|
| "Jacquard coefficients" | **"Haplotype identity state frequencies"** or **"IBS-derived identity states"** | Distinguishes observed patterns from theoretical IBD |
| "Delta_1 through Delta_9" | **"Identity State S1 through S9"** | Neutral terminology that doesn't imply true IBD |
| "Jacquard-style metrics" | **"Diploid identity state distribution"** | Describes what we measure without theoretical claims |

### State Definitions (Observable IBS Configurations)

For two diploid individuals A (haplotypes a1, a2) and B (haplotypes b1, b2), we observe whether each of the six haplotype pairs shows IBS in a given window. The nine identity states are:

| State | Configuration | Haplotype Blocks | Description |
|-------|--------------|------------------|-------------|
| S1 | All four identical | {a1, a2, b1, b2} | Complete identity |
| S2 | Within-individual identical, between different | {a1, a2}, {b1, b2} | Autozygosity in both |
| S3 | Three-way cluster + singleton | {a1, a2, b1}, {b2} | Partial sharing |
| S4 | a1, a2 identical; b1, b2 different | {a1, a2}, {b1}, {b2} | A homozygous by state |
| S5 | Three-way cluster + singleton | {a1, b1, b2}, {a2} | Partial sharing |
| S6 | b1, b2 identical; a1, a2 different | {b1, b2}, {a1}, {a2} | B homozygous by state |
| S7 | Cross-matching pairs | {a1, b1}, {a2, b2} or {a1, b2}, {a2, b1} | Complementary sharing |
| S8 | One cross-pair identical | {a1, b1}, {a2}, {b2} or similar | Single haplotype match |
| S9 | All four different | {a1}, {a2}, {b1}, {b2} | No IBS observed |

### Important Distinction: IBS vs IBD

- **IBS (Identity-By-State)**: Observable sequence similarity. Two haplotypes show IBS if they are identical (or nearly identical) in sequence.

- **IBD (Identity-By-Descent)**: Inferred common ancestry. Two haplotypes are IBD if they descended from the same ancestral haplotype without recombination.

IBS is necessary but not sufficient for IBD. Two haplotypes may be IBS because:
1. They are IBD (share recent common ancestry)
2. They carry the same common variant by chance (identity by chance)
3. The region has very low diversity (coalescent identity)

Our HMM-based approach uses patterns of IBS across sliding windows to *infer* IBD, leveraging the expectation that true IBD segments will show:
- Sustained high identity across many consecutive windows
- Contiguous patterns consistent with shared haplotype blocks

---

## 3. Relationship Between Binary IBD and the Nine Identity States

### How the States Relate to Pairwise Binary IBD

The nine identity states describe configurations of four haplotypes, but each configuration is *composed of* pairwise binary IBD relationships. For each pair of haplotypes, IBD is binary (0 or 1).

Given four haplotypes and six possible pairs:
- Within individual A: (a1, a2)
- Within individual B: (b1, b2)
- Between individuals: (a1, b1), (a1, b2), (a2, b1), (a2, b2)

Each identity state corresponds to a specific pattern of which pairs are IBD (or in our observable case, IBS):

| State | Pairs in IBS | Within-A | Within-B | Cross-pairs |
|-------|-------------|----------|----------|-------------|
| S1 | All 6 | Yes | Yes | All 4 |
| S2 | 2 | Yes | Yes | None |
| S3 | 4 | Yes | No | 2 |
| S4 | 1 | Yes | No | None |
| S5 | 4 | No | Yes | 2 |
| S6 | 1 | No | Yes | None |
| S7 | 2 | No | No | 2 (complementary) |
| S8 | 1 | No | No | 1 |
| S9 | 0 | No | No | None |

### From Binary to Aggregate

The pipeline workflow reflects this hierarchy:
1. **Window-level IBS**: For each genomic window, `impg similarity` identifies which haplotype pairs show IBS
2. **State classification**: The union-find algorithm groups IBS-connected haplotypes into blocks, determining the identity state
3. **State frequencies**: Aggregating across windows yields the distribution of identity states
4. **IBD inference**: The HMM uses these patterns to call IBD segments for specific haplotype pairs

---

## 4. Advantages of Pangenome-Based IBD Detection

### 4.1 No Phasing Errors

**Traditional approach**: Short-read variant calling produces unphased genotypes. Statistical or reference-based phasing introduces errors that propagate into IBD inference. Switch errors can create false IBD breakpoints or merge distinct IBD segments.

**Pangenome approach**: Long-read assemblies are inherently phased. Each assembly represents a single haplotype path through the genome. There is no phasing step that could introduce errors - the phase is determined by physical sequencing of continuous DNA molecules.

### 4.2 Complete Structural Variant Representation

**Traditional approach**: VCF representation struggles with complex structural variants (SVs), inversions, and repetitive regions. Many SVs are missed entirely, and those detected may be incorrectly genotyped. IBD analysis based on incomplete variant information may miss true IBD or call false positives.

**Pangenome approach**: Assemblies capture the complete sequence, including:
- Large insertions and deletions
- Inversions and complex rearrangements
- Tandem repeat expansions and contractions
- Mobile element insertions

Sequence identity in pangenome comparisons naturally incorporates SV sharing, providing a more complete picture of haplotype relationships.

### 4.3 Direct Haplotype Comparison

**Traditional approach**: IBD detection algorithms must model diploid genotypes and phase uncertainty. This adds computational complexity and introduces modeling assumptions that may not hold.

**Pangenome approach**: Direct comparison of two sequences is conceptually simpler and computationally more tractable. The `impg` tool provides efficient all-to-all haplotype comparison within the pangenome graph framework.

### 4.4 Resolution of Complex Regions

**Traditional approach**: Regions with high repeat content, segmental duplications, or copy number variation are often masked or produce unreliable genotype calls. IBD detection effectively has "blind spots" in these regions.

**Pangenome approach**: Modern assemblers (like hifiasm) can resolve many complex regions that confound short-read methods. While some regions remain challenging, the overall coverage and accuracy in complex regions is substantially improved.

### 4.5 Population-Scale Efficiency

The Human Pangenome Reference Consortium (HPRC) provides:
- High-quality diploid assemblies for hundreds of individuals
- Consistent assembly methodology enabling fair comparisons
- Graph-based tools (like impg) for efficient pairwise similarity computation

This infrastructure makes population-scale pangenome IBD analysis practical and reproducible.

---

## 5. Summary of Key Concepts

1. **Binary IBD is the correct model** when comparing resolved haplotypes. The IBD0/IBD1/IBD2 trichotomy applies to diploid genotype comparisons where phase is unknown.

2. **Terminology should distinguish IBS from IBD**. What we observe is Identity-By-State; what we infer is Identity-By-Descent. The nine "Jacquard-style" states are better termed "haplotype identity state frequencies" or "IBS-derived identity states."

3. **The nine identity states are composed of binary pairwise relationships**. Each state describes a specific pattern of which haplotype pairs show IBS, but the fundamental unit is the binary question: do these two haplotypes share identity?

4. **Pangenome assemblies provide advantages** over VCF-based approaches: no phasing errors, complete SV representation, direct haplotype comparison, and resolution of complex regions.

5. **The inference pipeline** moves from observation (IBS in windows) to inference (IBD segments via HMM), using the sustained patterns of sequence identity to distinguish true IBD from sporadic IBS.

---

## References

- Jacquard, A. (1974). *The Genetic Structure of Populations*. Springer-Verlag.
- Thompson, E. A. (2013). Identity by descent: variation in meiosis, across genomes, and in populations. *Genetics*, 194(2), 301-326.
- Browning, S. R., & Browning, B. L. (2012). Identity by descent between distant relatives: detection and applications. *Annual Review of Genetics*, 46, 617-633.
- Liao, W. W., et al. (2023). A draft human pangenome reference. *Nature*, 617(7960), 312-324.
