# Haplotype-Based IBS/IBD Analysis for HPRCv2

A computational pipeline for detecting Identity-By-State (IBS) and inferring Identity-By-Descent (IBD) segments directly from pangenome assemblies. This approach leverages the Human Pangenome Reference Consortium (HPRC) data and the implicit pangenome graph (impg) tool to enable haplotype-level IBD analysis without the limitations of VCF-based methods.

---

## Overview

This repository provides tools for:
- Detecting pairwise IBS across sliding genomic windows
- Computing Jacquard-style identity state frequencies
- Inferring IBD segments using Hidden Markov Models
- Generating reports and visualizations for population-scale analyses

---

## Pipeline Workflow

```
                          HPRCv2-IBD Pipeline
    ======================================================================

    INPUT DATA
    ----------
    [AGC Archive] -----> [PAF Alignments] -----> [impg Index]
         |                      |                      |
         +----------------------+----------------------+
                               |
                               v
    ======================================================================
    STEP 1: Window-Based IBS Detection
    ======================================================================

    Reference Chromosome (e.g., chr20)
    |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
      W1    W2    W3    W4    W5    W6    W7    W8    W9   W10  ...
                               |
                               v
                    +---------------------+
                    |   impg similarity   |
                    |   (per window)      |
                    +---------------------+
                               |
                               v
                    IBS Window Table
                    +------------------------------------------+
                    | chrom | start | end | hap_a | hap_b | id |
                    |-------|-------|-----|-------|-------|----|
                    | chr20 | 1     | 5k  | A#1   | B#1   |.999|
                    | chr20 | 1     | 5k  | A#1   | B#2   |.998|
                    | ...   | ...   | ... | ...   | ...   |... |
                    +------------------------------------------+

    ======================================================================
    STEP 2: Identity State Classification
    ======================================================================

    For diploid pair (A, B) with haplotypes {a1, a2} x {b1, b2}:

         IBS Pairs in Window          Union-Find Clustering
         +-----------------+          +-------------------+
         | (a1,a2): Yes    |          |    {a1,a2,b1}    |  --> State S3
         | (a1,b1): Yes    |   --->   |       {b2}       |
         | (a2,b1): Yes    |          +-------------------+
         | others: No      |
         +-----------------+

                               |
                               v
                    Identity State Frequencies
                    +--------------------------------+
                    | State | Frequency | Count      |
                    |-------|-----------|------------|
                    | S1    | 0.001     | 12         |
                    | S2    | 0.002     | 24         |
                    | ...   | ...       | ...        |
                    | S9    | 0.850     | 10,200     |
                    +--------------------------------+

    ======================================================================
    STEP 3: HMM-Based IBD Inference
    ======================================================================

    Identity Track for Haplotype Pair:

    Window:  1    2    3    4    5    6    7    8    9   10   11   12
    Score: 0.5  0.6  0.99 0.99 0.99 0.99 0.99 0.5  0.4  0.99 0.99 0.5
             |    |    |____|____|____|____|    |    |    |____|    |
             |    |         IBD Segment         |    |    IBD Seg   |
             |____|_______non-IBD_______________|____|________|_____|

                               |
                         Viterbi Algorithm
                               |
                               v
                    IBD Segment Table
                    +------------------------------------------------+
                    | chrom | start   | end     | hap_a | hap_b | id |
                    |-------|---------|---------|-------|-------|----|
                    | chr20 | 10,001  | 30,000  | A#1   | B#1   |.992|
                    | chr20 | 45,001  | 55,000  | A#1   | B#1   |.995|
                    +------------------------------------------------+

    ======================================================================
    OUTPUT
    ======================================================================

    +------------------+     +---------------------+     +----------------+
    | IBS Windows TSV  |     | Jacquard States TSV |     | IBD Segments   |
    | (per-window IBS) |     | (per-pair states)   |     | (final calls)  |
    +------------------+     +---------------------+     +----------------+
             |                        |                         |
             +------------------------+-------------------------+
                                      |
                                      v
                         +------------------------+
                         |   Analysis Notebooks   |
                         |   Reports & Figures    |
                         +------------------------+
```

---

## Quick Start Guide

### Prerequisites

- Rust toolchain (1.70+)
- `impg` executable in PATH ([github.com/pangenome/impg](https://github.com/pangenome/impg))
- GNU coreutils, awk, parallel
- R with HMM packages (for R-based IBD calling)

### Step 1: Download Required Data

```bash
# Sequence archive (AGC format)
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc

# Alignments to CHM13
wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz
wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz.gzi

# Implicit graph index
wget https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg
```

### Step 2: Build the CLI Tools

```bash
cd production/ibs-cli
cargo build --release
```

### Step 3: Run IBS Detection

```bash
# Single chromosome analysis
cd production/ibs-cli/scripts
./ibs.sh \
  --sequence-files /path/to/HPRC_r2_assemblies_0.6.1.agc \
  -a /path/to/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr20:1-64444167 \
  -size 5000 \
  --subset-sequence-list /path/to/sample_list.txt \
  --output /output/ibs_chr20.tsv
```

### Step 4: Compute Identity States (Optional)

```bash
./jacquard_coeffs.sh \
  --ibs ../ibs_for_ibd.out \
  --hap-a1 HG00096#1 --hap-a2 HG00096#2 \
  --hap-b1 HG00097#1 --hap-b2 HG00097#2
```

### Step 5: Call IBD Segments

```bash
./ibd.sh \
  --sequence-files /path/to/HPRC_r2_assemblies_0.6.1.agc \
  -a /path/to/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr20:1-64444167 \
  -size 5000 \
  --subset-sequence-list /path/to/sample_list.txt \
  --output /output/ibd_segments.tsv
```

---

## Repository Structure

```
HPRCv2-IBD/
|
+-- production/
|   +-- ibs-cli/              # Main Rust CLI tools
|       +-- src/
|       |   +-- main.rs       # IBS detection binary
|       |   +-- hmm.rs        # Hidden Markov Model
|       |   +-- segment.rs    # Segment detection
|       |   +-- stats.rs      # Statistical utilities
|       |   +-- bin/
|       |       +-- jacquard.rs   # Identity state calculator
|       |       +-- ibd.rs        # IBD calling binary
|       +-- scripts/
|           +-- ibs.sh            # IBS detection wrapper
|           +-- ibd.sh            # IBD calling wrapper
|           +-- jacquard_coeffs.sh
|           +-- run_full.sh       # Parallel chromosome runner
|
+-- analysis/
|   +-- ibd-network/          # Analysis notebooks and scripts
|       +-- scripts/
|       +-- notebooks/
|
+-- docs/
|   +-- paper_concepts/       # Publication documentation
|   |   +-- conceptual_framework.md
|   |   +-- methods.md
|   |   +-- results_template.md
|   |   +-- limitations.md
|   +-- tutorials/            # Step-by-step guides
|   +-- reports/              # Generated reports
|
+-- data/                     # Data directory (not in git)
```

---

## Documentation

### Conceptual Documentation

| Document | Description |
|----------|-------------|
| [Conceptual Framework](docs/paper_concepts/conceptual_framework.md) | Theoretical foundations of pangenome-based IBD |
| [Methods](docs/paper_concepts/methods.md) | Detailed algorithms with pseudocode and complexity analysis |
| [Results Template](docs/paper_concepts/results_template.md) | Standardized reporting templates |
| [Limitations](docs/paper_concepts/limitations.md) | Known caveats and edge cases |

### Tutorials

| Tutorial | Description |
|----------|-------------|
| [IBS Detection](docs/tutorials/ibs.md) | Running the IBS sliding window analysis |
| [IBD Calling](docs/tutorials/ibd.md) | HMM-based IBD segment detection |
| [Jacquard Coefficients](docs/tutorials/jacquard_coeffs.md) | Computing identity state frequencies |
| [Full Pipeline](docs/tutorials/run_full.md) | Running the complete chromosome analysis |
| [Pairwise impg](docs/tutorials/run_pairwise_impg.md) | Direct pairwise similarity queries |

---

## Key Concepts

### Why Pangenome-Based IBD?

Traditional IBD detection uses VCF files with diploid genotypes, requiring statistical phasing that introduces errors. Pangenome assemblies provide:

1. **Inherent phasing** - Each assembly is a single haplotype
2. **Structural variant inclusion** - Full representation of SVs
3. **Binary IBD model** - Direct haplotype comparison (IBD = 0 or 1)
4. **No reference bias** - Graph-based representation

### Identity States (Jacquard-Style)

For two diploid individuals with haplotypes {a1, a2} and {b1, b2}:

| State | Configuration | Description |
|-------|---------------|-------------|
| S1 | {a1, a2, b1, b2} | All four identical |
| S2 | {a1, a2}, {b1, b2} | Within-individual identity only |
| S3-S6 | Various 3+1 or 2+2 | Partial sharing patterns |
| S7 | {a1, b1}, {a2, b2} | Cross-pair matching |
| S8 | {a1, b1}, {a2}, {b2} | Single cross-pair |
| S9 | {a1}, {a2}, {b1}, {b2} | No observed IBS |

---

## Citation

If you use this pipeline, please cite:

```
[Citation to be added upon publication]
```

And the underlying tools:
- impg: [Garrison et al., implicit pangenome graph]
- HPRC: [Liao et al., 2023, Nature]

---

## License

[License information]

---

## Contact

For questions or issues, please open a GitHub issue or contact [maintainer information].
