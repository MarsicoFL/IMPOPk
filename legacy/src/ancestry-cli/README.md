# ancestry-cli

Local Ancestry Inference (LAI) using HMM for pangenome data.

## Overview

Given a pangenome with:
- **Query samples**: Individuals of unknown/mixed ancestry (e.g., TBG bats)
- **Reference populations**: Known ancestral species/populations with multiple haplotypes

This tool infers which ancestral population each genomic segment of each query sample most likely derives from.

## Pangenome Context

In a pangenome alignment:
- All sequences are aligned to a **coordinate reference** (e.g., soricina#HAP1)
- The reference is only for **coordinate system**, not for similarity calculation
- `impg similarity` computes pairwise identity between **any two sequences** that align to the same region
- Similarities reflect true sequence identity, independent of which sequence is the reference

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  impg similarity (per window)                                       │
│  ─────────────────────────────                                      │
│  Input: region soricina#HAP1#super15:1-5000                         │
│  Output: ALL pairwise similarities for sequences in that region     │
│                                                                     │
│  TBG_5116#1 vs TBG_5117#1      → 0.92  (query vs query)            │
│  TBG_5116#1 vs commissarisi#HAP1 → 0.87  (query vs reference) ✓    │
│  TBG_5116#1 vs mutica#A        → 0.72  (query vs reference) ✓      │
│  commissarisi#HAP1 vs mutica#A → 0.65  (ref vs ref)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Filter: keep only query vs reference
┌─────────────────────────────────────────────────────────────────────┐
│  AncestryObservation (per sample, per window)                       │
│  ────────────────────────────────────────────                       │
│  sample: TBG_5116#1                                                 │
│  window: super15:1-5000                                             │
│  similarities: {                                                    │
│      commissarisi#HAP1 → 0.87                                       │
│      commissarisi#HAP2 → 0.85                                       │
│      mutica#A          → 0.72                                       │
│      mutica#B          → 0.71                                       │
│      soricina#HAP1     → 0.78                                       │
│      soricina#HAP2     → 0.76                                       │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HMM inference
┌─────────────────────────────────────────────────────────────────────┐
│  HMM with 3 states (one per ancestral population)                   │
│  ────────────────────────────────────────────────                   │
│                                                                     │
│  States: [commissarisi] ←→ [mutica] ←→ [soricina]                  │
│                                                                     │
│  For each window:                                                   │
│    - Emission P(obs|state): higher similarity to state's refs       │
│    - Transition: high prob stay, low prob switch (switch_prob)      │
│                                                                     │
│  Algorithms:                                                        │
│    - Viterbi: most likely state sequence                            │
│    - Forward-Backward: posterior P(state|all observations)          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Output: Ancestry Segments                                          │
│  ────────────────────────────                                       │
│  sample      chrom    start      end        ancestry      posterior │
│  TBG_5116#1  super15  1          500000     commissarisi  0.95      │
│  TBG_5116#1  super15  500001     1200000    soricina      0.89      │
│  TBG_5116#1  super15  1200001    2000000    commissarisi  0.97      │
└─────────────────────────────────────────────────────────────────────┘
```

## HMM Model Details

### States
One state per ancestral population. For Glossophaga bats:
- State 0: *G. commissarisi* (references: HAP1, HAP2)
- State 1: *G. mutica* (references: A, B)
- State 2: *G. soricina* (references: HAP1, HAP2)

### Emissions
For state `s` with population `P`, the emission probability is based on the **maximum similarity** to any haplotype in `P`:

```
max_sim = max(similarity to P's haplotypes)
P(observation | state=P) = Gaussian(max_sim; mean=μ_same, std=σ)
```

Where:
- `μ_same` ≈ 0.95 (expected similarity when ancestry matches)
- `σ` ≈ 0.03 (estimated from data)

### Transitions
```
P(stay in same state) = 1 - switch_prob
P(switch to other state) = switch_prob / (n_states - 1)
```

Default `switch_prob = 0.001` means expected segment length of ~1000 windows (~5 Mb with 5kb windows).

### Inference
- **Viterbi**: Returns most likely state sequence (MAP estimate)
- **Forward-Backward**: Returns posterior P(state | all data) for each window

## Usage

### Generate similarity data
```bash
./scripts/generate_ancestry_similarities.sh super15
```

### Run ancestry inference
```bash
./scripts/run_ancestry.sh super15 --posteriors
```

### CLI options
```
ancestry [OPTIONS] --sequence-files <AGC> -a <PAF> -r <REF> --region <REGION> --query-samples <FILE> -o <OUTPUT>

Options:
  --sequence-files <AGC>       AGC archive with assemblies
  -a <PAF>                     Alignment file (PAF)
  -r <REF>                     Reference name for coordinates (e.g., soricina#HAP1)
  --region <REGION>            Region (e.g., super15:1-1000000)
  --window-size <BP>           Window size [default: 5000]
  --query-samples <FILE>       File with query sample IDs
  --populations <FILE>         Population definitions (optional)
  --similarity-file <FILE>     Pre-computed similarities (faster)
  -o <OUTPUT>                  Output ancestry segments
  --posteriors-output <FILE>   Per-window posteriors (optional)
  --switch-prob <FLOAT>        Ancestry switch probability [default: 0.001]
  --min-len-bp <BP>            Minimum segment length [default: 10000]
  --min-windows <N>            Minimum windows per segment [default: 3]
  -t <N>                       Threads [default: 4]
```

## Output Files

### Ancestry segments (`-o`)
```
chrom    start    end    sample    ancestry    n_windows    mean_similarity    mean_posterior
```

### Per-window posteriors (`--posteriors-output`)
```
chrom    start    end    sample    P(commissarisi)    P(mutica)    P(soricina)
```

### Similarity backup (from generate script)
Raw `impg similarity` output with all query-vs-reference pairs.

## Biological Interpretation

- **High posterior for one population**: Strong signal of ancestry from that species
- **Ancestry switches**: Ancient recombination, introgression events, or incomplete lineage sorting (ILS)
- **Low posteriors everywhere**: Ambiguous ancestry, possibly equidistant from all references

## Example: Glossophaga Bats

```bash
cd experiments/phase2_cases/bats

# 1. Generate similarities for super15 (28 Mb, ~5600 windows)
./scripts/generate_ancestry_similarities.sh super15

# 2. Run ancestry inference
./scripts/run_ancestry.sh super15 --posteriors

# 3. View results
head output/ancestry_super15_*.tsv
```
