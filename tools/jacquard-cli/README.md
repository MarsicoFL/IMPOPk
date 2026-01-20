# Jacquard CLI

Compute Jacquard delta coefficients (Δ1-Δ9) from IBS windows for HPRC pangenome data.

## Overview

The Jacquard coefficients describe 9 possible identity-by-descent (IBD) states for two diploid individuals (4 haplotypes total). This tool takes IBS window data and classifies each locus into one of the 9 delta states.

## The 9 Delta States

| Delta | Pattern | Description |
|-------|---------|-------------|
| Δ1 | All 4 connected | Complete identity (A1=A2=B1=B2) |
| Δ2 | {A1,A2} + {B1,B2} | Within-individual identity only |
| Δ3 | {A1,A2,B1} + {B2} | 3 connected including both A |
| Δ4 | {A1,A2} + singletons | Only A haplotypes identical |
| Δ5 | {A1,B1,B2} + {A2} | 3 connected including both B |
| Δ6 | {B1,B2} + singletons | Only B haplotypes identical |
| Δ7 | {A1,B1} + {A2,B2} | Cross-individual pairs |
| Δ8 | {Ai,Bj} + singletons | Single cross-pair |
| Δ9 | 4 singletons | No identity |

## Usage

```bash
jacquard \
  --ibs ibs_windows.tsv \
  --hap-a1 "SAMPLE1#1" \
  --hap-a2 "SAMPLE1#2" \
  --hap-b1 "SAMPLE2#1" \
  --hap-b2 "SAMPLE2#2"
```

## Input Format

TSV file with columns:
- `chrom`: Chromosome
- `start`: Start position
- `end`: End position
- `group.a`: First haplotype in IBS pair
- `group.b`: Second haplotype in IBS pair

## Output

Prints delta state counts and proportions to stdout.

## Build

```bash
cargo build --release
```

## Algorithm

1. **Union-Find**: Groups haplotypes by IBS connectivity per locus
2. **Block Analysis**: Counts A vs B haplotypes in each connected component
3. **State Classification**: Maps block patterns to delta states

## Dependencies

- Input: IBS windows from `ibs-cli`
