# IBD CLI

Identity-By-Descent (IBD) segment detection using Hidden Markov Model (HMM) for HPRC pangenome data.

## Overview

This tool detects IBD segments between haplotypes using a 2-state HMM:
- **State 0 (non-IBD)**: Background similarity (~99.9% identity)
- **State 1 (IBD)**: Shared ancestry (~99.9% identity)

## Algorithm

1. **IBS Collection**: Runs `impg similarity` over sliding windows
2. **HMM Inference**: Viterbi algorithm finds optimal state sequence per haplotype pair
3. **Segment Extraction**: Run-length encoding extracts contiguous IBD regions
4. **Filtering**: Applies minimum length and window count thresholds

## Usage

```bash
ibd \
  --sequence-files data/assemblies.agc \
  -a data/alignments.paf.gz \
  -r CHM13 \
  --region chr2:130787850-140837183 \
  --size 5000 \
  --subset-sequence-list samples.txt \
  --output ibd_segments.tsv \
  --ibs-output ibs_windows.tsv
```

## HMM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--expected-seg-windows` | 50 | Expected IBD segment length in windows |
| `--p-enter-ibd` | 0.0001 | P(non-IBD → IBD) per window |
| `--min-len-bp` | 2000000 | Minimum segment length (bp). Default 2 Mb for reliable IBD detection. |
| `--min-windows` | 3 | Minimum windows per segment |

## Output Format

**IBD Segments TSV:**
```
chrom   start   end     group.a group.b n_windows   mean_identity
chr2    1000    50000   HAP1    HAP2    10          0.9995
```

**IBS Windows TSV (optional):**
```
chrom   start   end     group.a group.b estimated.identity
chr2    1000    5999    HAP1    HAP2    0.9996
```

## Build

```bash
cargo build --release
```

## Modules

- `hmm.rs`: HMM parameters, Viterbi algorithm, segment extraction
- `stats.rs`: Gaussian distributions, k-means clustering
- `segment.rs`: Segment detection and merging
- `lib.rs`: Pipeline overview, module re-exports (hmm, stats, segment, concordance, hapibd)

## Dependencies

- Input: Requires `impg` in PATH
- Output: Used by downstream analysis
