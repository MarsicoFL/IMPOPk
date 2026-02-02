# Haplotype Relatedness Analysis Tutorial

This tutorial demonstrates how to use the ancestry HMM model to determine **which reference haplotype each genomic segment of a query individual is most similar to**. This is a form of relatedness analysis that sits between traditional IBD detection and population-level ancestry inference.

## Use Case

Given:
- A **query individual** (both haplotypes)
- Multiple **reference haplotypes** from related or potentially related individuals

The model determines, for each genomic window, which reference haplotype the query is most similar to. This can reveal:
- Shared haplotype segments between individuals
- Patterns of inheritance
- Potential relatedness

## Prerequisites

### Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| **impg** | Pangenome similarity queries | `cargo install impg` or [github.com/ekg/impg](https://github.com/ekg/impg) |
| **ancestry** | HMM inference | `cargo build --release --bin ancestry` (this repo) |
| **GNU parallel** | Parallel window processing | `apt install parallel` |
| **Python 3** | Plotting | With pandas, matplotlib, numpy |

### Data

Download to `data/` directory (see [data/README.md](../../data/README.md)):

| File | Size | Download |
|------|------|----------|
| `HPRC_r2_assemblies_0.6.1.agc` | 3.1 GB | [Link](https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/) |
| `hprc465vschm13.aln.paf.gz` | 5.3 GB | [Link](https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz) |
| `hprc465vschm13.aln.paf.gz.impg` | 315 MB | [Link](https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg) |

Place files in:
```
data/
├── assemblies/
│   └── HPRC_r2_assemblies_0.6.1.agc
└── alignments/
    ├── hprc465vschm13.aln.paf.gz
    └── hprc465vschm13.aln.paf.gz.impg
```

## Example Setup

We'll analyze:
- **Query**: HG00344 (EUR ancestry) - both haplotypes
- **References**: HG00099 and HG00097 (EUR ancestry) - 4 haplotypes total
- **Region**: chr1:50,000,001-60,000,000 (10 Mb)
- **Window size**: 5,000 bp

### Create Sample Files

```bash
mkdir -p tutorial_relatedness/samples

# Query haplotypes
cat > tutorial_relatedness/samples/query.txt << 'EOF'
HG00344#1
HG00344#2
EOF

# Reference haplotypes
cat > tutorial_relatedness/samples/references.txt << 'EOF'
HG00099#1
HG00099#2
HG00097#1
HG00097#2
EOF

# All samples combined
cat tutorial_relatedness/samples/query.txt tutorial_relatedness/samples/references.txt > tutorial_relatedness/samples/all.txt
```

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Pairwise Similarity Calculation                    │
│  Tool: impg similarity                                      │
│  Output: All pairwise similarities per window              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Extract Query vs Reference Matrix                  │
│  Tool: Python script                                        │
│  Output: Matrix of query haplotypes vs reference haplotypes│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: HMM Inference                                      │
│  Tool: ancestry                                            │
│  Output: Most likely reference for each segment            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Visualization                                      │
│  Tool: Python script                                       │
│  Output: Chromosome painting + statistics                  │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Guide

### Step 1: Generate Pairwise Similarities

Calculate sequence similarity between all pairs of haplotypes for each genomic window.

```bash
# Configuration
AGC="data/assemblies/HPRC_r2_assemblies_0.6.1.agc"
PAF="data/alignments/hprc465vschm13.aln.paf.gz"
SAMPLES="tutorial_relatedness/samples/all.txt"
OUTDIR="tutorial_relatedness/results"
mkdir -p "$OUTDIR"

CHROM="chr1"
START=50000001
END=60000000
WINDOW_SIZE=5000
JOBS=8

# Generate window coordinates
TMPDIR=$(mktemp -d)
pos=$START
idx=0
while [[ $pos -le $END ]]; do
    win_end=$((pos + WINDOW_SIZE - 1))
    [[ $win_end -gt $END ]] && win_end=$END
    echo "$idx $pos $win_end"
    pos=$((win_end + 1))
    idx=$((idx + 1))
done > "$TMPDIR/windows.txt"

echo "Total windows: $(wc -l < $TMPDIR/windows.txt)"

# Create processing script
cat > "$TMPDIR/process.sh" << 'SCRIPT'
#!/bin/bash
idx=$1; start=$2; end=$3
impg similarity \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "${CHROM}:${start}-${end}" \
    --subset-sequence-list "$SAMPLES" \
    --force-large-region \
    -t 1 -v 0 2>/dev/null | tail -n +2 > "$TMPDIR/w_${idx}.tsv"
SCRIPT
chmod +x "$TMPDIR/process.sh"

# Write header
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity" > "$OUTDIR/similarities.tsv"

# Run in parallel
export AGC PAF CHROM SAMPLES TMPDIR
cat "$TMPDIR/windows.txt" | parallel -j $JOBS --colsep ' ' "$TMPDIR/process.sh" {1} {2} {3}

# Combine results
for f in "$TMPDIR"/w_*.tsv; do
    [[ -s "$f" ]] && cat "$f" >> "$OUTDIR/similarities.tsv"
done

rm -rf "$TMPDIR"
echo "Similarities: $(wc -l < $OUTDIR/similarities.tsv) lines"
```

### Step 2: Extract Query vs Reference Matrix

Create a Python script to extract the relevant comparisons:

```bash
cat > tutorial_relatedness/scripts/extract_query_vs_ref.py << 'PYTHON'
#!/usr/bin/env python3
"""Extract query-vs-reference similarities from impg output."""

import sys
import argparse
from collections import defaultdict

def extract_sample_id(full_id):
    """Extract sample#haplotype from full ID."""
    parts = full_id.split('#')
    if len(parts) >= 2:
        return f"{parts[0]}#{parts[1]}"
    return full_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input similarities file')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--references', required=True)
    args = parser.parse_args()

    with open(args.queries) as f:
        query_samples = set(line.strip() for line in f if line.strip())
    with open(args.references) as f:
        reference_haplotypes = [line.strip() for line in f if line.strip()]

    data = defaultdict(dict)

    with open(args.input) as f:
        header = f.readline().strip().split('\t')
        col_idx = {name: i for i, name in enumerate(header)}

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) <= col_idx['estimated.identity']:
                continue

            chrom = fields[col_idx['chrom']]
            start = fields[col_idx['start']]
            end = fields[col_idx['end']]
            id_a = extract_sample_id(fields[col_idx['group.a']])
            id_b = extract_sample_id(fields[col_idx['group.b']])
            identity = fields[col_idx['estimated.identity']]

            if id_a in query_samples and id_b in reference_haplotypes:
                data[(chrom, start, end, id_a)][id_b] = identity
            elif id_b in query_samples and id_a in reference_haplotypes:
                data[(chrom, start, end, id_b)][id_a] = identity

    with open(args.output, 'w') as out:
        out.write('\t'.join(['chrom', 'start', 'end', 'sample'] + reference_haplotypes) + '\n')
        for key in sorted(data.keys(), key=lambda x: (x[0], int(x[1]), x[3])):
            chrom, start, end, sample = key
            row = [chrom, start, end, sample] + [data[key].get(ref, 'NA') for ref in reference_haplotypes]
            out.write('\t'.join(row) + '\n')

    print(f"Output: {len(data)} windows", file=sys.stderr)

if __name__ == '__main__':
    main()
PYTHON
chmod +x tutorial_relatedness/scripts/extract_query_vs_ref.py
```

Run it:

```bash
python3 tutorial_relatedness/scripts/extract_query_vs_ref.py \
    "$OUTDIR/similarities.tsv" \
    -o "$OUTDIR/query_vs_ref.tsv" \
    --queries tutorial_relatedness/samples/query.txt \
    --references tutorial_relatedness/samples/references.txt
```

### Step 3: Run HMM Inference

```bash
# Build if needed
cargo build --release --bin ancestry

# Run ancestry HMM
./target/release/ancestry \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "chm13#chr1" \
    --region "1:${START}-${END}" \
    --region-length $((END - START + 1)) \
    --window-size $WINDOW_SIZE \
    --query-samples tutorial_relatedness/samples/query.txt \
    -o "$OUTDIR/relatedness.tsv" \
    --similarity-file "$OUTDIR/similarities.tsv" \
    --estimate-params \
    --smooth-min-windows 3 \
    --min-posterior 0.7 \
    --posteriors-output "$OUTDIR/posteriors.tsv" \
    -t $JOBS
```

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `--estimate-params` | Automatically estimate HMM emission parameters from data |
| `--smooth-min-windows 3` | Merge segments shorter than 3 windows |
| `--min-posterior 0.7` | Only report segments with posterior probability >= 0.7 |

**Output format:**

```
chrom   start     end       sample     ancestry   mean_posterior  n_windows  discriminability
chr1    50000001  50125000  HG00344#1  HG00099#1  0.92           25         0.0045
chr1    50125001  50350000  HG00344#1  HG00097#2  0.88           45         0.0038
```

### Step 4: Visualization

Create plotting script:

```bash
cat > tutorial_relatedness/scripts/plot_relatedness.py << 'PYTHON'
#!/usr/bin/env python3
"""Plot relatedness results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection
import argparse

def get_colors(haplotypes):
    base = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    return {hap: base[i % len(base)] for i, hap in enumerate(haplotypes)}

def plot_painting(df, output, title=None):
    samples = sorted(df['sample'].unique())
    ancestries = sorted(df['ancestry'].unique())
    colors = get_colors(ancestries)

    fig, ax = plt.subplots(figsize=(16, max(6, len(samples) * 0.8)))
    patches, patch_colors = [], []

    for i, sample in enumerate(samples):
        for _, row in df[df['sample'] == sample].iterrows():
            rect = mpatches.Rectangle((row['start'], i - 0.4), row['end'] - row['start'], 0.8)
            patches.append(rect)
            patch_colors.append(colors.get(row['ancestry'], '#95A5A6'))

    ax.add_collection(PatchCollection(patches, facecolors=patch_colors, edgecolors='none', alpha=0.9))
    ax.set_xlim(0, df['end'].max())
    ax.set_ylim(-0.5, len(samples) - 0.5)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples)
    ax.set_xlabel('Position (Mb)')
    ax.xaxis.set_major_formatter(lambda x, p: f'{x/1e6:.1f}')
    ax.set_title(title or 'Haplotype Relatedness')
    ax.legend(handles=[mpatches.Patch(color=colors[a], label=a) for a in ancestries],
              loc='upper right', title='Reference')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")

def plot_stats(df, output):
    ancestries = sorted(df['ancestry'].unique())
    colors = get_colors(ancestries)
    df['length'] = df['end'] - df['start']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df['ancestry'].value_counts()
    axes[0].bar(range(len(counts)), counts.values, color=[colors[a] for a in counts.index])
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=45, ha='right')
    axes[0].set_ylabel('Segments')
    axes[0].set_title('Segment Count by Reference')

    lengths = df.groupby('ancestry')['length'].sum() / 1e6
    axes[1].bar(range(len(lengths)), lengths.values, color=[colors[a] for a in lengths.index])
    axes[1].set_xticks(range(len(lengths)))
    axes[1].set_xticklabels(lengths.index, rotation=45, ha='right')
    axes[1].set_ylabel('Total Length (Mb)')
    axes[1].set_title('Total Length by Reference')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved: {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ancestry_file')
    parser.add_argument('-o', '--output', default='relatedness')
    parser.add_argument('--title')
    args = parser.parse_args()

    df = pd.read_csv(args.ancestry_file, sep='\t')
    plot_painting(df, f'{args.output}_painting.png', args.title)
    plot_stats(df, f'{args.output}_stats.png')
PYTHON
chmod +x tutorial_relatedness/scripts/plot_relatedness.py
```

Generate plots:

```bash
python3 tutorial_relatedness/scripts/plot_relatedness.py \
    "$OUTDIR/relatedness.tsv" \
    -o "$OUTDIR/relatedness" \
    --title "Haplotype Relatedness: HG00344 vs HG00099/HG00097 (chr1:50-60Mb)"
```

## Output Interpretation

### Chromosome Painting

Each horizontal bar represents a query haplotype (HG00344#1 and HG00344#2). Colors indicate which reference haplotype that segment is most similar to.

### Discriminability

The `discriminability` column indicates confidence:
- **High (>0.05)**: Clear winner among references
- **Low (<0.05)**: Multiple references have similar similarity (ambiguous)

## Complete Script

For convenience, here's a complete script that runs all steps:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration
AGC="data/assemblies/HPRC_r2_assemblies_0.6.1.agc"
PAF="data/alignments/hprc465vschm13.aln.paf.gz"
CHROM="chr1"
START=50000001
END=60000000
WINDOW_SIZE=5000
JOBS=8

WORKDIR="tutorial_relatedness"
mkdir -p "$WORKDIR"/{samples,results,scripts}

# [Include all steps from above...]
```

See `bin/run_impg_ped.sh` for the complete runnable script.

## Extending the Analysis

### Different Samples

Edit the sample files to analyze different individuals:

```bash
# Your query
echo "SAMPLE#1" > samples/query.txt
echo "SAMPLE#2" >> samples/query.txt

# Potential relatives as references
echo "REF1#1" > samples/references.txt
echo "REF1#2" >> samples/references.txt
# ... add more
```

### Full Chromosome

```bash
CHROM="chr1"
START=1
END=248387328  # Full chr1
```

### Multiple Chromosomes

Loop over chromosomes and combine results.
