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

---

## Installation

### Step 1: Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

Verify installation:

```bash
rustc --version  # Should be 1.70+
cargo --version
```

### Step 2: Clone and Build HPRCv2-IBD

```bash
git clone https://github.com/MarsicoFL/HPRCv2-IBD.git
cd HPRCv2-IBD

# Build all tools (including ancestry)
cargo build --release

# Verify the ancestry binary was built
./target/release/ancestry --help
```

The `ancestry` binary will be at `./target/release/ancestry`.

### Step 3: Install impg

`impg` is the pangenome similarity tool required for computing pairwise similarities.

```bash
cargo install impg

# Verify installation
impg --version
```

Alternatively, build from source:

```bash
git clone https://github.com/ekg/impg.git
cd impg
cargo build --release
# Binary at ./target/release/impg
```

### Step 4: Install GNU Parallel

For parallel window processing:

```bash
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel

# Verify
parallel --version
```

### Step 5: Install Python Dependencies

```bash
pip install pandas matplotlib numpy
```

### Step 6: Download HPRC Data

Create data directories and download required files:

```bash
cd HPRCv2-IBD
mkdir -p data/assemblies data/alignments

# Download AGC file (3.1 GB) - compressed assemblies
wget -P data/assemblies/ \
  https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc

# Download PAF alignment file (5.3 GB)
wget -P data/alignments/ \
  https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz

# Download IMPG index (315 MB) - speeds up queries
wget -P data/alignments/ \
  https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg
```

Verify the data structure:

```bash
ls -lh data/assemblies/
# HPRC_r2_assemblies_0.6.1.agc (3.1G)

ls -lh data/alignments/
# hprc465vschm13.aln.paf.gz (5.3G)
# hprc465vschm13.aln.paf.gz.impg (315M)
```

---

## Quick Start

Once everything is installed, run the example analysis:

```bash
cd HPRCv2-IBD
./bin/run_impg_ped.sh
```

This will:
1. Create sample files for HG00344 (query) vs HG00099/HG00097 (references)
2. Compute pairwise similarities for chr1:50-60Mb
3. Run HMM inference
4. Generate plots

Results will be in `tutorial_relatedness/results/`.

---

## Example Setup

We'll analyze:
- **Query**: HG00344 (EUR ancestry) - both haplotypes
- **References**: HG00099 and HG00097 (EUR ancestry) - 4 haplotypes total
- **Region**: chr1:50,000,001-60,000,000 (10 Mb)
- **Window size**: 5,000 bp

### Important: Reference Format

In HPRC data, the reference chromosome format is `CHM13#0#chr1`, not just `chr1`. This is critical for `impg` queries.

### Create Sample Files

```bash
mkdir -p tutorial_relatedness/{samples,results}

# Query haplotypes (the individual to analyze)
cat > tutorial_relatedness/samples/query.txt << 'EOF'
HG00344#1
HG00344#2
EOF

# Reference haplotypes (potential relatives)
cat > tutorial_relatedness/samples/references.txt << 'EOF'
HG00099#1
HG00099#2
HG00097#1
HG00097#2
EOF

# All samples combined (for impg)
cat tutorial_relatedness/samples/query.txt \
    tutorial_relatedness/samples/references.txt \
    > tutorial_relatedness/samples/all.txt

# Populations file (each haplotype as its own "population")
# Format: population_name<TAB>haplotype_id
cat > tutorial_relatedness/samples/populations.tsv << 'EOF'
HG00099#1	HG00099#1
HG00099#2	HG00099#2
HG00097#1	HG00097#1
HG00097#2	HG00097#2
EOF
```

---

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

---

## Step-by-Step Guide

### Step 1: Generate Pairwise Similarities

Calculate sequence similarity between all pairs of haplotypes for each genomic window.

```bash
# Configuration
AGC="data/assemblies/HPRC_r2_assemblies_0.6.1.agc"
PAF="data/alignments/hprc465vschm13.aln.paf.gz"
SAMPLES="tutorial_relatedness/samples/all.txt"
OUTDIR="tutorial_relatedness/results"

# IMPORTANT: Use CHM13#0#chr1 format for HPRC data
CHROM="CHM13#0#chr1"
START=50000001
END=60000000
WINDOW_SIZE=5000
JOBS=8

# Test a single window first
impg similarity \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "${CHROM}:${START}-$((START + WINDOW_SIZE - 1))" \
    --subset-sequence-list "$SAMPLES" \
    --force-large-region \
    -t 1

# If that works, proceed with parallel processing...
```

Generate all windows in parallel:

```bash
TMPDIR=$(mktemp -d)

# Generate window coordinates
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
cat > "$TMPDIR/process.sh" << SCRIPT
#!/bin/bash
idx=\$1; start=\$2; end=\$3
impg similarity \\
    --sequence-files "$AGC" \\
    -a "$PAF" \\
    -r "${CHROM}:\${start}-\${end}" \\
    --subset-sequence-list "$SAMPLES" \\
    --force-large-region \\
    -t 1 -v 0 2>/dev/null | tail -n +2 > "$TMPDIR/w_\${idx}.tsv"
SCRIPT
chmod +x "$TMPDIR/process.sh"

# Write header
echo -e "chrom\tstart\tend\tgroup.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\tjaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity" > "$OUTDIR/similarities.tsv"

# Run in parallel
cat "$TMPDIR/windows.txt" | parallel -j $JOBS --colsep ' ' "$TMPDIR/process.sh" {1} {2} {3}

# Combine results
for f in "$TMPDIR"/w_*.tsv; do
    [[ -s "$f" ]] && cat "$f" >> "$OUTDIR/similarities.tsv"
done

rm -rf "$TMPDIR"
echo "Similarities: $(wc -l < $OUTDIR/similarities.tsv) lines"
```

### Step 2: Extract Query vs Reference Matrix

Use the provided script:

```bash
python3 bin/extract_query_vs_ref_similarities.py \
    "$OUTDIR/similarities.tsv" \
    -o "$OUTDIR/query_vs_ref.tsv" \
    --queries tutorial_relatedness/samples/query.txt \
    --references tutorial_relatedness/samples/references.txt
```

### Step 3: Run HMM Inference

```bash
./target/release/ancestry \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "CHM13#0" \
    --region "chr1:${START}-${END}" \
    --region-length $((END - START + 1)) \
    --window-size $WINDOW_SIZE \
    --query-samples tutorial_relatedness/samples/query.txt \
    --populations tutorial_relatedness/samples/populations.tsv \
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
| `-r "CHM13#0"` | Reference prefix (coordinates are relative to this) |
| `--region "chr1:..."` | Region to analyze (chromosome:start-end) |
| `--populations` | TSV file defining reference "populations" (haplotypes) |
| `--estimate-params` | Automatically estimate HMM parameters from data |
| `--smooth-min-windows 3` | Merge segments shorter than 3 windows |
| `--min-posterior 0.7` | Only report segments with posterior >= 0.7 |

**Output format (relatedness.tsv):**

```
chrom   start     end       sample     ancestry   mean_posterior  n_windows  discriminability
chr1    50000001  50125000  HG00344#1  HG00099#1  0.92           25         0.0045
chr1    50125001  50350000  HG00344#1  HG00097#2  0.88           45         0.0038
```

### Step 4: Visualization

```bash
python3 bin/plot_relatedness.py \
    "$OUTDIR/relatedness.tsv" \
    -o "$OUTDIR/relatedness" \
    --title "Haplotype Relatedness: HG00344 vs HG00099/HG00097 (chr1:50-60Mb)"
```

This generates:
- `relatedness_painting.png` - Chromosome painting
- `relatedness_stats.png` - Summary statistics

---

## Output Interpretation

### Chromosome Painting

Each horizontal bar represents a query haplotype (HG00344#1 and HG00344#2). Colors indicate which reference haplotype that segment is most similar to.

### Discriminability

The `discriminability` column indicates confidence:
- **High (>0.05)**: Clear winner among references
- **Low (<0.05)**: Multiple references have similar similarity (ambiguous)

---

## Troubleshooting

### "Sequence 'chr1' not found in index"

Use the full reference format: `CHM13#0#chr1` instead of `chr1`.

```bash
# Wrong
impg similarity -r "chr1:1-10000" ...

# Correct
impg similarity -r "CHM13#0#chr1:1-10000" ...
```

### "Using default Glossophaga populations"

You need to provide a `--populations` file. Without it, the tool uses bat species as default.

```bash
# Create populations file
cat > populations.tsv << 'EOF'
HG00099#1	HG00099#1
HG00099#2	HG00099#2
EOF

# Use it
ancestry --populations populations.tsv ...
```

### Empty similarities file

Check that:
1. Sample IDs match what's in the AGC (e.g., `HG00344#1` not `HG00344_1`)
2. Region format is correct (`CHM13#0#chr1:start-end`)
3. Samples exist in the alignment file

Test with:
```bash
impg similarity --sequence-files $AGC -a $PAF -r "CHM13#0#chr1:50000001-50005000" -t 1
```

### impg not found

```bash
# Check if installed
which impg

# If not, install
cargo install impg
```

---

## Extending the Analysis

### Different Samples

Edit the sample files:

```bash
# Your query
echo "YOUR_SAMPLE#1" > samples/query.txt
echo "YOUR_SAMPLE#2" >> samples/query.txt

# References
echo "REF1#1" > samples/references.txt
echo "REF1#2" >> samples/references.txt

# Update populations.tsv accordingly
```

### Full Chromosome

```bash
CHROM="CHM13#0#chr1"
START=1
END=248387328  # Full chr1 length
```

### Available EUR Samples

See `data/samples/EUR.txt` for the list of European ancestry samples in HPRC:
```
HG00097, HG00099, HG00126, HG00128, HG00133, HG00140, HG00146, ...
```
