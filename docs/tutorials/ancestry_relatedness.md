# Haplotype Relatedness Analysis

Determine which reference haplotype each genomic segment of a query individual is most similar to. Useful for relatedness and pedigree analysis.

---

## Table of Contents

1. [Installation](#installation)
2. [Reproducible Example (HPRC)](#reproducible-example-hprc)
3. [Using Your Own Data](#using-your-own-data)
4. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify
rustc --version   # Should be 1.70+
```

### 2. Install impg

```bash
cargo install impg

# Verify
impg --version
```

### 3. Install GNU Parallel

```bash
# Ubuntu/Debian
sudo apt install parallel

# macOS
brew install parallel
```

### 4. Install Python packages

```bash
pip install pandas matplotlib numpy
```

### 5. Clone and build this repository

```bash
git clone https://github.com/MarsicoFL/HPRCv2-IBD.git
cd HPRCv2-IBD
cargo build --release

# Verify
./target/release/ancestry --help
```

---

## Reproducible Example (HPRC)

This example analyzes relatedness between three EUR individuals from the Human Pangenome Reference Consortium (HPRC).

### Download HPRC Data

```bash
cd HPRCv2-IBD
mkdir -p data/assemblies data/alignments

# AGC file (3.1 GB) - compressed genome assemblies
wget -P data/assemblies/ \
  https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc

# PAF file (5.3 GB) - alignments to CHM13 reference
wget -P data/alignments/ \
  https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz

# Optional: Create IMPG index (speeds up queries significantly)
impg index data/alignments/hprc465vschm13.aln.paf.gz
```

### Run the Example

```bash
./bin/run_impg_ped.sh
```

This analyzes:
- **Query**: HG00344 (both haplotypes)
- **References**: HG00099 and HG00097 (4 haplotypes)
- **Region**: chr1:50-60 Mb (10 Mb test region)

Results appear in `tutorial_relatedness/results/`.

### Expected Output

The script will show progress and finish with a summary:

```
============================================================
RELATEDNESS ANALYSIS SUMMARY
============================================================

Total segments: 5
Total length: 5.83 Mb
Query samples: 2
Reference haplotypes: 4

--- Segments by reference haplotype ---
  HG00097#1: 1 segments, 1.78 Mb
  HG00097#2: 1 segments, 1.06 Mb
  HG00099#1: 2 segments, 2.03 Mb
  HG00099#2: 1 segments, 0.95 Mb

--- Per query sample ---

  HG00344#1 (0.84 Mb total):
    HG00099#1: 0.84 Mb (100.0%)

  HG00344#2 (4.99 Mb total):
    HG00097#1: 1.78 Mb (35.8%)
    HG00099#1: 1.18 Mb (23.7%)
    HG00097#2: 1.06 Mb (21.3%)
    HG00099#2: 0.95 Mb (19.1%)
```

### Expected Output Files

| File | Description |
|------|-------------|
| `relatedness.tsv` | Segments with assigned reference haplotype |
| `relatedness_painting.png` | Chromosome painting visualization |
| `relatedness_stats.png` | Segment counts and lengths by reference |
| `relatedness_proportions.png` | Proportions per query haplotype |
| `posteriors.tsv` | Per-window posterior probabilities |
| `similarities.tsv` | Raw pairwise similarities |

### Expected relatedness.tsv

```
chrom          start      end        sample     ancestry   n_windows  mean_similarity  mean_posterior  discriminability
CHM13#0#chr1   50000001   51785000   HG00344#2  HG00097#1  357        0.9997           0.748           0.001
CHM13#0#chr1   53515001   54470000   HG00344#2  HG00099#2  191        0.9993           0.843           0.002
CHM13#0#chr1   57475001   58660000   HG00344#2  HG00099#1  237        0.9991           0.915           0.003
CHM13#0#chr1   58935001   60000000   HG00344#2  HG00097#2  213        0.9991           0.836           0.004
CHM13#0#chr1   54915001   55760000   HG00344#1  HG00099#1  169        0.9985           0.832           0.002
```

### Interpretation

- **HG00344#1** shows 100% similarity to HG00099#1 in the detected segment
- **HG00344#2** has segments matching all 4 reference haplotypes, with HG00097#1 being the most common (36%)
- The `mean_posterior` values (0.75-0.91) indicate moderate to high confidence
- Low `discriminability` values (<0.005) suggest references have similar overall similarity, which is expected for unrelated EUR individuals

---

## Using Your Own Data

### Requirements

You need:
1. **AGC file** (`your.agc`) - Compressed assemblies ([AGC format](https://github.com/refresh-bio/agc))
2. **PAF file** (`your.paf.gz`) - Alignments to a reference genome

Optional but recommended:
- **IMPG index** - Create with `impg index your.paf.gz` (speeds up queries)

### Step 1: Identify your reference format

Check your PAF file to find the reference sequence names:

```bash
zcat your.paf.gz | head -5 | cut -f6
```

Example outputs:
- HPRC human data: `CHM13#0#chr1`
- Custom data: `reference#chr1` or just `chr1`

**Important**: Use the exact format shown when running `impg`.

### Step 2: Create sample files

Create a working directory:

```bash
mkdir -p my_analysis/{samples,results}
```

Create query file (the individual to analyze):

```bash
cat > my_analysis/samples/query.txt << 'EOF'
SAMPLE_A#1
SAMPLE_A#2
EOF
```

Create references file (potential relatives):

```bash
cat > my_analysis/samples/references.txt << 'EOF'
SAMPLE_B#1
SAMPLE_B#2
SAMPLE_C#1
SAMPLE_C#2
EOF
```

Combine all samples:

```bash
cat my_analysis/samples/query.txt my_analysis/samples/references.txt > my_analysis/samples/all.txt
```

Create populations file (each reference haplotype as its own "population"):

```bash
cat > my_analysis/samples/populations.tsv << 'EOF'
SAMPLE_B#1	SAMPLE_B#1
SAMPLE_B#2	SAMPLE_B#2
SAMPLE_C#1	SAMPLE_C#1
SAMPLE_C#2	SAMPLE_C#2
EOF
```

### Step 3: Test impg works with your data

Test a single region to verify the format:

```bash
impg similarity \
    --sequence-files your.agc \
    -a your.paf.gz \
    -r "YOUR_REFERENCE:1-10000" \
    --subset-sequence-list my_analysis/samples/all.txt \
    --force-large-region \
    -t 1
```

If you get "Sequence not found", check the reference format from Step 1.

### Step 4: Run the analysis

Set your parameters:

```bash
# Data files
AGC="your.agc"
PAF="your.paf.gz"

# Reference format (from Step 1)
REFERENCE="YOUR_REFERENCE"   # e.g., "CHM13#0#chr1" or "ref#chr1"

# Region to analyze
CHROM="chr1"                 # Chromosome name in your reference
START=1
END=10000000                 # 10 Mb

# Processing
WINDOW_SIZE=5000
JOBS=8

WORKDIR="my_analysis"
```

Generate similarities:

```bash
TMPDIR=$(mktemp -d)
OUTDIR="$WORKDIR/results"
mkdir -p "$OUTDIR"

# Generate windows
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
    -r "${REFERENCE}:\${start}-\${end}" \\
    --subset-sequence-list "$WORKDIR/samples/all.txt" \\
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

Run HMM inference:

```bash
./target/release/ancestry \
    --sequence-files "$AGC" \
    -a "$PAF" \
    -r "${REFERENCE%#*}" \
    --region "${CHROM}:${START}-${END}" \
    --region-length $((END - START + 1)) \
    --window-size $WINDOW_SIZE \
    --query-samples "$WORKDIR/samples/query.txt" \
    --populations "$WORKDIR/samples/populations.tsv" \
    -o "$OUTDIR/relatedness.tsv" \
    --similarity-file "$OUTDIR/similarities.tsv" \
    --estimate-params \
    --smooth-min-windows 3 \
    --min-posterior 0.7 \
    --posteriors-output "$OUTDIR/posteriors.tsv" \
    -t $JOBS
```

Generate plots:

```bash
python3 bin/plot_relatedness.py \
    "$OUTDIR/relatedness.tsv" \
    -o "$OUTDIR/relatedness"
```

### Output Files

| File | Description |
|------|-------------|
| `similarities.tsv` | Pairwise similarities per window |
| `relatedness.tsv` | Segments with best-matching reference haplotype |
| `posteriors.tsv` | Per-window posterior probabilities |
| `relatedness_painting.png` | Chromosome painting |
| `relatedness_stats.png` | Summary statistics |

### Output Format

`relatedness.tsv` columns:

```
chrom   start     end       sample     ancestry   mean_posterior  n_windows  discriminability
chr1    1         500000    SAMPLE_A#1 SAMPLE_B#1 0.92           100        0.045
chr1    500001    1200000   SAMPLE_A#1 SAMPLE_C#2 0.88           140        0.038
```

- **ancestry**: Which reference haplotype this segment matches best
- **mean_posterior**: Confidence (0-1)
- **discriminability**: Difference between best and second-best match

---

## Troubleshooting

### "Sequence 'X' not found in index"

Your reference format is wrong. Check your PAF file:

```bash
zcat your.paf.gz | head -1 | cut -f6
```

Use that exact format.

### "Using default Glossophaga populations"

You forgot `--populations`. The tool defaults to bat species without it.

```bash
ancestry --populations my_analysis/samples/populations.tsv ...
```

### Empty results

1. Check sample IDs match your AGC:
   ```bash
   # List samples in AGC (if you have agc tool)
   agc listset your.agc | grep "SAMPLE"
   ```

2. Test impg manually:
   ```bash
   impg similarity --sequence-files your.agc -a your.paf.gz \
       -r "YOUR_REF:1-10000" -t 1
   ```

### impg is slow

- Use fewer parallel jobs if memory is limited
- Create an IMPG index:
  ```bash
  impg index your.paf.gz
  ```

---

## Parameters Reference

### ancestry

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window-size` | 5000 | Window size in bp |
| `--populations` | (required) | TSV file: population_name, haplotype_id |
| `--estimate-params` | off | Auto-estimate HMM parameters |
| `--smooth-min-windows` | 0 | Merge short segments |
| `--min-posterior` | 0.0 | Minimum confidence to report |
| `-t` | 4 | Number of threads |

### impg similarity

| Parameter | Description |
|-----------|-------------|
| `--sequence-files` | AGC file |
| `-a` | PAF alignment file |
| `-r` | Region (reference:start-end) |
| `--subset-sequence-list` | File with sample IDs to include |
| `--force-large-region` | Allow regions > 100kb |
