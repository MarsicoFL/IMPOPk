# chr1_full: Complete Pipeline Instructions

## Quick Start

```bash
cd /path/to/ibd-cli/experiments/validation/chr1_full/scripts

# Step 1: Generate pairwise identity data (run once, takes ~10-20 hrs)
./01_generate_identity_data.sh

# Step 2: Run IBD inference with HMM (after Step 1 completes)
python3 02_ibd_hmm_inference.py --populations EUR AFR

# Step 3: Generate figures (after Step 2 completes)
python3 03_generate_figures.py
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  01_generate_identity_data.sh                                       │
│  ─────────────────────────────                                      │
│  Uses: pairwise-identity.sh (in ibd-cli/scripts/)                  │
│  Input: HPRC assemblies + alignments + sample lists                │
│  Output: data/EUR_chr1_full.tsv (~10 GB)                           │
│          data/AFR_chr1_full.tsv (~50 GB)                           │
│  Time: ~10-20 hours                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  02_ibd_hmm_inference.py                                            │
│  ───────────────────────                                            │
│  Uses: ibd_inference.py (HMM core module)                          │
│  Input: data/*_chr1_full.tsv                                       │
│  Output: results/json/*_emission_params.json                       │
│          results/json/*_ibd_results.json                           │
│          results/REPORT.md                                          │
│  Time: ~30-60 minutes per population                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  03_generate_figures.py                                             │
│  ──────────────────────                                             │
│  Uses: visualization.py (plotting utilities)                       │
│  Input: results/json/*.json                                        │
│  Output: results/figures/*.png, *.pdf                              │
│  Time: ~5 minutes                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Generate Pairwise Identity Data

### What it does

Computes ALL pairwise sequence identity values between haplotypes using `impg similarity`. No cutoff filtering - this captures the full distribution needed for proper HMM calibration.

### Run

```bash
./01_generate_identity_data.sh
```

Or run manually for a single population:

```bash
# From ibd-cli/scripts/
./pairwise-identity.sh \
  --sequence-files ../../ibs-cli/data/HPRC_r2_assemblies_0.6.1.agc \
  -a ../../ibs-cli/data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr1:1-248956422 \
  -size 5000 \
  --subset-sequence-list ../../ibs-cli/sample_lists/HPRCv2_EUR_full.txt \
  --output ../experiments/validation/chr1_full/data/EUR_chr1_full.tsv \
  -j 8
```

### Expected output

| File | Population | Individuals | Pairs | Est. Size |
|------|------------|-------------|-------|-----------|
| `EUR_chr1_full.tsv` | EUR | 30 | 1,770 | ~10 GB |
| `AFR_chr1_full.tsv` | AFR | 67 | 8,911 | ~50 GB |

### Monitor progress

```bash
# Check if files are being written
ls -lh data/*.tsv

# Watch file growth
watch -n 60 'ls -lh data/*.tsv'
```

---

## Step 2: Run IBD HMM Inference

### What it does

1. Loads full identity distribution from TSV
2. Estimates emission parameters empirically:
   - Non-IBD distribution (mean, std) from bulk of data
   - IBD distribution from high-identity tail (≥0.9995)
   - Calculates d' separability metric
3. Runs HMM for selected pairs:
   - Forward-backward → posterior P(IBD) per window
   - Viterbi → MAP state sequence
   - Segment extraction with statistics
4. Generates summary report

### Run

```bash
# Both populations (recommended)
python3 02_ibd_hmm_inference.py --populations EUR AFR

# Single population
python3 02_ibd_hmm_inference.py --populations EUR

# With custom parameters
python3 02_ibd_hmm_inference.py \
  --populations EUR AFR \
  --max-pairs 100 \
  --expected-ibd-length 100 \
  --min-segment-windows 20

# Quick test with sampled data
python3 02_ibd_hmm_inference.py --populations EUR --max-pairs 10 --sample-frac 0.1
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--populations` | EUR AFR | Which populations to analyze |
| `--max-pairs` | 50 | Max haplotype pairs for full HMM |
| `--expected-ibd-length` | 50 | Expected IBD length in windows (250 kb) |
| `--min-segment-windows` | 10 | Min segment = 10 windows = 50 kb |
| `--sample-frac` | 1.0 | Fraction of data (for testing) |

### Expected output

```
results/
├── REPORT.md
└── json/
    ├── EUR_emission_params.json   # HMM emission parameters
    ├── EUR_ibd_results.json       # All IBD segments found
    ├── AFR_emission_params.json
    └── AFR_ibd_results.json
```

---

## Step 3: Generate Figures

### Run

```bash
python3 03_generate_figures.py
```

### Expected output

```
results/figures/
├── fig1_distribution_analysis.png
├── fig2_ibd_tracks.png
├── fig3_population_comparison.png
└── fig4_validation_summary.png
```

---

## File Structure

```
chr1_full/
├── PIPELINE.md                      # This file
├── README.md                        # Experiment overview
│
├── data/                            # Step 1 output
│   ├── EUR_chr1_full.tsv           # ~10 GB
│   └── AFR_chr1_full.tsv           # ~50 GB
│
├── results/                         # Steps 2-3 output
│   ├── REPORT.md
│   ├── json/
│   │   ├── EUR_emission_params.json
│   │   ├── EUR_ibd_results.json
│   │   ├── AFR_emission_params.json
│   │   └── AFR_ibd_results.json
│   └── figures/
│       └── *.png, *.pdf
│
└── scripts/
    ├── 01_generate_identity_data.sh  # Step 1
    ├── 02_ibd_hmm_inference.py       # Step 2
    ├── 03_generate_figures.py        # Step 3
    ├── ibd_inference.py              # HMM core module
    └── visualization.py              # Plotting utilities
```

---

## Expected Results

Based on exp02 (chr2:1-50Mb), scaled to full chr1:

| Metric | EUR | AFR |
|--------|-----|-----|
| d' separability | ~1.5-2.0 | ~1.5-2.0 |
| Pairs with IBD ≥2Mb | ~95% | ~28% |
| Mean segment length | ~3.0 Mb | ~2.5 Mb |
| Segments per pair | ~3-6x more | baseline |

The EUR/AFR difference reflects:
- EUR: Recent bottleneck, smaller Ne, more recent common ancestry
- AFR: Larger Ne, deeper coalescence, more genetic diversity
