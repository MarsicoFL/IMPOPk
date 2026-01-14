# Tutorial: IBD Segment Detection with HMM

## Overview

The IBD (Identity-By-Descent) detection pipeline uses a Hidden Markov Model (HMM) to identify genomic segments where two haplotypes share recent common ancestry. Unlike simple IBS detection, the HMM integrates evidence across consecutive windows to distinguish true IBD from sporadic sequence similarity.

This tutorial covers:
- **Rust binary**: `cargo run --bin ibd` (recommended for production)
- **Shell script**: `scripts/ibd.sh` (R-based HMM implementation)

---

## How It Works

The IBD pipeline:
1. Collects sequence identity observations across sliding windows (same as IBS)
2. Groups observations by haplotype pair
3. Estimates HMM emission parameters using k-means clustering
4. Runs the Viterbi algorithm to find the most likely state sequence
5. Extracts contiguous IBD segments meeting length/quality thresholds

### The Two-State HMM Model

The HMM uses two hidden states:

| State | Name | Description | Typical Identity |
|-------|------|-------------|------------------|
| 0 | Non-IBD | Haplotypes do NOT share recent ancestry | ~0.5 (random) |
| 1 | IBD | Haplotypes DO share recent common ancestry | ~0.999 (identical) |

**Key insight**: True IBD segments show sustained high identity across many consecutive windows, while sporadic IBS appears as isolated high-identity windows.

---

## Prerequisites

### Required Tools
| Tool | Purpose | Installation |
|------|---------|--------------|
| **impg** | Pangenome similarity queries | `cargo install impg` |
| **Rust toolchain** | Building the binary | [rustup.rs](https://rustup.rs) |
| **Rscript** (optional) | For shell script HMM | `apt install r-base` |

### Required Data Files
Same as IBS pipeline: AGC archive, PAF alignment, subset list.

---

## CLI Reference

### Rust Binary Arguments

```
ibd - IBD segment detection using HMM

USAGE:
    ibd [OPTIONS] --sequence-files <FILE> -a <FILE> -r <NAME> --region <REGION> --size <BP> --subset-sequence-list <FILE> --output <FILE>

OPTIONS:
    --sequence-files <FILE>       Path to AGC/FASTA sequence archive (required)
    -a <FILE>                     Alignment file (.paf/.paf.gz) (required)
    -r <NAME>                     Reference name, e.g., CHM13 (required)
    --region <REGION>             Target region: chr1:1-1000000 or chr1 (required)
    --size <BP>                   Window size in base pairs (required)
    --subset-sequence-list <FILE> Haplotypes to compare (required)
    --output <FILE>               Output IBD segments TSV (required)
    --ibs-output <FILE>           Optional: save intermediate IBS windows
    --region-length <BP>          Required if --region omits coordinates
    --min-len-bp <BP>             Minimum IBD segment length [default: 5000]
    --min-windows <N>             Minimum windows per segment [default: 3]
    --expected-seg-windows <N>    Expected IBD segment length in windows [default: 50]
    --p-enter-ibd <PROB>          Probability of entering IBD state [default: 0.0001]
    -h, --help                    Print help information
```

---

## HMM Parameters Explained

### Transition Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `expected_seg_windows` | `--expected-seg-windows` | 50 | Expected length of IBD segments in windows |
| `p_enter_ibd` | `--p-enter-ibd` | 0.0001 | Probability of transitioning from non-IBD to IBD |

**How they affect the model**:

- **expected_seg_windows**: Controls how "sticky" the IBD state is. Higher values mean the model expects longer IBD segments and is less likely to break segments on single low-identity windows.

  ```
  p_stay_ibd = 1 - 1/expected_seg_windows

  Example:
    expected_seg_windows = 50  ->  p_stay_ibd = 0.98 (98% chance to stay in IBD)
    expected_seg_windows = 10  ->  p_stay_ibd = 0.90 (90% chance to stay in IBD)
  ```

- **p_enter_ibd**: Controls how easily the model enters the IBD state. Lower values make it harder to call new IBD segments (more conservative).

### Emission Parameters

The HMM uses Gaussian emission distributions that are **automatically estimated** from the data using k-means clustering:

| State | Typical Mean | Typical Std | Description |
|-------|--------------|-------------|-------------|
| Non-IBD | ~0.5 | ~0.2 | Background random similarity |
| IBD | ~0.99 | ~0.01 | High identity from shared ancestry |

**Adaptive estimation**: The model automatically clusters observed identities into two groups, handling datasets with different overall identity distributions.

### Segment Filtering Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `min_len_bp` | `--min-len-bp` | 5000 | Minimum segment length in base pairs |
| `min_windows` | `--min-windows` | 3 | Minimum number of windows |

---

## Parameter Tuning Guide

### For Different IBD Lengths

| Expected IBD | Recommended Settings | Rationale |
|--------------|---------------------|-----------|
| Short (recent admixture) | `--expected-seg-windows 20 --min-windows 3` | More sensitive to short segments |
| Medium (typical) | `--expected-seg-windows 50 --min-windows 5` | Balanced default |
| Long (isolated populations) | `--expected-seg-windows 200 --min-windows 10` | Stricter, fewer false positives |

### For Different Population Contexts

| Context | Recommended Settings | Rationale |
|---------|---------------------|-----------|
| Closely related samples | `--p-enter-ibd 0.001` | Higher prior for IBD |
| Diverse populations | `--p-enter-ibd 0.0001` | Conservative default |
| Outgroup comparisons | `--p-enter-ibd 0.00001` | Very conservative |

### For Different Window Sizes

| Window Size | Segment Windows | Rationale |
|-------------|-----------------|-----------|
| 1 kb | 100-200 | More windows for same physical length |
| 5 kb | 20-50 | Typical setting |
| 10 kb | 10-25 | Fewer windows needed |

---

## Usage Examples

### Example 1: Basic IBD Detection

```bash
cd /path/to/HPRCv2-IBD/production/ibs-cli

./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-10000000 \
  --size 5000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibd_segments.tsv
```

### Example 2: Save Intermediate IBS Data

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-50000000 \
  --size 5000 \
  --subset-sequence-list sample_lists/HPRCv2_AFRsubset.txt \
  --output /results/ibd_segments.tsv \
  --ibs-output /results/ibs_windows.tsv
```

### Example 3: Sensitive Detection (Short Segments)

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-100000000 \
  --size 5000 \
  --subset-sequence-list sample_lists/ibs_example.txt \
  --output /tmp/ibd_sensitive.tsv \
  --expected-seg-windows 20 \
  --min-windows 3 \
  --min-len-bp 10000
```

### Example 4: Conservative Detection (Reduce False Positives)

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-100000000 \
  --size 5000 \
  --subset-sequence-list sample_lists/HPRCv2_EURsubset.txt \
  --output /tmp/ibd_conservative.tsv \
  --expected-seg-windows 100 \
  --min-windows 10 \
  --min-len-bp 50000 \
  --p-enter-ibd 0.00001
```

---

## Output Format

The output is a tab-separated file:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome name |
| `start` | Segment start position |
| `end` | Segment end position |
| `group.a` | First haplotype identifier |
| `group.b` | Second haplotype identifier |
| `n_windows` | Number of windows in segment |
| `mean_identity` | Average sequence identity across segment |

### Example Output

```
chrom	start	end	group.a	group.b	n_windows	mean_identity
chr20	5001	75000	HG01167#1	NA19682#1	15	0.999200
chr20	150001	325000	HG01167#1	NA19682#2	35	0.998900
chr20	500001	650000	HG01167#2	NA19682#1	30	0.999500
```

### Interpreting IBD Output

| Metric | Interpretation |
|--------|----------------|
| `n_windows` | Confidence indicator - more windows = more evidence |
| `mean_identity` | Quality indicator - higher = stronger signal |
| Segment length | Long segments (>1 Mb) suggest recent common ancestry |
| Multiple segments | May indicate same IBD block split by recombination or gaps |

---

## Output Interpretation Examples

### Case 1: Strong IBD Signal
```
chr20	1000000	5000000	SampleA#1	SampleB#1	800	0.9995
```
- **Length**: 4 Mb (very long)
- **Windows**: 800 (strong evidence)
- **Identity**: 99.95% (excellent)
- **Interpretation**: High-confidence IBD, likely close relatives or recent shared ancestry

### Case 2: Moderate IBD Signal
```
chr20	1000000	1100000	SampleA#1	SampleC#2	20	0.9980
```
- **Length**: 100 kb (moderate)
- **Windows**: 20 (sufficient)
- **Identity**: 99.80% (good)
- **Interpretation**: Probable IBD, may need validation

### Case 3: Borderline Signal
```
chr20	1000000	1025000	SampleA#1	SampleD#1	5	0.9950
```
- **Length**: 25 kb (short)
- **Windows**: 5 (minimum)
- **Identity**: 99.50% (marginal)
- **Interpretation**: Possible false positive, consider increasing thresholds

---

## Troubleshooting

### No IBD Segments Detected

**Possible causes**:
1. Samples are not closely related
2. Parameters too conservative
3. Insufficient data coverage

**Solutions**:
```bash
# Lower thresholds
--min-windows 2 --min-len-bp 5000 --p-enter-ibd 0.001

# Check intermediate IBS data
--ibs-output debug_ibs.tsv
# Then examine: do high-identity windows exist?
```

### Too Many Short Segments

**Possible cause**: expected_seg_windows too low, causing fragmentation

**Solution**:
```bash
# Increase expected segment length
--expected-seg-windows 100 --min-windows 10
```

### Segments Breaking at Low-Identity Windows

**Possible cause**: Single dropout windows breaking true IBD segments

**Solution**:
```bash
# The HMM already handles this via state persistence
# Ensure expected-seg-windows is high enough
--expected-seg-windows 50  # or higher
```

---

## Algorithm Details

### Viterbi Algorithm

The Viterbi algorithm finds the most likely sequence of hidden states given the observations:

```
For each window t:
    delta[t][s] = max over previous states of:
        delta[t-1][prev] + log(transition[prev->s]) + log(emission[s](identity[t]))
```

The final state sequence is obtained by backtracking through the computed path.

### Emission Estimation via K-means

Before running Viterbi, the emission parameters are estimated:

1. Cluster all observed identities into 2 groups using k-means
2. Compute mean and standard deviation for each cluster
3. Assign lower cluster to non-IBD state, higher to IBD state

This adaptive approach handles different identity distributions across datasets.

---

## See Also

- [IBS Tutorial](ibs.md) - Understanding the input data
- [Jacquard Tutorial](jacquard_coeffs.md) - Computing diploid identity states
- [Conceptual Framework](../paper_concepts/conceptual_framework.md) - Theoretical background on binary IBD
- [TESTING_GUIDE.md](../../TESTING_GUIDE.md) - Testing the IBD pipeline
