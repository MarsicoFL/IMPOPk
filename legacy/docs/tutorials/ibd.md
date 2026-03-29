# Tutorial: IBD Segment Detection with HMM

## Overview

The IBD (Identity-By-Descent) detection pipeline uses a Hidden Markov Model (HMM) to identify genomic segments where two haplotypes share recent common ancestry. Unlike simple IBS detection, the HMM integrates evidence across consecutive windows to distinguish true IBD from sporadic sequence similarity.

The Rust binary includes:
- **Viterbi algorithm**: Finds the most likely state sequence (MAP estimate)
- **Forward-backward algorithm**: Computes posterior P(IBD|data) for each window
- **Posterior-based filtering**: Filter segments by confidence

---

## How It Works

The IBD pipeline:
1. Collects sequence identity observations across sliding windows (same as IBS)
2. Groups observations by haplotype pair
3. Estimates HMM emission parameters using k-means clustering with population priors
4. Runs Viterbi for state sequence AND forward-backward for posteriors
5. Extracts IBD segments filtered by length, windows, and posterior threshold

### The Two-State HMM Model

The HMM uses two hidden states:

| State | Name | Description | Typical Identity |
|-------|------|-------------|------------------|
| 0 | Non-IBD | Haplotypes do NOT share recent ancestry | ~0.999 (population-specific) |
| 1 | IBD | Haplotypes DO share recent common ancestry | ~0.9997 (sequencing error rate) |

**Key insight**: In humans, both IBD and non-IBD have very high identity (~0.999). The difference is only ~0.05-0.1%. The HMM accumulates evidence over many windows to distinguish them.

---

## Prerequisites

### Required Tools
| Tool | Purpose | Installation |
|------|---------|--------------|
| **impg** | Pangenome similarity queries | `cargo install impg` |
| **Rust toolchain** | Building the binary | [rustup.rs](https://rustup.rs) |

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
    --min-len-bp <BP>             Minimum IBD segment length [default: 2000000]
    --min-windows <N>             Minimum windows per segment [default: 400]
    --expected-seg-windows <N>    Expected IBD segment length in windows [default: 50]
    --p-enter-ibd <PROB>          Probability of entering IBD state [default: 0.0001]
    --population <POP>            Population for HMM parameters [default: Generic]
                                  Options: AFR, EUR, EAS, CSA, AMR, InterPop, Generic
    --posterior-threshold <PROB>  Minimum mean P(IBD) for segment [default: 0.0]
    --output-posteriors <FILE>    Output per-window posteriors (optional)
    -t, --threads <N>             Number of parallel threads [default: auto]
    -h, --help                    Print help information
```

---

## HMM Parameters Explained

### Population-Specific Parameters

The non-IBD emission depends on nucleotide diversity (pi):

| Population | Pi | Non-IBD Mean | Description |
|------------|-----|--------------|-------------|
| AFR | 0.00125 | 0.99875 | African (highest diversity) |
| EUR | 0.00085 | 0.99915 | European |
| EAS | 0.00080 | 0.99920 | East Asian |
| CSA | 0.00095 | 0.99905 | Central/South Asian |
| AMR | 0.00100 | 0.99900 | American (admixed) |
| InterPop | 0.00110 | 0.99890 | Cross-population comparison |
| Generic | 0.00100 | 0.99900 | Default |

**Why it matters**: Using the correct population ensures the HMM correctly distinguishes IBD from background similarity.

### Transition Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `expected_seg_windows` | `--expected-seg-windows` | 50 | Expected length of IBD segments in windows |
| `p_enter_ibd` | `--p-enter-ibd` | 0.0001 | Probability of transitioning from non-IBD to IBD |

**How they affect the model**:

- **expected_seg_windows**: Controls how "sticky" the IBD state is.
  ```
  p_stay_ibd = 1 - 1/expected_seg_windows

  Example:
    expected_seg_windows = 50  ->  p_stay_ibd = 0.98 (98% stay in IBD)
    expected_seg_windows = 10  ->  p_stay_ibd = 0.90 (90% stay in IBD)
  ```

- **p_enter_ibd**: Controls how easily the model enters IBD. Lower = more conservative.

### Posterior Filtering (NEW)

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `posterior_threshold` | `--posterior-threshold` | 0.0 | Minimum mean P(IBD) for segment |

The forward-backward algorithm computes P(IBD|all data) for each window. Segments are filtered by their mean posterior.

**Recommended values**:
- `0.0`: No filtering (report all Viterbi-detected segments)
- `0.5`: Moderate confidence
- `0.8`: High confidence
- `0.9`: Very high confidence

### Segment Filtering Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `min_len_bp` | `--min-len-bp` | 2000000 | Minimum segment length in base pairs (2 Mb) |
| `min_windows` | `--min-windows` | 400 | Minimum number of windows |

---

## Usage Examples

### Example 1: Basic IBD Detection

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-10000000 \
  --size 5000 \
  --subset-sequence-list data/samples/EUR.txt \
  --population EUR \
  --output /tmp/ibd_segments.tsv
```

### Example 2: High-Confidence Detection with Posteriors

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-50000000 \
  --size 5000 \
  --subset-sequence-list data/samples/AFR.txt \
  --population AFR \
  --output /results/ibd_high_conf.tsv \
  --posterior-threshold 0.8 \
  --output-posteriors /results/posteriors.tsv
```

### Example 3: Save All Intermediate Data

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr20:1-50000000 \
  --size 5000 \
  --subset-sequence-list data/samples/EUR.txt \
  --population EUR \
  --output /results/ibd_segments.tsv \
  --ibs-output /results/ibs_windows.tsv \
  --output-posteriors /results/posteriors.tsv
```

### Example 4: Sensitive Detection (Short Segments)

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-100000000 \
  --size 5000 \
  --subset-sequence-list data/samples/EUR.txt \
  --population EUR \
  --output /tmp/ibd_sensitive.tsv \
  --expected-seg-windows 20 \
  --min-windows 3 \
  --min-len-bp 10000
```

### Example 5: Conservative Detection (Reduce False Positives)

```bash
./target/release/ibd \
  --sequence-files /data/HPRC_r2_assemblies_0.6.1.agc \
  -a /data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  --region chr1:1-100000000 \
  --size 5000 \
  --subset-sequence-list data/samples/EUR.txt \
  --population EUR \
  --output /tmp/ibd_conservative.tsv \
  --expected-seg-windows 100 \
  --min-windows 10 \
  --min-len-bp 50000 \
  --p-enter-ibd 0.00001 \
  --posterior-threshold 0.9
```

---

## Output Format

### IBD Segments Output

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome name |
| `start` | Segment start position |
| `end` | Segment end position |
| `group.a` | First haplotype identifier |
| `group.b` | Second haplotype identifier |
| `n_windows` | Number of windows in segment |
| `mean_identity` | Average sequence identity across segment |
| `mean_posterior` | Mean P(IBD) across segment (from forward-backward) |
| `min_posterior` | Minimum P(IBD) in segment |
| `max_posterior` | Maximum P(IBD) in segment |

**Example**:
```
chrom	start	end	group.a	group.b	n_windows	mean_identity	mean_posterior	min_posterior	max_posterior
chr20	5001	75000	HG01167#1	NA19682#1	15	0.999200	0.9523	0.8901	0.9812
chr20	150001	325000	HG01167#1	NA19682#2	35	0.998900	0.8845	0.7234	0.9567
```

### Posteriors Output (--output-posteriors)

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome name |
| `start` | Window start position |
| `end` | Window end position |
| `group.a` | First haplotype identifier |
| `group.b` | Second haplotype identifier |
| `identity` | Sequence identity for this window |
| `posterior` | P(IBD) for this window given all data |

**Example**:
```
chrom	start	end	group.a	group.b	identity	posterior
chr20	1	5000	HG01167#1	NA19682#1	0.998734	0.1234
chr20	5001	10000	HG01167#1	NA19682#1	0.999812	0.8923
```

---

## Interpreting Results

### Using Posterior Values

| Mean Posterior | Interpretation |
|----------------|----------------|
| > 0.95 | Very high confidence IBD |
| 0.8 - 0.95 | High confidence IBD |
| 0.5 - 0.8 | Moderate confidence, may need validation |
| < 0.5 | Low confidence, likely false positive |

### Example Analysis

```bash
# Filter for high-confidence segments
awk -F'\t' 'NR==1 || $8 > 0.9' ibd_segments.tsv > high_conf_segments.tsv

# Count segments by confidence tier
awk -F'\t' 'NR>1 {
  if ($8 > 0.95) tier="very_high";
  else if ($8 > 0.8) tier="high";
  else if ($8 > 0.5) tier="moderate";
  else tier="low";
  print tier
}' ibd_segments.tsv | sort | uniq -c
```

---

## Algorithm Details

### Viterbi Algorithm

Finds the most likely state sequence:
```
delta[t][s] = max over prev of:
    delta[t-1][prev] + log(transition[prev->s]) + log(emission[s](obs[t]))
```

### Forward-Backward Algorithm

Computes posterior P(state=IBD | all observations):
```
P(IBD at t | all obs) = alpha[t][IBD] * beta[t][IBD] / P(all obs)

Where:
  alpha[t][s] = P(obs[0..t], state[t]=s)     [forward]
  beta[t][s]  = P(obs[t+1..n] | state[t]=s)  [backward]
```

**Key difference**:
- Viterbi: Best single path (global decoding)
- Forward-backward: Probability at each position (marginal decoding)

Both are computed for comprehensive analysis.

---

## Troubleshooting

### No IBD Segments Detected

**Check**:
1. Are samples related? Use `--posterior-threshold 0` to see all candidates
2. Are parameters too conservative? Try `--min-windows 2 --min-len-bp 5000`
3. Check intermediate IBS data: `--ibs-output debug.tsv`

### Too Many Low-Confidence Segments

**Solution**: Use posterior filtering
```bash
--posterior-threshold 0.8
```

### Segments Breaking at Low-Identity Windows

**Solution**: Increase expected segment length
```bash
--expected-seg-windows 100
```

---

## See Also

- [IBS Tutorial](ibs.md) - Understanding the input data
- [Jacquard Tutorial](jacquard_coeffs.md) - Computing diploid identity states
