# ancestry-cli

Local Ancestry Inference (LAI) using an N-state HMM for HPRC pangenome data.

## Overview

Given pairwise identity data from a pangenome alignment, this tool infers the ancestral population of each genomic segment for query haplotypes. It uses a Hidden Markov Model with one state per reference population, where emission probabilities are based on the maximum similarity between a query haplotype and each population's reference haplotypes.

Key features:
- N-state HMM (one state per ancestral population)
- Viterbi decoding for MAP state sequences; forward-backward for posterior probabilities
- Auto-configuration of emission temperature, switch probability, and pairwise weights
- Two-pass inference with posterior feedback for improved accuracy
- Deconvolution of admixed reference populations

## Usage

```bash
ancestry \
  --similarity-file chr1_similarity.tsv \
  --populations populations.tsv \
  --query-samples query_haplotypes.txt \
  --auto-configure \
  -o ancestry_segments.tsv \
  --posteriors-output ancestry_posteriors.tsv
```

The `--populations` file maps reference haplotypes to population labels. The `--query-samples` file lists query haplotype IDs. With `--auto-configure`, emission and transition parameters are estimated from the data automatically.

When pre-computed similarities are not available, you can compute them on-the-fly:

```bash
ancestry \
  --sequence-files data/assemblies.agc \
  -a data/alignments.paf.gz \
  -r CHM13 \
  --region chr1:1-248956422 \
  --populations populations.tsv \
  --query-samples query_haplotypes.txt \
  --auto-configure \
  -o ancestry_segments.tsv
```

## Output Files

### Ancestry segments (`-o`)

10 columns:

```
chrom   start   end   sample   ancestry   n_windows   mean_similarity   mean_posterior   discriminability   lod_score
```

- **discriminability**: mean difference between highest and lowest population similarity across windows
- **lod_score**: log-odds score for the assigned ancestry vs. alternatives

### Per-window posteriors (`--posteriors-output`)

```
chrom   start   end   sample   P(pop1)   P(pop2)   ...   margin   entropy
```

- **margin**: difference between the top-two posterior probabilities (confidence measure)
- **entropy**: Shannon entropy of the posterior distribution (lower = more certain)

## Tutorials

See the `tutorials/` directory for detailed walkthroughs with HPRC data, including ancestry inference on human superpopulations.
