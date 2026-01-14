# HPRCv2-IBD

IBS detection and IBD inference from HPRC pangenome assemblies using haplotype-level identity analysis.

## What it does

- **IBS detection**: Sliding window analysis over chromosomes using `impg` similarity
- **IBD inference**: 2-state HMM (Viterbi) to distinguish true IBD from sporadic IBS
- **Identity states**: Jacquard-style coefficients for diploid pairs

## Structure

```
production/ibs-cli/    # Rust CLI tools (ibs, ibd, jacquard binaries)
analysis/              # Notebooks and Python scripts
docs/                  # Methods documentation and tutorials
data/                  # AGC archives, PAF alignments, impg indices
```

## Requirements

- Rust 1.70+
- [impg](https://github.com/pangenome/impg)
- GNU coreutils, parallel

## Usage

```bash
# Build
cd production/ibs-cli && cargo build --release

# Run IBS detection
./scripts/ibs.sh \
  --sequence-files /path/to/hprc.agc \
  -a /path/to/alignments.paf.gz \
  -r CHM13 \
  -region chr20:1-64444167 \
  -size 5000 \
  --subset-sequence-list samples.txt \
  --output ibs.tsv

# Run IBD calling
./scripts/ibd.sh \
  --sequence-files /path/to/hprc.agc \
  -a /path/to/alignments.paf.gz \
  -r CHM13 \
  -region chr20:1-64444167 \
  -size 5000 \
  --subset-sequence-list samples.txt \
  --output ibd.tsv
```

## Data

Download required files:
```bash
# Sequence archive
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/HPRC_r2_assemblies_0.6.1.agc

# Alignments and index
wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz
wget https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz.gzi
wget https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg
```
