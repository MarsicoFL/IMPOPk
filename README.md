# impopₖ

IBD detection and local ancestry inference from pangenome assemblies, without VCF or statistical phasing.

## Overview

A suite of Rust CLI tools that compute haplotype identity directly from pangenome alignments (PAF) and use Hidden Markov Models to infer IBD segments and local ancestry. Alternative to VCF-based methods (hap-ibd, RFMix).

- **IBS Detection**: Pairwise identity computation from PAF alignments (>100x faster than impg)
- **IBD Inference**: 2-state HMM (Viterbi + forward-backward + Baum-Welch) for IBD segment detection
- **Local Ancestry**: N-state HMM with softmax emissions, pairwise contrast, auto-configuration
- **Jacquard Coefficients**: Delta coefficient estimation for relatedness analysis
- **eGRM Output**: GCTA-compatible genetic relationship matrices from assemblies

## Tools

| Tool | Binary | Description |
|------|--------|-------------|
| [ibs-cli](src/ibs-cli/) | `ibs` | Window-based IBS detection via impg |
| [ibs-cli](src/ibs-cli/) | `ibs-from-paf` | Direct PAF identity (no impg needed, >100x faster) |
| [ibs-cli](src/ibs-cli/) | `ibs-from-tpa` | TPA-indexed identity with O(1) regional access (6x faster) |
| [ibs-cli](src/ibs-cli/) | `tpa-spatial-index` | Build spatial index over TPA files |
| [ibs-cli](src/ibs-cli/) | `tpa-validate` | Numerical validation between IBS outputs |
| [ibd-cli](src/ibd-cli/) | `ibd` | HMM-based IBD inference |
| [ibd-cli](src/ibd-cli/) | `ibd-validate` | IBD validation against gold standard |
| [ancestry-cli](src/ancestry-cli/) | `ancestry` | N-state HMM local ancestry inference |
| [jacquard-cli](src/jacquard-cli/) | `jacquard` | Jacquard delta coefficients |

## Installation

### Requirements

- **Rust** 1.70+ ([rustup.rs](https://rustup.rs/))

### Build

```bash
git clone https://github.com/MarsicoFL/impopk.git
cd impopk
cargo build --release
```

Binaries: `target/release/{ibs,ibs-from-paf,ibs-from-tpa,tpa-spatial-index,tpa-validate,ibd,ibd-validate,ancestry,jacquard}`

See [INSTALL.md](INSTALL.md) for detailed instructions including optional dependencies (impg, AGC).

## Quick Start

### IBS from PAF (fast)

```bash
ibs-from-paf \
    -a alignments.paf.gz \
    -r CHM13 \
    --region chr12:1-133000000 \
    --size 50000 \
    --output ibs_chr12.tsv
```

### IBS from TPA (recommended, fastest for regional queries)

Pre-index once with [cigzip](https://github.com/AndreaGuarracino/cigzip) + `tpa-spatial-index`, then run regional queries in seconds instead of minutes:

```bash
# One-time: convert PAF to TPA (requires cigzip)
cigzip encode --paf alignments.paf.gz | cigzip compress -i - -o alignments.tpa

# One-time: build spatial index (~15s)
tpa-spatial-index --tpa alignments.tpa --output alignments.sidx

# Fast regional queries (6x faster than ibs-from-paf)
ibs-from-tpa \
    --tpa alignments.tpa \
    --spatial-index alignments.sidx \
    --agc assemblies.agc \
    --region chr12:1000000-2000000 \
    --size 10000 \
    --output ibs_chr12.tsv
```

Output is identical to `ibs-from-paf` — drop-in replacement for `ancestry` and `ibd`.

### Local Ancestry (auto-configured)

```bash
ancestry \
    --similarity-file similarities.tsv \
    --populations populations.tsv \
    --query-samples query.txt \
    --auto-configure \
    --identity-floor 0.9 \
    -o ancestry_tracts.tsv
```

The populations file is a two-column TSV mapping haplotype IDs to populations (`AFR\tHG01884#1`).

### IBD Detection

```bash
ibd-validate \
    --input ibs_output.tsv \
    --output ibd_segments.tsv \
    --population EUR \
    --window-size 10000 \
    --baum-welch-iters 20 \
    --identity-floor 0.9 \
    --min-len-bp 2000000
```

### Jacquard Coefficients

```bash
jacquard \
    --ibs ibs_results.tsv \
    --hap-a1 HG00097#1 --hap-a2 HG00097#2 \
    --hap-b1 HG00099#1 --hap-b2 HG00099#2
```

### Try Now (bundled test data, no downloads needed)

```bash
# Ancestry inference on bundled mini dataset
ancestry \
    --similarity-file test/ibs_ancestry_mini.tsv \
    --populations test/populations.tsv \
    --query-samples test/query_amr_mini.txt \
    --auto-configure --identity-floor 0.9 \
    -o my_first_ancestry.tsv

# IBD detection
ibd --similarity-file test/ibs_paf_5Mb_EUR.tsv \
    --output my_first_ibd.tsv \
    --identity-floor 0.9 --min-lod 2.0 \
    --baum-welch-iters 10 --min-len-bp 500000

# Run full test suite
bash test/run_mini_tests.sh
```

## Validation Results

| Benchmark | Metric | impopₖ | Gold standard |
|-----------|--------|--------|---------------|
| Simulated ancestry (3-way) | Concordance | **97.95%** | RFMix: 95.9% |
| HPRC real ancestry (3-way) | Concordance | **76.45%** | vs RFMix |
| IBD detection (chr10/11/12) | Top-10% ranking | **100%** (11/11) | vs hap-ibd |
| Platinum pedigree (4-state) | Accuracy | **99.49%** | Mendelian inheritance |
| IBD simulation (pair F1) | F1 | **0.514** | hap-ibd: 0.489 |
| Full genome (22 autosomes) | Time / RAM | **195s / 101MB** | -- |
| PAF-direct vs impg | Speedup | **1,090x** | -- |

## Data

### Bundled Data

The repository includes lightweight data files that do not require download:

- **`data/samples/`**: Population sample lists (AFR, EUR, EAS, CSA, AMR) with haplotype IDs
- **`data/genetic_maps/`**: Plink genetic maps for GRCh38 (chr10, chr11, chr12, chr20)

### Required External Data

Large data files must be downloaded separately. Scripts are provided in `scripts/`:

| File | Size | Script |
|------|------|--------|
| HPRC r2 assemblies (AGC) | ~3.1 GB | `scripts/download_hprc.sh` |
| Pangenome alignments (PAF) | ~5.3 GB | `scripts/download_hprc.sh` |
| CHM13 v2.0 reference | ~900 MB | `scripts/download_reference.sh` |
| Platinum pedigree | varies | `scripts/download_platinum.sh` |
| Validation VCFs | varies | `scripts/download_vcf.sh` |

Download everything at once:

```bash
./scripts/download_all.sh
```

Or with `--dry-run` to see what will be downloaded without actually fetching anything.

### Input Data

The tools require:

1. **Alignments**: PAF alignments to a reference genome (e.g., from minigraph-cactus)
2. **Assemblies** (optional): AGC-compressed genome assemblies (only for impg-based mode)
3. **Sample lists**: Text files with haplotype IDs, one per population

### Included Sample Lists

Population sample lists in [`data/samples/`](data/samples/):

| Population | Individuals | Haplotypes |
|------------|-------------|------------|
| AFR | 70 | 140 |
| EUR | 31 | 62 |
| EAS | 51 | 102 |
| CSA | 36 | 72 |
| AMR | 44 | 88 |

## Tutorials

Step-by-step tutorials covering all tools and analysis modes. All tutorials use real HPRC data and require running the download scripts first.

| Tutorial | Topic |
|----------|-------|
| [01_installation.md](tutorials/01_installation.md) | Build impopₖ and install optional dependencies |
| [02_data_preparation.md](tutorials/02_data_preparation.md) | Download data, verify checksums, explore structure |
| [03_ibs_computation.md](tutorials/03_ibs_computation.md) | Compute pairwise identity from PAF with ibs-from-paf |
| [04_ibd_detection.md](tutorials/04_ibd_detection.md) | Detect IBD segments with the 2-state HMM |
| [05_ancestry_inference.md](tutorials/05_ancestry_inference.md) | 3-way local ancestry inference with auto-configure |
| [06_platinum_pedigree.md](tutorials/06_platinum_pedigree.md) | Founder painting in CEPH 1463 (99.49% accuracy) |
| [07_simulation.md](tutorials/07_simulation.md) | Simulation framework with msprime + pangenome_sim |
| [08_advanced_features.md](tutorials/08_advanced_features.md) | eGRM, demographic inference, cross-chromosome, Jacquard |

## Paper

The `paper/` directory contains the full manuscript in LaTeX:

- `HPRCv2_IBD_paper.tex` -- main document
- Individual section files: `abstract.tex`, `introduction.tex`, `methods_*.tex`, `results.tex`, `discussion.tex`
- `supplementary.tex` -- detailed supplementary material
- `references.bib` -- bibliography
- `HPRCv2_IBD_paper.pdf` -- pre-compiled PDF

To compile from source:

```bash
cd paper
pdflatex HPRCv2_IBD_paper
bibtex HPRCv2_IBD_paper
pdflatex HPRCv2_IBD_paper
pdflatex HPRCv2_IBD_paper
```

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for a scientific overview of the approach, including the HMM formulations, emission models, and comparison with VCF-based methods.

## Testing

```bash
cargo test --workspace          # 7334 tests
cargo clippy --workspace -- -D warnings
```

## Citation

```bibtex
@article{marsico2026impopk,
    title   = {impopk: Identity-by-Descent Detection and Local Ancestry
               Inference from Pangenome Alignments},
    author  = {Marsico, Franco},
    journal = {TBD},
    year    = {2026}
}
```

## License

MIT License
