# impopₖ

IBD detection and local ancestry inference from pangenome assemblies, without VCF or statistical phasing.

## Overview

A suite of Rust CLI tools that compute haplotype identity directly from pangenome alignments (PAF) and use Hidden Markov Models to infer IBD segments and local ancestry. Alternative to VCF-based methods (hap-ibd, RFMix).

- **IBS Detection**: Pairwise identity computation from PAF alignments (>1,000x faster than impg)
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
git clone https://github.com/MarsicoFL/IMPOPk.git
cd impopk
cargo build --release
```

Binaries: `target/release/{ibs,ibs-from-paf,ibs-from-tpa,tpa-spatial-index,tpa-validate,ibd,ibd-validate,ancestry,jacquard}`



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

Interactive HTML tutorial covering all tools and analysis modes: **[docs/tutorials.html](docs/tutorials.html)**

Topics: Installation, Data Preparation, IBS Computation, IBD Detection, Ancestry Inference, Platinum Pedigree Validation, Simulation Framework, Advanced Features (eGRM, demographics, cross-chromosome, Jacquard).

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
