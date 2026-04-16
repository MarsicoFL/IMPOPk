# impopₖ

**Local ancestry and IBD inference directly from pangenome-derived alignments.**

`impopk` is a small suite of Rust CLI tools that compute windowed pairwise
sequence identity from haplotype-resolved assembly alignments and feed that
signal to Hidden Markov Models for IBD detection, local ancestry inference,
and kinship estimation. It does **not** require phased VCFs, variant
calling, or a population-specific SNP panel.

The approach is described in:

> Marsico et al. *Local ancestry and IBD inference directly from
> pangenome-derived alignments*. [Manuscript in preparation.]

## What impopₖ does

Given a pangenome alignment (PAF) and a haplotype-resolved assembly archive
(AGC), `impopk` offers four inference modes:

| Mode | Binary | Input | Output |
|------|--------|-------|--------|
| **IBS** | `ibs` | PAF + AGC + region + subset list | Windowed pairwise identity TSV |
| **IBD** | `ibd` | IBS TSV + sample pairs | Detected IBD segments per pair |
| **Local ancestry** | `ancestry` | IBS TSV + population map + query list | Painted ancestry tracts per query |
| **Kinship** | `jacquard` | IBD segment TSV | Jacquard Δ coefficients per pair |

The `ibs` binary is a wrapper over `impg similarity` that filters, deduplicates,
and canonicalizes its output into a stable TSV format. Everything downstream
(`ibd`, `ancestry`, `jacquard`) reads that TSV.

## Installation

Rust 1.70+ and `impg` are required. `impg` provides the pangenome graph
query used to compute pairwise identity.

```bash
# 1. Install impg (https://github.com/pangenome/impg)
cargo install impg

# 2. Build impopk
git clone https://github.com/MarsicoFL/IMPOPk.git
cd IMPOPk
cargo build --release
```

Binaries are placed in `target/release/`:

```
target/release/ibs
target/release/ibd
target/release/ibd-validate
target/release/ancestry
target/release/jacquard
```

## Quick start (precomputed examples)

The `data/examples/` folder ships precomputed pairwise-identity TSVs for
each inference mode, so you can run the HMMs without first setting up a
pangenome alignment. Each subfolder contains a ready-to-run shell recipe:

```bash
cd data/examples/ibd && bash run.sh            # 2-state IBD HMM
cd data/examples/ancestry && bash run.sh       # N-state ancestry HMM
cd data/examples/pedigree && bash run.sh       # founder painting (4-state)
cd data/examples/ibs && bash run.sh            # IBS window enrichment
```

See `data/examples/README.md` for a walkthrough.

## Full pipeline from a pangenome

If you already have a pangenome alignment (PAF) and assemblies (AGC), the
full pipeline looks like this:

### 1. Compute windowed pairwise identity (IBS)

```bash
ibs \
  --alignment    data/alignments/hprc_chr12.paf.gz \
  --sequence-files data/assemblies/HPRC_r2.agc \
  --region       chr12:1-133324548 \
  --size         10000 \
  --subset-list  data/panel_subset.txt \
  --threads      8 \
  --output       ibs_chr12.tsv
```

### 2. IBD detection

```bash
ibd \
  --similarity-file ibs_chr12.tsv \
  --window-size     10000 \
  --estimate-params \
  --threads         8 \
  --output          ibd_chr12.tsv
```

### 3. Local ancestry

```bash
ancestry \
  --similarity-file ibs_chr12.tsv \
  --window-size     10000 \
  --populations     populations.tsv \
  --query-samples   queries.txt \
  --emission-model  max \
  --estimate-params \
  --threads         8 \
  --output          ancestry_chr12.tsv
```

### 4. Kinship

```bash
jacquard \
  --ibd-file ibd_chr12.tsv \
  --output   jacquard_chr12.tsv
```

## Tutorials

End-to-end tutorials covering each mode are in `docs/tutorials.html`.

## License

MIT. See `LICENSE`.
