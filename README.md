# impopₖ
v.0.0.1
**Local ancestry and IBD inference directly from pangenome-derived alignments.**

`impopk` is a small suite of Rust CLI tools that compute windowed pairwise
sequence identity from haplotype-resolved assembly alignments and feed that
signal to Hidden Markov Models for IBD detection, local ancestry inference,
and kinship estimation. It does **not** require phased VCFs, variant
calling, or a population-specific SNP panel.

## What impopₖ does

Given a pangenome alignment (PAF) and a haplotype-resolved assembly archive
(AGC), `impopk` offers four inference modes:

| Mode | Binary | Input | Output |
|------|--------|-------|--------|
| **IBS** | `ibs` | PAF + AGC + region + subset list | Windowed pairwise identity TSV |
| **IBD** | `ibd` | IBS TSV + sample pairs | Detected IBD segments per pair |
| **Local ancestry** | `ancestry` | IBS TSV + population map + query list | Painted ancestry tracts per query |
| **Kinship (scalar θ)** | `scripts/kinship_from_ibd.py` | IBD segments TSV + chromosome length | θ̂ = Σ L_IBD / 4·L per diploid pair |
| **Kinship (Δ states)** | `jacquard` | IBS TSV + 4 haplotype IDs | Nine Jacquard Δ coefficients for that pair |

The `ibs` binary is a wrapper over `impg similarity` that filters, deduplicates,
and canonicalizes its output into a stable TSV format. Everything downstream
(`ibd`, `ancestry`, `jacquard`) reads that TSV.

## Installation

### Dependencies

The only hard requirement to run the HMMs on precomputed inputs (see
`data/examples/`) is a working Rust toolchain.

To run the full pipeline starting from a pangenome you also need:

- **[impg](https://github.com/pangenome/impg)** ≥ 0.3 — pangenome graph query
- **[AGC](https://github.com/refresh-bio/agc)** ≥ 3.2 — compressed assembly archive (C++, used by `impg`)

### Build

```bash
# Dependencies
cargo install impg                                    # Rust crate
# AGC must be built from source (C++); see its repo for instructions.

# impopk
git clone https://github.com/MarsicoFL/IMPOPk.git
cd IMPOPk
cargo build --release
```

Binaries are placed in `target/release/`:

```
target/release/ibs          # windowed pairwise identity (wraps impg)
target/release/ibd          # 2-state IBD HMM
target/release/ibd-validate # compare ibd output against a gold-standard IBD TSV
target/release/ancestry     # N-state local ancestry / founder painting HMM
target/release/jacquard     # 9 Jacquard Δ coefficients for a diploid pair
```

`ibd-validate` is a developer utility: given a detected-IBD TSV and a
ground-truth TSV (same columns), it reports recall, precision, and
boundary accuracy. Useful when tuning parameters against simulated data.

### Naming

Three forms of the name appear and are intentional:
`impopₖ` is the stylised display form, `impopk` is the prose and
code form, and `IMPOPk` is the GitHub repository slug. They all refer
to the same project.

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
  -a                      data/alignments/hprc_chr12.paf.gz \
  --sequence-files        data/assemblies/HPRC_r2.agc \
  -r                      CHM13 \
  --region                chr12:1-133324548 \
  --size                  10000 \
  --subset-sequence-list  data/panel_subset.txt \
  --threads               8 \
  --output                ibs_chr12.tsv
```

### 2. IBD detection

```bash
ibd \
  --similarity-file ibs_chr12.tsv \
  --region          chr12:1-133324548 \
  --region-length   133324548 \
  --size            10000 \
  --min-len-bp      2000000 \
  --population      Generic \
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

### 4a. Kinship scalar from detected IBD

The kinship formula
θ̂ = Σ<sub>α,β</sub> L<sub>IBD</sub>(A<sub>α</sub>, B<sub>β</sub>) / 4·L
is implemented as a thin post-processor over the `ibd` output:

```bash
python3 scripts/kinship_from_ibd.py \
  --ibd          ibd_chr12.tsv \
  --chrom-length 133324548 \
  --output       kinship_chr12.tsv
```

Output columns: `individual_a`, `individual_b`, `total_ibd_bp`, `theta_hat`.

### 4b. Full Jacquard Δ coefficients (nine condensed states)

`jacquard` reads the windowed identity TSV (not IBD segments) and takes the
four haplotypes of the pair as arguments:

```bash
jacquard \
  --ibs    ibs_chr12.tsv \
  --hap-a1 HG00097#1 --hap-a2 HG00097#2 \
  --hap-b1 HG00099#1 --hap-b2 HG00099#2
```

The nine condensed-identity deltas are printed to stdout. Use this when you
need the full identity-state decomposition rather than the scalar θ.

## File formats

### Input: `ibs.tsv` (produced by `ibs`)

Tab-separated, one row per pair-window:

```
chrom    start    end    group.a    group.b    estimated.identity    [group.a.length]    [group.b.length]
```

- `chrom` can be either a bare chromosome (`chr12`) or a PanSN path
  (`CHM13#0#chr12`) — downstream tools accept both.
- `group.a`, `group.b` are haplotype identifiers. Either short form
  (`HG00097#1`) or full PanSN contig (`HG00097#1#CM087323.1:1-248000`)
  is accepted. The downstream HMMs strip everything after the second `#`
  to reduce to a haplotype identity.
- The trailing coverage-length columns are optional (present when `ibs`
  is run with `--coverage-feature` downstream).

### Input: `populations.tsv` (for `ancestry` and pedigree painting)

Tab-separated, two columns, no header:

```
EUR     HG00097#1
EUR     HG00097#2
AFR     HG01884#1
...
```

The first column is the state label used by the HMM; the second is the
haplotype ID. For founder painting (pedigree mode) each founder is its
own state — see `data/examples/pedigree/input/populations.tsv`.

### Input: `queries.txt` (for `ancestry` and pedigree)

One haplotype ID per line.

### Input: `subset-sequence-list` (for `ibs`)

One haplotype ID (or sample ID) per line. Accepted: `HG00097`,
`HG00097#1`, or full PanSN contigs.

### Output: `ibd.tsv` (from `ibd`)

```
chrom  start  end  group.a  group.b  n_windows  mean_identity  mean_posterior
min_posterior  max_posterior  lod_score
```

### Output: `ancestry.tsv` (from `ancestry`)

```
chrom  start  end  sample  ancestry  n_windows  mean_similarity  mean_posterior
discriminability  lod_score
```

- `discriminability`: gap between the top two state probabilities
  (high = clear call, low = ambiguous).
- `lod_score`: log-odds of the top state versus the next-best alternative.

## Tutorials

End-to-end tutorials covering each mode are in `docs/tutorials.html`.

## Integration tests

A short end-to-end sanity script exercises all five binaries on the
bundled mini fixtures and runs `cargo test --workspace` + `cargo clippy`:

```bash
bash test/run_mini_tests.sh
```

Expect `21/21 passed`. Use this after any local modification to catch
regressions.

## License

MIT. See `LICENSE`.
