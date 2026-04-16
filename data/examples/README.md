# impopₖ — Example datasets

Each subfolder contains a precomputed pairwise-identity table and a
`run.sh` recipe so you can exercise one inference mode of `impopk`
without first running `impg similarity` yourself.

| Folder | Mode | What it shows |
|--------|------|---------------|
| `ibs/` | IBS enrichment | Reading a pairwise identity TSV and summarising high-identity windows per population. |
| `ibd/` | IBD detection | The 2-state HMM (`ibd`) detecting IBD segments from the precomputed identity TSV. |
| `ancestry/` | Local ancestry | The N-state HMM (`ancestry`) painting a query haplotype with ancestry labels. |
| `pedigree/` | Founder painting | Running the same N-state HMM where each population slot is one grandparent. |

## What is in `input/`

Each `input/ibs.tsv` is the output of the `ibs` binary on a small panel of
HPRC chr12 haplotypes. Columns:

```
chrom  start  end  group.a  group.b  estimated.identity  [group.a.length]  [group.b.length]
```

`populations.tsv` maps haplotypes to population labels (TSV: population, haplotype).
`queries.txt` lists query haplotypes for ancestry/pedigree, one per line.

## Running an example

From the repository root, build the binaries once:

```bash
cargo build --release
```

Then cd into any example folder and run:

```bash
cd data/examples/ibd
bash run.sh
```

Outputs are written to the local `output/` folder. Reference outputs from
a previous run are in `expected_output/` for sanity comparison.

## Reproducing from scratch

If you want to reproduce the `input/ibs.tsv` tables from the underlying
pangenome, see the full pipeline in the top-level `README.md`. In brief:

```bash
ibs --alignment PAF --sequence-files AGC \
    --region chr12:15000000-20000000 --size 10000 \
    --subset-sequence-list subset.txt --output input/ibs.tsv
```
