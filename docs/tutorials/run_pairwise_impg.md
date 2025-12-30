# Tutorial: `analysis/ibd-network/scripts/run_pairwise_impg.sh`

## Purpose
This research helper consumes a BED file of windows, calls `impg similarity` on
each region, and emits an augmented table with the original `impg` columns plus
`REGION/CHR/START/END/LENGTH`. It feeds the Python prototypes in
`analysis/ibd-network/scripts/ibd.py`.

## Prerequisites
- `impg` installed.
- BED file defining windows to evaluate.
- AGC and PAF inputs (defaults point to `../../data/`).
- Optional subset list specifying the haplotypes to compare.

## Flags
| Flag | Description |
| --- | --- |
| `-b` | BED file with `chr start end` |
| `-p` | PAF alignment (`--sequence-files` uses AGC) |
| `-s` | Sequence archive (`.agc`) |
| `-u` | Optional subset list |
| `-P` | Region prefix (default `CHM13#0#`) |
| `-o` | Output TSV (defaults to `stdout`) |
| `-v` | Verbose logging |

Environment variables `PAF_FILE`, `SEQUENCE_FILES`, `REGION_PREFIX`, and
`SUBSET_LIST` can predefine the same options.

## How to run
```bash
cd analysis/ibd-network/scripts
./run_pairwise_impg.sh \
  -b windows.bed \
  -p ../../data/hprc465vschm13.aln.paf.gz \
  -s ../../data/HPRC_r2_assemblies_0.6.1.agc \
  -u subset.txt \
  -o pairwise.tsv -v
```

## Output
A TSV whose first five columns are `REGION CHR START END LENGTH` followed by the
unmodified `impg similarity` header. Each row corresponds to one haplotype pair
within a BED-defined window, making it suitable for downstream aggregation and
IBD calling experiments.
