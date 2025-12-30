# Tutorial: `production/ibs-cli/scripts/ibd.sh`

## Purpose
`ibd.sh` expands on the IBS sliding-window pipeline by keeping all
`impg similarity` rows, aggregating them per haplotype pair, and running a
lightweight R-based Hidden Markov Model to call IBD segments along each
chromosome.

## Prerequisites
- `impg` and `Rscript` available in `PATH`.
- Same AGC/PAF inputs used for the IBS wrapper.
- Subset list enumerating the haplotypes of interest.
- GNU coreutils, `awk`, and `parallel` (already required by `ibs.sh`).

## Required inputs
| Flag | Description |
| --- | --- |
| `--sequence-files` | Sequence archive(s) for `impg` |
| `-a` | Alignment file for `impg -p` |
| `-r` | Reference name |
| `-region` | Interval (`chr:start-end` or chromosome name) |
| `-size` | Window size in bp |
| `--subset-sequence-list` | Haplotypes to compare |
| `--output` | Final IBD-segment TSV |
| `--region-length` | Needed when `-region` omits coordinates |
| `--ibs-output` | Optional path to dump intermediate per-window identities |
| `--min-len-bp` | Minimum IBD length to keep (bp) |
| `--expected-seg-windows` | Expected IBD segment length (windows) for the HMM |

## How to run
1. Move into `production/ibs-cli/scripts`.
2. Launch the pipeline with the desired region and parameters. Example:
   ```bash
   cd production/ibs-cli/scripts
   ./ibd.sh \
     --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \
     -a ../data/human/hprc465vschm13.aln.paf.gz \
     -r CHM13 \
     -region chr20:1-60000000 \
     -size 5000 \
     --subset-sequence-list ../sample_lists/ibs_example.txt \
     --ibs-output /tmp/ibs_windows.tsv \
     --output /tmp/ibd_segments.tsv
   ```
3. The script first streams IBS windows (same format as `ibs.sh`) and then runs
   the embedded R HMM to emit segments with state, length, and supporting stats.

## Output
`--output` receives a tabular file where each row represents an IBD segment with
columns for chromosome, start, end, haplotypes, mean identity, and related
metrics. If `--ibs-output` is provided, the intermediate per-window IBS table is
also saved for troubleshooting or downstream use.
