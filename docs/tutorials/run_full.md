# Tutorial: `production/ibs-cli/scripts/run_full.sh`

## Purpose
`run_full.sh` automates chromosome-wide IBS tiling. It divides a region into
non-overlapping windows, launches multiple `ibs.sh` processes via GNU Parallel,
and merges the partial outputs into a single sorted TSV.

## Prerequisites
- Same requirements as `ibs.sh` (`impg`, AGC/PAF files, subset list).
- GNU Parallel available in `PATH`.
- Sufficient CPU and disk bandwidth for the chosen `JOBS` value.

## Configuration via environment variables
| Variable | Default | Description |
| --- | --- | --- |
| `AGC` | `../data/human/HPRC_r2_assemblies_0.6.1.agc` | Sequence archive |
| `PAF` | `../data/human/hprc465vschm13.aln.paf.gz` | Alignment file |
| `SUB` | `../sample_lists/ibs_example.txt` | Haplotypes to compare |
| `REF` | `CHM13` | Reference name |
| `CHR` | `chr20` | Chromosome |
| `START` | `1` | Start coordinate |
| `END` | `60000000` | End coordinate |
| `SIZE` | `5000` | Window size |
| `JOBS` | `10` | Parallel workers |

## How to run
1. `cd production/ibs-cli/scripts`.
2. Override any relevant environment variables (optional).
3. Execute the wrapper:
   ```bash
   cd production/ibs-cli/scripts
   AGC=/data/HPRC.agc PAF=/data/hprc.paf.gz \
   CHR=chr7 START=1 END=159345973 SIZE=10000 JOBS=16 \
   ./run_full.sh
   ```
4. The script writes temporary partial files in a scratch directory, merges them
   (sorted by `chrom/start/end`), and saves the combined result as
   `production/ibs-cli/ibs_for_ibd.out`.

## Output
`production/ibs-cli/ibs_for_ibd.out` is a sorted TSV identical to the output of
`ibs.sh` but covering the entire tiled region. Remove or archive the file before
running again if you need multiple variants.
