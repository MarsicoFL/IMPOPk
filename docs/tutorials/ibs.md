# Tutorial: `production/ibs-cli/scripts/ibs.sh`

## Purpose
`ibs.sh` slides a fixed-size window along a reference chromosome, invokes
`impg similarity` on each region, filters the results, and streams them into a
single TSV describing IBS-positive haplotype pairs. It mirrors the Rust `ibs`
binary and is useful for quick prototypes or parity tests.

## Prerequisites
- `impg` executable available in `PATH`.
- Sequence archive (e.g. `.agc`) and corresponding alignment `.paf/.paf.gz`.
- Optional subset list with haplotypes to compare (plain text, one ID per line).

## Required inputs
| Flag | Description |
| --- | --- |
| `--sequence-files` | Path(s) to AGC/FASTA archives fed into `impg` |
| `-a` | Alignment file passed to `impg` as `-p` |
| `-r` | Reference name (e.g. `CHM13`) |
| `-region` | Target interval (`chr:start-end` or `chr`) |
| `-size` | Window length in bp |
| `--output` | Destination TSV file |
| `--region-length` | Only needed when `-region` omits coordinates (e.g. `chr1`) |
| `--subset-sequence-list` | Optional haplotype allowlist |

## How to run
1. Change into `production/ibs-cli/scripts` so relative paths behave as expected.
2. Provide AGC/PAF/region/window parameters. Example:
   ```bash
   cd production/ibs-cli/scripts
   ./ibs.sh \
     --sequence-files ../data/human/HPRC_r2_assemblies_0.6.1.agc \
     -a ../data/human/hprc465vschm13.aln.paf.gz \
     -r CHM13 \
     -region chr20:1-15000000 \
     -size 5000 \
     --subset-sequence-list ../sample_lists/ibs_example.txt \
     --output /tmp/ibs_chr20.tsv
   ```
3. Monitor stderr for per-window progress messages. The script streams each
   processed window directly to `--output`.

## Output
A TSV headed with `chrom start end group.a group.b estimated.identity`. Each row
lists an IBS-positive haplotype pair in a window that met the identity cutoff
(default `1.0`, adjustable with `-c`).
