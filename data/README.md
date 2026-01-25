# HPRCv2-IBD Input Data

This directory contains input data for IBD analysis. Large files are stored as symlinks to external locations.

## Directory Structure

```
data/
├── README.md            # This file
├── assemblies/          # HPRC genome assemblies
│   └── HPRC_r2_assemblies_0.6.1.agc -> (symlink)
├── alignments/          # Alignments to CHM13 reference
│   ├── hprc465vschm13.aln.paf.gz -> (symlink)
│   └── hprc465vschm13.aln.paf.gz.impg -> (symlink)
└── samples/             # Population sample lists
    ├── AFR.txt          # African (67 individuals, 134 haplotypes)
    ├── EUR.txt          # European (30 individuals, 60 haplotypes)
    ├── EAS.txt          # East Asian (50 individuals, 100 haplotypes)
    ├── CSA.txt          # Central/South Asian (36 individuals, 72 haplotypes)
    └── AMR.txt          # Admixed American (44 individuals, 88 haplotypes)
```

## Data Sources

### Assemblies

| File | Size | Description |
|------|------|-------------|
| `HPRC_r2_assemblies_0.6.1.agc` | 3.1 GB | AGC-compressed HPRC v2 assemblies |

**Source:** Human Pangenome Reference Consortium (HPRC) v2 release

Contains phased diploid assemblies from 227 individuals (454 haplotypes):
- Mat (maternal) and Pat (paternal) haplotypes for trio-phased samples
- Hap1 and Hap2 for non-trio samples

### Alignments

| File | Size | Description |
|------|------|-------------|
| `hprc465vschm13.aln.paf.gz` | 5.3 GB | PAF alignments to CHM13 reference |
| `hprc465vschm13.aln.paf.gz.impg` | 315 MB | IMPG index for fast region queries |

**Reference:** CHM13 T2T assembly (complete human reference)

Minimap2 alignments of all haplotypes against the CHM13 v2.0 reference genome.

### Population Sample Lists

| File | Individuals | Haplotypes | Description |
|------|-------------|------------|-------------|
| `AFR.txt` | 67 | 134 | African ancestry |
| `EUR.txt` | 30 | 60 | European ancestry |
| `EAS.txt` | 50 | 100 | East Asian ancestry |
| `CSA.txt` | 36 | 72 | Central/South Asian ancestry |
| `AMR.txt` | 44 | 88 | Admixed American ancestry |
| **Total** | **227** | **454** | |

## Sample List Format

Each file contains haplotype identifiers, one per line:

```
HG00096#1
HG00096#2
HG00097#1
...
```

Format: `{sample_id}#{haplotype_number}` (1 = hap1/mat, 2 = hap2/pat)

## Usage Examples

### IBS Analysis
```bash
../src/ibs-cli/target/release/ibs \
    --sequence-files assemblies/HPRC_r2_assemblies_0.6.1.agc \
    -a alignments/hprc465vschm13.aln.paf.gz \
    --subset-sequence-list samples/EUR.txt \
    --region chr1:1-10000000 \
    --size 5000 \
    -t 0.999 -m cosine
```

### IBD Inference
```bash
../src/ibd-cli/target/release/ibd-hmm inference \
    --input ibs_results.tsv \
    --output ibd_segments.json
```

## Symlink Targets

Files point to:
```
/home/franco/Escritorio/genomica/impop/data/human/
```

Ensure these files exist before running analyses.

## Citation

If using HPRC data, please cite the Human Pangenome Reference Consortium.
