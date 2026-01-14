# Data Files for IBS/IBD Analysis

This directory contains symlinks to the HPRC v2 data files required for IBS/IBD analysis.

## Files

| File | Size | Description |
|------|------|-------------|
| `HPRC_r2_assemblies_0.6.1.agc` | 3.1 GB | AGC (Assembly Graph Container) with all HPRC v2 assemblies |
| `hprc465vschm13.aln.paf.gz` | 5.3 GB | PAF alignments of 465 HPRC haplotypes to CHM13 reference |
| `hprc465vschm13.aln.paf.gz.impg` | 315 MB | IMPG index for fast region queries |

## Source Location

Original files are located at:
```
/home/franco/Escritorio/genomica/impop/data/human/
```

## Usage

These files are used by the `impg similarity` tool to compute pairwise sequence identity between haplotypes in specified genomic regions.

### Example Command

```bash
impg similarity \
  --sequence-files data/HPRC_r2_assemblies_0.6.1.agc \
  -a data/hprc465vschm13.aln.paf.gz \
  -r CHM13#0#chr2:130787850-130792849 \
  --subset-sequence-list sample_lists/HPRCv2_EURsubset.txt
```

## Data Contents

### AGC File
Contains phased diploid assemblies from 233 individuals (465 haplotypes + CHM13 reference):
- Mat (maternal) and Pat (paternal) haplotypes for trio-phased samples
- Hap1 and Hap2 for non-trio samples

### PAF Alignments
Minimap2 alignments of all haplotypes against the CHM13 v2.0 reference genome.

### IMPG Index
Pre-computed index enabling fast projection of reference coordinates to query haplotype coordinates.

## Population Subsets

Sample lists for population-specific analyses are in `../sample_lists/`:
- `HPRCv2_AFRsubset.txt` - African ancestry (32 samples, 64 haplotypes)
- `HPRCv2_EURsubset.txt` - European ancestry (30 samples, 60 haplotypes)
- `HPRCv2_EASsubset.txt` - East Asian ancestry (27 samples, 54 haplotypes)
- `HPRCv2_CSAsubset.txt` - Central/South Asian ancestry (11 samples, 22 haplotypes)
- `HPRCv2_AMRsubset.txt` - Americas ancestry (3 samples, 6 haplotypes)
