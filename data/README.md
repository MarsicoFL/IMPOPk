# Data Directory

This directory contains sample metadata bundled with impopk and serves as the
target location for downloaded datasets.  After running the download scripts,
the layout should look like this:

```
data/
├── README.md                       # This file
├── samples/                        # Bundled — population sample lists
│   ├── AFR.txt                     #   70 individuals, 140 haplotypes
│   ├── EUR.txt                     #   31 individuals,  62 haplotypes
│   ├── EAS.txt                     #   51 individuals, 102 haplotypes
│   ├── CSA.txt                     #   36 individuals,  72 haplotypes
│   └── AMR.txt                     #   44 individuals,  88 haplotypes
├── genetic_maps/                   # Bundled — plink-format recombination maps
│   ├── plink.chr10.GRCh38.map
│   ├── plink.chr11.GRCh38.map
│   ├── plink.chr12.GRCh38.map
│   └── plink.chr20.GRCh38.map      # (full set: chr1-chr22 via download script)
├── assemblies/                     # Downloaded — ~3.1 GB
│   └── HPRC_r2_assemblies_0.6.1.agc
├── alignments/                     # Downloaded — ~5.3 GB
│   └── hprc465vschm13.aln.paf.gz
├── reference/                      # Downloaded — ~0.9 GB
│   ├── chm13v2.0.fa
│   ├── chm13v2.0.fa.fai
│   └── chm13v2.0_SD.bed
├── vcf/                            # Downloaded — ~2 GB (validation only)
│   └── (chr10, chr11, chr12 subsets)
└── platinumPed/                    # Downloaded — ~1 GB (validation only)
    └── (CEPH 1463 pedigree data)
```

## What is bundled vs. what needs downloading

| Component | Bundled? | Size | Download script |
|-----------|----------|------|-----------------|
| Sample lists (`samples/`) | Yes | 5 KB | `scripts/generate_sample_lists.sh` (regenerate) |
| Genetic maps (`genetic_maps/`) | Partial (4 chr) | 16 MB | `scripts/download_genetic_maps.sh` (full 22 chr) |
| HPRC AGC assemblies | No | 3.1 GB | `scripts/download_hprc.sh` |
| HPRC PAF alignments | No | 5.3 GB | `scripts/download_hprc.sh` |
| CHM13 v2.0 reference | No | 0.9 GB | `scripts/download_reference.sh` |
| Validation VCFs | No | ~2 GB | `scripts/download_vcf.sh` |
| Platinum pedigree | No | ~1 GB | `scripts/download_platinum.sh` |

**Quick start**: run `scripts/download_all.sh` to fetch everything (~12 GB total).
Use `--dry-run` to preview what will be downloaded.


## Sample lists

The files in `data/samples/` contain haplotype identifiers for 232 HPRC
individuals (464 haplotypes), classified by continental superpopulation:

| Population | Code | Individuals | Haplotypes | Source populations |
|------------|------|-------------|------------|-------------------|
| African | AFR | 70 | 140 | ACB, ASW, ESN, GWD, MSL, YRI, MKK + African American |
| European | EUR | 31 | 62 | GBR, FIN, TSI + Ashkenazi Jewish (GIAB HG002) |
| East Asian | EAS | 51 | 102 | CHS, CDX, KHV, JPT + Han Chinese (GIAB HG005) |
| Central/South Asian | CSA | 36 | 72 | BEB, GIH, ITU, PJL, STU |
| Admixed American | AMR | 44 | 88 | CLM, MXL, PEL, PUR |
| **Total** | | **232** | **464** | |

Each file lists one haplotype ID per line in the format `SAMPLEID#1` and
`SAMPLEID#2`, matching the query names used in the PAF alignment file.

### Data sources for population classification

Sample-to-superpopulation assignments were derived from three sources,
applied in priority order:

1. **1000 Genomes 3202-sample PED file** (223 of 232 samples):
   `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt`

2. **HPRC Year 1 sample metadata** (4 additional samples with superpopulation):
   `https://github.com/human-pangenomics/HPP_Year1_Assemblies/blob/main/sample_metadata/hprc_year1_assemblies_v2_sample_metadata.txt`

3. **Manual annotation from Coriell catalog** (5 samples):
   - HG002: GIAB Ashkenazi Jewish trio son (Coriell NA24385) -> EUR
   - HG005: GIAB Han Chinese trio son (Coriell NA24631) -> EAS
   - HG03471: Mende in Sierra Leone (MSL, Coriell HG03471) -> AFR
   - HG06807: African American, St. Louis MO (Coriell HG06807) -> AFR
   - NA21309: Maasai in Kinyawa, Kenya (MKK, HPRC HAPMAP annotation) -> AFR

The script `scripts/generate_sample_lists.sh` automates this process.  It
extracts sample IDs from a PAF file, downloads the metadata sources above,
cross-references them, and writes the per-population files.

### Note on CSA vs SAS naming

This project uses **CSA** (Central/South Asian) instead of the standard 1000
Genomes code **SAS** (South Asian).  The CSA label is used throughout the
codebase and HMM population parameters.  The `generate_sample_lists.sh` script
handles this mapping automatically.


## Genetic maps

Plink-format genetic maps for GRCh38 from the Browning Lab
(University of Washington).  Four chromosomes (10, 11, 12, 20) are bundled;
the full set of 22 autosomes can be downloaded with:

```bash
./scripts/download_genetic_maps.sh
```

Source: https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/


## Assemblies and alignments

The HPRC Release 2 pangenome data consists of:

- **AGC file** (`HPRC_r2_assemblies_0.6.1.agc`): Compressed collection of 465
  phased haplotype assemblies in AGC format.  Required by `impg` for
  sequence-level pairwise identity computation.

- **PAF alignment** (`hprc465vschm13.aln.paf.gz`): Minigraph-cactus whole-genome
  alignment of all 465 haplotypes against CHM13 v2.0 reference.  This is the
  primary input for all impopk tools.

Source: https://humanpangenome.org/data/


## Reference genome

CHM13 v2.0 (T2T Consortium), used as the coordinate system for all analyses:

- `chm13v2.0.fa` — Reference FASTA
- `chm13v2.0.fa.fai` — FASTA index
- `chm13v2.0_SD.bed` — Segmental duplication regions (for masking)

Source: https://github.com/marbl/CHM13


## Validation data (optional)

These datasets are only needed for reproducing the validation results
reported in the paper.

### VCFs
Phased VCF subsets for chromosomes 10, 11, and 12, used to run hap-ibd and
RFMix as gold-standard comparisons.

### Platinum pedigree
CEPH family 1463 pedigree data from the Platinum Genomes project, used
for the 4-state founder painting validation (99.53% accuracy).

Source: s3://platinum-pedigree-data/
