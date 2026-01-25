# Data

Sample lists and required data for running HPRCv2-IBD tools.

## Population Sample Lists

| File | Individuals | Haplotypes |
|------|-------------|------------|
| `samples/AFR.txt` | 67 | 134 |
| `samples/EUR.txt` | 30 | 60 |
| `samples/EAS.txt` | 50 | 100 |
| `samples/CSA.txt` | 36 | 72 |
| `samples/AMR.txt` | 44 | 88 |
| **Total** | **227** | **454** |

### Format

One sample ID per line:
```
HG00097
HG00099
HG00126
```

Tools expand these to haplotype IDs (`HG00097#1`, `HG00097#2`).

## Required External Data

Download these files to run analyses (not included due to size):

| File | Size | Download |
|------|------|----------|
| `HPRC_r2_assemblies_0.6.1.agc` | 3.1 GB | [Link](https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=submissions/B4174A5F-F20E-4DCF-8470-F8A907B640BC--HPRCv2_0.6.1_pr_agc_submission/) |
| `hprc465vschm13.aln.paf.gz` | 5.3 GB | [Link](https://garrisonlab.s3.amazonaws.com/hprcv2/pafs/hprc465vschm13.aln.paf.gz) |
| `hprc465vschm13.aln.paf.gz.impg` | 315 MB | [Link](https://garrisonlab.s3.amazonaws.com/hprcv2/impg/hprc465vschm13.aln.paf.gz.impg) |

## Usage

```bash
# IBS detection
ibs --sequence-files assemblies/HPRC_r2.agc \
    -a alignments/hprc465vschm13.aln.paf.gz \
    --subset-sequence-list samples/EUR.txt \
    --region chr1:1-10000000 \
    --size 5000 -t 0.999

# IBD inference
ibd-hmm inference --input ibs_results.tsv --output ibd.json
```
