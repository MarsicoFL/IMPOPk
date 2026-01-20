# chr1_full: Full Chromosome 1 IBD Analysis

## Objective

Perform complete IBD inference across the entire chromosome 1 using full pairwise identity distributions (no cutoff filtering) for EUR and AFR populations.

## Chromosome 1 Specifications

| Parameter | Value |
|-----------|-------|
| Chromosome | chr1 |
| Length (CHM13) | 248,956,422 bp (~249 Mb) |
| Window size | 5,000 bp |
| Total windows | 49,791 |
| Reference | CHM13 T2T |

## Populations

| Population | Individuals | Haplotypes | Pairwise Comparisons |
|------------|-------------|------------|----------------------|
| EUR | 30 | 60 | 1,770 |
| AFR | 67 | 134 | 8,911 |

## Data Generation

```bash
# Navigate to ibd-cli scripts
cd /path/to/ibd-cli/scripts

# Generate EUR full identity data
./pairwise-identity.sh \
  --sequence-files ../../ibs-cli/data/HPRC_r2_assemblies_0.6.1.agc \
  -a ../../ibs-cli/data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr1:1-248956422 \
  -size 5000 \
  --subset-sequence-list ../../ibs-cli/sample_lists/HPRCv2_EUR_full.txt \
  --output ../experiments/validation/chr1_full/data/EUR_chr1_full.tsv \
  -j 8

# Generate AFR full identity data
./pairwise-identity.sh \
  --sequence-files ../../ibs-cli/data/HPRC_r2_assemblies_0.6.1.agc \
  -a ../../ibs-cli/data/hprc465vschm13.aln.paf.gz \
  -r CHM13 \
  -region chr1:1-248956422 \
  -size 5000 \
  --subset-sequence-list ../../ibs-cli/sample_lists/HPRCv2_AFR_full.txt \
  --output ../experiments/validation/chr1_full/data/AFR_chr1_full.tsv \
  -j 8
```

Or use the provided run script:
```bash
./scripts/run_data_generation.sh
```

## Expected Output Size

| Population | Windows | Pairs | Records per Window | Total Records | Est. Size |
|------------|---------|-------|-------------------|---------------|-----------|
| EUR | 49,791 | 1,770 | ~1,770 | ~88M | ~8-10 GB |
| AFR | 49,791 | 8,911 | ~8,911 | ~444M | ~40-50 GB |

**Total estimated**: ~50-60 GB

## Analysis Pipeline

1. **Generate pairwise identity data**:
   ```bash
   ./scripts/run_data_generation.sh
   ```

2. **Estimate emission parameters**:
   ```bash
   python3 scripts/estimate_emissions.py
   ```

3. **Run IBD inference**:
   ```bash
   python3 scripts/run_analysis.py
   ```

4. **Generate figures**:
   ```bash
   python3 scripts/generate_figures.py
   ```

## Expected Results

### Population Differences

Based on exp02_chr2_50Mb_full results, we expect:

| Metric | EUR | AFR | EUR/AFR Ratio |
|--------|-----|-----|---------------|
| Pairs with IBD >= 2Mb | ~95% | ~28% | ~3.4x |
| Mean segment length | ~3.0 Mb | ~2.5 Mb | 1.2x |
| Total segments | ~5-6x more | baseline | — |

### Key Biological Questions

1. **Genome-wide IBD patterns**: Are the patterns observed in chr2:1-50Mb representative of the whole genome?
2. **Centromeric regions**: How does IBD vary near the chr1 centromere?
3. **Selection signatures**: Are there regions of unusually high/low IBD indicating selection?

## Directory Structure

```
chr1_full/
├── README.md
├── data/
│   ├── EUR_chr1_full.tsv    # ~10 GB
│   └── AFR_chr1_full.tsv    # ~50 GB
├── results/
│   ├── figures/
│   └── json/
│       ├── EUR_emission_params.json
│       ├── AFR_emission_params.json
│       ├── EUR_ibd_results.json
│       └── AFR_ibd_results.json
└── scripts/
    ├── run_data_generation.sh
    ├── estimate_emissions.py
    ├── run_analysis.py
    └── generate_figures.py
```

## Comparison with exp02_chr2_50Mb_full

| Aspect | exp02 (chr2:1-50Mb) | chr1_full |
|--------|---------------------|-----------|
| Region size | 50 Mb | 249 Mb (~5x) |
| Windows | 10,000 | 49,791 (~5x) |
| Est. runtime | ~2-4 hrs | ~10-20 hrs |
| Est. output | ~4 GB | ~60 GB |

## Notes

- This experiment will take significantly longer than exp02 due to the 5x larger region
- Ensure sufficient disk space (~100 GB free recommended)
- Consider running overnight or on a server with adequate resources
