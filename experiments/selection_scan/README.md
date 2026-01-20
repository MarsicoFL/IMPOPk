# Multi-Region Selection Scan

IBS analysis across known selection loci to detect signatures of positive selection.

## Regions

| Region | Chromosome | Coordinates | Size | Gene | Target Pop | Selection Type |
|--------|------------|-------------|------|------|------------|----------------|
| LCT | chr2 | 130,787,850-140,837,183 | 10 Mb | Lactase | EUR | Lactase persistence |
| SLC24A5 | chr15 | 48,000,000-50,000,000 | 2 Mb | Skin pigment | EUR | Light skin |
| EDAR | chr2 | 108,000,000-110,000,000 | 2 Mb | Hair morphology | EAS | Hair thickness |
| HBB | chr11 | 5,200,000-5,300,000 | 100 kb | Hemoglobin-β | AFR | Malaria resistance |
| DARC | chr1 | 159,000,000-160,000,000 | 1 Mb | Duffy antigen | AFR | Malaria resistance |

## Methodology

For each region:
1. Run IBS on target population (expected elevated IBS due to selection)
2. Run IBS on AFR control (baseline diversity)
3. Compare IBS rates between target and control

## Run

```bash
./scripts/run_selection_scan.sh           # Run all regions
./scripts/run_selection_scan.sh LCT       # Run single region
```

## Expected Results

- **LCT**: EUR >> AFR (strong positive selection ~10 kya)
- **SLC24A5**: EUR >> AFR (strong positive selection)
- **EDAR**: EAS >> AFR (strong positive selection)
- **HBB**: AFR elevated (balancing selection)
- **DARC**: AFR elevated (strong positive selection)
