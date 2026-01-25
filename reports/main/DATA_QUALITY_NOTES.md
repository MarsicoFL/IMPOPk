# HPRCv2-IBD Data Quality Assessment

## Critical Finding: AFR IBD Segment Data is Invalid

During quality assessment of the analysis results, we identified that **AFR IBD segment data is invalid** and should not be used for biological interpretation.

## Summary of Data Validity

| Data Component | Status | Evidence |
|----------------|--------|----------|
| EUR IBD segments | **VALID** | Identity 99.97-99.99%, distributed across chromosome |
| AFR IBD segments | **INVALID** | Identity 27-45% (should be ~99.99%), 95.5% in centromere |
| EUR emission params (v2) | **VALID** | d' = 5.37, excellent separation |
| AFR emission params (v2) | **VALID** | d' = 5.29, excellent separation |
| Selection scan IBS rates | **VALID** | Real measurements from pangenome data |

## Evidence of AFR IBD Segment Invalidity

### 1. Segment Identity Values

**EUR segments (VALID):**
- Mean identity: 0.9998 (99.98%)
- Range: 0.9997 - 0.9999
- This is expected for true IBD segments

**AFR segments (INVALID):**
- Mean identity: 0.35 (35%)
- Range: 0.27 - 0.45
- **This is NOT IBD** - true IBD should have ~99.99% identity

### 2. Spatial Distribution

**EUR segments:**
- In centromere: 30.4% (14 segments)
- Outside centromere: 69.6% (32 segments)
- Distributed across the chromosome

**AFR segments:**
- In centromere: **95.5%** (107/112 segments)
- Outside centromere: 4.5% (5 segments)
- **Concentrated in centromere - indicates artifact detection**

### 3. Model Separation (d')

The emission parameters used for segment inference were not quality-filtered:

**Original parameters (used for segment detection):**
- EUR d' = 0.53 (marginal but functional)
- AFR d' = **0.0009** (essentially zero - no separation)

**v2 corrected parameters:**
- EUR d' = 5.37 (excellent)
- AFR d' = 5.29 (excellent)

The AFR segments were detected with d' ~ 0, meaning the HMM had no ability to distinguish IBD from non-IBD states.

## Root Cause

The AFR emission parameters were not filtered for quality before segment inference:

```json
// AFR_ibd_results.json - Original (INVALID) parameters
{
  "ibd": {
    "mean": 1.0199,  // Impossible for identity (>1)
    "std": 140.207   // Absurd standard deviation
  },
  "d_prime": 0.0008954  // Essentially zero
}
```

```json
// AFR_summary_v2.json - Corrected (VALID) parameters
{
  "ibd": {
    "mean": 0.9999876,  // Correct
    "std": 0.0001       // Reasonable
  },
  "d_prime": 5.287      // Excellent
}
```

## Impact on Previous Analyses

The following analyses in the original report are **invalid**:

1. **EUR/AFR IBD comparison** (63x ratio) - Cannot be validated
2. **Centromere exclusion analysis** - Based on invalid AFR data
3. **AFR chromosome arm analysis** - Invalid AFR segments
4. **Any AFR-based IBD conclusions** - All invalid

## What Remains Valid

1. **EUR IBD segment analysis** - All EUR segments are valid
2. **v2 emission parameters** - Valid for both populations
3. **Selection scan IBS rates** - Real measurements, not dependent on IBD segments
4. **HMM methodology** - Correct, but needs rerunning with v2 parameters for AFR

## Required Actions

1. **Re-run AFR IBD inference** using v2 corrected emission parameters
2. **Update reports** to only use validated data (done in corrected version)
3. **Add centromere masking** to avoid structural artifacts
4. **Full pair analysis** for unbiased population estimates

## Files Affected

### Original (contains invalid AFR data):
- `experiments/chr1_full/results/json/AFR_ibd_results.json`
- `reports/HPRCv2_IBD_Analysis_Report.tex`
- `reports/generate_detailed_figures.py`
- `reports/figures_science/*.png`

### Corrected (uses only valid data):
- `reports/HPRCv2_IBD_Analysis_Report_Corrected.tex`
- `reports/generate_corrected_figures.py`
- `reports/figures_corrected/*.png`

## Verification Code

To verify the data quality issues:

```python
import json

# Load AFR results
with open('experiments/chr1_full/results/json/AFR_ibd_results.json') as f:
    afr = json.load(f)

# Check segment identities
for pair in afr['results'][:5]:
    for seg in pair['segments']:
        print(f"Identity: {seg['mean_identity']:.4f}")  # Will show ~0.30-0.45

# Check emission parameters
print(f"AFR d': {afr['emission_params']['d_prime']:.6f}")  # Will show ~0.0009

# Load EUR results for comparison
with open('experiments/chr1_full/results/json/EUR_ibd_results.json') as f:
    eur = json.load(f)

for pair in eur['results'][:5]:
    for seg in pair['segments']:
        print(f"Identity: {seg['mean_identity']:.4f}")  # Will show ~0.9998
```

## Conclusion

This data quality assessment demonstrates the importance of validating emission parameters before biological interpretation. The v2 corrected parameters show that the HMM methodology is sound - both populations achieve d' > 5 with proper quality filtering. However, the AFR IBD segment inference must be re-run with corrected parameters before any AFR/EUR comparisons can be made.

---
*Last updated: January 2026*
