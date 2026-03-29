# Tutorial: Jacquard Identity State Coefficients

## Overview

The Jacquard coefficient calculator computes the nine possible identity states between two diploid individuals based on IBS (Identity-By-State) observations from pangenome haplotypes. This provides a comprehensive view of haplotype sharing patterns between individuals.

This tutorial covers:
- **Rust binary**: `cargo run --bin jacquard` (recommended)
- **Shell script**: `scripts/jacquard_coeffs.sh`

---

## Biological Background

### The Nine Jacquard Identity States

When comparing two diploid individuals A (with haplotypes a1, a2) and B (with haplotypes b1, b2), there are exactly nine possible configurations of identity among the four haplotypes. These are called **Jacquard's condensed identity coefficients** or **Delta states**.

**Important distinction**: In this pipeline, we observe IBS (sequence similarity) rather than true IBD (shared ancestry). The Delta values represent empirical frequencies of observed identity patterns, which approximate the theoretical Jacquard coefficients when IBS reflects true IBD.

### Visual Representation of the Nine States

```
Delta1: All four identical          Delta2: Within-individual pairs only
  [a1=a2=b1=b2]                       [a1=a2] [b1=b2]
     ||||                                ||     ||

Delta3: Three-way + singleton       Delta4: Individual A homozygous
  [a1=a2=b1] [b2]                      [a1=a2] [b1] [b2]
     |||      |                           ||    |    |

Delta5: Three-way + singleton       Delta6: Individual B homozygous
  [a1=b1=b2] [a2]                      [b1=b2] [a1] [a2]
     |||      |                           ||    |    |

Delta7: Cross-matching pairs        Delta8: Single cross-match
  [a1=b1] [a2=b2]                      [a1=b1] [a2] [b2]
     ||     ||                            ||    |    |

Delta9: All four different
  [a1] [a2] [b1] [b2]
    |    |    |    |
```

---

## Biological Interpretation of Each State

### Delta1: Complete Identity (All Four Haplotypes Identical)

```
Configuration: a1 = a2 = b1 = b2
```

**Biological meaning**: All four haplotypes are identical by state at this locus.

**When this occurs**:
- Highly conserved genomic regions with very low diversity
- Regions under strong purifying selection
- Recent common ancestry in all four lineages
- Fixed haplotypes in the population

**Expected frequency**: Rare in diverse populations; more common in bottlenecked or inbred populations.

---

### Delta2: Both Individuals Homozygous, Different from Each Other

```
Configuration: a1 = a2, b1 = b2, but {a1,a2} != {b1,b2}
```

**Biological meaning**: Both individuals are homozygous at this locus, but for different haplotypes.

**When this occurs**:
- Both individuals are autozygous (have runs of homozygosity)
- Different ancestral haplotypes became fixed in each individual's ancestry
- Common in regions of low recombination within consanguineous individuals

**Population genetics context**: High Delta2 between individuals from different isolated populations can indicate population structure.

---

### Delta3: Three Haplotypes Identical, One Different (Individual A's Context)

```
Configuration: a1 = a2 = b1, b2 different
```

**Biological meaning**: Individual A is homozygous, and one of B's haplotypes matches A's.

**When this occurs**:
- A is autozygous and shares ancestry with one of B's parental lineages
- Half-sibling relationship where shared parent's haplotype became homozygous in A
- One haplotype is common in the population

---

### Delta4: Individual A Homozygous, No Cross-Individual IBS

```
Configuration: a1 = a2, b1 and b2 all different from each other and from A
```

**Biological meaning**: Individual A is homozygous, but shares no haplotypes with B.

**When this occurs**:
- A has a run of homozygosity from inbreeding
- A and B are from different genetic backgrounds
- Common when comparing individuals from different populations

---

### Delta5: Three Haplotypes Identical, One Different (Individual B's Context)

```
Configuration: a1 = b1 = b2, a2 different
```

**Biological meaning**: Individual B is homozygous, and one of A's haplotypes matches B's.

**When this occurs**: Mirror image of Delta3 - B is autozygous and shares ancestry with one of A's parental lineages.

---

### Delta6: Individual B Homozygous, No Cross-Individual IBS

```
Configuration: b1 = b2, a1 and a2 all different from each other and from B
```

**Biological meaning**: Individual B is homozygous, but shares no haplotypes with A.

**When this occurs**: Mirror image of Delta4 - B has runs of homozygosity but no shared ancestry with A.

---

### Delta7: Cross-Matching Pairs (Reciprocal Haplotype Sharing)

```
Configuration: a1 = b1, a2 = b2 (or a1 = b2, a2 = b1)
```

**Biological meaning**: Each haplotype in A matches a haplotype in B, forming two distinct identity groups.

**When this occurs**:
- **Full siblings**: Share both parental haplotypes in IBD
- **Double first cousins**: Share ancestors on both sides
- **Same ancestral population**: Two common haplotypes segregating

**Key relationship indicator**: High Delta7 is a strong signal of close biological relationship (siblings, double cousins).

---

### Delta8: Single Haplotype Shared Between Individuals

```
Configuration: a1 = b1 (or any single cross-match), all others different
```

**Biological meaning**: Exactly one haplotype from A matches exactly one from B.

**When this occurs**:
- **Half-siblings**: Share one parent
- **Avuncular relationships**: Uncle/aunt with nephew/niece
- **Cousins**: Share grandparents
- **Common haplotype**: Population-level sharing

**Most common IBD state**: In outbred populations, this is often the most frequent state indicating biological relatedness.

---

### Delta9: No Identity Among Four Haplotypes

```
Configuration: a1, a2, b1, b2 all different
```

**Biological meaning**: No sequence identity detected among any pair of haplotypes.

**When this occurs**:
- Unrelated individuals from diverse populations
- High-diversity genomic regions
- Regions with high mutation rate or recombination
- Missing data (windows without IBS calls are assigned to Delta9)

**Note**: High Delta9 does not necessarily mean "unrelated" - it may simply indicate a region with insufficient identity signal.

---

## CLI Reference

### Rust Binary Arguments

```
jacquard - Compute Jacquard delta coefficients from IBS windows

USAGE:
    jacquard --ibs <FILE> --hap-a1 <ID> --hap-a2 <ID> --hap-b1 <ID> --hap-b2 <ID>

OPTIONS:
    --ibs <FILE>     IBS windows file (TSV with chrom/start/end/group.a/group.b)
    --hap-a1 <ID>    First haplotype of individual A
    --hap-a2 <ID>    Second haplotype of individual A
    --hap-b1 <ID>    First haplotype of individual B
    --hap-b2 <ID>    Second haplotype of individual B
    -h, --help       Print help information
```

---

## Usage Examples

### Example 1: Basic Jacquard Calculation

```bash
cd /path/to/HPRCv2-IBD

./target/release/jacquard \
  --ibs /results/ibs_windows.tsv \
  --hap-a1 "HG01167#1" \
  --hap-a2 "HG01167#2" \
  --hap-b1 "NA19682#1" \
  --hap-b2 "NA19682#2"
```

### Example 2: Using Test Fixture

```bash
./target/release/jacquard \
  --ibs tests/data/jacquard_toy.tsv \
  --hap-a1 "HGA#1" \
  --hap-a2 "HGA#2" \
  --hap-b1 "HGB#1" \
  --hap-b2 "HGB#2"
```

### Example 3: Shell Script Version

```bash
cd bin/jacquard

./jacquard_coeffs.sh \
  --ibs /results/ibs_for_ibd.out \
  --hap-a1 HG01167#1 --hap-a2 HG01167#2 \
  --hap-b1 NA19682#1 --hap-b2 NA19682#2
```

---

## Output Format

### Standard Output (stdout)

Nine lines showing each Delta state's frequency:

```
Delta1	0.00000000	(count=0)
Delta2	0.00500000	(count=10)
Delta3	0.00000000	(count=0)
Delta4	0.02000000	(count=40)
Delta5	0.00000000	(count=0)
Delta6	0.01500000	(count=30)
Delta7	0.05000000	(count=100)
Delta8	0.31000000	(count=620)
Delta9	0.60000000	(count=1200)
```

**Columns**:
1. State name (Delta1-Delta9)
2. Fraction of total windows
3. Raw count of windows in this state

### Diagnostic Output (stderr)

```
# chrom	chr20	min_start	1	max_end	9999999	win_size	5000
# total_windows	2000	loci_with_IBS_fourhaps	800	missing_windows_as_Delta9	1200	unclassified	0
```

**Fields explained**:
- `total_windows`: Expected number of windows based on region size
- `loci_with_IBS_fourhaps`: Windows where at least one IBS pair was observed among the four haplotypes
- `missing_windows_as_Delta9`: Windows with no IBS data (counted as Delta9)
- `unclassified`: Windows that could not be classified (should be 0)

---

## Interpreting Results

### Example Output Analysis

```
Delta1	0.00100000	(count=2)     # Very rare - complete identity
Delta2	0.00050000	(count=1)     # Rare - both homozygous, different
Delta3	0.00000000	(count=0)     # Not observed
Delta4	0.01000000	(count=20)    # A homozygous, no sharing
Delta5	0.00000000	(count=0)     # Not observed
Delta6	0.00500000	(count=10)    # B homozygous, no sharing
Delta7	0.05000000	(count=100)   # Cross-matching pairs - siblings?
Delta8	0.25000000	(count=500)   # Single haplotype shared - related
Delta9	0.68350000	(count=1367)  # No identity - unrelated/diverse
```

### Relationship Inference

| Relationship | Expected Pattern |
|--------------|------------------|
| **Monozygotic twins** | Delta1 >> all others |
| **Full siblings** | High Delta7 + Delta8, low Delta9 |
| **Half-siblings** | Moderate Delta8, low Delta7, high Delta9 |
| **Parent-child** | High Delta8 (exactly 50%), rest Delta9 |
| **First cousins** | Low Delta8, very high Delta9 |
| **Unrelated** | Almost all Delta9 |

### Population Structure Indicators

| Pattern | Interpretation |
|---------|----------------|
| High Delta2 | Both from inbred/isolated populations, different ancestry |
| High Delta4/Delta6 | One individual from inbred background |
| Elevated Delta7 | Close relationship OR shared population with few haplotypes |
| Very high Delta8 | Some recent shared ancestry |

---

## Visualization Suggestions

### Bar Plot of Delta Frequencies

```python
import matplotlib.pyplot as plt
import pandas as pd

# Parse output
data = {
    'Delta1': 0.001, 'Delta2': 0.0005, 'Delta3': 0.0,
    'Delta4': 0.01, 'Delta5': 0.0, 'Delta6': 0.005,
    'Delta7': 0.05, 'Delta8': 0.25, 'Delta9': 0.6835
}

plt.figure(figsize=(10, 6))
plt.bar(data.keys(), data.values())
plt.xlabel('Identity State')
plt.ylabel('Frequency')
plt.title('Jacquard Identity State Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('jacquard_distribution.png')
```

### Heatmap for Multiple Pairs

```python
import seaborn as sns
import numpy as np

# Example: comparing multiple sample pairs
pairs = ['A-B', 'A-C', 'B-C', 'A-D']
states = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']

# Data matrix (rows=pairs, cols=states)
data = np.array([
    [0.001, 0.001, 0, 0.01, 0, 0.01, 0.05, 0.25, 0.678],
    [0, 0, 0, 0.02, 0, 0.01, 0.01, 0.10, 0.86],
    [0, 0, 0, 0.015, 0, 0.015, 0.02, 0.15, 0.80],
    [0, 0, 0, 0.01, 0, 0.02, 0.001, 0.05, 0.919],
])

plt.figure(figsize=(12, 6))
sns.heatmap(data, xticklabels=states, yticklabels=pairs,
            cmap='YlOrRd', annot=True, fmt='.3f')
plt.title('Identity State Comparison Across Sample Pairs')
plt.savefig('jacquard_heatmap.png')
```

### Pie Chart for Single Comparison

```python
# Focus on non-Delta9 states for clearer visualization
labels = ['D1', 'D2', 'D4', 'D6', 'D7', 'D8']
sizes = [0.001, 0.0005, 0.01, 0.005, 0.05, 0.25]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Identity States (excluding Delta9)')
plt.savefig('jacquard_pie.png')
```

---

## Troubleshooting

### All Zeros Except Delta9

**Cause**: No IBS pairs found among the four specified haplotypes.

**Solutions**:
1. Verify haplotype names match exactly (case-sensitive)
2. Check that haplotypes exist in the IBS file
3. Ensure the IBS file was generated with a suitable identity threshold

```bash
# Debug: check what haplotypes are in the IBS file
cut -f4,5 ibs_windows.tsv | sort -u | head -20
```

### Fractions Do Not Sum to 1

**Cause**: Usually rounding display; computationally they should sum to 1.

**Note**: This is normal due to floating-point formatting.

### Unclassified Windows

**Cause**: Unusual identity patterns that do not fit the nine states.

**Solution**: This should be rare; if common, check for data quality issues.

---

## Mathematical Details

### Union-Find Classification

The tool uses a union-find algorithm to cluster the four haplotypes based on observed IBS pairs:

1. Initialize each haplotype as its own cluster
2. For each IBS pair (a, b), merge their clusters
3. Count resulting clusters and their composition (how many from A, how many from B)
4. Map cluster configuration to Delta state

### State Classification Logic

| Clusters | Sizes | Composition | State |
|----------|-------|-------------|-------|
| 1 | [4] | - | Delta1 |
| 2 | [2,2] | Each has 2A or 2B | Delta2 |
| 2 | [2,2] | Each has 1A+1B | Delta7 |
| 2 | [3,1] | Triplet has 2A+1B | Delta3 |
| 2 | [3,1] | Triplet has 1A+2B | Delta5 |
| 3 | [2,1,1] | Pair is 2A | Delta4 |
| 3 | [2,1,1] | Pair is 2B | Delta6 |
| 3 | [2,1,1] | Pair is 1A+1B | Delta8 |
| 4 | [1,1,1,1] | - | Delta9 |

---

## See Also

- [IBS Tutorial](ibs.md) - Generating the input IBS windows
- [IBD Tutorial](ibd.md) - HMM-based IBD segment detection
- [Conceptual Framework](../paper_concepts/conceptual_framework.md) - IBS vs IBD distinction
