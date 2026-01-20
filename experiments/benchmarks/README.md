# IBS Scalability Benchmarks

Benchmarking suite for IBS runtime and output scaling analysis.

## Haplotype Scaling

Tests how IBS runtime scales with increasing sample count.

| Haplotypes | Pairs (C(n,2)) | Window Size |
|------------|----------------|-------------|
| 2 | 1 | 5 kb |
| 10 | 45 | 5 kb |
| 50 | 1,225 | 5 kb |
| 100 | 4,950 | 5 kb |
| 150 | 11,175 | 5 kb |
| 200 | 19,900 | 5 kb |

**Expected**: Output scales O(n²) with haplotypes; runtime is dominated by impg indexing overhead.

### Run

```bash
./haplotype_scaling/scripts/run_haplotype_benchmark.sh
```

## Window Scaling

Tests how IBS runtime scales with window resolution.

| Window Size | Windows (10 Mb) | Haplotypes |
|-------------|-----------------|------------|
| 2 kb | 5,024 | 8 |
| 5 kb | 2,009 | 8 |
| 7 kb | 1,435 | 8 |
| 10 kb | 1,005 | 8 |

**Expected**: Runtime scales linearly with number of windows.

### Run

```bash
./window_scaling/scripts/run_window_benchmark.sh
```

## Analysis

```bash
python3 analysis/benchmark_analysis.py
```

Generates:
- `analysis/output/haplotype_scaling.png` - Runtime and output vs haplotypes/pairs
- `analysis/output/window_scaling.png` - Runtime vs window count
