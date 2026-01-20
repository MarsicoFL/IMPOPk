# Pending Benchmark Experiments

**Status**: NOT COMPLETED
**Date**: 2026-01-15
**Reason**: Killed to free CPU resources for selection scan experiments

---

## Overview

Two benchmark suites were prepared but not fully executed due to resource constraints (all 24 CPUs at 100% load running 3 experiment batches simultaneously).

---

## 1. Haplotype Scaling Benchmark

**Objective**: Measure how IBS runtime and output scale with the number of haplotypes (O(n²) pair complexity).

**Location**: `experiments/benchmarks/haplotype_scaling/`

**Configuration**:
- Region: LCT (chr2:130787850-140837183, ~10 Mb)
- Window size: 5000 bp (fixed)
- Haplotype counts: 2, 10, 50, 100, 150, 200

**Completed runs** (before kill):
| Haplotypes | Pairs | Runtime | Records |
|------------|-------|---------|---------|
| 2 | 1 | 1263s | 191 |
| 10 | 45 | 1803s | 11,032 |

**Pending runs**: 50, 100, 150, 200 haplotypes

**Sample files** (already created):
```
experiments/benchmarks/haplotype_scaling/data/
├── random_002hap.txt  (1 individual)
├── random_010hap.txt  (5 individuals)
├── random_050hap.txt  (25 individuals)
├── random_100hap.txt  (50 individuals)
├── random_150hap.txt  (75 individuals)
└── random_200hap.txt  (100 individuals)
```

---

## 2. Window Scaling Benchmark

**Objective**: Measure resolution vs runtime tradeoff with different window sizes.

**Location**: `experiments/benchmarks/window_scaling/`

**Configuration**:
- Region: LCT (chr2:130787850-140837183, ~10 Mb)
- Haplotypes: 8 (fixed, 4 individuals)
- Window sizes: 2000, 5000, 7000, 10000 bp

**Completed runs** (before kill):
| Window Size | Windows | Runtime | Records |
|-------------|---------|---------|---------|
| 2000 bp | 5024 | 3485s | 40,680 |

**Pending runs**: 5000, 7000, 10000 bp windows

**Sample file** (already created):
```
experiments/benchmarks/window_scaling/data/fixed_008hap.txt
```

---

## How to Run These Benchmarks

### Prerequisites

1. Ensure `impg` is installed and in PATH
2. Ensure `GNU parallel` is installed
3. Ensure data files exist:
   - `data/HPRC_r2_assemblies_0.6.1.agc`
   - `data/hprc465vschm13.aln.paf.gz`

### Commands

**Option A: Run both benchmarks sequentially (recommended)**
```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli

# Haplotype scaling (~4-6 hours)
./experiments/benchmarks/haplotype_scaling/scripts/run_haplotype_benchmark.sh

# Window scaling (~2-3 hours)
./experiments/benchmarks/window_scaling/scripts/run_window_benchmark.sh
```

**Option B: Run in background**
```bash
cd /home/franco/Escritorio/trabajadores/HPRCv2-IBD/production/ibs-cli

nohup ./experiments/benchmarks/haplotype_scaling/scripts/run_haplotype_benchmark.sh > hap_bench.log 2>&1 &
nohup ./experiments/benchmarks/window_scaling/scripts/run_window_benchmark.sh > win_bench.log 2>&1 &
```

### Expected Output

Results will be written to:
- `experiments/benchmarks/haplotype_scaling/results/benchmark_metrics.tsv`
- `experiments/benchmarks/window_scaling/results/benchmark_metrics.tsv`

Individual IBS files:
- `experiments/benchmarks/haplotype_scaling/results/bench_hap{N}_ibs.tsv`
- `experiments/benchmarks/window_scaling/results/bench_win{N}_ibs.tsv`

---

## Running Benchmarks

To complete the benchmarks:

```bash
# Run haplotype scaling benchmark
./experiments/benchmarks/haplotype_scaling/scripts/run_haplotype_benchmark.sh

# Run window scaling benchmark
./experiments/benchmarks/window_scaling/scripts/run_window_benchmark.sh
showing how runtime and output scale with haplotypes and window size.
```

---

## Analysis Script

An analysis script exists at:
```
experiments/benchmarks/analysis/benchmark_analysis.py
```

This script will:
1. Load metrics from both benchmark results
2. Plot runtime scaling curves
3. Plot output size scaling
4. Calculate theoretical vs observed O(n²) scaling

Run after benchmarks complete:
```bash
python3 experiments/benchmarks/analysis/benchmark_analysis.py
```

---

## Technical Details

### Parallelization

Both scripts use `ibs_parallel.sh` which:
- Processes windows in parallel using GNU parallel
- Uses `nproc - 3` cores (leaves 3 for system)
- Achieves ~20x speedup vs sequential processing

### Expected Scaling

- **Haplotype scaling**: O(n²) for pairs, roughly O(n²) for output records
- **Window scaling**: O(w) for runtime where w = windows, output roughly constant per window

### Resource Requirements

- CPU: Will use all available cores minus 3
- RAM: ~1-2 GB per parallel impg process
- Disk: ~50-200 MB per experiment depending on parameters

---

## Partial Results Location

The partial results from the killed run are in:
- `experiments/benchmarks/haplotype_scaling/results/bench_hap2_ibs.tsv`
- `experiments/benchmarks/haplotype_scaling/results/bench_hap10_ibs.tsv`
- `experiments/benchmarks/window_scaling/results/bench_win2000_ibs.tsv`

These can be kept or deleted before re-running.
