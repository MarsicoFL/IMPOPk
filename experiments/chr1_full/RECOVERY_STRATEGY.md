# chr1_full Recovery Strategy

## Problem Encountered

Running `pairwise-identity.sh` with `-j 8` (8 parallel chunks) caused several chunk
processes to die mid-execution, likely due to memory pressure or I/O contention.
The system has 62 GB RAM but with 8 concurrent impg processes (each using ~1-6 GB),
plus awk filters and disk I/O, some processes were terminated.

## Decision: Incremental Chunk Processing

Instead of running all 8 chunks in parallel, we adopt an incremental strategy:
1. Complete chunks that are >80% done first (quick wins)
2. Use fewer parallel jobs (2-4) to reduce system pressure
3. Continue from exact positions where chunks left off
4. Merge completed chunks progressively

## Current State (2026-01-18)

### Chunk Boundaries (chr1:1-248956422, 5kb windows, 6224 windows/chunk)
```
chunk_000:         1 -  31,120,001
chunk_001: 31,120,001 -  62,240,001
chunk_002: 62,240,001 -  93,360,001
chunk_003: 93,360,001 - 124,480,001
chunk_004: 124,480,001 - 155,600,001
chunk_005: 155,600,001 - 186,720,001
chunk_006: 186,720,001 - 217,840,001
chunk_007: 217,840,001 - 248,960,001
```

### EUR Status (30 individuals, 60 haplotypes)
| Chunk | Max Position | End Position | Progress | Status |
|-------|-------------|--------------|----------|--------|
| 000 | 31,120,000 | 31,120,001 | **100%** | COMPLETE |
| 001 | 58,400,000 | 62,240,001 | 88% | needs continuation |
| 002 | 93,360,000 | 93,360,001 | **100%** | COMPLETE |
| 003 | 123,575,000 | 124,480,001 | 97% | needs continuation |
| 004 | 147,965,000 | 155,600,001 | 75% | needs continuation |
| 005 | 186,720,000 | 186,720,001 | **100%** | COMPLETE |
| 006 | 217,840,000 | 217,840,001 | **100%** | COMPLETE |
| 007 | - | 248,960,001 | 0% | needs full run |

**EUR Summary:** 4/8 chunks COMPLETE. Need to finish: 001 (12%), 003 (3%), 004 (25%), 007 (100%)

### AFR Status (67 individuals, 134 haplotypes)
| Chunk | Max Position | End Position | Progress | Status |
|-------|-------------|--------------|----------|--------|
| 000 | 18,140,000 | 31,120,001 | 58% | needs continuation |
| 001 | 60,425,000 | 62,240,001 | 94% | **>80% - priority** |
| 002 | 93,115,000 | 93,360,001 | 99% | **>80% - priority** |
| 003 | 109,640,000 | 124,480,001 | 52% | needs continuation |
| 004 | 142,850,000 | 155,600,001 | 59% | needs continuation |
| 005 | 183,385,000 | 186,720,001 | 89% | **>80% - priority** |
| 006 | 217,530,000 | 217,840,001 | 99% | **>80% - priority** |
| 007 | - | 248,960,001 | 0% | needs full run |

**AFR Summary:** 0/8 complete. Priority: 001, 002, 005, 006 (all >80%)

## Data Location

Partial outputs preserved in:
```
data/partial_runs/
├── EUR_run1/          # First EUR attempt
│   ├── chunk_000      # BED file
│   ├── out_000.tsv    # Output (complete)
│   ├── ...
└── AFR_run1/          # First AFR attempt
    ├── chunk_000
    ├── out_000.tsv
    └── ...
```

## Continuation Process

### Step 1: Complete >80% Chunks (Quick Wins)

For each incomplete chunk, create a continuation BED file starting from max_position+1:

```bash
# Example for AFR chunk_001 (94% done, needs 60,425,001 - 62,240,001)
# Generate BED for remaining windows:
./scripts/generate_continuation_bed.sh AFR 001 60425001 62240001
```

### Step 2: Run Continuation Jobs

```bash
# Run single chunk at a time (safest) or max 2-4 parallel
./scripts/continue_chunk.sh AFR 001
```

### Step 3: Merge Outputs

After all chunks complete:
```bash
# For each population
cat data/partial_runs/EUR_run1/out_*.tsv | sort -k1,1 -k2,2n > data/EUR_chr1_full.tsv
```

## Scripts

See `scripts/` directory:
- `continue_chunk.sh` - Continue a single chunk from a given position
- `merge_completed.sh` - Merge all completed chunk outputs
- `check_progress.sh` - Check current state of all chunks

## Recommended Execution Order

### Phase 1: AFR >80% chunks (fastest to complete)
```bash
# Run 2 at a time max
./scripts/continue_chunk.sh AFR 002  # 99% done, ~245kb remaining
./scripts/continue_chunk.sh AFR 006  # 99% done, ~310kb remaining
# Wait, then:
./scripts/continue_chunk.sh AFR 001  # 94% done, ~1.8Mb remaining
./scripts/continue_chunk.sh AFR 005  # 89% done, ~3.3Mb remaining
```

### Phase 2: EUR near-complete chunks
```bash
./scripts/continue_chunk.sh EUR 003  # 97% done, ~905kb remaining
./scripts/continue_chunk.sh EUR 001  # 88% done, ~3.8Mb remaining
```

### Phase 3: Remaining chunks
```bash
# One at a time to be safe
./scripts/continue_chunk.sh EUR 004  # 75%
./scripts/continue_chunk.sh AFR 000  # 58%
./scripts/continue_chunk.sh AFR 003  # 52%
./scripts/continue_chunk.sh AFR 004  # 59%
./scripts/continue_chunk.sh EUR 007  # 0%
./scripts/continue_chunk.sh AFR 007  # 0%
```

## Notes

- AFR has more haplotypes (134 vs 60), so produces ~5x more output per window
- Expected output sizes: EUR ~10-12 GB total, AFR ~50-60 GB total
- With 2-4 parallel jobs, memory should stay under 20 GB
- Monitor with: `watch -n 30 'ps aux | grep impg | grep -v grep'`

## Important Fix (2026-01-18)

### Chromosome Length Issue

The original chunk boundaries used chr1:1-248956422 (the commonly cited length), but the
actual CHM13 chr1 sequence length is 248,387,328 bp. Chunk 007 boundaries were updated:

**Before:** 217,840,001 - 248,960,001 (would cause impg error)
**After:** 217,840,001 - 248,387,328 (correct)

This affects chunk_007 only. The continue_chunk.sh script has been updated with the
correct CHUNK_END[007] value.
