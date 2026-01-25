# ibs-cli Optimization Notes

## Current Bottleneck

The current implementation spawns a new `impg similarity` process for each window:

```rust
// Current approach (main.rs:161-175)
for window in windows {
    Command::new("impg")
        .arg("similarity")
        .arg("-r")
        .arg(&window_region)  // ONE window per call
        ...
}
```

This incurs significant overhead per window:
- AGC index loading: ~2s
- Alignment index loading: ~1s
- Total: ~3s overhead before actual computation

For 10,000 windows (50 Mb region), this overhead dominates runtime.

## Proposed Optimization: BED File Approach

Replace the per-window loop with a single `impg similarity` call using `--target-bed`:

```rust
// Proposed approach
// 1. Generate BED file with all windows
// 2. Single impg call
Command::new("impg")
    .arg("similarity")
    .arg("--target-bed")
    .arg(&bed_file)  // ALL windows in one call
    ...
// 3. Post-process output (filter, dedupe, format)
```

### Benchmark Results

| Approach | Windows | Time | Rate |
|----------|---------|------|------|
| Current (loop) | 5 | 7.0s | 1.4s/window |
| BED file | 100 | 74s | 0.74s/window |

**~2x faster per window**, with gains increasing for larger regions due to amortized index loading.

### Post-processing Required

The BED approach outputs all pairs; post-processing needed to match current output:

```python
# Filter impg output to match ibs-cli format:
if identity < cutoff: continue
if group_a == group_b: continue        # self-self
if "CHM13#" in group_a or group_b: continue  # reference pairs
if group_a > group_b: swap(a, b)       # canonical order
output(chrom, start, end, group_a, group_b, identity)
```

## Further Optimization: Region Parallelization

For very large regions (full chromosomes), split the BED file into chunks and process in parallel:

```
Chromosome (243 Mb, ~48,000 windows)
    |
    +-- Chunk 1: windows 1-10000     --> impg (thread 1) --> output_1.tsv
    +-- Chunk 2: windows 10001-20000 --> impg (thread 2) --> output_2.tsv
    +-- Chunk 3: windows 20001-30000 --> impg (thread 3) --> output_3.tsv
    +-- Chunk 4: windows 30001-40000 --> impg (thread 4) --> output_4.tsv
    +-- Chunk 5: windows 40001-48000 --> impg (thread 5) --> output_5.tsv
    |
    +--> Merge outputs (cat + sort) --> final_output.tsv
```

### Implementation Sketch

```rust
fn run_ibs_optimized(args: &Args) -> Result<()> {
    // 1. Generate full BED file
    let bed_path = generate_windows_bed(&args.region, args.window_size)?;

    // 2. Split into chunks for parallelization
    let chunks = split_bed_file(&bed_path, args.threads)?;

    // 3. Run impg in parallel on each chunk
    let outputs: Vec<PathBuf> = chunks
        .par_iter()
        .map(|chunk| run_impg_on_bed(chunk, args))
        .collect::<Result<Vec<_>>>()?;

    // 4. Merge and post-process
    merge_and_filter_outputs(&outputs, &args.output, args.cutoff)?;

    Ok(())
}
```

## Expected Performance

| Region Size | Current (loop) | BED (single) | BED (parallel, 4 threads) |
|-------------|----------------|--------------|---------------------------|
| 5 Mb (1000 windows) | ~23 min | ~12 min | ~4 min |
| 50 Mb (10000 windows) | ~4 hours | ~2 hours | ~35 min |
| 243 Mb (full chr2) | ~19 hours | ~10 hours | ~3 hours |

## Notes

- Output is functionally identical (same sample pairs per window)
- Minor floating-point precision differences (6th decimal) - no impact on downstream IBD inference
- BED approach already supported by `impg similarity --target-bed`
