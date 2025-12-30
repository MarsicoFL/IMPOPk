# IBS CLI and Production Scripts

This directory encapsulates the "production" side of the project: the Rust CLI
(`ibs`) plus the operational shell scripts that wrap `impg similarity` for large
runs and Jacquard coefficient summaries.

## Layout
- `src/` and `Cargo.toml`: the Rust crate for the streaming IBS window caller.
  Build and run with `cargo run -- --help` from this directory.
- `scripts/`: shell wrappers kept alongside the crate to guarantee they evolve
  together. They assume they are executed from within this directory.
  - `ibs.sh`: bash port of the Rust CLI, left here while we transition workloads.
  - `ibd.sh`: extends IBS windows with an HMM-based IBD caller.
  - `run_full.sh`: convenience launcher that tiles a chromosome and launches
    multiple `ibs.sh` workers in parallel. Paths are resolved relative to the
    repository root and can be overridden via environment variables (`AGC`,
    `PAF`, `SUB`, `REF`, `CHR`, `START`, `END`, `SIZE`, `JOBS`).
  - `jacquard_coeffs.sh`: computes Jacquard delta coefficients from the IBS
    windows.
- `sample_lists/`: curated lists of haplotypes or subsets used when invoking the
  CLI or scripts.
- `examples/`: tiny example IBS/IBD outputs that double as fixtures when testing
  parsing logic.

By treating this production stack as an isolated package we can add CI/tests in
one place without disturbing exploratory notebooks.
