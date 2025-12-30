# IBS CLI and Production Scripts

`production/ibs-cli/` hosts the production-ready tooling used to generate IBS
windows and downstream summaries. The Rust binaries and shell wrappers share the
same CLI surface so researchers can prototype in Bash/Python and transition into
Rust when necessary.

- `src/` and `Cargo.toml`: the Rust crate hosts multiple binaries. Today we ship
  the original streaming IBS caller (`cargo run --bin ibs -- --help`) and a
  Jacquard delta calculator (`cargo run --bin jacquard -- --help`).
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

## Porting framework
The repository contains a spec-driven parity harness to make it trivial to port
Bash/Python scripts into Rust while guaranteeing behaviour. Specs live under
`tests/parity/*.toml`, fixtures under `tests/data/`, and the parity test itself
is implemented in `tests/parity.rs`. See `docs/PORTING.md` for the full
workflow. Once you add a spec you can run `cargo test --test parity` to ensure
the legacy script and the Rust binary produce identical stdout/stderr.

Treating this production stack as an isolated package keeps CI/tests focused in
one location while leaving exploratory notebooks untouched.
