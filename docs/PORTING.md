# Porting Shell/Python Prototypes into `ibs-cli`

Experimental ideas typically begin as short Bash or Python scripts. The
`ibs-cli` package includes a repeatable framework that (1) records the contract
of a script, (2) scaffolds a Rust port, and (3) continuously tests that both
implementations behave the same.

## Workflow overview
1. **Prototype** in `production/ibs-cli/scripts/` (Bash) or `analysis/.../scripts/`
   (Python). Keep the CLI stable; the Rust command will mirror it.
2. **Document the contract** by adding a spec under
   `production/ibs-cli/tests/parity/<name>.toml`. Each spec declares the script
   path, Rust binary name, working directory, optional environment variables,
   and one or more test cases (arguments + fixtures).
3. **Port to Rust** by creating `production/ibs-cli/src/bin/<name>.rs`. Reuse the
   existing dependencies (`anyhow`, `clap`) and keep the CLI identical.
4. **Add fixtures** under `production/ibs-cli/tests/data/` so the spec does not
   depend on external datasets (impg/AGC/PAF). Fixtures should be minimal yet
   expressive enough to hit the interesting branches of the algorithm.
5. **Run the parity suite**: `cargo test --test parity`. The test harness reads
   every TOML spec, runs the legacy script and the Rust binary, and asserts that
   their stdout/stderr/exit codes match. Specs double as machine-readable
   documentation and allow us to extend coverage gradually.

This workflow allows fast iteration in shell/Python while the Rust binary is
under development. Once the parity tests pass, the legacy script can be removed
(or kept for familiarity) knowing that CI guards against regressions.

## Spec format
```toml
name = "jacquard"
workdir = "."               # relative to `production/ibs-cli`
script = "./scripts/jacquard_coeffs.sh"
rust_bin = "jacquard"        # Cargo binary name

[[tests]]
name = "toy"
args = [
  "--ibs", "tests/data/jacquard_toy.tsv",
  "--hap-a1", "HGA#1",
  "--hap-a2", "HGA#2",
  "--hap-b1", "HGB#1",
  "--hap-b2", "HGB#2",
]
```
Optional `env` tables can be attached to each test if the CLI requires
additional configuration. The harness normalizes stdout/stderr before comparing
so whitespace and ordering mismatches surface immediately during `cargo test`.

## Example: Jacquard coefficients
- Legacy script: `production/ibs-cli/scripts/jacquard_coeffs.sh`
- Rust port: `cargo run --bin jacquard -- --help`
- Spec: `production/ibs-cli/tests/parity/jacquard.toml`
- Fixture: `production/ibs-cli/tests/data/jacquard_toy.tsv`

`cargo test --test parity` runs both implementations and guarantees that the
Delta1..9 table, warnings, and summary metadata always match.
