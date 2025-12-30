# Identity-By-State and Identity-By-Descent Analysis in HPRCv2

We use public HPRC assemblies plus impg-based similarity to explore IBS/IBD
relationships between haplotypes. The repository now separates reporting assets
from production tooling so each can evolve independently.

## Repository layout
- `production/ibs-cli/` – Rust CLI (`cargo run -- --help`) together with the
  operational bash wrappers for large impg jobs and Jacquard summaries. See
  `production/ibs-cli/README.md`.
- `analysis/ibd-network/` – exploratory notebooks and lightweight scripts used
  while drafting HPRC reports. Documentation lives in `analysis/ibd-network/README.md`.
- `docs/reports/` – published artifacts that were delivered to the HPRC
  consortium (PDFs, slide decks, etc.).
- `data/` – small metadata tracked in git. Large AGC/PAF inputs are expected to
  live in sibling folders under `data/` but are not committed; point the scripts
  at your local copies via CLI flags or environment variables.

## Production IBS/IBD tooling
1. `cd production/ibs-cli`.
2. Build the Rust CLI: `cargo build --release`.
3. Run the streaming IBS caller via `cargo run --bin ibs -- ...` or the bash
   wrapper `scripts/ibs.sh`.
4. Use `scripts/run_full.sh` when you want to tile a chromosome into windows and
   dispatch multiple workers via GNU Parallel. Override defaults with env vars
   (e.g. `AGC=/path/to.agc CHR=chr7 scripts/run_full.sh`).
5. Feed the resulting IBS windows to either the Rust Jacquard port (`cargo run
   --bin jacquard -- ...`) or the legacy bash script
   `scripts/jacquard_coeffs.sh` for Delta summaries, and continue with
   `scripts/ibd.sh` for HMM-based IBD calling when needed.
6. For new utilities start in Bash/Python, describe the CLI contract in
   `tests/parity/*.toml`, and rely on `cargo test --test parity` to guarantee
   that the eventual Rust binary matches the prototype. See `docs/PORTING.md`.

## Analysis and reporting assets
- Notebooks plus helper scripts live under `analysis/ibd-network`. They consume
  the per-window IBS tables generated above and were the source for the deliverables stored in `docs/reports/` (e.g. `HPRCv2_IBD.pdf`).
- Keep heavyweight raw data out of git; drop them under `data/` or supply
  explicit paths when executing the scripts.

This split should make it easier to add tests/CI for the production pipeline
while keeping exploratory work organized for future HPRC reports.
