# IBD Network Analysis Sandbox

This area hosts all exploratory work used when producing HPRC network reports.
It is intentionally separate from the production `ibs` tooling so we can keep
notebooks, scripts, and derived artifacts isolated from the CLI.

## Layout
- `notebooks/` interactive development notebooks. Currently contains
  `communitiesv3_hprc.ipynb`.
- `scripts/` lightweight helpers:
  - `run_pairwise_impg.sh` wraps `impg similarity` over a BED of windows and
    writes the per-window identities expected by downstream analysis.
  - `ibd.py` prototypes several IBD calling strategies using the pairwise table
    produced above.
  - `generate_toys.py` emits toy IBS tables for rapid experimentation.
- `inputs/` static assets needed by the notebooks (for instance `vertex_map.csv`).
- `outputs/` cached CSV exports used in the HPRC report.

To keep the repository lean, heavy dependencies (impg builds, AGC/PAF files)
are fetched outside of version control. The scripts assume those live under the
repository `data/` directory; override with environment variables or CLI flags
when you call them.
