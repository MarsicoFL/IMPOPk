#!/usr/bin/env python3
"""Generate toy IBS tables to sanity-check downstream tooling."""

import argparse
import csv
from typing import Iterable


HEADER = [
    "REGION",
    "CHR",
    "START",
    "END",
    "LENGTH",
    "group.a",
    "group.b",
    "estimated.identity",
]


def write_row(writer: csv.writer, chrom: str, start: int, end: int, hap_a: str, hap_b: str, ident: float) -> None:
    """Emit a single IBS-positive interval to ``writer``."""

    writer.writerow([
        f"CHM13#0#{chrom}:{start}-{end}",
        chrom,
        start,
        end,
        end - start,
        hap_a,
        hap_b,
        f"{ident:.4f}",
    ])

def toy1(path: str) -> None:
    win = 5000
    w = csv.writer(open(path, "w", newline=""), delimiter="\t")
    w.writerow(HEADER)
    for i in range(10):
        s = i*win; e = s+win
        ident = 0.9950
        if 1 <= i <= 6:  # windows 1..6 inclusive => 5k..35k
            ident = 0.9998 if i != 4 else 0.9997
        write_row(w, "chr1", s, e, "A", "B", ident)
    for i in range(0,7):
        s = i*win; e = s+win
        write_row(w, "chr1", s, e, "A", "C", 0.9950)

def toy2(path: str) -> None:
    win = 5000
    w = csv.writer(open(path, "w", newline=""), delimiter="\t")
    w.writerow(HEADER)
    vals = [0.9970,0.9990,0.9960,0.9970,0.9960,0.9970,0.9960,0.9970,0.9960,0.9970]
    for i,ident in enumerate(vals):
        s=i*win; e=s+win
        write_row(w, "chr1", s, e, "A", "B", ident)

def toy3(path: str) -> None:
    win = 5000
    w = csv.writer(open(path, "w", newline=""), delimiter="\t")
    w.writerow(HEADER)
    write_row(w, "chr1", 0, 5000, "A", "B", 0.9950)
    write_row(w, "chr1", 5000, 10000, "A", "B", 0.9997)
    write_row(w, "chr1", 10000, 15000, "A", "B", 0.9998)
    write_row(w, "chr1", 25000, 30000, "A", "B", 0.9998)
    write_row(w, "chr1", 30000, 35000, "A", "B", 0.9998)
    write_row(w, "chr1", 35000, 40000, "A", "B", 0.9950)

def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry-point used from notebooks and automated tests."""

    ap = argparse.ArgumentParser(description="Generate toy pairwise per-window identities")
    ap.add_argument("--which", choices=["toy1", "toy2", "toy3", "all"], default="all")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)
    if args.which in ("toy1", "all"):
        toy1(f"{args.outdir}/toy1.pairwise.tsv")
    if args.which in ("toy2", "all"):
        toy2(f"{args.outdir}/toy2.pairwise.tsv")
    if args.which in ("toy3", "all"):
        toy3(f"{args.outdir}/toy3.pairwise.tsv")
    print(
        "Done. Files:",
        ", ".join([f"{args.outdir}/toy{i}.pairwise.tsv" for i in (1, 2, 3)]),
    )

if __name__ == "__main__":
    main()
