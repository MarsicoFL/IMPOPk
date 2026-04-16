#!/usr/bin/env python3
"""Compute the kinship coefficient θ = Σ L_IBD / (4·L) from an ibd.tsv output.

This implements the formula used in the impopk manuscript:
    θ̂ = Σ_{α,β} L_IBD(A_α, B_β) / 4·L
where (α, β) index the four (hap-A, hap-B) pairings of a diploid pair
(A, B), L_IBD is the total detected IBD length per pair of haplotypes,
and L is the chromosome length.

Usage
-----
    python3 scripts/kinship_from_ibd.py \
        --ibd      ibd.tsv \
        --chrom-length 133324548 \
        [--pairs pedigree_pairs.tsv] \
        [--output  kinship.tsv]

If `--pairs` is omitted, the script computes θ for every diploid pair
that has at least one detected IBD segment between their haplotypes.
`pedigree_pairs.tsv` is a 2-column TSV with `individual_a <tab> individual_b`.

Output columns: individual_a, individual_b, total_ibd_bp, theta_hat
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def ind_of(haplotype: str) -> str:
    """Strip the trailing '#hap' from a haplotype ID.

    Handles both raw sample IDs (e.g. HG00097#1) and full PanSN contig
    names (e.g. HG00097#1#CM087323.1:...). In both cases we take the
    substring before the first '#' to recover the individual ID.
    """
    return haplotype.split("#", 1)[0]


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ibd", required=True, help="ibd.tsv from the `ibd` binary")
    ap.add_argument("--chrom-length", type=int, required=True,
                    help="Chromosome length L in bp (denominator 4·L)")
    ap.add_argument("--pairs", help="Optional TSV with two columns: individual_a, individual_b")
    ap.add_argument("--output", help="Output TSV path (default: stdout)")
    args = ap.parse_args()

    # 1. Sum detected IBD per diploid pair
    total = defaultdict(int)
    with open(args.ibd) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            a = ind_of(row["group.a"])
            b = ind_of(row["group.b"])
            if a == b:
                continue  # skip within-individual segments (not kinship)
            length = int(row["end"]) - int(row["start"])
            total[pair_key(a, b)] += length

    # 2. Restrict to requested pairs if --pairs provided
    if args.pairs:
        wanted = set()
        with open(args.pairs) as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 2 or row[0].startswith("#"):
                    continue
                wanted.add(pair_key(row[0], row[1]))
        selected = [(p, total.get(p, 0)) for p in wanted]
    else:
        selected = list(total.items())

    # 3. θ̂ = total_ibd / (4 · L)
    L = args.chrom_length
    rows = []
    for (a, b), total_bp in sorted(selected):
        theta = total_bp / (4.0 * L)
        rows.append((a, b, total_bp, theta))

    # 4. Write
    out = open(args.output, "w") if args.output else None
    sink = out if out else __import__("sys").stdout
    sink.write("individual_a\tindividual_b\ttotal_ibd_bp\ttheta_hat\n")
    for a, b, total_bp, theta in rows:
        sink.write(f"{a}\t{b}\t{total_bp}\t{theta:.6f}\n")
    if out:
        out.close()


if __name__ == "__main__":
    main()
