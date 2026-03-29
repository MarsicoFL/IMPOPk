#!/usr/bin/env python3
"""
Extract query-vs-reference similarities from impg similarity output.

Creates a matrix of similarities for each query haplotype against each reference,
organized by window. This is the input for ancestry/relatedness HMM inference.

Output format:
  chrom, start, end, sample, ref1#1, ref1#2, ref2#1, ref2#2, ...
"""

import sys
import argparse
from collections import defaultdict


def extract_sample_id(full_id):
    """Extract sample#haplotype from full ID with scaffold:coords suffix."""
    parts = full_id.split('#')
    if len(parts) >= 2:
        return f"{parts[0]}#{parts[1]}"
    return full_id


def main():
    parser = argparse.ArgumentParser(description='Extract query-vs-reference similarities')
    parser.add_argument('input', help='Input similarities file from impg')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument('--queries', required=True, help='File with query sample IDs')
    parser.add_argument('--references', required=True, help='File with reference haplotype IDs')
    args = parser.parse_args()

    # Load query samples
    with open(args.queries) as f:
        query_samples = set(line.strip() for line in f if line.strip())
    print(f"Query samples: {len(query_samples)}", file=sys.stderr)

    # Load reference haplotypes
    with open(args.references) as f:
        reference_haplotypes = [line.strip() for line in f if line.strip()]
    print(f"Reference haplotypes: {reference_haplotypes}", file=sys.stderr)

    # Data structure: {(chrom, start, end, sample): {ref: similarity}}
    data = defaultdict(dict)

    # Parse input
    with open(args.input) as f:
        header = f.readline().strip().split('\t')

        # Find column indices
        col_idx = {name: i for i, name in enumerate(header)}
        required = ['chrom', 'start', 'end', 'group.a', 'group.b', 'estimated.identity']
        for col in required:
            if col not in col_idx:
                print(f"ERROR: Missing column: {col}", file=sys.stderr)
                sys.exit(1)

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) <= col_idx['estimated.identity']:
                continue

            chrom = fields[col_idx['chrom']]
            start = fields[col_idx['start']]
            end = fields[col_idx['end']]
            group_a = fields[col_idx['group.a']]
            group_b = fields[col_idx['group.b']]
            identity = fields[col_idx['estimated.identity']]

            id_a = extract_sample_id(group_a)
            id_b = extract_sample_id(group_b)

            # Check if query vs reference
            if id_a in query_samples and id_b in reference_haplotypes:
                key = (chrom, start, end, id_a)
                data[key][id_b] = identity
            elif id_b in query_samples and id_a in reference_haplotypes:
                key = (chrom, start, end, id_b)
                data[key][id_a] = identity

    print(f"Windows with data: {len(data)}", file=sys.stderr)

    # Write output
    with open(args.output, 'w') as out:
        # Header
        out.write('\t'.join(['chrom', 'start', 'end', 'sample'] + reference_haplotypes) + '\n')

        # Sort by chrom, start, sample
        sorted_keys = sorted(data.keys(), key=lambda x: (x[0], int(x[1]), x[3]))

        for key in sorted_keys:
            chrom, start, end, sample = key
            sims = data[key]

            row = [chrom, start, end, sample]
            for ref in reference_haplotypes:
                row.append(sims.get(ref, 'NA'))

            out.write('\t'.join(row) + '\n')

    print(f"Output written to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
