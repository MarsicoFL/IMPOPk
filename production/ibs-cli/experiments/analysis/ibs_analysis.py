#!/usr/bin/env python3
"""
IBS Enrichment Analysis for LCT Region Experiments

This script analyzes IBS (Identity-By-State) patterns across populations
in the chr2 LCT gene region using HPRC v2 data.

Key features:
- Properly normalized IBS rate accounting for sample size differences
- Deduplication of multiple alignments per (window, pair)
- Fold enrichment calculation relative to AFR baseline

Usage:
    python ibs_analysis.py [--results-dir PATH] [--output-dir PATH]

Author: HPRCv2-IBD Project
Date: January 2026
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

REGION_START = 130787850
REGION_END = 140837183
WINDOW_SIZE = 5000
TOTAL_WINDOWS = (REGION_END - REGION_START) // WINDOW_SIZE  # 2009

# Sample counts per population (haplotypes)
HAPLOTYPE_COUNTS = {
    'AFR': 8,  # 4 diploid individuals
    'EUR': 8,
    'EAS': 8,
    'CSA': 8,
    'AMR': 6   # 3 diploid individuals
}

POPULATION_COLORS = {
    'AFR': '#e74c3c',
    'EUR': '#3498db',
    'EAS': '#2ecc71',
    'CSA': '#9b59b6',
    'AMR': '#f39c12'
}


# =============================================================================
# Data Loading
# =============================================================================

def get_possible_pairs(pop: str) -> int:
    """Calculate C(n,2) for a population."""
    n = HAPLOTYPE_COUNTS[pop]
    return n * (n - 1) // 2


def normalize_pair(group_a: str, group_b: str) -> tuple:
    """
    Create normalized pair ID from group identifiers.
    Handles A-B == B-A by sorting.

    Input format: "SAMPLE#HAP#ACCESSION:coords"
    Output: sorted tuple of ("SAMPLE#HAP", "SAMPLE#HAP")
    """
    def extract_id(g):
        parts = str(g).split('#')
        return f"{parts[0]}#{parts[1]}" if len(parts) >= 2 else g

    a, b = extract_id(group_a), extract_id(group_b)
    return tuple(sorted([a, b]))


def load_intra_results(results_dir: Path) -> dict:
    """
    Load intra-population IBS results with deduplication.

    Returns dict mapping population -> deduplicated DataFrame
    """
    results = {}
    col_names = ['chrom', 'start', 'end', 'group.a', 'group.b', 'estimated.identity']

    for pop in HAPLOTYPE_COUNTS.keys():
        tsv_path = results_dir / f"{pop}_intra" / f"{pop}_intra_ibs.tsv"

        if not tsv_path.exists():
            print(f"  Warning: {tsv_path} not found, skipping {pop}")
            continue

        # Check for header
        with open(tsv_path, 'r') as f:
            has_header = f.readline().strip().startswith('chrom')

        # Load data
        if has_header:
            df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip', low_memory=False)
            df = df[df['chrom'] != 'chrom']  # Remove duplicate headers
        else:
            df = pd.read_csv(tsv_path, sep='\t', names=col_names,
                           on_bad_lines='skip', low_memory=False)

        # Clean numeric columns
        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df = df.dropna(subset=['start'])
        df['start'] = df['start'].astype(int)

        # Create normalized pair ID and deduplicate
        df['norm_pair'] = df.apply(
            lambda r: normalize_pair(r['group.a'], r['group.b']), axis=1
        )
        df_dedup = df.drop_duplicates(subset=['start', 'norm_pair'])

        results[pop] = df_dedup
        print(f"  {pop}: {len(df_dedup):,} unique (window, pair) records")

    return results


# =============================================================================
# Analysis
# =============================================================================

def compute_ibs_metrics(results: dict) -> pd.DataFrame:
    """
    Compute normalized IBS metrics for each population.

    IBS Rate = unique_ibs_records / (possible_pairs × total_windows)
    """
    metrics = []

    for pop, df in results.items():
        n_records = len(df)
        possible_pairs = get_possible_pairs(pop)
        unique_windows = df['start'].nunique()

        # Normalized IBS rate
        ibs_rate = n_records / (possible_pairs * TOTAL_WINDOWS)

        metrics.append({
            'population': pop,
            'n_individuals': HAPLOTYPE_COUNTS[pop] // 2,
            'n_haplotypes': HAPLOTYPE_COUNTS[pop],
            'possible_pairs': possible_pairs,
            'unique_records': n_records,
            'unique_windows': unique_windows,
            'ibs_rate': ibs_rate
        })

    return pd.DataFrame(metrics)


def compute_fold_enrichment(metrics_df: pd.DataFrame, baseline: str = 'AFR') -> pd.DataFrame:
    """Add fold enrichment column relative to baseline population."""
    baseline_rate = metrics_df[metrics_df['population'] == baseline]['ibs_rate'].values[0]
    metrics_df['fold_enrichment'] = metrics_df['ibs_rate'] / baseline_rate
    metrics_df['baseline'] = baseline
    return metrics_df, baseline_rate


# =============================================================================
# Visualization
# =============================================================================

def plot_enrichment(metrics_df: pd.DataFrame, baseline_rate: float,
                    output_path: Path) -> None:
    """Generate publication-quality enrichment figure."""

    df = metrics_df.sort_values('ibs_rate', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: IBS Rate
    ax = axes[0]
    y_pos = range(len(df))
    colors = [POPULATION_COLORS[p] for p in df['population']]

    ax.barh(y_pos, df['ibs_rate'] * 100, color=colors,
            edgecolor='black', linewidth=0.5, height=0.7)
    ax.axvline(x=baseline_rate * 100, color='#e74c3c', linestyle='--',
               linewidth=2, alpha=0.7, label=f'AFR baseline: {baseline_rate*100:.2f}%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['population'], fontsize=12, fontweight='bold')
    ax.set_xlabel('IBS Rate (%)', fontsize=11)
    ax.set_title('A. Normalized IBS Rate', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max(df['ibs_rate'] * 100) * 1.15)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['ibs_rate'] * 100 + 0.5, i, f"{row['ibs_rate']*100:.2f}%",
                va='center', fontsize=10, fontweight='bold')

    # Panel B: Fold Enrichment
    ax = axes[1]

    ax.barh(y_pos, df['fold_enrichment'], color=colors,
            edgecolor='black', linewidth=0.5, height=0.7)
    ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=2,
               alpha=0.7, label='AFR baseline (1.0×)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['population'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Fold Enrichment vs AFR', fontsize=11)
    ax.set_title('B. IBS Fold Enrichment', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max(df['fold_enrichment']) * 1.15)

    for i, (_, row) in enumerate(df.iterrows()):
        marker = '**' if row['fold_enrichment'] > 1.5 else ('*' if row['fold_enrichment'] > 1.2 else '')
        ax.text(row['fold_enrichment'] + 0.03, i, f"{row['fold_enrichment']:.2f}×{marker}",
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save both formats
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def save_metrics_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Save metrics as TSV."""
    metrics_df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='IBS Enrichment Analysis for LCT Region',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python ibs_analysis.py --results-dir ../results --output-dir ./output
        """
    )
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR,
                        help='Directory containing IBS result TSV files')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for figures and tables')
    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IBS Enrichment Analysis - LCT Region")
    print("=" * 70)
    print(f"\nRegion: chr2:{REGION_START:,}-{REGION_END:,} ({TOTAL_WINDOWS} windows)")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    print("\n[1/3] Loading intra-population results...")
    results = load_intra_results(args.results_dir)

    if len(results) == 0:
        print("Error: No results found!")
        sys.exit(1)

    # Compute metrics
    print("\n[2/3] Computing metrics...")
    metrics_df = compute_ibs_metrics(results)
    metrics_df, baseline_rate = compute_fold_enrichment(metrics_df)

    print("\n" + "-" * 70)
    print("Results Summary")
    print("-" * 70)
    for _, row in metrics_df.sort_values('ibs_rate', ascending=False).iterrows():
        print(f"  {row['population']:5s}: {row['unique_records']:6,} records | "
              f"Rate: {row['ibs_rate']*100:5.2f}% | "
              f"Fold: {row['fold_enrichment']:.2f}×")
    print(f"\n  Baseline (AFR): {baseline_rate*100:.2f}%")

    # Generate outputs
    print("\n[3/3] Generating outputs...")

    fig_path = args.output_dir / "ibs_enrichment_rates"
    plot_enrichment(metrics_df, baseline_rate, fig_path)
    print(f"  Figure: {fig_path}.png/.pdf")

    table_path = args.output_dir / "ibs_metrics.tsv"
    save_metrics_table(metrics_df, table_path)
    print(f"  Table: {table_path}")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
