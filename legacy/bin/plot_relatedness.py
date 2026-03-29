#!/usr/bin/env python3
"""
Plot relatedness/ancestry results for human haplotype analysis.
Shows which reference haplotype each segment is most similar to.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection
import argparse


def get_color_palette(haplotypes):
    """Generate distinct colors for each reference haplotype."""
    base_colors = [
        '#E74C3C',  # Red
        '#3498DB',  # Blue
        '#2ECC71',  # Green
        '#9B59B6',  # Purple
        '#F39C12',  # Orange
        '#1ABC9C',  # Teal
        '#E91E63',  # Pink
        '#00BCD4',  # Cyan
    ]
    return {hap: base_colors[i % len(base_colors)] for i, hap in enumerate(haplotypes)}


def plot_relatedness_painting(ancestry_file, output_file=None, title=None):
    """Chromosome painting showing which reference each segment matches."""
    df = pd.read_csv(ancestry_file, sep='\t')

    samples = sorted(df['sample'].unique())
    n_samples = len(samples)

    # Get unique ancestries (reference haplotypes)
    ancestries = sorted(df['ancestry'].unique())
    colors = get_color_palette(ancestries)

    chrom_len = df['end'].max()

    fig, ax = plt.subplots(figsize=(16, max(6, n_samples * 0.8)))

    bar_height = 0.8
    patches = []
    patch_colors = []

    for i, sample in enumerate(samples):
        sample_df = df[df['sample'] == sample]

        for _, row in sample_df.iterrows():
            start = row['start']
            end = row['end']
            ancestry = row['ancestry']

            rect = mpatches.Rectangle(
                (start, i - bar_height/2),
                end - start,
                bar_height,
            )
            patches.append(rect)
            patch_colors.append(colors.get(ancestry, '#95A5A6'))

    if patches:
        collection = PatchCollection(patches, facecolors=patch_colors,
                                     edgecolors='none', alpha=0.9)
        ax.add_collection(collection)

    ax.set_xlim(0, chrom_len)
    ax.set_ylim(-0.5, n_samples - 0.5)
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(samples, fontsize=10)

    # X-axis in Mb
    xticks = np.arange(0, chrom_len + 1, max(1_000_000, chrom_len // 10))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x/1e6:.1f}' for x in xticks])
    ax.set_xlabel('Position (Mb)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Haplotype Relatedness - Which reference is each segment most similar to?', fontsize=14)

    # Legend
    legend_patches = [mpatches.Patch(color=colors[anc], label=anc) for anc in ancestries]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
              title='Reference Haplotype')

    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


def plot_stats(ancestry_file, output_file=None):
    """Summary statistics of relatedness analysis."""
    df = pd.read_csv(ancestry_file, sep='\t')

    ancestries = sorted(df['ancestry'].unique())
    colors = get_color_palette(ancestries)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Segment count by reference
    ax1 = axes[0]
    counts = df['ancestry'].value_counts()
    bars = ax1.bar(range(len(counts)), counts.values, color=[colors[a] for a in counts.index])
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Number of Segments')
    ax1.set_title('Segment Count by Reference Haplotype')

    # Add counts on bars
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=9)

    # Right: Total length by reference
    ax2 = axes[1]
    df['length'] = df['end'] - df['start']
    lengths = df.groupby('ancestry')['length'].sum() / 1e6  # Convert to Mb
    lengths = lengths.reindex(ancestries)
    bars = ax2.bar(range(len(lengths)), lengths.values, color=[colors[a] for a in lengths.index])
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels(lengths.index, rotation=45, ha='right')
    ax2.set_ylabel('Total Length (Mb)')
    ax2.set_title('Total Matching Length by Reference Haplotype')

    # Add lengths on bars
    for bar, length in zip(bars, lengths.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{length:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


def plot_proportions(ancestry_file, output_file=None):
    """Plot proportions per query sample."""
    df = pd.read_csv(ancestry_file, sep='\t')

    samples = sorted(df['sample'].unique())
    ancestries = sorted(df['ancestry'].unique())
    colors = get_color_palette(ancestries)

    fig, ax = plt.subplots(figsize=(12, 6))

    df['length'] = df['end'] - df['start']

    proportions = []
    for sample in samples:
        sample_df = df[df['sample'] == sample]
        total_len = sample_df['length'].sum()
        props = {}
        for anc in ancestries:
            anc_len = sample_df[sample_df['ancestry'] == anc]['length'].sum()
            props[anc] = anc_len / total_len if total_len > 0 else 0
        proportions.append(props)

    x = np.arange(len(samples))
    width = 0.8
    bottom = np.zeros(len(samples))

    for anc in ancestries:
        values = [p.get(anc, 0) for p in proportions]
        ax.bar(x, values, width, label=anc, bottom=bottom, color=colors[anc])
        bottom += values

    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_xlabel('Query Sample', fontsize=12)
    ax.set_title('Relatedness Proportions by Query Haplotype', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', title='Reference Haplotype')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


def print_summary(ancestry_file):
    """Print summary statistics."""
    df = pd.read_csv(ancestry_file, sep='\t')
    df['length'] = df['end'] - df['start']

    print("\n" + "="*60)
    print("RELATEDNESS ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nTotal segments: {len(df)}")
    print(f"Total length: {df['length'].sum() / 1e6:.2f} Mb")
    print(f"Query samples: {df['sample'].nunique()}")
    print(f"Reference haplotypes: {df['ancestry'].nunique()}")

    print("\n--- Segments by reference haplotype ---")
    for anc in sorted(df['ancestry'].unique()):
        count = len(df[df['ancestry'] == anc])
        length = df[df['ancestry'] == anc]['length'].sum() / 1e6
        print(f"  {anc}: {count} segments, {length:.2f} Mb")

    print("\n--- Per query sample ---")
    for sample in sorted(df['sample'].unique()):
        sample_df = df[df['sample'] == sample]
        total_len = sample_df['length'].sum() / 1e6
        print(f"\n  {sample} ({total_len:.2f} Mb total):")
        for anc in sorted(sample_df['ancestry'].unique()):
            anc_len = sample_df[sample_df['ancestry'] == anc]['length'].sum() / 1e6
            pct = 100 * anc_len / total_len
            print(f"    {anc}: {anc_len:.2f} Mb ({pct:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot relatedness/ancestry results')
    parser.add_argument('ancestry_file', help='Ancestry TSV file')
    parser.add_argument('-o', '--output', default='relatedness', help='Output prefix')
    parser.add_argument('--title', help='Plot title')
    args = parser.parse_args()

    print_summary(args.ancestry_file)
    plot_relatedness_painting(args.ancestry_file, f'{args.output}_painting.png', args.title)
    plot_stats(args.ancestry_file, f'{args.output}_stats.png')
    plot_proportions(args.ancestry_file, f'{args.output}_proportions.png')

    print(f"\nPlots saved with prefix: {args.output}")
