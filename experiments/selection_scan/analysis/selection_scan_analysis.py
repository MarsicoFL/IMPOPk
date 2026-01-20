#!/usr/bin/env python3
"""
Selection Scan Analysis - Publication Quality Figures
Generates Nature/Science style plots for IBS-based selection detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette (colorblind-friendly, Nature style)
COLORS = {
    'EUR': '#0072B2',  # Blue
    'EAS': '#D55E00',  # Orange/Red
    'AFR': '#009E73',  # Green
    'CSA': '#CC79A7',  # Pink
    'AMR': '#F0E442',  # Yellow
    'highlight': '#E69F00',
    'neutral': '#999999'
}

BASE_DIR = Path(__file__).parent.parent
REGIONS_DIR = BASE_DIR / 'regions'
OUTPUT_DIR = BASE_DIR / 'analysis' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region metadata
REGIONS = {
    'LCT': {
        'chrom': 'chr2', 'start': 130787850, 'end': 140837183,
        'gene': 'LCT', 'target': 'EUR',
        'title': 'Lactase persistence',
        'description': 'Positive selection for lactase persistence in Europeans'
    },
    'SLC24A5': {
        'chrom': 'chr15', 'start': 48000000, 'end': 50000000,
        'gene': 'SLC24A5', 'target': 'EUR',
        'title': 'Skin pigmentation',
        'description': 'Selection for light skin pigmentation in Europeans'
    },
    'EDAR': {
        'chrom': 'chr2', 'start': 108000000, 'end': 110000000,
        'gene': 'EDAR', 'target': 'EAS',
        'title': 'Hair morphology',
        'description': 'Selection for hair thickness in East Asians'
    },
    'HBB': {
        'chrom': 'chr11', 'start': 5200000, 'end': 5300000,
        'gene': 'HBB', 'target': 'AFR',
        'title': 'Sickle cell (HbS)',
        'description': 'Balancing selection for malaria resistance'
    },
    'DARC': {
        'chrom': 'chr1', 'start': 159000000, 'end': 160000000,
        'gene': 'DARC', 'target': 'AFR',
        'title': 'Duffy-null',
        'description': 'Selection for Plasmodium vivax resistance'
    }
}

# Population sample sizes
POP_SIZES = {
    'EUR': {'ind': 30, 'hap': 60, 'pairs': 1770},
    'EAS': {'ind': 50, 'hap': 100, 'pairs': 4950},
    'AFR': {'ind': 67, 'hap': 134, 'pairs': 8911},
    'CSA': {'ind': 36, 'hap': 72, 'pairs': 2556},
    'AMR': {'ind': 44, 'hap': 88, 'pairs': 3828}
}


def load_ibs_data(region, pop):
    """Load IBS data for a region/population"""
    filepath = REGIONS_DIR / region / f'{region}_{pop}_ibs.tsv'
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, sep='\t')
    return df


def calculate_window_ibs_rate(df, pop):
    """Calculate IBS rate per window"""
    pairs = POP_SIZES[pop]['pairs']
    window_counts = df.groupby(['chrom', 'start', 'end']).size().reset_index(name='ibs_count')
    window_counts['ibs_rate'] = window_counts['ibs_count'] / pairs
    window_counts['position'] = (window_counts['start'] + window_counts['end']) / 2
    return window_counts


def create_manhattan_plot(region_name, ax=None):
    """Create Manhattan-style plot for a single region"""
    info = REGIONS[region_name]
    target = info['target']

    # Load data
    target_df = load_ibs_data(region_name, target)
    afr_df = load_ibs_data(region_name, 'AFR') if target != 'AFR' else None

    if target_df is None:
        return None

    # Calculate rates
    target_rates = calculate_window_ibs_rate(target_df, target)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot target population
    x = (target_rates['position'] - info['start']) / 1e6
    ax.fill_between(x, 0, target_rates['ibs_rate'],
                    alpha=0.7, color=COLORS[target], label=target)

    # Plot AFR control if available
    if afr_df is not None:
        afr_rates = calculate_window_ibs_rate(afr_df, 'AFR')
        x_afr = (afr_rates['position'] - info['start']) / 1e6
        ax.plot(x_afr, afr_rates['ibs_rate'],
                color=COLORS['AFR'], alpha=0.8, linewidth=0.8, label='AFR')

    # Formatting
    ax.set_xlim(0, (info['end'] - info['start']) / 1e6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_xlabel(f"Position on {info['chrom']} (Mb from {info['start']/1e6:.1f})")
    ax.set_ylabel('IBS rate')
    ax.set_title(f"{info['gene']} — {info['title']}", fontweight='bold', loc='left')
    ax.legend(loc='upper right', frameon=False)

    return ax


def create_fold_enrichment_barplot():
    """Create bar plot of fold enrichment across regions"""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    results = []
    for region, info in REGIONS.items():
        target = info['target']
        target_df = load_ibs_data(region, target)

        if target_df is None:
            continue

        target_pairs = POP_SIZES[target]['pairs']
        windows = target_df.groupby(['start']).ngroups
        target_rate = len(target_df) / (target_pairs * windows)

        # Get AFR baseline
        if target != 'AFR':
            afr_df = load_ibs_data(region, 'AFR')
            if afr_df is not None:
                afr_pairs = POP_SIZES['AFR']['pairs']
                afr_rate = len(afr_df) / (afr_pairs * windows)
                fold = target_rate / afr_rate if afr_rate > 0 else 0
            else:
                fold = 0
        else:
            fold = 1.0  # AFR vs AFR
            afr_rate = target_rate

        results.append({
            'region': region,
            'gene': info['gene'],
            'target': target,
            'target_rate': target_rate,
            'afr_rate': afr_rate if target != 'AFR' else target_rate,
            'fold': fold
        })

    df = pd.DataFrame(results)
    df = df.sort_values('fold', ascending=True)

    # Create horizontal bar plot
    y_pos = np.arange(len(df))
    colors = [COLORS[row['target']] for _, row in df.iterrows()]

    bars = ax.barh(y_pos, df['fold'], color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        if row.fold > 1:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{row.fold:.2f}×', va='center', fontsize=8, fontweight='bold')

    # Add baseline
    ax.axvline(x=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(1.05, len(df)-0.5, 'AFR baseline', fontsize=7, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['gene']} ({row['target']})" for _, row in df.iterrows()])
    ax.set_xlabel('Fold enrichment vs AFR')
    ax.set_title('Selection signal strength', fontweight='bold', loc='left')
    ax.set_xlim(0, max(df['fold']) * 1.3)

    plt.tight_layout()
    return fig, df


def create_multi_region_figure():
    """Create main figure with all regions"""
    fig = plt.figure(figsize=(7.2, 9))  # Nature single column width

    # Layout: 5 rows for regions + 1 for summary
    gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1.2],
                          hspace=0.4, wspace=0.3)

    # Plot each region
    region_order = ['LCT', 'SLC24A5', 'EDAR', 'HBB', 'DARC']
    for i, region in enumerate(region_order):
        ax = fig.add_subplot(gs[i, :])
        create_manhattan_plot(region, ax)

        # Add panel label
        ax.text(-0.08, 1.1, chr(65+i), transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # Add fold enrichment summary
    ax_bar = fig.add_subplot(gs[5, 0])

    # Calculate fold enrichments
    results = []
    for region in region_order:
        info = REGIONS[region]
        target = info['target']
        target_df = load_ibs_data(region, target)
        if target_df is None:
            continue

        target_pairs = POP_SIZES[target]['pairs']
        windows = target_df.groupby(['start']).ngroups
        target_rate = len(target_df) / (target_pairs * windows)

        if target != 'AFR':
            afr_df = load_ibs_data(region, 'AFR')
            if afr_df is not None:
                afr_pairs = POP_SIZES['AFR']['pairs']
                afr_rate = len(afr_df) / (afr_pairs * windows)
                fold = target_rate / afr_rate
            else:
                fold = 1
        else:
            fold = 1

        results.append({'region': region, 'target': target, 'fold': fold})

    df_fold = pd.DataFrame(results)
    colors = [COLORS[t] for t in df_fold['target']]
    bars = ax_bar.bar(df_fold['region'], df_fold['fold'], color=colors,
                      edgecolor='black', linewidth=0.5)

    ax_bar.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_bar.set_ylabel('Fold enrichment\nvs AFR')
    ax_bar.set_title('F. Selection signal summary', fontweight='bold', loc='left')

    for bar, fold in zip(bars, df_fold['fold']):
        if fold > 1:
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{fold:.1f}×', ha='center', fontsize=7, fontweight='bold')

    # Legend
    ax_legend = fig.add_subplot(gs[5, 1])
    ax_legend.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['EUR'], edgecolor='black', label='European (EUR)'),
        mpatches.Patch(facecolor=COLORS['EAS'], edgecolor='black', label='East Asian (EAS)'),
        mpatches.Patch(facecolor=COLORS['AFR'], edgecolor='black', label='African (AFR)'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', frameon=False,
                     title='Population', title_fontsize=9)

    # Add text summary
    ax_legend.text(0.5, 0.2,
                   'IBS rate = IBS pairs / (total pairs × windows)\n'
                   'Threshold: sequence identity ≥ 99.9%\n'
                   'Window size: 5 kb',
                   transform=ax_legend.transAxes, ha='center', va='center',
                   fontsize=7, style='italic', alpha=0.7)

    return fig


def create_heatmap_figure():
    """Create IBS rate heatmap across regions and populations"""
    fig, ax = plt.subplots(figsize=(5, 4))

    # Build matrix
    regions = ['LCT', 'SLC24A5', 'EDAR', 'HBB', 'DARC']
    pops = ['EUR', 'EAS', 'AFR']

    matrix = np.zeros((len(regions), len(pops)))
    matrix[:] = np.nan

    for i, region in enumerate(regions):
        for j, pop in enumerate(pops):
            df = load_ibs_data(region, pop)
            if df is not None:
                pairs = POP_SIZES[pop]['pairs']
                windows = df.groupby(['start']).ngroups
                rate = len(df) / (pairs * windows)
                matrix[i, j] = rate

    # Plot
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(len(pops)))
    ax.set_yticks(np.arange(len(regions)))
    ax.set_xticklabels(pops)
    ax.set_yticklabels([f"{r} ({REGIONS[r]['gene']})" for r in regions])

    # Add values
    for i in range(len(regions)):
        for j in range(len(pops)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if matrix[i, j] > 0.2 else 'black')

    ax.set_title('IBS rate by region and population', fontweight='bold', loc='left')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('IBS rate', fontsize=8)

    plt.tight_layout()
    return fig


def generate_statistics_table():
    """Generate comprehensive statistics table"""
    stats = []

    for region, info in REGIONS.items():
        target = info['target']
        target_df = load_ibs_data(region, target)

        if target_df is None:
            continue

        target_pairs = POP_SIZES[target]['pairs']
        windows = target_df.groupby(['start']).ngroups
        target_records = len(target_df)
        target_rate = target_records / (target_pairs * windows)

        # AFR baseline
        afr_records = 0
        afr_rate = 0
        fold = 1.0

        if target != 'AFR':
            afr_df = load_ibs_data(region, 'AFR')
            if afr_df is not None:
                afr_pairs = POP_SIZES['AFR']['pairs']
                afr_records = len(afr_df)
                afr_rate = afr_records / (afr_pairs * windows)
                fold = target_rate / afr_rate if afr_rate > 0 else np.inf

        region_size = info['end'] - info['start']

        stats.append({
            'Region': region,
            'Gene': info['gene'],
            'Chr': info['chrom'],
            'Size (kb)': region_size / 1000,
            'Windows': windows,
            'Target Pop': target,
            'Target Haplotypes': POP_SIZES[target]['hap'],
            'Target Pairs': target_pairs,
            'Target IBS Records': target_records,
            'Target IBS Rate': target_rate,
            'AFR IBS Records': afr_records,
            'AFR IBS Rate': afr_rate,
            'Fold Enrichment': fold,
            'Biological Function': info['description']
        })

    return pd.DataFrame(stats)


def main():
    print("=" * 70)
    print("SELECTION SCAN ANALYSIS - Publication Quality Figures")
    print("=" * 70)
    print()

    # Generate main multi-panel figure
    print("Generating main figure (Fig. 1)...")
    fig1 = create_multi_region_figure()
    fig1.savefig(OUTPUT_DIR / 'figure1_selection_scan.png', bbox_inches='tight', facecolor='white')
    fig1.savefig(OUTPUT_DIR / 'figure1_selection_scan.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUTPUT_DIR / 'figure1_selection_scan.png'}")

    # Generate fold enrichment bar plot
    print("Generating fold enrichment plot (Fig. 2)...")
    fig2, fold_df = create_fold_enrichment_barplot()
    fig2.savefig(OUTPUT_DIR / 'figure2_fold_enrichment.png', bbox_inches='tight', facecolor='white')
    fig2.savefig(OUTPUT_DIR / 'figure2_fold_enrichment.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUTPUT_DIR / 'figure2_fold_enrichment.png'}")

    # Generate heatmap
    print("Generating heatmap (Fig. S1)...")
    fig3 = create_heatmap_figure()
    fig3.savefig(OUTPUT_DIR / 'figureS1_heatmap.png', bbox_inches='tight', facecolor='white')
    fig3.savefig(OUTPUT_DIR / 'figureS1_heatmap.pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUTPUT_DIR / 'figureS1_heatmap.png'}")

    # Generate statistics table
    print("Generating statistics table...")
    stats_df = generate_statistics_table()
    stats_df.to_csv(OUTPUT_DIR / 'selection_scan_statistics.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'selection_scan_statistics.csv'}")

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(stats_df[['Region', 'Gene', 'Target Pop', 'Target IBS Rate', 'AFR IBS Rate', 'Fold Enrichment']].to_string(index=False))
    print()

    # Key findings
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    max_fold = stats_df.loc[stats_df['Fold Enrichment'].idxmax()]
    print(f"Strongest selection signal: {max_fold['Gene']} in {max_fold['Target Pop']}")
    print(f"  Fold enrichment: {max_fold['Fold Enrichment']:.2f}× vs AFR baseline")
    print()

    for _, row in stats_df.iterrows():
        if row['Fold Enrichment'] > 1.5:
            print(f"• {row['Gene']}: {row['Target Pop']} shows {row['Fold Enrichment']:.2f}× enrichment")
            print(f"  {row['Biological Function']}")
            print()

    plt.close('all')
    print("Analysis complete!")
    return stats_df


if __name__ == '__main__':
    stats = main()
