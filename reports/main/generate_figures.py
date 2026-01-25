#!/usr/bin/env python3
"""
Generate CORRECTED figures for HPRCv2-IBD analysis.

CRITICAL NOTE: This script only uses validated data:
- EUR IBD segments: VALID (identity ~99.97-99.99%)
- AFR IBD segments: INVALID (identity ~30-40%, concentrated in centromere)
- v2 emission parameters: VALID for both populations
- Selection scan IBS rates: VALID

Figures removed due to invalid AFR data:
- fig_chr1_ibd_signal_track (AFR panels invalid)
- fig_chr1_arms_analysis (AFR panels invalid)
- fig_excluding_centromere (based on invalid AFR data)
- fig_selection_position_tracks (synthetic data - replaced with bar chart)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

POPULATION_COLORS = {
    'AFR': '#E64B35',
    'EUR': '#4DBBD5',
    'EAS': '#00A087',
    'CSA': '#3C5488',
    'AMR': '#F39B7F',
}

# Chromosome 1 regions
CHR1_LENGTH = 248956422
CENTROMERE_START = 121500000
CENTROMERE_END = 142200000


def load_data():
    """Load all data files."""
    base_path = Path('/home/franco/Escritorio/trabajadores/HPRCv2-IBD')

    # v2 corrected parameters (VALID)
    with open(base_path / 'experiments/chr1_full/results/json/EUR_summary_v2.json') as f:
        eur_summary_v2 = json.load(f)
    with open(base_path / 'experiments/chr1_full/results/json/AFR_summary_v2.json') as f:
        afr_summary_v2 = json.load(f)

    # IBD results (EUR valid, AFR invalid)
    with open(base_path / 'experiments/chr1_full/results/json/EUR_ibd_results.json') as f:
        eur_results = json.load(f)
    with open(base_path / 'experiments/chr1_full/results/json/AFR_ibd_results.json') as f:
        afr_results = json.load(f)

    # Selection scan (VALID)
    with open(base_path / 'experiments/selection_scan/analysis/figures/expanded_statistics.json') as f:
        selection_data = json.load(f)

    return {
        'eur_summary_v2': eur_summary_v2,
        'afr_summary_v2': afr_summary_v2,
        'eur_results': eur_results,
        'afr_results': afr_results,  # Keep for comparison showing invalidity
        'selection': selection_data,
    }


def extract_segments(results):
    """Extract all segments from results."""
    segments = []
    for pair in results['results']:
        for seg in pair['segments']:
            segments.append({
                'start': seg['start_bp'],
                'end': seg['end_bp'],
                'length': seg['length_bp'],
                'identity': seg['mean_identity'],
                'posterior': seg['mean_posterior'],
                'sample': pair['sample_a'],
                'hap': pair['hap_a']
            })
    return segments


def compute_coverage_track(segments, bin_size=500000):
    """Compute IBD coverage track along chromosome."""
    bins = np.arange(0, CHR1_LENGTH + bin_size, bin_size)
    coverage = np.zeros(len(bins) - 1)

    for seg in segments:
        for i, (b_start, b_end) in enumerate(zip(bins[:-1], bins[1:])):
            if seg['start'] < b_end and seg['end'] > b_start:
                overlap = min(seg['end'], b_end) - max(seg['start'], b_start)
                coverage[i] += overlap / bin_size

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, coverage


def compute_segment_density(segments, bin_size=1000000):
    """Compute segment count per bin."""
    bins = np.arange(0, CHR1_LENGTH + bin_size, bin_size)
    density = np.zeros(len(bins) - 1)

    for seg in segments:
        mid = (seg['start'] + seg['end']) / 2
        bin_idx = int(mid / bin_size)
        if bin_idx < len(density):
            density[bin_idx] += 1

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, density


def fig_emission_distributions_v2(data, output_dir):
    """
    Figure 1: Emission distributions from v2 corrected parameters.
    This shows the CORRECT separation between IBD and non-IBD states.
    """
    eur = data['eur_summary_v2']['emission_params']
    afr = data['afr_summary_v2']['emission_params']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.linspace(0.994, 1.0005, 1000)

    # EUR
    ax = axes[0]
    y_non_ibd = stats.norm.pdf(x, eur['non_ibd']['mean'], eur['non_ibd']['std'])
    y_ibd = stats.norm.pdf(x, eur['ibd']['mean'], eur['ibd']['std'])

    ax.fill_between(x, y_non_ibd, alpha=0.5, color=POPULATION_COLORS['EUR'], label='non-IBD')
    ax.fill_between(x, y_ibd, alpha=0.5, color='green', label='IBD')
    ax.plot(x, y_non_ibd, color=POPULATION_COLORS['EUR'], linewidth=1)
    ax.plot(x, y_ibd, color='green', linewidth=1)

    ax.set_xlabel('Identity')
    ax.set_ylabel('Density')
    ax.set_title(f"A) EUR emission distributions (d' = {eur['d_prime']:.2f})", loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim(0.994, 1.0005)

    # Add v2 badge
    ax.text(0.98, 0.95, 'v2 corrected', transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            ha='right', va='top')

    # AFR
    ax = axes[1]
    y_non_ibd = stats.norm.pdf(x, afr['non_ibd']['mean'], afr['non_ibd']['std'])
    y_ibd = stats.norm.pdf(x, afr['ibd']['mean'], afr['ibd']['std'])

    ax.fill_between(x, y_non_ibd, alpha=0.5, color=POPULATION_COLORS['AFR'], label='non-IBD')
    ax.fill_between(x, y_ibd, alpha=0.5, color='green', label='IBD')
    ax.plot(x, y_non_ibd, color=POPULATION_COLORS['AFR'], linewidth=1)
    ax.plot(x, y_ibd, color='green', linewidth=1)

    ax.set_xlabel('Identity')
    ax.set_ylabel('Density')
    ax.set_title(f"B) AFR emission distributions (d' = {afr['d_prime']:.2f})", loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim(0.994, 1.0005)

    ax.text(0.98, 0.95, 'v2 corrected', transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            ha='right', va='top')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_emission_distributions.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_emission_distributions.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig1_emission_distributions (v2 corrected)")


def fig_eur_ibd_landscape(data, output_dir):
    """
    Figure 2: EUR IBD landscape - VALID DATA ONLY.
    Shows EUR IBD coverage and density across chromosome 1.
    """
    eur_segments = extract_segments(data['eur_results'])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # Panel A: EUR IBD coverage
    ax = axes[0]
    pos, cov = compute_coverage_track(eur_segments)
    ax.fill_between(pos/1e6, cov, alpha=0.7, color=POPULATION_COLORS['EUR'])
    ax.plot(pos/1e6, cov, color=POPULATION_COLORS['EUR'], linewidth=0.5)
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_ylabel('IBD coverage')
    ax.set_title('A) EUR IBD coverage along chromosome 1', loc='left', fontweight='bold')
    ax.text(132, ax.get_ylim()[1]*0.8, 'CEN', fontsize=7, ha='center', color='gray')

    # Add data quality badge
    ax.text(0.98, 0.95, 'VALID DATA', transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            ha='right', va='top')

    # Panel B: EUR segment density
    ax = axes[1]
    pos, dens = compute_segment_density(eur_segments)
    ax.bar(pos/1e6, dens, width=0.9, color=POPULATION_COLORS['EUR'], alpha=0.7, edgecolor='none')
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_ylabel('Segments per Mb')
    ax.set_title('B) EUR segment density', loc='left', fontweight='bold')

    # Panel C: EUR segment identity by position
    ax = axes[2]
    if eur_segments:
        positions = [(s['start'] + s['end'])/2/1e6 for s in eur_segments]
        identities = [s['identity'] for s in eur_segments]
        ax.scatter(positions, identities, c=POPULATION_COLORS['EUR'], alpha=0.4, s=8, edgecolors='none')
        ax.axhline(np.mean(identities), color='red', linestyle='--', linewidth=0.8,
                   label=f'Mean: {np.mean(identities):.6f}')
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Segment identity')
    ax.set_title('C) EUR segment identity (all ~99.97-99.99%)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', frameon=False)
    ax.set_ylim(0.9995, 1.0001)

    ax.set_xlim(0, CHR1_LENGTH/1e6)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_eur_ibd_landscape.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_eur_ibd_landscape.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig2_eur_ibd_landscape (EUR only - valid data)")


def fig_data_quality_comparison(data, output_dir):
    """
    Figure 3: Data quality comparison showing why AFR IBD data is invalid.
    This figure is CRITICAL for scientific honesty.
    """
    eur_segments = extract_segments(data['eur_results'])
    afr_segments = extract_segments(data['afr_results'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Identity distribution comparison
    ax = axes[0, 0]
    eur_identities = [s['identity'] for s in eur_segments]
    afr_identities = [s['identity'] for s in afr_segments]

    ax.hist(eur_identities, bins=30, alpha=0.7, color=POPULATION_COLORS['EUR'],
            label=f'EUR (n={len(eur_identities)})', density=True)
    ax.hist(afr_identities, bins=30, alpha=0.7, color=POPULATION_COLORS['AFR'],
            label=f'AFR (n={len(afr_identities)})', density=True)
    ax.axvline(0.9999, color='green', linestyle='--', linewidth=1.5, label='Expected IBD (~0.9999)')
    ax.set_xlabel('Segment identity')
    ax.set_ylabel('Density')
    ax.set_title('A) Segment identity distributions', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False, fontsize=7)

    # Add annotation
    ax.annotate('EUR: VALID\n(~99.97-99.99%)', xy=(0.9998, 15), fontsize=8, color='blue',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.annotate('AFR: INVALID\n(~27-45%)', xy=(0.35, 8), fontsize=8, color='red',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Panel B: Position distribution
    ax = axes[0, 1]
    eur_positions = [(s['start'] + s['end'])/2/1e6 for s in eur_segments]
    afr_positions = [(s['start'] + s['end'])/2/1e6 for s in afr_segments]

    ax.hist(eur_positions, bins=50, alpha=0.7, color=POPULATION_COLORS['EUR'],
            label=f'EUR (n={len(eur_positions)})', range=(0, 250))
    ax.hist(afr_positions, bins=50, alpha=0.7, color=POPULATION_COLORS['AFR'],
            label=f'AFR (n={len(afr_positions)})', range=(0, 250))
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.3, color='gray')
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Segment count')
    ax.set_title('B) Segment position distributions', loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=7)

    # Add centromere annotation
    ax.text(132, ax.get_ylim()[1]*0.9, 'CEN', fontsize=8, ha='center', color='gray')
    ax.annotate('95.5% of AFR\n"segments" here', xy=(135, ax.get_ylim()[1]*0.6),
                fontsize=7, color='red', ha='center')

    # Panel C: d' comparison (v2 corrected vs original)
    ax = axes[1, 0]

    # Original (bad) parameters
    eur_orig_dprime = data['eur_results']['emission_params']['d_prime']
    afr_orig_dprime = data['afr_results']['emission_params']['d_prime']

    # v2 corrected parameters
    eur_v2_dprime = data['eur_summary_v2']['emission_params']['d_prime']
    afr_v2_dprime = data['afr_summary_v2']['emission_params']['d_prime']

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [eur_orig_dprime, afr_orig_dprime], width,
                   label='Original (segments used)', color=['lightblue', 'lightcoral'])
    bars2 = ax.bar(x + width/2, [eur_v2_dprime, afr_v2_dprime], width,
                   label='v2 corrected', color=[POPULATION_COLORS['EUR'], POPULATION_COLORS['AFR']])

    ax.axhline(4, color='green', linestyle='--', linewidth=1, label="d' > 4 (excellent)")
    ax.set_xticks(x)
    ax.set_xticklabels(['EUR', 'AFR'])
    ax.set_ylabel("d' (separation)")
    ax.set_title("C) Model separation: original vs v2 corrected", loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=7)
    ax.set_ylim(0, 7)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', fontsize=7)

    # Panel D: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'EUR', 'AFR', 'Status'],
        ['Segments', f'{len(eur_segments)}', f'{len(afr_segments)}', ''],
        ['Mean identity', f'{np.mean(eur_identities):.4f}', f'{np.mean(afr_identities):.4f}', ''],
        ['Expected identity', '~0.9999', '~0.9999', ''],
        ["d' (original)", f'{eur_orig_dprime:.2f}', f'{afr_orig_dprime:.4f}', ''],
        ["d' (v2 corrected)", f'{eur_v2_dprime:.2f}', f'{afr_v2_dprime:.2f}', ''],
        ['In centromere (%)', '30.4%', '95.5%', ''],
        ['Data validity', 'VALID', 'INVALID', ''],
    ]

    colors = [['white']*4,
              ['lightgreen', 'lightgreen', 'lightcoral', 'white'],
              ['lightgreen', 'lightgreen', 'lightcoral', 'white'],
              ['white']*4,
              ['lightyellow', 'lightyellow', 'lightcoral', 'white'],
              ['lightgreen', 'lightgreen', 'lightgreen', 'white'],
              ['lightgreen', 'lightgreen', 'lightcoral', 'white'],
              ['lightgreen', 'lightgreen', 'lightcoral', 'white']]

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    ax.set_title('D) Data quality summary', loc='left', fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_data_quality_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_data_quality_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig3_data_quality_comparison")


def fig_selection_scan_ibs_rates(data, output_dir):
    """
    Figure 4: Selection scan with REAL IBS rates (not synthetic position tracks).
    """
    selection = data['selection']

    loci = ['LCT', 'SLC24A5', 'EDAR', 'HBB', 'DARC']
    populations = ['AFR', 'EUR', 'EAS', 'CSA', 'AMR']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Heatmap of IBS rates
    ax = axes[0]

    ibs_matrix = np.array([[selection['ibs_rates'][pop][locus] for pop in populations]
                           for locus in loci])

    im = ax.imshow(ibs_matrix, cmap='YlOrRd', aspect='auto', vmin=0.05, vmax=0.30)

    ax.set_xticks(range(len(populations)))
    ax.set_xticklabels(populations)
    ax.set_yticks(range(len(loci)))
    ax.set_yticklabels(loci)

    # Add value labels
    for i in range(len(loci)):
        for j in range(len(populations)):
            val = ibs_matrix[i, j]
            color = 'white' if val > 0.2 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=8)

    # Mark expected targets
    targets = {'LCT': 'EUR', 'SLC24A5': 'EUR', 'EDAR': 'EAS', 'HBB': 'AFR', 'DARC': 'AFR'}
    for locus, target in targets.items():
        i = loci.index(locus)
        j = populations.index(target)
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                    edgecolor='blue', linewidth=2))

    ax.set_title('A) IBS rates by locus and population', loc='left', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('IBS rate')

    ax.text(0.02, -0.15, 'Blue boxes: expected selection targets', transform=ax.transAxes,
            fontsize=7, style='italic')

    # Panel B: Fold enrichment vs AFR baseline
    ax = axes[1]

    x = np.arange(len(loci))
    width = 0.18

    for idx, pop in enumerate(['EUR', 'EAS', 'CSA', 'AMR']):
        enrichments = []
        for locus in loci:
            afr_rate = selection['ibs_rates']['AFR'][locus]
            pop_rate = selection['ibs_rates'][pop][locus]
            enrichments.append(pop_rate / afr_rate)

        bars = ax.bar(x + idx*width - 1.5*width, enrichments, width,
                      label=pop, color=POPULATION_COLORS[pop])

        # Mark expected targets
        for i, locus in enumerate(loci):
            if targets[locus] == pop:
                ax.scatter([x[i] + idx*width - 1.5*width], [enrichments[i] + 0.1],
                          marker='*', color='blue', s=100, zorder=5)

    ax.axhline(1, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(loci)
    ax.set_ylabel('Fold enrichment vs AFR')
    ax.set_title('B) Fold enrichment relative to AFR baseline', loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=8)

    ax.text(0.02, -0.15, '* = expected selection target', transform=ax.transAxes,
            fontsize=7, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_selection_scan.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_selection_scan.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig4_selection_scan (real IBS rates)")


def fig_population_summary(data, output_dir):
    """
    Figure 5: Population-level summary with clear data validity indicators.
    """
    eur = data['eur_summary_v2']
    afr = data['afr_summary_v2']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: d' comparison (v2 corrected - both valid)
    ax = axes[0]
    pops = ['EUR', 'AFR']
    dprimes = [eur['emission_params']['d_prime'], afr['emission_params']['d_prime']]
    colors = [POPULATION_COLORS['EUR'], POPULATION_COLORS['AFR']]

    bars = ax.bar(pops, dprimes, color=colors)
    ax.axhline(4, color='green', linestyle='--', linewidth=1, label="d' > 4 (excellent)")
    ax.set_ylabel("d' (state separation)")
    ax.set_title("A) Model performance (v2 corrected)", loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylim(0, 7)

    for bar, d in zip(bars, dprimes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{d:.2f}', ha='center', fontsize=9)

    ax.text(0.5, 0.02, 'Both populations: VALID v2 parameters', transform=ax.transAxes,
            ha='center', fontsize=7, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel B: IBD segment counts (with validity indicator)
    ax = axes[1]
    segments = [eur['total_segments'], afr['total_segments']]

    bars = ax.bar(pops, segments, color=colors)
    ax.set_ylabel('Total IBD segments')
    ax.set_title('B) Detected IBD segments', loc='left', fontweight='bold')

    for bar, s in zip(bars, segments):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{s:,}', ha='center', fontsize=9)

    # Add validity badges
    ax.text(0, segments[0] * 0.5, 'VALID', ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.9))
    ax.text(1, segments[1] * 0.5, 'INVALID*', ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))

    ax.text(0.5, 0.02, '*AFR segments have 30-40% identity (should be ~99.99%)',
            transform=ax.transAxes, ha='center', fontsize=6, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel C: Mean IBD per pair (with validity indicator)
    ax = axes[2]
    mean_ibd = [eur['mean_ibd_mb'], afr['mean_ibd_mb']]

    bars = ax.bar(pops, mean_ibd, color=colors)
    ax.set_ylabel('Mean IBD per pair (Mb)')
    ax.set_title('C) Mean IBD sharing', loc='left', fontweight='bold')

    for bar, m in zip(bars, mean_ibd):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{m:.2f}', ha='center', fontsize=9)

    ax.text(0, mean_ibd[0] * 0.5, 'VALID', ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.9))
    ax.text(1, mean_ibd[1] * 2, 'NOT REAL\nIBD', ha='center', fontsize=7, color='white',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_population_summary.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_population_summary.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig5_population_summary")


def fig_selection_bars(data, output_dir):
    """
    Figure 6: Selection scan bar comparison.
    """
    selection = data['selection']

    loci = ['LCT', 'SLC24A5', 'EDAR', 'HBB', 'DARC']
    populations = ['AFR', 'EUR', 'EAS', 'CSA', 'AMR']

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(loci))
    width = 0.15

    for idx, pop in enumerate(populations):
        rates = [selection['ibs_rates'][pop][locus] for locus in loci]
        bars = ax.bar(x + idx*width - 2*width, rates, width,
                      label=pop, color=POPULATION_COLORS[pop])

    # Mark expected targets
    targets = {'LCT': 'EUR', 'SLC24A5': 'EUR', 'EDAR': 'EAS', 'HBB': 'AFR', 'DARC': 'AFR'}
    for locus, target in targets.items():
        i = loci.index(locus)
        j = populations.index(target)
        target_x = i + j*width - 2*width
        target_y = selection['ibs_rates'][target][locus]
        ax.scatter([target_x], [target_y + 0.015], marker='*', color='black', s=100, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(loci)
    ax.set_ylabel('IBS rate')
    ax.set_title('IBS rates by population at known selection loci', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, ncol=5)

    ax.text(0.02, -0.12, '* = expected selection target', transform=ax.transAxes,
            fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_selection_bars.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig8_selection_bars.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig8_selection_bars")


def fig_segment_distribution(data, output_dir):
    """
    Figure: Segment distribution - EUR only (valid data).
    """
    eur_segments = extract_segments(data['eur_results'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: EUR length distribution
    ax = axes[0, 0]
    lengths = [s['length']/1e3 for s in eur_segments]  # kb
    ax.hist(lengths, bins=30, color=POPULATION_COLORS['EUR'], alpha=0.7, edgecolor='white')
    ax.axvline(np.median(lengths), color='red', linestyle='--', linewidth=1,
               label=f'Median: {np.median(lengths):.1f} kb')
    ax.set_xlabel('Segment length (kb)')
    ax.set_ylabel('Count')
    ax.set_title('A) EUR segment length distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False)

    # Panel B: EUR spatial distribution
    ax = axes[0, 1]
    positions = [(s['start'] + s['end'])/2/1e6 for s in eur_segments]
    ax.hist(positions, bins=50, color=POPULATION_COLORS['EUR'], alpha=0.7, edgecolor='white', range=(0, 250))
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.3, color='gray')
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Segment count')
    ax.set_title('B) EUR spatial distribution', loc='left', fontweight='bold')
    ax.text(132, ax.get_ylim()[1]*0.9, 'CEN', fontsize=8, ha='center', color='gray')

    # Panel C: EUR length vs position
    ax = axes[1, 0]
    ax.scatter(positions, lengths, c=POPULATION_COLORS['EUR'], alpha=0.4, s=10, edgecolors='none')
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Segment length (kb)')
    ax.set_title('C) EUR segment length by position', loc='left', fontweight='bold')
    ax.set_xlim(0, 250)

    # Panel D: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Classify by region
    in_cen = sum(1 for s in eur_segments if CENTROMERE_START <= (s['start']+s['end'])/2 <= CENTROMERE_END)
    p_arm = sum(1 for s in eur_segments if (s['start']+s['end'])/2 < CENTROMERE_START)
    q_arm = sum(1 for s in eur_segments if (s['start']+s['end'])/2 > CENTROMERE_END)

    stats_text = f"""EUR IBD Segment Statistics (VALID DATA)

Total segments: {len(eur_segments):,}
Mean length: {np.mean(lengths):.1f} kb
Median length: {np.median(lengths):.1f} kb
Total IBD: {sum(lengths)/1e3:.2f} Mb

Spatial distribution:
  p-arm (0-121.5 Mb): {p_arm} segments ({100*p_arm/len(eur_segments):.1f}%)
  Centromere: {in_cen} segments ({100*in_cen/len(eur_segments):.1f}%)
  q-arm (142.2-249 Mb): {q_arm} segments ({100*q_arm/len(eur_segments):.1f}%)

Identity: {np.mean([s['identity'] for s in eur_segments]):.6f} (expected ~0.9999)
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.set_title('D) Summary statistics', loc='left', fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_segment_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_segment_distribution.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig3_segment_distribution (EUR only)")


def fig_posterior_quality(data, output_dir):
    """
    Figure: Posterior probability quality - EUR only (valid data).
    """
    eur_segments = extract_segments(data['eur_results'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Mean posterior distribution
    ax = axes[0]
    posteriors = [s['posterior'] for s in eur_segments]
    ax.hist(posteriors, bins=30, color=POPULATION_COLORS['EUR'], alpha=0.7, edgecolor='white')
    ax.axvline(0.95, color='green', linestyle='--', linewidth=1, label='95% threshold')
    ax.set_xlabel('Mean posterior probability')
    ax.set_ylabel('Count')
    ax.set_title('A) EUR mean posterior distribution', loc='left', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)

    # Calculate percentage above 0.95
    pct_high = 100 * sum(1 for p in posteriors if p > 0.95) / len(posteriors)
    ax.text(0.95, 0.9, f'{pct_high:.1f}% > 0.95', transform=ax.transAxes,
            fontsize=9, ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Posterior vs position
    ax = axes[1]
    positions = [(s['start'] + s['end'])/2/1e6 for s in eur_segments]
    colors = ['green' if p > 0.95 else 'orange' if p > 0.8 else 'red' for p in posteriors]
    ax.scatter(positions, posteriors, c=colors, alpha=0.5, s=8, edgecolors='none')
    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.axhline(0.95, color='green', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(0.80, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_ylabel('Mean posterior probability')
    ax.set_title('B) EUR posterior by position', loc='left', fontweight='bold')
    ax.set_xlim(0, 250)
    ax.set_ylim(0.5, 1.02)

    # Legend
    ax.scatter([], [], c='green', s=20, label='>0.95 (high)')
    ax.scatter([], [], c='orange', s=20, label='0.80-0.95')
    ax.scatter([], [], c='red', s=20, label='<0.80 (low)')
    ax.legend(loc='lower right', frameon=False, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_posterior_quality.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_posterior_quality.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig7_posterior_quality (EUR only)")


def fig_ibd_tracks(data, output_dir):
    """
    Figure: Individual IBD tracks - EUR only (valid data).
    """
    eur_results = data['eur_results']['results']

    # Sort by total IBD
    eur_sorted = sorted(eur_results, key=lambda x: x['total_ibd_bp'], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(12, 4))

    for idx, pair in enumerate(eur_sorted):
        y = idx
        for seg in pair['segments']:
            start = seg['start_bp'] / 1e6
            end = seg['end_bp'] / 1e6
            ax.barh(y, end - start, left=start, height=0.6,
                    color=POPULATION_COLORS['EUR'], alpha=0.7, edgecolor='white', linewidth=0.5)

        label = f"{pair['sample_a']}_{pair['hap_a']} vs {pair['sample_b']}_{pair['hap_b']}"
        ax.text(-5, y, label, fontsize=7, ha='right', va='center')

    ax.axvspan(CENTROMERE_START/1e6, CENTROMERE_END/1e6, alpha=0.15, color='gray')
    ax.set_xlim(-50, CHR1_LENGTH/1e6)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('Chromosome 1 position (Mb)')
    ax.set_yticks([])
    ax.set_title('EUR top 5 pairs by IBD (VALID DATA)', fontweight='bold')

    ax.text(132, 4.7, 'CEN', fontsize=8, ha='center', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_ibd_tracks_detailed.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_ibd_tracks_detailed.png', bbox_inches='tight')
    plt.close()
    print("Generated: fig6_ibd_tracks_detailed (EUR only)")


def main():
    print("="*70)
    print("CORRECTED FIGURE GENERATION - USING ONLY VALID DATA")
    print("="*70)
    print("\nData validity status:")
    print("  - EUR IBD segments: VALID (identity ~99.97-99.99%)")
    print("  - AFR IBD segments: INVALID (identity ~30-40%)")
    print("  - v2 emission parameters: VALID (both populations)")
    print("  - Selection scan IBS rates: VALID")
    print()

    print("Loading data...")
    data = load_data()

    output_dir = Path('/home/franco/Escritorio/trabajadores/HPRCv2-IBD/reports/figures_corrected')
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating corrected figures to: {output_dir}")
    print("-"*70)

    # Generate figures
    fig_emission_distributions_v2(data, output_dir)
    fig_eur_ibd_landscape(data, output_dir)
    fig_data_quality_comparison(data, output_dir)
    fig_selection_scan_ibs_rates(data, output_dir)
    fig_population_summary(data, output_dir)
    fig_selection_bars(data, output_dir)
    fig_segment_distribution(data, output_dir)
    fig_posterior_quality(data, output_dir)
    fig_ibd_tracks(data, output_dir)

    print("-"*70)
    print(f"\nAll corrected figures saved to: {output_dir}")

    # Print summary of what was removed
    print("\n" + "="*70)
    print("FIGURES REMOVED DUE TO INVALID DATA:")
    print("="*70)
    print("  - fig_excluding_centromere.png (based on invalid AFR data)")
    print("  - fig_selection_position_tracks.png (synthetic data)")
    print("  - AFR panels from multi-panel figures")
    print()
    print("All figures now clearly indicate data validity status.")
    print("="*70)


if __name__ == '__main__':
    main()
