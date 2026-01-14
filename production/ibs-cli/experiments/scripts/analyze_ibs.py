#!/usr/bin/env python3
"""
IBS Analysis and Plotting Script for Population Comparisons
Generates comprehensive reports with statistical analysis and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_ibs_data(filepath):
    """Load IBS results from TSV file."""
    df = pd.read_csv(filepath, sep='\t')
    # Extract sample IDs from group names
    df['sample_a'] = df['group.a'].apply(lambda x: x.split('#')[0] if pd.notna(x) else None)
    df['sample_b'] = df['group.b'].apply(lambda x: x.split('#')[0] if pd.notna(x) else None)
    return df

def compute_statistics(df):
    """Compute summary statistics for IBS values."""
    stats = {
        'n_windows': len(df),
        'mean_identity': df['estimated.identity'].mean(),
        'median_identity': df['estimated.identity'].median(),
        'std_identity': df['estimated.identity'].std(),
        'min_identity': df['estimated.identity'].min(),
        'max_identity': df['estimated.identity'].max(),
        'q25': df['estimated.identity'].quantile(0.25),
        'q75': df['estimated.identity'].quantile(0.75),
        'pct_high_ibs': (df['estimated.identity'] >= 0.999).mean() * 100,
        'pct_perfect': (df['estimated.identity'] == 1.0).mean() * 100,
    }
    return stats

def plot_identity_distribution(df, output_path, title="IBS Identity Distribution"):
    """Plot histogram of identity values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df['estimated.identity'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.999, color='red', linestyle='--', label='IBS threshold (0.999)')
    axes[0].axvline(x=df['estimated.identity'].mean(), color='green', linestyle='-', label=f'Mean ({df["estimated.identity"].mean():.4f})')
    axes[0].set_xlabel('Estimated Identity')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].legend()

    # Box plot by genomic position bins
    df['pos_bin'] = pd.cut(df['start'], bins=10)
    axes[1].boxplot([group['estimated.identity'].values for name, group in df.groupby('pos_bin', observed=True)],
                    labels=[f'{int(i.left/1e6)}-{int(i.right/1e6)}Mb' for i in df['pos_bin'].cat.categories])
    axes[1].set_xlabel('Genomic Position (Mb)')
    axes[1].set_ylabel('Estimated Identity')
    axes[1].set_title(f'{title} - By Position')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def plot_heatmap(df, output_path, title="Pairwise Mean IBS"):
    """Create heatmap of mean IBS between sample pairs."""
    # Get unique samples
    samples = sorted(set(df['sample_a'].unique()) | set(df['sample_b'].unique()))

    # Create matrix
    matrix = pd.DataFrame(index=samples, columns=samples, dtype=float)
    matrix.values[:] = np.nan

    for (sa, sb), group in df.groupby(['sample_a', 'sample_b']):
        mean_ibs = group['estimated.identity'].mean()
        matrix.loc[sa, sb] = mean_ibs
        matrix.loc[sb, sa] = mean_ibs

    # Fill diagonal with 1.0
    np.fill_diagonal(matrix.values, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = matrix.isna()
    sns.heatmap(matrix.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
                mask=mask, ax=ax, vmin=0.8, vmax=1.0,
                cbar_kws={'label': 'Mean IBS Identity'})
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def plot_position_profile(df, output_path, title="IBS Along Genomic Position"):
    """Plot IBS identity along genomic positions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get unique sample pairs
    df['pair'] = df['sample_a'] + ' vs ' + df['sample_b']

    for pair, group in df.groupby('pair'):
        group_sorted = group.sort_values('start')
        ax.plot(group_sorted['start'] / 1e6, group_sorted['estimated.identity'],
                alpha=0.6, linewidth=0.8, label=pair if len(df['pair'].unique()) <= 10 else None)

    ax.axhline(y=0.999, color='red', linestyle='--', alpha=0.5, label='IBS threshold')
    ax.set_xlabel('Genomic Position (Mb)')
    ax.set_ylabel('Estimated Identity')
    ax.set_title(title)
    ax.set_ylim(0.7, 1.02)

    if len(df['pair'].unique()) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def compare_populations(data_dict, output_path):
    """Create comparison plot across populations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Violin plot of identity distributions
    all_data = []
    for pop, df in data_dict.items():
        temp = df[['estimated.identity']].copy()
        temp['Population'] = pop
        all_data.append(temp)

    combined = pd.concat(all_data, ignore_index=True)

    sns.violinplot(data=combined, x='Population', y='estimated.identity', ax=axes[0, 0])
    axes[0, 0].axhline(y=0.999, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('IBS Identity Distribution by Population')
    axes[0, 0].set_ylabel('Estimated Identity')

    # 2. Box plot
    sns.boxplot(data=combined, x='Population', y='estimated.identity', ax=axes[0, 1])
    axes[0, 1].axhline(y=0.999, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('IBS Identity Summary by Population')
    axes[0, 1].set_ylabel('Estimated Identity')

    # 3. Mean IBS bar chart
    means = {pop: df['estimated.identity'].mean() for pop, df in data_dict.items()}
    stds = {pop: df['estimated.identity'].std() for pop, df in data_dict.items()}

    pops = list(means.keys())
    x = range(len(pops))
    axes[1, 0].bar(x, [means[p] for p in pops], yerr=[stds[p] for p in pops], capsize=5, alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(pops)
    axes[1, 0].set_ylabel('Mean IBS Identity')
    axes[1, 0].set_title('Mean IBS Identity by Population')
    axes[1, 0].set_ylim(0.9, 1.0)

    # 4. High IBS percentage
    high_ibs = {pop: (df['estimated.identity'] >= 0.999).mean() * 100 for pop, df in data_dict.items()}
    axes[1, 1].bar(x, [high_ibs[p] for p in pops], alpha=0.7, color='green')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(pops)
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Percentage of Windows with IBS >= 0.999')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path

def generate_report(data_dict, output_dir, experiment_name):
    """Generate comprehensive markdown report."""
    report_path = output_dir / f"{experiment_name}_report.md"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(f"# IBS Analysis Report: {experiment_name}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Population | N Windows | Mean IBS | Median IBS | Std | High IBS (%) | Perfect (%) |\n")
        f.write("|------------|-----------|----------|------------|-----|--------------|-------------|\n")

        all_stats = {}
        for pop, df in data_dict.items():
            stats = compute_statistics(df)
            all_stats[pop] = stats
            f.write(f"| {pop} | {stats['n_windows']} | {stats['mean_identity']:.4f} | "
                   f"{stats['median_identity']:.4f} | {stats['std_identity']:.4f} | "
                   f"{stats['pct_high_ibs']:.1f} | {stats['pct_perfect']:.1f} |\n")

        f.write("\n## Detailed Statistics\n\n")
        for pop, stats in all_stats.items():
            f.write(f"### {pop}\n\n")
            f.write(f"- **Total windows**: {stats['n_windows']}\n")
            f.write(f"- **Identity range**: [{stats['min_identity']:.4f}, {stats['max_identity']:.4f}]\n")
            f.write(f"- **Interquartile range**: [{stats['q25']:.4f}, {stats['q75']:.4f}]\n")
            f.write(f"- **Windows with IBS >= 0.999**: {stats['pct_high_ibs']:.1f}%\n")
            f.write(f"- **Windows with perfect identity**: {stats['pct_perfect']:.1f}%\n\n")

        # Generate plots
        f.write("## Visualizations\n\n")

        if len(data_dict) > 1:
            comparison_plot = plots_dir / f"{experiment_name}_comparison.png"
            compare_populations(data_dict, comparison_plot)
            f.write(f"### Population Comparison\n\n")
            f.write(f"![Population Comparison]({comparison_plot.name})\n\n")

        for pop, df in data_dict.items():
            f.write(f"### {pop} Analysis\n\n")

            # Distribution plot
            dist_plot = plots_dir / f"{experiment_name}_{pop}_distribution.png"
            plot_identity_distribution(df, dist_plot, f"{pop} IBS Distribution")
            f.write(f"![{pop} Distribution]({dist_plot.name})\n\n")

            # Position profile
            pos_plot = plots_dir / f"{experiment_name}_{pop}_position.png"
            plot_position_profile(df, pos_plot, f"{pop} IBS Along Position")
            f.write(f"![{pop} Position Profile]({pos_plot.name})\n\n")

            # Heatmap (if multiple samples)
            if df['sample_a'].nunique() > 1:
                heat_plot = plots_dir / f"{experiment_name}_{pop}_heatmap.png"
                plot_heatmap(df, heat_plot, f"{pop} Pairwise Mean IBS")
                f.write(f"![{pop} Heatmap]({heat_plot.name})\n\n")

        f.write("## Conclusions\n\n")

        # Find population with highest/lowest mean IBS
        if all_stats:
            highest_pop = max(all_stats.items(), key=lambda x: x[1]['mean_identity'])
            lowest_pop = min(all_stats.items(), key=lambda x: x[1]['mean_identity'])

            f.write(f"- **Highest mean IBS**: {highest_pop[0]} ({highest_pop[1]['mean_identity']:.4f})\n")
            f.write(f"- **Lowest mean IBS**: {lowest_pop[0]} ({lowest_pop[1]['mean_identity']:.4f})\n")

            # Overall interpretation
            overall_mean = np.mean([s['mean_identity'] for s in all_stats.values()])
            f.write(f"- **Overall mean across populations**: {overall_mean:.4f}\n\n")

            if overall_mean > 0.99:
                f.write("The high IBS values suggest this region has limited variation or potential IBD sharing.\n")
            elif overall_mean > 0.95:
                f.write("Moderate IBS values indicate typical genetic variation in this region.\n")
            else:
                f.write("Lower IBS values suggest high genetic diversity or structural variation.\n")

    return report_path

def main():
    parser = argparse.ArgumentParser(description='Analyze IBS results and generate reports')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input IBS TSV files (format: POP:filepath)')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--name', '-n', default='ibs_analysis', help='Experiment name')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dict = {}
    for item in args.input:
        if ':' in item:
            pop, filepath = item.split(':', 1)
        else:
            pop = Path(item).stem
            filepath = item

        if Path(filepath).exists():
            data_dict[pop] = load_ibs_data(filepath)
            print(f"Loaded {pop}: {len(data_dict[pop])} windows")
        else:
            print(f"Warning: File not found: {filepath}")

    if not data_dict:
        print("No data loaded!")
        return

    # Generate report
    report_path = generate_report(data_dict, output_dir, args.name)
    print(f"\nReport generated: {report_path}")

if __name__ == '__main__':
    main()
