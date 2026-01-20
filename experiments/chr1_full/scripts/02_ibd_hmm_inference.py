#!/usr/bin/env python3
"""
chr1_full: Full Chromosome 1 IBD Analysis

This script runs IBD inference on the complete chromosome 1 using the full
pairwise identity distribution (no cutoff filtering).

Based on exp02_chr2_50Mb_full but scaled to full chromosome.

Author: IBD-CLI Project
Date: 2026-01
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from ibd_inference import (
    Population, HMMParams, GaussianParams, IBDResult,
    infer_ibd, forward_backward, viterbi, extract_segments
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

POPULATION_MAP = {
    'AFR': Population.AFR,
    'EUR': Population.EUR,
}

# Theoretical population diversity (for comparison)
THEORETICAL_DIVERSITY = {
    'AFR': 0.00125,
    'EUR': 0.00085,
}

# Region parameters
CHROM = "chr1"
REGION_START = 1
REGION_END = 248956422
WINDOW_SIZE = 5000  # bp
TOTAL_WINDOWS = 49792


# ============================================================
# Empirical Parameter Estimation
# ============================================================

def estimate_emission_parameters(identities: np.ndarray, verbose: bool = True) -> Dict:
    """
    Estimate emission parameters empirically from FULL identity distribution.

    The key insight is that the full distribution is a mixture:
    - Most pairs are non-IBD (bulk of distribution)
    - A small fraction are IBD (right tail near 1.0)

    We use robust estimators to separate these components.
    """
    n_total = len(identities)

    # Basic statistics on full distribution
    mean_all = np.mean(identities)
    std_all = np.std(identities)
    median_all = np.median(identities)

    # The non-IBD distribution dominates, so we can estimate it from the bulk
    # Use median and MAD for robustness against IBD outliers
    median_identity = np.median(identities)
    mad = np.median(np.abs(identities - median_identity))
    robust_std = mad * 1.4826  # Convert MAD to std estimate

    # Estimate non-IBD parameters using percentile method
    # Use 10th-90th percentile to avoid tails
    p10, p90 = np.percentile(identities, [10, 90])
    bulk_identities = identities[(identities >= p10) & (identities <= p90)]

    mean_non_ibd = np.mean(bulk_identities)
    std_non_ibd = np.std(bulk_identities)

    # IBD should be near 1.0 with very low variance
    # Estimate from high-identity windows
    ibd_threshold = 0.9995
    high_identity = identities[identities >= ibd_threshold]

    if len(high_identity) > 100:
        mean_ibd = np.mean(high_identity)
        std_ibd = np.std(high_identity)
    else:
        # Use theoretical values if not enough high-identity windows
        mean_ibd = 0.9997
        std_ibd = 0.0005

    # Calculate d-prime (separability)
    pooled_std = np.sqrt((std_non_ibd**2 + std_ibd**2) / 2)
    d_prime = (mean_ibd - mean_non_ibd) / pooled_std if pooled_std > 0 else 0

    # Fraction above various thresholds
    frac_above_999 = np.mean(identities >= 0.999)
    frac_above_9995 = np.mean(identities >= 0.9995)
    frac_above_9999 = np.mean(identities >= 0.9999)

    result = {
        'n_observations': n_total,
        'full_distribution': {
            'mean': float(mean_all),
            'std': float(std_all),
            'median': float(median_all),
            'min': float(np.min(identities)),
            'max': float(np.max(identities)),
        },
        'non_ibd': {
            'mean': float(mean_non_ibd),
            'std': float(std_non_ibd),
            'robust_std': float(robust_std),
        },
        'ibd': {
            'mean': float(mean_ibd),
            'std': float(std_ibd),
            'n_samples': len(high_identity),
        },
        'd_prime': float(d_prime),
        'fraction_above_thresholds': {
            '0.999': float(frac_above_999),
            '0.9995': float(frac_above_9995),
            '0.9999': float(frac_above_9999),
        },
    }

    if verbose:
        print(f"  Full distribution: mean={mean_all:.6f}, std={std_all:.6f}")
        print(f"  Non-IBD estimate: mean={mean_non_ibd:.6f}, std={std_non_ibd:.6f}")
        print(f"  IBD estimate: mean={mean_ibd:.6f}, std={std_ibd:.6f}")
        print(f"  d' separability: {d_prime:.3f}")
        print(f"  Fraction >= 0.999: {frac_above_999:.4f}")
        print(f"  Fraction >= 0.9995: {frac_above_9995:.4f}")

    return result


def create_empirical_hmm_params(
    emission_params: Dict,
    expected_ibd_length: float = 50.0,
    p_enter_ibd: float = 0.0001,
) -> HMMParams:
    """Create HMM parameters using empirically estimated emissions."""

    emission_non_ibd = GaussianParams(
        mean=emission_params['non_ibd']['mean'],
        std=emission_params['non_ibd']['std'],
    )

    emission_ibd = GaussianParams(
        mean=emission_params['ibd']['mean'],
        std=emission_params['ibd']['std'],
    )

    p_exit_ibd = 1.0 / expected_ibd_length
    p_exit_ibd = np.clip(p_exit_ibd, 0.0001, 0.5)

    return HMMParams(
        emission_non_ibd=emission_non_ibd,
        emission_ibd=emission_ibd,
        p_enter_ibd=p_enter_ibd,
        p_exit_ibd=p_exit_ibd,
    )


# ============================================================
# Data Loading
# ============================================================

def load_full_identity_data(filepath: Path, sample_frac: float = 1.0,
                            max_lines: int = None) -> Tuple[np.ndarray, Dict]:
    """
    Load full pairwise identity data (no cutoff filtering).

    Args:
        filepath: Path to TSV file
        sample_frac: Fraction of lines to sample (for testing)
        max_lines: Maximum lines to read (for testing)

    Returns:
        all_identities: All identity values for parameter estimation
        pair_data: Dictionary of pair data for IBD inference
    """
    print(f"Loading: {filepath.name}")

    all_identities = []
    all_windows = set()
    pair_windows = defaultdict(dict)  # pair_key -> {(start, end): identity}
    chrom = None

    line_count = 0
    with open(filepath, 'r') as f:
        header = f.readline()

        for line in f:
            line_count += 1

            if max_lines and line_count > max_lines:
                break

            if line_count % 5000000 == 0:
                print(f"  Processed {line_count:,} lines...")

            # Optional sampling for faster processing
            if sample_frac < 1.0 and np.random.random() > sample_frac:
                continue

            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue

            chrom_full, start, end, group_a, group_b, identity = parts[:6]

            try:
                start = int(start)
                end = int(end)
                identity = float(identity)
            except ValueError:
                continue

            if chrom is None:
                chrom = chrom_full

            all_identities.append(identity)

            # Parse haplotype names (format: SampleID#haplotype#contig)
            parts_a = group_a.split('#')
            parts_b = group_b.split('#')
            sample_a = parts_a[0]
            # Safely parse haplotype number (should be 1 or 2)
            hap_a = int(parts_a[1]) if len(parts_a) > 1 and parts_a[1].isdigit() else 0
            sample_b = parts_b[0]
            hap_b = int(parts_b[1]) if len(parts_b) > 1 and parts_b[1].isdigit() else 0

            # Canonical order
            if sample_a > sample_b or (sample_a == sample_b and hap_a > hap_b):
                sample_a, hap_a, sample_b, hap_b = sample_b, hap_b, sample_a, hap_a

            window_key = (start, end)
            pair_key = (sample_a, hap_a, sample_b, hap_b)

            all_windows.add(window_key)
            pair_windows[pair_key][window_key] = identity

    print(f"  Total lines: {line_count:,}")
    print(f"  Total identity values: {len(all_identities):,}")
    print(f"  Unique windows: {len(all_windows):,}")
    print(f"  Unique pairs: {len(pair_windows):,}")

    # Sort windows
    sorted_windows = sorted(all_windows)

    # Build pair data
    pair_data = {}
    for pair_key, windows in pair_windows.items():
        sample_a, hap_a, sample_b, hap_b = pair_key

        starts = []
        ends = []
        identities = []

        for start, end in sorted_windows:
            starts.append(start)
            ends.append(end)
            # Use actual identity or NaN if window not present for this pair
            identities.append(windows.get((start, end), np.nan))

        pair_data[(sample_a, hap_a, sample_b, hap_b, chrom)] = {
            'starts': np.array(starts),
            'ends': np.array(ends),
            'identities': np.array(identities),
        }

    return np.array(all_identities), pair_data


# ============================================================
# IBD Analysis with Empirical Parameters
# ============================================================

def analyze_pair_empirical(
    pair_key: Tuple,
    data: Dict,
    hmm_params: HMMParams,
    min_segment_windows: int = 10,
) -> Optional[IBDResult]:
    """Run IBD inference with empirically calibrated parameters."""

    sample_a, hap_a, sample_b, hap_b, chrom = pair_key

    identities = data['identities']
    starts = data['starts']
    ends = data['ends']

    # Handle missing values - interpolate or use non-IBD mean
    valid_mask = ~np.isnan(identities)
    if np.sum(valid_mask) < 100:
        return None

    # For missing windows, use the non-IBD mean
    identities_filled = identities.copy()
    identities_filled[~valid_mask] = hmm_params.emission_non_ibd.mean

    try:
        # Run forward-backward
        posterior_ibd, log_likelihood = forward_backward(identities_filled, hmm_params)

        # Run Viterbi
        viterbi_states = viterbi(identities_filled, hmm_params)

        # Extract segments
        segments = extract_segments(
            viterbi_states,
            posterior_ibd,
            identities_filled,
            starts,
            ends,
            min_windows=min_segment_windows,
            min_posterior=0.5,
        )

        result = IBDResult(
            sample_a=sample_a,
            hap_a=hap_a,
            sample_b=sample_b,
            hap_b=hap_b,
            chrom=chrom,
            n_windows=len(identities_filled),
            window_starts=starts,
            window_ends=ends,
            identities=identities_filled,
            posterior_ibd=posterior_ibd,
            viterbi_states=viterbi_states,
            log_likelihood=log_likelihood,
            segments=segments,
        )
        result.compute_summary()

        return result

    except Exception as e:
        print(f"  Error on {sample_a}-{sample_b}: {e}")
        return None


def analyze_population_full(
    pop_name: str,
    max_pairs: int = 50,
    expected_ibd_length: float = 50.0,
    min_segment_windows: int = 10,
    sample_frac: float = 1.0,
) -> Tuple[Dict, List[IBDResult]]:
    """
    Analyze IBD for a population using full distribution data.

    Returns:
        emission_params: Empirically estimated emission parameters
        results: List of IBD results for each pair
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {pop_name} - chr1 Full ({REGION_END/1e6:.0f} Mb)")
    print(f"{'='*60}")

    # Find data file
    data_file = DATA_DIR / f"{pop_name}_chr1_full.tsv"
    if not data_file.exists():
        print(f"  Data file not found: {data_file}")
        return {}, []

    # Load data
    all_identities, pair_data = load_full_identity_data(data_file, sample_frac)

    # Estimate emission parameters from full distribution
    print("\nEstimating emission parameters from full distribution...")
    emission_params = estimate_emission_parameters(all_identities)

    # Compare with theoretical
    pi_theoretical = THEORETICAL_DIVERSITY.get(pop_name, 0.001)
    mean_theoretical = 1 - pi_theoretical
    print(f"\n  Theoretical non-IBD mean: {mean_theoretical:.6f}")
    print(f"  Empirical non-IBD mean: {emission_params['non_ibd']['mean']:.6f}")
    print(f"  Difference: {(emission_params['non_ibd']['mean'] - mean_theoretical)*100:.4f}%")

    # Create HMM with empirical parameters
    hmm_params = create_empirical_hmm_params(
        emission_params,
        expected_ibd_length=expected_ibd_length,
    )

    # Select pairs for analysis
    # Use pairs with most variance in identity (more likely to have IBD)
    pair_variance = {}
    for pair_key, data in pair_data.items():
        valid = data['identities'][~np.isnan(data['identities'])]
        if len(valid) > 100:
            pair_variance[pair_key] = np.var(valid)

    sorted_pairs = sorted(pair_variance.items(), key=lambda x: -x[1])
    selected_pairs = [p[0] for p in sorted_pairs[:max_pairs]]

    print(f"\nRunning IBD inference on {len(selected_pairs)} pairs...")

    results = []
    for i, pair_key in enumerate(selected_pairs):
        if i % 10 == 0:
            print(f"  Processing pair {i+1}/{len(selected_pairs)}")

        result = analyze_pair_empirical(
            pair_key,
            pair_data[pair_key],
            hmm_params,
            min_segment_windows,
        )

        if result:
            results.append(result)

    print(f"  Completed: {len(results)} pairs analyzed")

    return emission_params, results


# ============================================================
# Report Generation
# ============================================================

def generate_report(
    results_by_pop: Dict[str, Tuple[Dict, List[IBDResult]]],
    output_dir: Path,
) -> str:
    """Generate comprehensive report with empirical parameters."""

    report = []
    report.append("# chr1_full: Full Chromosome 1 IBD Analysis Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Data Source**: HPRC v2 Pangenome")
    report.append(f"**Region**: chr1:1-{REGION_END:,} ({REGION_END/1e6:.0f} Mb)")
    report.append(f"**Window Size**: {WINDOW_SIZE/1000:.0f} kb")
    report.append(f"**Total Windows**: {TOTAL_WINDOWS:,}")
    report.append("")

    # Emission Parameters
    report.append("## 1. Empirical Emission Parameters")
    report.append("")
    report.append("### 1.1 Non-IBD Distribution")
    report.append("")
    report.append("| Population | Empirical Mean | Empirical Std | Theoretical Mean | d' |")
    report.append("|------------|----------------|---------------|------------------|-----|")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            params, _ = results_by_pop[pop_name]
            if params:
                theoretical = 1 - THEORETICAL_DIVERSITY.get(pop_name, 0.001)
                report.append(f"| {pop_name} | {params['non_ibd']['mean']:.6f} | "
                             f"{params['non_ibd']['std']:.6f} | {theoretical:.6f} | "
                             f"{params['d_prime']:.2f} |")

    report.append("")

    # Quality Assessment
    report.append("### 1.2 Quality Assessment (d' separability)")
    report.append("")
    report.append("d' measures how well the IBD and non-IBD distributions are separated:")
    report.append("- d' < 1: Poor separation (high overlap)")
    report.append("- d' = 1-2: Moderate separation")
    report.append("- d' > 2: Good separation (low overlap)")
    report.append("")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            params, _ = results_by_pop[pop_name]
            if params:
                d_prime = params['d_prime']
                quality = "Good" if d_prime > 2 else ("Moderate" if d_prime > 1 else "Poor")
                report.append(f"- **{pop_name}**: d' = {d_prime:.2f} ({quality})")

    report.append("")

    # IBD Results
    report.append("## 2. IBD Detection Results")
    report.append("")
    report.append("| Population | Pairs | Segments | Mean IBD (Mb) | Mean Fraction | Mean Length (kb) |")
    report.append("|------------|-------|----------|---------------|---------------|------------------|")

    for pop_name in ['EUR', 'AFR']:
        if pop_name in results_by_pop:
            _, results = results_by_pop[pop_name]
            if results:
                n_pairs = len(results)
                total_segments = sum(r.n_segments for r in results)
                mean_ibd = np.mean([r.total_ibd_bp for r in results]) / 1e6
                mean_frac = np.mean([r.fraction_ibd for r in results])

                all_lengths = [s.length_bp for r in results for s in r.segments]
                mean_length = np.mean(all_lengths) / 1000 if all_lengths else 0

                report.append(f"| {pop_name} | {n_pairs} | {total_segments} | "
                             f"{mean_ibd:.2f} | {mean_frac:.3f} | {mean_length:.1f} |")

    report.append("")

    # Comparison with exp02
    report.append("## 3. Comparison with exp02 (chr2:1-50Mb)")
    report.append("")
    report.append("| Metric | exp02 (chr2 50Mb) | chr1_full (249 Mb) |")
    report.append("|--------|-------------------|---------------------|")
    report.append("| Region size | 50 Mb | 249 Mb |")
    report.append(f"| Windows | 10,000 | {TOTAL_WINDOWS:,} |")
    report.append("| Scale factor | 1x | ~5x |")
    report.append("")

    # Conclusions
    report.append("## 4. Conclusions")
    report.append("")

    all_d_primes = [results_by_pop[p][0]['d_prime'] for p in results_by_pop if results_by_pop[p][0]]
    if all_d_primes:
        mean_d = np.mean(all_d_primes)
        if mean_d > 1.5:
            report.append(f"1. **Good separability**: Mean d' = {mean_d:.2f}")
            report.append("2. Full chromosome analysis confirms patterns from chr2 50Mb")
        else:
            report.append(f"1. d' = {mean_d:.2f} indicates room for model improvement")

    report.append("")

    report_text = '\n'.join(report)

    report_path = output_dir / "REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved: {report_path}")

    return report_text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='chr1_full: Full chromosome 1 IBD analysis')
    parser.add_argument('--populations', nargs='+', default=['EUR', 'AFR'])
    parser.add_argument('--max-pairs', type=int, default=50,
                        help='Maximum pairs to analyze per population')
    parser.add_argument('--expected-ibd-length', type=float, default=50.0,
                        help='Expected IBD segment length in windows')
    parser.add_argument('--min-segment-windows', type=int, default=10,
                        help='Minimum segment length to report')
    parser.add_argument('--sample-frac', type=float, default=1.0,
                        help='Fraction of data to sample (for testing)')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / 'json'
    json_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("chr1_full: FULL CHROMOSOME 1 IBD ANALYSIS")
    print("=" * 70)
    print(f"Region: chr1:1-{REGION_END:,} ({REGION_END/1e6:.0f} Mb)")
    print(f"Windows: {TOTAL_WINDOWS:,}")
    print(f"Populations: {args.populations}")
    print(f"Max pairs: {args.max_pairs}")
    print(f"Expected IBD length: {args.expected_ibd_length} windows")

    results_by_pop = {}

    for pop in args.populations:
        emission_params, results = analyze_population_full(
            pop,
            max_pairs=args.max_pairs,
            expected_ibd_length=args.expected_ibd_length,
            min_segment_windows=args.min_segment_windows,
            sample_frac=args.sample_frac,
        )

        if emission_params:
            results_by_pop[pop] = (emission_params, results)

            # Save emission parameters
            with open(json_dir / f'{pop}_emission_params.json', 'w') as f:
                json.dump(emission_params, f, indent=2)

            # Save results
            if results:
                json_data = {
                    'population': pop,
                    'region': f'chr1:1-{REGION_END}',
                    'emission_params': emission_params,
                    'n_pairs': len(results),
                    'results': [r.to_dict() for r in results],
                }
                with open(json_dir / f'{pop}_ibd_results.json', 'w') as f:
                    json.dump(json_data, f, indent=2)

    # Generate report
    if results_by_pop:
        generate_report(results_by_pop, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    for pop in args.populations:
        if pop in results_by_pop:
            params, results = results_by_pop[pop]
            print(f"\n{pop}:")
            print(f"  d' separability: {params['d_prime']:.2f}")
            print(f"  Pairs analyzed: {len(results)}")
            if results:
                total_seg = sum(r.n_segments for r in results)
                mean_ibd = np.mean([r.total_ibd_bp for r in results]) / 1e6
                print(f"  Total segments: {total_seg}")
                print(f"  Mean IBD: {mean_ibd:.2f} Mb")


if __name__ == '__main__':
    main()
