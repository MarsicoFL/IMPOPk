#!/usr/bin/env python3
"""
IBD Inference Module for Pangenome Data

This module implements Hidden Markov Model-based IBD detection with:
- Forward-backward algorithm for posterior probabilities
- Viterbi algorithm for MAP state sequence
- Population-specific emission parameters
- Comprehensive segment statistics

Author: IBD-CLI Project
Date: 2026-01
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json


class Population(Enum):
    """Human populations with associated genetic diversity parameters."""
    AFR = "AFR"  # African
    EUR = "EUR"  # European
    EAS = "EAS"  # East Asian
    CSA = "CSA"  # Central/South Asian
    AMR = "AMR"  # American (admixed)
    INTER = "INTER"  # Inter-population comparison


# Population-specific nucleotide diversity (π) from 1000 Genomes
POPULATION_DIVERSITY = {
    Population.AFR: 0.00125,
    Population.EUR: 0.00085,
    Population.EAS: 0.00080,
    Population.CSA: 0.00095,
    Population.AMR: 0.00100,
    Population.INTER: 0.00110,
}


@dataclass
class GaussianParams:
    """Parameters for Gaussian emission distribution."""
    mean: float
    std: float

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density."""
        z = (x - self.mean) / self.std
        return -0.5 * z * z - np.log(self.std) - 0.5 * np.log(2 * np.pi)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density."""
        return np.exp(self.log_pdf(x))


@dataclass
class HMMParams:
    """Parameters for two-state IBD Hidden Markov Model."""
    # Emission distributions
    emission_non_ibd: GaussianParams
    emission_ibd: GaussianParams

    # Transition probabilities
    p_enter_ibd: float  # P(IBD | non-IBD)
    p_exit_ibd: float   # P(non-IBD | IBD)

    # Initial probabilities
    p_initial_ibd: float = 0.01

    @classmethod
    def from_population(
        cls,
        population: Population,
        expected_ibd_length: float = 50.0,  # windows
        p_enter_ibd: float = 0.0001,
    ) -> 'HMMParams':
        """
        Create HMM parameters for a specific population.

        Args:
            population: Population for non-IBD emission parameters
            expected_ibd_length: Expected IBD segment length in windows
            p_enter_ibd: Probability of transitioning into IBD state
        """
        # Non-IBD emission from population diversity
        pi = POPULATION_DIVERSITY.get(population, 0.001)
        mean_non_ibd = 1.0 - pi

        # Std derived from Poisson variance with LD correction
        window_size = 5000.0
        ld_correction = 3.0
        std_non_ibd = np.sqrt(pi / window_size * ld_correction)

        # IBD emission (based on Browning error rate ε ≈ 0.0003)
        mean_ibd = 0.9997
        std_ibd = 0.0005

        # Exit probability from expected length
        p_exit_ibd = 1.0 / expected_ibd_length
        p_exit_ibd = np.clip(p_exit_ibd, 0.0001, 0.5)

        return cls(
            emission_non_ibd=GaussianParams(mean_non_ibd, std_non_ibd),
            emission_ibd=GaussianParams(mean_ibd, std_ibd),
            p_enter_ibd=p_enter_ibd,
            p_exit_ibd=p_exit_ibd,
        )

    @property
    def log_transition(self) -> np.ndarray:
        """Log transition matrix [from_state, to_state]."""
        return np.array([
            [np.log(1 - self.p_enter_ibd), np.log(self.p_enter_ibd)],
            [np.log(self.p_exit_ibd), np.log(1 - self.p_exit_ibd)],
        ])

    @property
    def log_initial(self) -> np.ndarray:
        """Log initial state probabilities."""
        return np.array([
            np.log(1 - self.p_initial_ibd),
            np.log(self.p_initial_ibd),
        ])


@dataclass
class IBDSegment:
    """Detected IBD segment with statistics."""
    start_idx: int      # Start window index
    end_idx: int        # End window index (inclusive)
    start_bp: int       # Start position in base pairs
    end_bp: int         # End position in base pairs
    n_windows: int      # Number of windows
    length_bp: int      # Length in base pairs
    mean_identity: float
    mean_posterior: float  # Mean P(IBD) in segment
    max_posterior: float   # Max P(IBD) in segment
    min_posterior: float   # Min P(IBD) in segment

    def to_dict(self) -> dict:
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_bp': self.start_bp,
            'end_bp': self.end_bp,
            'n_windows': self.n_windows,
            'length_bp': self.length_bp,
            'mean_identity': self.mean_identity,
            'mean_posterior': self.mean_posterior,
            'max_posterior': self.max_posterior,
            'min_posterior': self.min_posterior,
        }


@dataclass
class IBDResult:
    """Complete IBD inference result for a haplotype pair."""
    sample_a: str
    hap_a: int
    sample_b: str
    hap_b: int
    chrom: str

    # Window information
    n_windows: int
    window_starts: np.ndarray
    window_ends: np.ndarray
    identities: np.ndarray

    # HMM results
    posterior_ibd: np.ndarray      # P(IBD) for each window
    viterbi_states: np.ndarray     # MAP state sequence
    log_likelihood: float          # Total log-likelihood

    # Detected segments
    segments: List[IBDSegment] = field(default_factory=list)

    # Summary statistics
    total_ibd_bp: int = 0
    fraction_ibd: float = 0.0
    n_segments: int = 0
    mean_segment_length: float = 0.0

    def compute_summary(self):
        """Compute summary statistics from segments."""
        if self.segments:
            self.n_segments = len(self.segments)
            self.total_ibd_bp = sum(s.length_bp for s in self.segments)
            total_bp = self.window_ends[-1] - self.window_starts[0]
            self.fraction_ibd = self.total_ibd_bp / total_bp if total_bp > 0 else 0
            self.mean_segment_length = self.total_ibd_bp / self.n_segments

    def to_dict(self) -> dict:
        return {
            'sample_a': self.sample_a,
            'hap_a': self.hap_a,
            'sample_b': self.sample_b,
            'hap_b': self.hap_b,
            'chrom': self.chrom,
            'n_windows': self.n_windows,
            'log_likelihood': self.log_likelihood,
            'n_segments': self.n_segments,
            'total_ibd_bp': self.total_ibd_bp,
            'fraction_ibd': self.fraction_ibd,
            'mean_segment_length': self.mean_segment_length,
            'segments': [s.to_dict() for s in self.segments],
        }


def forward_algorithm(
    observations: np.ndarray,
    params: HMMParams,
) -> Tuple[np.ndarray, float]:
    """
    Forward algorithm for HMM.

    Args:
        observations: Identity values (n_windows,)
        params: HMM parameters

    Returns:
        alpha: Forward probabilities (n_windows, 2) in log space
        log_likelihood: Total log-likelihood
    """
    n = len(observations)
    alpha = np.zeros((n, 2))

    # Log emission probabilities
    log_emit = np.zeros((n, 2))
    log_emit[:, 0] = params.emission_non_ibd.log_pdf(observations)
    log_emit[:, 1] = params.emission_ibd.log_pdf(observations)

    log_trans = params.log_transition
    log_init = params.log_initial

    # Initialization
    alpha[0] = log_init + log_emit[0]

    # Forward pass
    for t in range(1, n):
        for s in range(2):
            # log-sum-exp over previous states
            log_probs = alpha[t-1] + log_trans[:, s]
            max_log = np.max(log_probs)
            alpha[t, s] = max_log + np.log(np.sum(np.exp(log_probs - max_log)))
            alpha[t, s] += log_emit[t, s]

    # Total log-likelihood
    max_log = np.max(alpha[-1])
    log_likelihood = max_log + np.log(np.sum(np.exp(alpha[-1] - max_log)))

    return alpha, log_likelihood


def backward_algorithm(
    observations: np.ndarray,
    params: HMMParams,
) -> np.ndarray:
    """
    Backward algorithm for HMM.

    Args:
        observations: Identity values (n_windows,)
        params: HMM parameters

    Returns:
        beta: Backward probabilities (n_windows, 2) in log space
    """
    n = len(observations)
    beta = np.zeros((n, 2))

    # Log emission probabilities
    log_emit = np.zeros((n, 2))
    log_emit[:, 0] = params.emission_non_ibd.log_pdf(observations)
    log_emit[:, 1] = params.emission_ibd.log_pdf(observations)

    log_trans = params.log_transition

    # Initialization (beta[n-1] = 0 in log space, i.e., prob = 1)
    beta[-1] = 0.0

    # Backward pass
    for t in range(n - 2, -1, -1):
        for s in range(2):
            # log-sum-exp over next states
            log_probs = log_trans[s, :] + log_emit[t+1] + beta[t+1]
            max_log = np.max(log_probs)
            beta[t, s] = max_log + np.log(np.sum(np.exp(log_probs - max_log)))

    return beta


def forward_backward(
    observations: np.ndarray,
    params: HMMParams,
) -> Tuple[np.ndarray, float]:
    """
    Forward-backward algorithm to compute posterior state probabilities.

    Args:
        observations: Identity values (n_windows,)
        params: HMM parameters

    Returns:
        posterior: P(state=1|observations) for each position
        log_likelihood: Total log-likelihood
    """
    alpha, log_likelihood = forward_algorithm(observations, params)
    beta = backward_algorithm(observations, params)

    # Posterior = alpha * beta / P(observations)
    # In log space: log_posterior = alpha + beta - log_likelihood
    log_gamma = alpha + beta - log_likelihood

    # Convert to probabilities
    # P(state=1) = exp(log_gamma[:, 1]) / sum(exp(log_gamma))
    posterior_ibd = np.exp(log_gamma[:, 1] - np.logaddexp(log_gamma[:, 0], log_gamma[:, 1]))

    return posterior_ibd, log_likelihood


def viterbi(
    observations: np.ndarray,
    params: HMMParams,
) -> np.ndarray:
    """
    Viterbi algorithm for MAP state sequence.

    Args:
        observations: Identity values (n_windows,)
        params: HMM parameters

    Returns:
        states: Most likely state sequence (0=non-IBD, 1=IBD)
    """
    n = len(observations)
    if n == 0:
        return np.array([], dtype=int)

    # Log emission probabilities
    log_emit = np.zeros((n, 2))
    log_emit[:, 0] = params.emission_non_ibd.log_pdf(observations)
    log_emit[:, 1] = params.emission_ibd.log_pdf(observations)

    log_trans = params.log_transition
    log_init = params.log_initial

    # Viterbi tables
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)

    # Initialization
    delta[0] = log_init + log_emit[0]

    # Forward pass
    for t in range(1, n):
        for s in range(2):
            scores = delta[t-1] + log_trans[:, s]
            psi[t, s] = np.argmax(scores)
            delta[t, s] = scores[psi[t, s]] + log_emit[t, s]

    # Backtracking
    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(delta[-1])

    for t in range(n - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def extract_segments(
    states: np.ndarray,
    posterior: np.ndarray,
    identities: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    min_windows: int = 5,
    min_posterior: float = 0.5,
) -> List[IBDSegment]:
    """
    Extract IBD segments from state sequence.

    Args:
        states: Viterbi state sequence
        posterior: Posterior P(IBD) values
        identities: Original identity values
        window_starts: Start positions of windows
        window_ends: End positions of windows
        min_windows: Minimum segment length in windows
        min_posterior: Minimum mean posterior for segment

    Returns:
        List of IBD segments
    """
    segments = []
    n = len(states)

    if n == 0:
        return segments

    in_segment = False
    start_idx = 0

    for i in range(n):
        if states[i] == 1 and not in_segment:
            in_segment = True
            start_idx = i
        elif states[i] == 0 and in_segment:
            in_segment = False
            # Check segment
            end_idx = i - 1
            if end_idx - start_idx + 1 >= min_windows:
                seg_posterior = posterior[start_idx:end_idx+1]
                if np.mean(seg_posterior) >= min_posterior:
                    segments.append(IBDSegment(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        start_bp=int(window_starts[start_idx]),
                        end_bp=int(window_ends[end_idx]),
                        n_windows=end_idx - start_idx + 1,
                        length_bp=int(window_ends[end_idx] - window_starts[start_idx]),
                        mean_identity=float(np.mean(identities[start_idx:end_idx+1])),
                        mean_posterior=float(np.mean(seg_posterior)),
                        max_posterior=float(np.max(seg_posterior)),
                        min_posterior=float(np.min(seg_posterior)),
                    ))

    # Handle segment at end
    if in_segment:
        end_idx = n - 1
        if end_idx - start_idx + 1 >= min_windows:
            seg_posterior = posterior[start_idx:end_idx+1]
            if np.mean(seg_posterior) >= min_posterior:
                segments.append(IBDSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_bp=int(window_starts[start_idx]),
                    end_bp=int(window_ends[end_idx]),
                    n_windows=end_idx - start_idx + 1,
                    length_bp=int(window_ends[end_idx] - window_starts[start_idx]),
                    mean_identity=float(np.mean(identities[start_idx:end_idx+1])),
                    mean_posterior=float(np.mean(seg_posterior)),
                    max_posterior=float(np.max(seg_posterior)),
                    min_posterior=float(np.min(seg_posterior)),
                ))

    return segments


def infer_ibd(
    identities: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    sample_a: str,
    hap_a: int,
    sample_b: str,
    hap_b: int,
    chrom: str,
    population: Population = Population.EUR,
    expected_ibd_length: float = 50.0,
    min_segment_windows: int = 5,
) -> IBDResult:
    """
    Complete IBD inference for a haplotype pair.

    Args:
        identities: Identity values per window
        window_starts: Start positions of windows
        window_ends: End positions of windows
        sample_a, hap_a: First haplotype identifier
        sample_b, hap_b: Second haplotype identifier
        chrom: Chromosome name
        population: Population for parameter estimation
        expected_ibd_length: Expected IBD segment length (windows)
        min_segment_windows: Minimum segment length to report

    Returns:
        IBDResult with posteriors, states, and segments
    """
    # Create HMM parameters
    params = HMMParams.from_population(population, expected_ibd_length)

    # Run forward-backward for posteriors
    posterior_ibd, log_likelihood = forward_backward(identities, params)

    # Run Viterbi for MAP states
    viterbi_states = viterbi(identities, params)

    # Extract segments
    segments = extract_segments(
        viterbi_states,
        posterior_ibd,
        identities,
        window_starts,
        window_ends,
        min_windows=min_segment_windows,
    )

    # Create result
    result = IBDResult(
        sample_a=sample_a,
        hap_a=hap_a,
        sample_b=sample_b,
        hap_b=hap_b,
        chrom=chrom,
        n_windows=len(identities),
        window_starts=window_starts,
        window_ends=window_ends,
        identities=identities,
        posterior_ibd=posterior_ibd,
        viterbi_states=viterbi_states,
        log_likelihood=log_likelihood,
        segments=segments,
    )

    result.compute_summary()

    return result


# ============================================================
# Testing and validation
# ============================================================

def test_inference():
    """Test IBD inference on synthetic data."""
    np.random.seed(42)

    # Generate synthetic data
    n_windows = 500
    window_size = 5000

    # True IBD regions: windows 100-150 and 300-380
    true_ibd = np.zeros(n_windows, dtype=bool)
    true_ibd[100:151] = True
    true_ibd[300:381] = True

    # Generate identities
    identities = np.zeros(n_windows)
    for i in range(n_windows):
        if true_ibd[i]:
            identities[i] = np.random.normal(0.9997, 0.0005)
        else:
            identities[i] = np.random.normal(0.99875, 0.00087)  # AFR params

    identities = np.clip(identities, 0.95, 1.0)

    # Window positions
    window_starts = np.arange(n_windows) * window_size
    window_ends = window_starts + window_size - 1

    # Run inference
    result = infer_ibd(
        identities=identities,
        window_starts=window_starts,
        window_ends=window_ends,
        sample_a="TEST1",
        hap_a=1,
        sample_b="TEST2",
        hap_b=1,
        chrom="chr20",
        population=Population.AFR,
    )

    print("=== IBD Inference Test ===")
    print(f"Windows: {result.n_windows}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Segments found: {result.n_segments}")
    print(f"Total IBD: {result.total_ibd_bp / 1e6:.2f} Mb")
    print(f"Fraction IBD: {result.fraction_ibd:.3f}")

    # Check detection accuracy
    detected_ibd = result.viterbi_states == 1
    tp = np.sum(detected_ibd & true_ibd)
    fp = np.sum(detected_ibd & ~true_ibd)
    fn = np.sum(~detected_ibd & true_ibd)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetection accuracy:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")

    print("\nSegments:")
    for seg in result.segments:
        print(f"  {seg.start_idx}-{seg.end_idx}: {seg.length_bp/1000:.1f} kb, "
              f"mean_post={seg.mean_posterior:.3f}")

    return result


if __name__ == '__main__':
    test_inference()
