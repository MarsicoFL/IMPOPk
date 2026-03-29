//! Multi-pulse demographic inference from ancestry tract lengths (T80)
//!
//! Fits a mixture of shifted exponentials to tract length distributions
//! via EM, with BIC model selection for number of pulses.
//! Single-pulse falls back to the closed-form estimator from T77.

use crate::AncestrySegment;

/// Result of demographic inference for one population
#[derive(Debug, Clone)]
pub struct PulseEstimate {
    /// Admixture time in generations
    pub generations: f64,
    /// Mixing proportion (fraction of tracts from this pulse)
    pub proportion: f64,
    /// Rate parameter λ = g × r
    pub rate: f64,
    /// Coefficient of variation for generation estimate: CV = 1/√(n_k × π_m)
    pub cv: f64,
}

/// Full demographic inference result for one population
#[derive(Debug, Clone)]
pub struct DemographicResult {
    /// Population name
    pub population: String,
    /// Number of tracts used
    pub n_tracts: usize,
    /// Number of pulses selected by BIC
    pub n_pulses: usize,
    /// Pulse estimates (sorted by generation, most recent first)
    pub pulses: Vec<PulseEstimate>,
    /// BIC value for the selected model
    pub bic: f64,
    /// Log-likelihood of the selected model
    pub log_likelihood: f64,
    /// KS statistic against single-exponential (if multi-pulse tested)
    pub ks_statistic: Option<f64>,
    /// Whether single-pulse was rejected by KS test
    pub single_pulse_rejected: bool,
    /// Warning if tract count is below minimum for reliable inference
    pub low_tract_warning: Option<String>,
}

/// Parameters for demographic inference
#[derive(Debug, Clone)]
pub struct DemographyParams {
    /// Recombination rate per bp per generation (default: 1e-8)
    pub recomb_rate: f64,
    /// Minimum tract length (L_min) — set to effective HMM detection limit
    pub l_min_bp: f64,
    /// Maximum number of pulses to test (default: 3)
    pub max_pulses: usize,
    /// EM convergence tolerance (default: 1e-8)
    pub em_tolerance: f64,
    /// Maximum EM iterations (default: 200)
    pub max_em_iters: usize,
    /// KS test significance level (default: 0.05)
    pub ks_alpha: f64,
    /// BW transition constraint: if Some(lambda_bw), constrain EM so that
    /// Σ π_m λ_m = lambda_bw (T80 §5.2). The value is p_switch / W_bp.
    pub bw_constraint: Option<f64>,
}

impl Default for DemographyParams {
    fn default() -> Self {
        Self {
            recomb_rate: 1e-8,
            l_min_bp: 20_000.0, // 2 windows × 10kb
            max_pulses: 3,
            em_tolerance: 1e-8,
            max_em_iters: 200,
            ks_alpha: 0.05,
            bw_constraint: None,
        }
    }
}

/// Extract tract lengths per population from ancestry segments
pub fn extract_tract_lengths(
    segments: &[AncestrySegment],
) -> std::collections::HashMap<String, Vec<f64>> {
    let mut tracts: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    for seg in segments {
        let length = (seg.end - seg.start) as f64;
        tracts.entry(seg.ancestry_name.clone()).or_default().push(length);
    }
    tracts
}

/// Log-likelihood of a single-component shifted exponential
fn log_likelihood_single(tracts: &[f64], l_min: f64) -> (f64, f64) {
    let n = tracts.len() as f64;
    let sum_excess: f64 = tracts.iter().map(|&l| l - l_min).sum();
    let lambda = n / sum_excess;
    let ll = n * lambda.ln() - lambda * sum_excess;
    (ll, lambda)
}

/// Log-likelihood of a mixture model at given parameters
fn log_likelihood_mixture(
    tracts: &[f64],
    l_min: f64,
    proportions: &[f64],
    rates: &[f64],
) -> f64 {
    tracts.iter().map(|&l| {
        let excess = l - l_min;
        let sum: f64 = proportions.iter().zip(rates.iter())
            .map(|(&pi, &lam)| pi * lam * (-lam * excess).exp())
            .sum();
        if sum > 0.0 { sum.ln() } else { f64::NEG_INFINITY }
    }).sum()
}

/// EM algorithm for exponential mixture on tract lengths
///
/// Returns (proportions, rates, log_likelihood, iterations)
fn em_exponential_mixture(
    tracts: &[f64],
    l_min: f64,
    m: usize,
    params: &DemographyParams,
) -> (Vec<f64>, Vec<f64>, f64, usize) {
    let n = tracts.len();
    if n == 0 || m == 0 {
        return (vec![], vec![], f64::NEG_INFINITY, 0);
    }

    // Quantile-based initialization (T80 §2.4)
    let mut sorted: Vec<f64> = tracts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut rates = Vec::with_capacity(m);
    let mut proportions = vec![1.0 / m as f64; m];

    for comp in 0..m {
        let lo = (n as f64 * comp as f64 / (m + 1) as f64) as usize;
        let hi = (n as f64 * (comp + 1) as f64 / (m + 1) as f64) as usize;
        let hi = hi.min(n);
        let lo = lo.min(hi.saturating_sub(1));
        let partition = &sorted[lo..hi];
        if partition.is_empty() {
            rates.push(1.0 / (sorted[n / 2] - l_min).max(1.0));
        } else {
            let sum_excess: f64 = partition.iter().map(|&l| (l - l_min).max(1.0)).sum();
            rates.push(partition.len() as f64 / sum_excess);
        }
    }

    // Ensure rates are distinct and positive
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for i in 1..rates.len() {
        if rates[i] <= rates[i - 1] {
            rates[i] = rates[i - 1] * 1.5;
        }
    }

    let mut responsibilities = vec![vec![0.0; m]; n];
    let mut prev_ll = f64::NEG_INFINITY;

    for iter in 0..params.max_em_iters {
        // E-step: compute responsibilities
        for (i, &l) in tracts.iter().enumerate() {
            let excess = l - l_min;
            let mut total = 0.0;
            for (j, (&pi, &lam)) in proportions.iter().zip(rates.iter()).enumerate() {
                let val = pi * lam * (-lam * excess).exp();
                responsibilities[i][j] = val;
                total += val;
            }
            if total > 0.0 {
                for val in responsibilities[i].iter_mut().take(m) {
                    *val /= total;
                }
            } else {
                // Uniform fallback
                for val in responsibilities[i].iter_mut().take(m) {
                    *val = 1.0 / m as f64;
                }
            }
        }

        // M-step: update proportions and rates
        for j in 0..m {
            let n_j: f64 = responsibilities.iter().map(|r| r[j]).sum();
            proportions[j] = (n_j / n as f64).max(1e-10);

            let weighted_excess: f64 = tracts.iter().enumerate()
                .map(|(i, &l)| responsibilities[i][j] * (l - l_min))
                .sum();

            if weighted_excess > 0.0 && n_j > 0.0 {
                rates[j] = n_j / weighted_excess;
            }
        }

        // Normalize proportions
        let prop_sum: f64 = proportions.iter().sum();
        for p in &mut proportions {
            *p /= prop_sum;
        }

        // BW constraint: Lagrange projection (T84) — minimize ||λ - λ*||²
        // subject to Σ πᵢλᵢ = λ_bw. Solution: λᵢ = λᵢ* + πᵢ × δ / Σ πⱼ²
        // where δ = λ_bw - Σ πⱼλⱼ*. Replaces uniform scaling (T80 §5.2)
        // which had ~5% rate bias for unequal proportions.
        if let Some(lambda_bw) = params.bw_constraint {
            if lambda_bw > 0.0 {
                let current_mean: f64 = proportions.iter()
                    .zip(rates.iter())
                    .map(|(&p, &r)| p * r)
                    .sum();
                let delta = lambda_bw - current_mean;
                let pi_sq_sum: f64 = proportions.iter().map(|&p| p * p).sum();
                if pi_sq_sum > 0.0 {
                    let nu = delta / pi_sq_sum;
                    for (r, &pi) in rates.iter_mut().zip(proportions.iter()) {
                        *r = (*r + nu * pi).max(1e-15);
                    }
                }
            }
        }

        // Sort components by rate (enforce ordering)
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| rates[a].partial_cmp(&rates[b]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_rates: Vec<f64> = indices.iter().map(|&i| rates[i]).collect();
        let sorted_props: Vec<f64> = indices.iter().map(|&i| proportions[i]).collect();
        rates = sorted_rates;
        proportions = sorted_props;

        // Check convergence
        let ll = log_likelihood_mixture(tracts, l_min, &proportions, &rates);
        if (ll - prev_ll).abs() < params.em_tolerance {
            return (proportions, rates, ll, iter + 1);
        }
        prev_ll = ll;
    }

    let ll = log_likelihood_mixture(tracts, l_min, &proportions, &rates);
    (proportions, rates, ll, params.max_em_iters)
}

/// BIC for model with k parameters and n data points
fn bic(log_likelihood: f64, n_params: usize, n_data: usize) -> f64 {
    -2.0 * log_likelihood + n_params as f64 * (n_data as f64).ln()
}

/// Kolmogorov-Smirnov test: compare tract lengths against single exponential
/// Returns (KS statistic, critical value at given alpha)
pub fn ks_test_exponential(tracts: &[f64], l_min: f64, alpha: f64) -> (f64, f64) {
    let n = tracts.len();
    if n == 0 {
        return (0.0, 1.0);
    }

    // Fit single exponential
    let sum_excess: f64 = tracts.iter().map(|&l| (l - l_min).max(0.0)).sum();
    let lambda = n as f64 / sum_excess;

    // Sort excess lengths
    let mut excess: Vec<f64> = tracts.iter().map(|&l| (l - l_min).max(0.0)).collect();
    excess.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // KS statistic: max |F_n(x) - F(x)|
    let mut d_max = 0.0_f64;
    for (i, &x) in excess.iter().enumerate() {
        let f_empirical = (i + 1) as f64 / n as f64;
        let f_theoretical = 1.0 - (-lambda * x).exp();
        let d1 = (f_empirical - f_theoretical).abs();
        let d2 = ((i as f64 / n as f64) - f_theoretical).abs();
        d_max = d_max.max(d1).max(d2);
    }

    // Critical value (Lilliefors-corrected for estimated parameter)
    // Using the standard approximation: c_alpha / sqrt(n)
    let c_alpha = if alpha <= 0.01 {
        1.63
    } else if alpha <= 0.05 {
        1.36
    } else {
        1.22
    };
    let critical = c_alpha / (n as f64).sqrt();

    (d_max, critical)
}

/// Run demographic inference for one population's tract lengths
pub fn infer_demography(
    tracts: &[f64],
    population: &str,
    params: &DemographyParams,
) -> DemographicResult {
    let n = tracts.len();

    // Not enough data for any inference
    if n < 3 {
        return DemographicResult {
            population: population.to_string(),
            n_tracts: n,
            n_pulses: 0,
            pulses: vec![],
            bic: f64::INFINITY,
            log_likelihood: f64::NEG_INFINITY,
            ks_statistic: None,
            single_pulse_rejected: false,
            low_tract_warning: if n == 0 { None } else { Some(format!("Only {} tracts, minimum 3 required", n)) },
        };
    }

    let l_min = params.l_min_bp;

    // Filter tracts that are too short (below detection limit)
    let valid_tracts: Vec<f64> = tracts.iter()
        .copied()
        .filter(|&l| l > l_min)
        .collect();

    if valid_tracts.len() < 3 {
        return DemographicResult {
            population: population.to_string(),
            n_tracts: valid_tracts.len(),
            n_pulses: 0,
            pulses: vec![],
            bic: f64::INFINITY,
            log_likelihood: f64::NEG_INFINITY,
            ks_statistic: None,
            single_pulse_rejected: false,
            low_tract_warning: Some(format!("Only {} valid tracts (above l_min), minimum 3 required", valid_tracts.len())),
        };
    }

    let n_valid = valid_tracts.len();

    // Fit M=1 (single pulse, closed-form)
    let (ll_1, lambda_1) = log_likelihood_single(&valid_tracts, l_min);
    let bic_1 = bic(ll_1, 1, n_valid); // 1 free parameter (lambda)

    // KS test: does single-exponential fit?
    let (ks_stat, ks_critical) = ks_test_exponential(&valid_tracts, l_min, params.ks_alpha);
    let single_rejected = ks_stat > ks_critical;

    let mut best_pulses = 1;
    let mut best_bic = bic_1;
    let mut best_ll = ll_1;
    let mut best_props = vec![1.0];
    let mut best_rates = vec![lambda_1];

    // Only try multi-pulse if single-pulse rejected and enough data
    if single_rejected && n_valid >= 20 {
        for m in 2..=params.max_pulses.min(3) {
            // Need enough tracts for reliable estimation
            if n_valid < 10 * m {
                break;
            }

            let (props, rates, ll, _iters) = em_exponential_mixture(
                &valid_tracts, l_min, m, params,
            );

            if ll.is_finite() {
                let bic_m = bic(ll, 2 * m - 1, n_valid);

                // Check rate separation (T80 §3.3: R > 1.5)
                let well_separated = if rates.len() >= 2 {
                    let r_min = rates.first().unwrap();
                    let r_max = rates.last().unwrap();
                    r_max / r_min > 1.5
                } else {
                    false
                };

                if bic_m < best_bic && well_separated {
                    best_pulses = m;
                    best_bic = bic_m;
                    best_ll = ll;
                    best_props = props;
                    best_rates = rates;
                }
            }
        }
    }

    // Convert rates to generation estimates
    let pulses: Vec<PulseEstimate> = best_rates.iter().zip(best_props.iter())
        .map(|(&rate, &prop)| {
            // CV = 1/√(n_k × π_m) where n_k is total tract count
            let cv = if n_valid > 0 && prop > 0.0 {
                1.0 / (n_valid as f64 * prop).sqrt()
            } else {
                f64::INFINITY
            };
            PulseEstimate {
                generations: rate / params.recomb_rate,
                proportion: prop,
                rate,
                cv,
            }
        })
        .collect();

    let low_tract_warning = if n_valid < 20 {
        Some(format!("{} tracts may be insufficient for reliable multi-pulse inference", n_valid))
    } else {
        None
    };

    DemographicResult {
        population: population.to_string(),
        n_tracts: n_valid,
        n_pulses: best_pulses,
        pulses,
        bic: best_bic,
        log_likelihood: best_ll,
        ks_statistic: Some(ks_stat),
        single_pulse_rejected: single_rejected,
        low_tract_warning,
    }
}

/// Run demographic inference for all populations across all samples
pub fn infer_all_demography(
    all_segments: &[&AncestrySegment],
    population_names: &[String],
    params: &DemographyParams,
) -> Vec<DemographicResult> {
    population_names.iter().map(|pop_name| {
        let tracts: Vec<f64> = all_segments.iter()
            .filter(|s| s.ancestry_name == *pop_name)
            .map(|s| (s.end - s.start) as f64)
            .collect();
        infer_demography(&tracts, pop_name, params)
    }).collect()
}

/// Format demographic inference results for stderr output
pub fn format_demography_report(results: &[DemographicResult]) -> String {
    let mut out = String::new();
    out.push_str("\n=== Demographic Inference (T80) ===\n");

    for result in results {
        if result.n_tracts < 3 {
            out.push_str(&format!("  {}: insufficient tracts (n={})\n",
                result.population, result.n_tracts));
            continue;
        }

        let ks_str = if let Some(ks) = result.ks_statistic {
            if result.single_pulse_rejected {
                format!(" [KS={:.3}, single-pulse REJECTED]", ks)
            } else {
                format!(" [KS={:.3}, single-pulse OK]", ks)
            }
        } else {
            String::new()
        };

        out.push_str(&format!("  {} (n={} tracts, {} pulse{}):{}\n",
            result.population,
            result.n_tracts,
            result.n_pulses,
            if result.n_pulses != 1 { "s" } else { "" },
            ks_str,
        ));

        for (i, pulse) in result.pulses.iter().enumerate() {
            out.push_str(&format!(
                "    Pulse {}: ĝ ≈ {:.0} generations ({:.1}% of tracts, λ={:.2e})\n",
                i + 1,
                pulse.generations,
                pulse.proportion * 100.0,
                pulse.rate,
            ));
        }

        out.push_str(&format!("    BIC={:.1}, LL={:.1}\n", result.bic, result.log_likelihood));
    }

    out
}

/// Per-sample demographic inference result
#[derive(Debug, Clone)]
pub struct SampleDemographicResult {
    /// Sample name (haplotype ID)
    pub sample: String,
    /// Per-population results
    pub results: Vec<DemographicResult>,
}

/// Run demographic inference per sample (not pooled).
///
/// Groups segments by sample, then runs EM per (sample, population) pair.
/// Returns one `SampleDemographicResult` per sample with all population results.
pub fn infer_per_sample_demography(
    all_segments: &[&AncestrySegment],
    population_names: &[String],
    params: &DemographyParams,
) -> Vec<SampleDemographicResult> {
    // Group segments by sample
    let mut by_sample: std::collections::BTreeMap<String, Vec<&AncestrySegment>> =
        std::collections::BTreeMap::new();
    for &seg in all_segments {
        by_sample.entry(seg.sample.clone()).or_default().push(seg);
    }

    by_sample.into_iter().map(|(sample, segments)| {
        let results = population_names.iter().map(|pop_name| {
            let tracts: Vec<f64> = segments.iter()
                .filter(|s| s.ancestry_name == *pop_name)
                .map(|s| (s.end - s.start) as f64)
                .collect();
            infer_demography(&tracts, pop_name, params)
        }).collect();

        SampleDemographicResult { sample, results }
    }).collect()
}

/// Write demographic inference results to a TSV file.
///
/// Outputs one row per (sample, population, pulse) with columns:
/// sample, population, n_tracts, n_pulses, pulse_idx, generations,
/// proportion, rate, bic, log_likelihood, ks_statistic, ks_rejected
///
/// If `per_sample` is empty, writes pooled results. If both are provided,
/// writes pooled first (sample="POOLED"), then per-sample.
pub fn write_demography_tsv(
    path: &std::path::Path,
    pooled: &[DemographicResult],
    per_sample: &[SampleDemographicResult],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

    writeln!(f, "sample\tpopulation\tn_tracts\tn_pulses\tpulse_idx\tgenerations\tproportion\trate\tbic\tlog_likelihood\tks_statistic\tks_rejected")?;

    // Helper closure to write rows for a set of results
    let write_results = |f: &mut std::io::BufWriter<std::fs::File>, sample: &str, results: &[DemographicResult]| -> std::io::Result<()> {
        for result in results {
            if result.pulses.is_empty() {
                // No pulses — write a single row with NA
                writeln!(f, "{}\t{}\t{}\t{}\tNA\tNA\tNA\tNA\t{}\t{}\t{}\t{}",
                    sample,
                    result.population,
                    result.n_tracts,
                    result.n_pulses,
                    if result.bic.is_finite() { format!("{:.1}", result.bic) } else { "NA".to_string() },
                    if result.log_likelihood.is_finite() { format!("{:.1}", result.log_likelihood) } else { "NA".to_string() },
                    result.ks_statistic.map_or("NA".to_string(), |v| format!("{:.4}", v)),
                    result.single_pulse_rejected,
                )?;
            } else {
                for (i, pulse) in result.pulses.iter().enumerate() {
                    writeln!(f, "{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.4}\t{:.2e}\t{:.1}\t{:.1}\t{}\t{}",
                        sample,
                        result.population,
                        result.n_tracts,
                        result.n_pulses,
                        i + 1,
                        pulse.generations,
                        pulse.proportion,
                        pulse.rate,
                        result.bic,
                        result.log_likelihood,
                        result.ks_statistic.map_or("NA".to_string(), |v| format!("{:.4}", v)),
                        result.single_pulse_rejected,
                    )?;
                }
            }
        }
        Ok(())
    };

    // Write pooled results first
    if !pooled.is_empty() {
        write_results(&mut f, "POOLED", pooled)?;
    }

    // Write per-sample results
    for sample_result in per_sample {
        write_results(&mut f, &sample_result.sample, &sample_result.results)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exponential_tracts(n: usize, lambda: f64, l_min: f64) -> Vec<f64> {
        // Deterministic pseudo-exponential: use quantile function
        // F^{-1}(p) = -ln(1-p)/lambda + l_min
        (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                -((1.0 - p).ln()) / lambda + l_min
            })
            .collect()
    }

    fn make_mixture_tracts(
        n1: usize, lambda1: f64,
        n2: usize, lambda2: f64,
        l_min: f64,
    ) -> Vec<f64> {
        let mut tracts = make_exponential_tracts(n1, lambda1, l_min);
        tracts.extend(make_exponential_tracts(n2, lambda2, l_min));
        tracts
    }

    #[test]
    fn test_single_pulse_mle() {
        let l_min = 20_000.0;
        let true_lambda = 1e-5; // corresponds to g=1000 at r=1e-8
        let tracts = make_exponential_tracts(200, true_lambda, l_min);

        let (ll, lambda) = log_likelihood_single(&tracts, l_min);
        assert!(ll.is_finite());
        // MLE should recover approximately the true rate
        let relative_error = (lambda - true_lambda).abs() / true_lambda;
        assert!(relative_error < 0.1, "lambda={}, true={}, err={}", lambda, true_lambda, relative_error);
    }

    #[test]
    fn test_em_single_component() {
        let l_min = 20_000.0;
        let true_lambda = 5e-6;
        let tracts = make_exponential_tracts(300, true_lambda, l_min);
        let params = DemographyParams::default();

        let (props, rates, ll, iters) = em_exponential_mixture(&tracts, l_min, 1, &params);
        assert_eq!(props.len(), 1);
        assert_eq!(rates.len(), 1);
        assert!(ll.is_finite());
        assert!(iters < params.max_em_iters);
        let relative_error = (rates[0] - true_lambda).abs() / true_lambda;
        assert!(relative_error < 0.1, "rate={}, true={}", rates[0], true_lambda);
    }

    #[test]
    fn test_em_two_components_well_separated() {
        let l_min = 20_000.0;
        // Component 1: recent admixture (g=50, lambda=5e-7)
        let lambda1 = 5e-7;
        // Component 2: ancient admixture (g=500, lambda=5e-6)
        let lambda2 = 5e-6;
        // Rate ratio = 10, well-separated

        let tracts = make_mixture_tracts(150, lambda1, 150, lambda2, l_min);
        let params = DemographyParams::default();

        let (props, rates, ll, _iters) = em_exponential_mixture(&tracts, l_min, 2, &params);
        assert_eq!(props.len(), 2);
        assert_eq!(rates.len(), 2);
        assert!(ll.is_finite());

        // Should recover roughly equal proportions
        assert!((props[0] - 0.5).abs() < 0.15, "prop[0]={}", props[0]);
        assert!((props[1] - 0.5).abs() < 0.15, "prop[1]={}", props[1]);

        // Should recover rates within 50% (mixture EM has wider tolerance)
        let rate_low = rates[0].min(rates[1]);
        let rate_high = rates[0].max(rates[1]);
        assert!(rate_low < lambda1 * 2.0 && rate_low > lambda1 * 0.5,
            "low rate {} not near true {}", rate_low, lambda1);
        assert!(rate_high < lambda2 * 2.0 && rate_high > lambda2 * 0.5,
            "high rate {} not near true {}", rate_high, lambda2);
    }

    #[test]
    fn test_ks_test_good_fit() {
        let l_min = 20_000.0;
        let lambda = 1e-5;
        let tracts = make_exponential_tracts(200, lambda, l_min);

        let (ks_stat, critical) = ks_test_exponential(&tracts, l_min, 0.05);
        // Perfect quantile data should pass the KS test
        assert!(ks_stat < critical, "KS={}, critical={}", ks_stat, critical);
    }

    #[test]
    fn test_ks_test_bad_fit() {
        let l_min = 20_000.0;
        // Mixture should fail single-exponential KS test
        let tracts = make_mixture_tracts(100, 1e-7, 100, 1e-5, l_min);

        let (ks_stat, critical) = ks_test_exponential(&tracts, l_min, 0.05);
        // Mixture data should reject single-exponential
        assert!(ks_stat > critical, "KS={}, critical={} (should reject)", ks_stat, critical);
    }

    #[test]
    fn test_bic_prefers_single_for_single_data() {
        let l_min = 20_000.0;
        let tracts = make_exponential_tracts(200, 1e-5, l_min);
        let params = DemographyParams::default();

        let result = infer_demography(&tracts, "TEST", &params);
        assert_eq!(result.n_pulses, 1, "BIC should prefer 1 pulse for single-exponential data");
    }

    #[test]
    fn test_infer_demography_minimum_tracts() {
        let params = DemographyParams::default();
        let result = infer_demography(&[30_000.0, 50_000.0], "TEST", &params);
        assert_eq!(result.n_tracts, 2);
        assert_eq!(result.n_pulses, 0); // Too few tracts
    }

    #[test]
    fn test_infer_demography_single_pulse() {
        let l_min = 20_000.0;
        let true_lambda = 2e-6; // g = 200 generations
        let tracts = make_exponential_tracts(100, true_lambda, l_min);
        let params = DemographyParams { l_min_bp: l_min, ..Default::default() };

        let result = infer_demography(&tracts, "EUR", &params);
        assert_eq!(result.n_pulses, 1);
        assert_eq!(result.pulses.len(), 1);

        let g = result.pulses[0].generations;
        let true_g = true_lambda / 1e-8;
        let rel_err = (g - true_g).abs() / true_g;
        assert!(rel_err < 0.15, "estimated g={}, true g={}", g, true_g);
    }

    #[test]
    fn test_extract_tract_lengths() {
        let segments = vec![
            AncestrySegment {
                chrom: "chr1".to_string(), start: 0, end: 100_000,
                sample: "S1".to_string(), ancestry_idx: 0, ancestry_name: "EUR".to_string(),
                n_windows: 10, mean_similarity: 0.95, mean_posterior: Some(0.9),
                discriminability: 0.1, lod_score: 5.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 100_000, end: 300_000,
                sample: "S1".to_string(), ancestry_idx: 1, ancestry_name: "AFR".to_string(),
                n_windows: 20, mean_similarity: 0.93, mean_posterior: Some(0.85),
                discriminability: 0.12, lod_score: 8.0,
            },
            AncestrySegment {
                chrom: "chr1".to_string(), start: 300_000, end: 500_000,
                sample: "S1".to_string(), ancestry_idx: 0, ancestry_name: "EUR".to_string(),
                n_windows: 20, mean_similarity: 0.96, mean_posterior: Some(0.92),
                discriminability: 0.15, lod_score: 10.0,
            },
        ];

        let tracts = extract_tract_lengths(&segments);
        assert_eq!(tracts["EUR"].len(), 2);
        assert_eq!(tracts["AFR"].len(), 1);
        assert_eq!(tracts["EUR"][0], 100_000.0);
        assert_eq!(tracts["EUR"][1], 200_000.0);
        assert_eq!(tracts["AFR"][0], 200_000.0);
    }

    #[test]
    fn test_format_demography_report() {
        let results = vec![
            DemographicResult {
                population: "EUR".to_string(),
                n_tracts: 100,
                n_pulses: 1,
                pulses: vec![PulseEstimate {
                    generations: 200.0, proportion: 1.0, rate: 2e-6, cv: 0.1,
                }],
                bic: -500.0,
                log_likelihood: -245.0,
                ks_statistic: Some(0.05),
                single_pulse_rejected: false,
                low_tract_warning: None,
            },
        ];

        let report = format_demography_report(&results);
        assert!(report.contains("EUR"));
        assert!(report.contains("200"));
        assert!(report.contains("1 pulse"));
        assert!(report.contains("KS=0.050"));
    }

    #[test]
    fn test_log_likelihood_mixture() {
        let l_min = 20_000.0;
        let tracts = make_exponential_tracts(50, 1e-5, l_min);
        let ll = log_likelihood_mixture(&tracts, l_min, &[1.0], &[1e-5]);
        assert!(ll.is_finite());
        assert!(ll < 0.0); // Log-likelihood should be negative
    }

    #[test]
    fn test_em_empty_tracts() {
        let params = DemographyParams::default();
        let (props, rates, ll, _) = em_exponential_mixture(&[], 20_000.0, 2, &params);
        assert!(props.is_empty());
        assert!(rates.is_empty());
        assert!(ll == f64::NEG_INFINITY);
    }

    fn make_segment(sample: &str, ancestry: &str, start: u64, end: u64) -> AncestrySegment {
        AncestrySegment {
            chrom: "chr1".to_string(), start, end,
            sample: sample.to_string(), ancestry_idx: 0,
            ancestry_name: ancestry.to_string(), n_windows: ((end - start) / 10_000) as usize,
            mean_similarity: 0.99, mean_posterior: Some(0.95),
            discriminability: 0.1, lod_score: 5.0,
        }
    }

    #[test]
    fn test_per_sample_demography() {
        let segments = vec![
            make_segment("sample_A", "EUR", 0, 100_000),
            make_segment("sample_A", "AFR", 100_000, 250_000),
            make_segment("sample_B", "EUR", 0, 300_000),
        ];
        let seg_refs: Vec<&AncestrySegment> = segments.iter().collect();
        let pop_names = vec!["EUR".to_string(), "AFR".to_string()];
        let params = DemographyParams::default();

        let results = infer_per_sample_demography(&seg_refs, &pop_names, &params);

        // BTreeMap ordering: sample_A before sample_B
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].sample, "sample_A");
        assert_eq!(results[1].sample, "sample_B");
        // Each sample has results for both populations
        assert_eq!(results[0].results.len(), 2);
        assert_eq!(results[1].results.len(), 2);
    }

    #[test]
    fn test_per_sample_demography_empty() {
        let seg_refs: Vec<&AncestrySegment> = vec![];
        let pop_names = vec!["EUR".to_string()];
        let params = DemographyParams::default();
        let results = infer_per_sample_demography(&seg_refs, &pop_names, &params);
        assert!(results.is_empty());
    }

    #[test]
    fn test_write_demography_tsv() {
        let pooled = vec![DemographicResult {
            population: "EUR".to_string(),
            n_tracts: 100,
            n_pulses: 1,
            pulses: vec![PulseEstimate {
                generations: 500.0,
                proportion: 1.0,
                rate: 5e-6,
                cv: 0.1,
            }],
            bic: 1234.5,
            log_likelihood: -600.0,
            ks_statistic: Some(0.05),
            single_pulse_rejected: false,
            low_tract_warning: None,
        }];
        let per_sample = vec![SampleDemographicResult {
            sample: "hap_A".to_string(),
            results: vec![DemographicResult {
                population: "EUR".to_string(),
                n_tracts: 20,
                n_pulses: 1,
                pulses: vec![PulseEstimate {
                    generations: 450.0,
                    proportion: 1.0,
                    rate: 4.5e-6,
                    cv: 0.224,
                }],
                bic: 300.0,
                log_likelihood: -140.0,
                ks_statistic: Some(0.08),
                single_pulse_rejected: false,
                low_tract_warning: None,
            }],
        }];

        let tmp = std::env::temp_dir().join("demog_test_output.tsv");
        write_demography_tsv(&tmp, &pooled, &per_sample).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // Header + 1 pooled + 1 per-sample = 3 lines
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("sample\t"));
        assert!(lines[1].starts_with("POOLED\t"));
        assert!(lines[2].starts_with("hap_A\t"));

        // Clean up
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_write_demography_tsv_no_pulses() {
        let pooled = vec![DemographicResult {
            population: "AFR".to_string(),
            n_tracts: 1,
            n_pulses: 0,
            pulses: vec![],
            bic: f64::INFINITY,
            log_likelihood: f64::NEG_INFINITY,
            ks_statistic: None,
            single_pulse_rejected: false,
            low_tract_warning: None,
        }];

        let tmp = std::env::temp_dir().join("demog_test_no_pulses.tsv");
        write_demography_tsv(&tmp, &pooled, &[]).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 row
        assert!(lines[1].contains("NA")); // pulse fields should be NA

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_write_demography_tsv_multi_pulse() {
        let pooled = vec![DemographicResult {
            population: "AMR".to_string(),
            n_tracts: 200,
            n_pulses: 2,
            pulses: vec![
                PulseEstimate { generations: 15.0, proportion: 0.6, rate: 1.5e-7, cv: 0.091 },
                PulseEstimate { generations: 50.0, proportion: 0.4, rate: 5e-7, cv: 0.112 },
            ],
            bic: 800.0,
            log_likelihood: -380.0,
            ks_statistic: Some(0.12),
            single_pulse_rejected: true,
            low_tract_warning: None,
        }];

        let tmp = std::env::temp_dir().join("demog_test_multi_pulse.tsv");
        write_demography_tsv(&tmp, &pooled, &[]).unwrap();

        let content = std::fs::read_to_string(&tmp).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        // Header + 2 pulse rows = 3 lines
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("POOLED"));
        assert!(lines[2].contains("POOLED"));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_constrained_em_satisfies_constraint() {
        // Two-component mixture: λ1=5e-6, λ2=2e-5
        let l_min = 20_000.0;
        let tracts = make_mixture_tracts(100, 5e-6, 100, 2e-5, l_min);

        // The unconstrained mixture mean
        let unconstrained_mean = 0.5 * 5e-6 + 0.5 * 2e-5; // 1.25e-5

        // Constrain to a different mean
        let target_mean = 1.0e-5;
        let mut params = DemographyParams::default();
        params.bw_constraint = Some(target_mean);

        let (props, rates, ll, _iters) = em_exponential_mixture(&tracts, l_min, 2, &params);

        // Check that the constraint is satisfied
        let actual_mean: f64 = props.iter().zip(rates.iter()).map(|(&p, &r)| p * r).sum();
        assert!((actual_mean - target_mean).abs() / target_mean < 0.05,
            "Constrained mean {:.2e} should be close to target {:.2e}", actual_mean, target_mean);
        assert!(ll.is_finite());
        assert!(props.len() == 2);

        // Compare: unconstrained should have different mean
        let mut params_free = DemographyParams::default();
        params_free.bw_constraint = None;
        let (props_free, rates_free, _, _) = em_exponential_mixture(&tracts, l_min, 2, &params_free);
        let free_mean: f64 = props_free.iter().zip(rates_free.iter()).map(|(&p, &r)| p * r).sum();
        // The unconstrained should be closer to the true 1.25e-5
        assert!((free_mean - unconstrained_mean).abs() < (actual_mean - unconstrained_mean).abs(),
            "Unconstrained mean {:.2e} should be closer to true mean {:.2e} than constrained {:.2e}",
            free_mean, unconstrained_mean, actual_mean);
    }

    #[test]
    fn test_lagrange_projection_unequal_proportions() {
        // T84: Lagrange projection with unequal proportions should satisfy
        // the constraint while keeping all rates positive and finite.
        let l_min = 20_000.0;
        // Unequal mixture: 80% from λ1=2e-6, 20% from λ2=1e-5
        let tracts = make_mixture_tracts(160, 2e-6, 40, 1e-5, l_min);

        // Get unconstrained rates for reference
        let params_free = DemographyParams::default();
        let (props_free, rates_free, _, _) = em_exponential_mixture(&tracts, l_min, 2, &params_free);
        let free_mean: f64 = props_free.iter().zip(rates_free.iter()).map(|(&p, &r)| p * r).sum();

        // Constrain to 80% of natural mean
        let target_mean = free_mean * 0.8;
        let params_c = DemographyParams {
            bw_constraint: Some(target_mean),
            ..Default::default()
        };
        let (props_c, rates_c, ll_c, _) = em_exponential_mixture(&tracts, l_min, 2, &params_c);
        assert!(ll_c.is_finite());
        assert_eq!(rates_c.len(), 2);

        // Constraint satisfied
        let actual_mean: f64 = props_c.iter().zip(rates_c.iter()).map(|(&p, &r)| p * r).sum();
        assert!((actual_mean - target_mean).abs() / target_mean < 0.05,
            "Constraint violated: actual={:.2e}, target={:.2e}", actual_mean, target_mean);

        // All rates positive
        assert!(rates_c.iter().all(|&r| r > 0.0), "All rates must be positive");

        // Rates should still be well-separated (ratio > 1.5)
        let ratio = rates_c[1] / rates_c[0];
        assert!(ratio > 1.0, "Rates should remain ordered: {:.2e} < {:.2e}", rates_c[0], rates_c[1]);

        // Constrained to 120% of natural mean
        let target_high = free_mean * 1.2;
        let params_h = DemographyParams {
            bw_constraint: Some(target_high),
            ..Default::default()
        };
        let (props_h, rates_h, ll_h, _) = em_exponential_mixture(&tracts, l_min, 2, &params_h);
        assert!(ll_h.is_finite());
        let actual_high: f64 = props_h.iter().zip(rates_h.iter()).map(|(&p, &r)| p * r).sum();
        assert!((actual_high - target_high).abs() / target_high < 0.05,
            "High constraint violated: actual={:.2e}, target={:.2e}", actual_high, target_high);
    }

    #[test]
    fn test_lagrange_projection_single_component() {
        // Single component: Lagrange projection should just set λ = λ_bw
        let l_min = 20_000.0;
        let tracts = make_exponential_tracts(200, 1e-5, l_min);
        let target = 8e-6;
        let params = DemographyParams {
            bw_constraint: Some(target),
            ..Default::default()
        };
        let (props, rates, ll, _) = em_exponential_mixture(&tracts, l_min, 1, &params);
        assert!(ll.is_finite());
        assert_eq!(props.len(), 1);
        // For M=1, π=1, so Lagrange reduces to λ = λ* + (λ_bw - λ*) = λ_bw
        assert!((rates[0] - target).abs() / target < 0.01,
            "Single-component rate {:.2e} should equal target {:.2e}", rates[0], target);
    }

    #[test]
    fn test_constrained_em_no_effect_when_none() {
        let l_min = 20_000.0;
        let tracts = make_exponential_tracts(100, 1e-5, l_min);

        let params_none = DemographyParams { bw_constraint: None, ..Default::default() };
        let (_, rates_none, ll_none, _) = em_exponential_mixture(&tracts, l_min, 1, &params_none);

        let params_zero = DemographyParams { bw_constraint: Some(0.0), ..Default::default() };
        let (_, rates_zero, ll_zero, _) = em_exponential_mixture(&tracts, l_min, 1, &params_zero);

        // With bw_constraint=0 the projection should be a no-op (0 is skipped)
        assert!((rates_none[0] - rates_zero[0]).abs() < 1e-10);
        assert!((ll_none - ll_zero).abs() < 1e-6);
    }
}
