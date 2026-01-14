# Methods: Pangenome-Based IBS/IBD Detection

## Overview

This document provides detailed algorithmic descriptions, complexity analyses, and mathematical foundations for the pangenome-based Identity-By-State (IBS) and Identity-By-Descent (IBD) detection pipeline implemented in HPRCv2-IBD.

---

## 1. Mathematical Notation

| Symbol | Definition |
|--------|------------|
| $h_i$ | A single haplotype assembly |
| $H = \{h_1, h_2, ..., h_n\}$ | Set of all haplotypes in the pangenome |
| $W_k$ | The k-th genomic window |
| $w$ | Window size in base pairs |
| $S(h_i, h_j, W_k)$ | Sequence similarity between haplotypes $h_i$ and $h_j$ in window $W_k$ |
| $\tau$ | Identity threshold for IBS classification |
| $I_{ij}^k$ | Binary IBS indicator: 1 if $S(h_i, h_j, W_k) \geq \tau$, else 0 |
| $\pi_{enter}$ | Transition probability: non-IBD to IBD state |
| $\pi_{exit}$ | Transition probability: IBD to non-IBD state |
| $\mu_0, \sigma_0$ | Gaussian emission parameters for non-IBD state |
| $\mu_1, \sigma_1$ | Gaussian emission parameters for IBD state |
| $L$ | Expected IBD segment length in windows |
| $N$ | Total number of windows in a chromosome |
| $P$ | Number of haplotype pairs |

---

## 2. Sliding Window IBS Detection

### 2.1 Algorithm Description

The IBS detection algorithm tiles a reference chromosome into non-overlapping windows and computes pairwise sequence similarity using the implicit pangenome graph (impg).

```
Algorithm 1: Sliding Window IBS Detection
-----------------------------------------
Input:
  - Pangenome graph G
  - Reference chromosome C with length L_C
  - Window size w
  - Identity threshold tau
  - Haplotype subset H' (optional)

Output:
  - IBS table T with columns (chrom, start, end, hap_a, hap_b, identity)

1:  T <- empty table
2:  n_windows <- floor(L_C / w)
3:
4:  for k = 0 to n_windows - 1 do
5:      start <- k * w + 1
6:      end <- (k + 1) * w
7:      region <- C:start-end
8:
9:      // Query impg for pairwise similarities in this window
10:     similarities <- impg_similarity(G, region, H')
11:
12:     for each (h_i, h_j, s) in similarities do
13:         if s >= tau then
14:             T.append(C, start, end, h_i, h_j, s)
15:         end if
16:     end for
17: end for
18:
19: return T
```

### 2.2 Time Complexity Analysis

| Component | Complexity | Description |
|-----------|------------|-------------|
| Window iteration | $O(N)$ | Linear in number of windows $N = L_C / w$ |
| impg similarity query | $O(P \cdot w)$ | Per-window comparison of $P$ haplotype pairs |
| Total | $O(N \cdot P \cdot w)$ | Equivalent to $O(L_C \cdot P)$ |

For a typical human chromosome with $L_C \approx 10^8$ bp, $P \approx 10^5$ haplotype pairs (for ~450 samples), and $w = 5000$ bp:
- Number of windows: $N \approx 20,000$
- Per-chromosome time: proportional to $2 \times 10^9$ pairwise comparisons

### 2.3 Space Complexity Analysis

| Structure | Complexity | Description |
|-----------|------------|-------------|
| Output table | $O(N \cdot P_{IBS})$ | Where $P_{IBS}$ is the number of IBS-positive pairs per window |
| Working memory | $O(P)$ | Similarity scores for current window |
| Total | $O(N \cdot P_{IBS})$ | Dominated by output size |

---

## 3. Identity State Classification (Jacquard-Style Metrics)

### 3.1 Union-Find for Haplotype Clustering

The identity state classification uses a Union-Find (Disjoint Set Union) data structure to efficiently group haplotypes based on observed IBS relationships within each window.

```
Algorithm 2: Union-Find with Path Compression
---------------------------------------------
Data Structure:
  - parent[]: maps each node to its parent

Initialize(nodes):
  for each node in nodes do
    parent[node] <- node
  end for

Find(node):
  if parent[node] != node then
    parent[node] <- Find(parent[node])  // Path compression
  end if
  return parent[node]

Union(a, b):
  root_a <- Find(a)
  root_b <- Find(b)
  if root_a != root_b then
    parent[root_b] <- root_a
  end if
```

### 3.2 Nine Identity States Classification

```
Algorithm 3: Identity State Classification
------------------------------------------
Input:
  - Window W with IBS pairs P = {(h_i, h_j) : I_ij = 1}
  - Haplotype set for individual A: {a_1, a_2}
  - Haplotype set for individual B: {b_1, b_2}

Output:
  - Identity state S in {1, 2, ..., 9}

1:  // Initialize Union-Find with 4 haplotypes
2:  uf <- UnionFind({a_1, a_2, b_1, b_2})
3:
4:  // Union all IBS-positive pairs
5:  for each (h_i, h_j) in P do
6:      if h_i in {a_1, a_2, b_1, b_2} and h_j in {a_1, a_2, b_1, b_2} then
7:          uf.Union(h_i, h_j)
8:      end if
9:  end for
10:
11: // Identify connected components (blocks)
12: blocks <- extract_connected_components(uf)
13:
14: // Classify based on block structure
15: return ClassifyBlocks(blocks)

ClassifyBlocks(blocks):
  n_blocks <- |blocks|

  if n_blocks == 1 and blocks[0].size == 4:
    return S1  // All four identical

  if n_blocks == 4 and all blocks have size 1:
    return S9  // All four different

  if n_blocks == 2:
    if both blocks have size 2:
      if block1 = {a_1, a_2} and block2 = {b_1, b_2}:
        return S2  // Within-individual identity only
      if each block contains one from A and one from B:
        return S7  // Cross-matching pairs
    if one block has size 3, other has size 1:
      if size-3 block has 2 from A, 1 from B:
        return S3
      if size-3 block has 1 from A, 2 from B:
        return S5

  if n_blocks == 3:
    // One block of size 2, two singletons
    pair_block <- block with size 2
    if pair_block = {a_1, a_2}:
      return S4  // A homozygous by state
    if pair_block = {b_1, b_2}:
      return S6  // B homozygous by state
    if pair_block contains one from A and one from B:
      return S8  // Single cross-pair match

  return UNCLASSIFIED
```

### 3.3 Complexity Analysis

| Operation | Complexity | Description |
|-----------|------------|-------------|
| Union-Find initialization | $O(4)$ | Constant for 4 haplotypes |
| Union operations | $O(6 \cdot \alpha(4))$ | At most 6 pairs, $\alpha$ is inverse Ackermann |
| Component extraction | $O(4 \cdot \alpha(4))$ | Path compression makes this nearly $O(1)$ |
| Per-window classification | $O(1)$ | Constant-time state lookup |
| Total for all windows | $O(N)$ | Linear in number of windows |

---

## 4. Hidden Markov Model for IBD Inference

### 4.1 Model Specification

The HMM consists of two hidden states representing IBD and non-IBD regions:

**States:**
- State 0: Non-IBD (background)
- State 1: IBD (shared ancestry)

**Transition Matrix:**

$$
A = \begin{pmatrix}
1 - \pi_{enter} & \pi_{enter} \\
\pi_{exit} & 1 - \pi_{exit}
\end{pmatrix}
$$

where:
- $\pi_{exit} = 1/L$ (inverse of expected segment length in windows)
- $\pi_{enter}$ is a tunable parameter (typically 0.001)

**Emission Distributions:**

Both states emit continuous observations (sequence identity scores) following Gaussian distributions:

$$
P(x | \text{state } s) = \frac{1}{\sigma_s \sqrt{2\pi}} \exp\left(-\frac{(x - \mu_s)^2}{2\sigma_s^2}\right)
$$

Default parameters:
- Non-IBD: $\mu_0 = 0.5$, $\sigma_0 = 0.2$
- IBD: $\mu_1 = 0.99$, $\sigma_1 = 0.01$

### 4.2 Viterbi Algorithm

```
Algorithm 4: Viterbi Decoding for IBD State Sequence
----------------------------------------------------
Input:
  - Observations O = (o_1, o_2, ..., o_N) - identity scores per window
  - HMM parameters: initial probabilities pi, transition matrix A, emission params

Output:
  - Most likely state sequence Q* = (q_1*, q_2*, ..., q_N*)

1:  // Initialization (log-space for numerical stability)
2:  for s in {0, 1} do
3:      delta[1][s] <- log(pi[s]) + log_emission(o_1, s)
4:      psi[1][s] <- 0
5:  end for
6:
7:  // Recursion
8:  for t = 2 to N do
9:      for s in {0, 1} do
10:         delta[t][s] <- max over s' of {delta[t-1][s'] + log(A[s'][s])} + log_emission(o_t, s)
11:         psi[t][s] <- argmax over s' of {delta[t-1][s'] + log(A[s'][s])}
12:     end for
13: end for
14:
15: // Termination
16: q_N* <- argmax over s of {delta[N][s]}
17:
18: // Backtracking
19: for t = N-1 downto 1 do
20:     q_t* <- psi[t+1][q_{t+1}*]
21: end for
22:
23: return Q*
```

### 4.3 Emission Parameter Estimation via K-Means

The pipeline optionally estimates emission parameters from observed data using 1D k-means clustering:

```
Algorithm 5: K-Means Emission Estimation
----------------------------------------
Input:
  - Observations O = (o_1, ..., o_N)
  - k = 2 (number of clusters)
  - max_iterations

Output:
  - Updated emission parameters (mu_0, sigma_0), (mu_1, sigma_1)

1:  // Initialize centers using quantiles
2:  sort(O)
3:  centers[0] <- O[0.25 * N]  // 25th percentile
4:  centers[1] <- O[0.75 * N]  // 75th percentile
5:
6:  for iter = 1 to max_iterations do
7:      // Assignment step
8:      for i = 1 to N do
9:          assignments[i] <- argmin over c of |o_i - centers[c]|
10:     end for
11:
12:     // Update step
13:     for c in {0, 1} do
14:         cluster_c <- {o_i : assignments[i] == c}
15:         centers[c] <- mean(cluster_c)
16:     end for
17:
18:     if no assignments changed then
19:         break
20:     end if
21: end for
22:
23: // Compute Gaussian parameters
24: for c in {0, 1} do
25:     cluster_c <- {o_i : assignments[i] == c}
26:     mu[c] <- mean(cluster_c)
27:     sigma[c] <- std(cluster_c)
28: end for
29:
30: // Ensure mu[0] < mu[1] (low cluster = non-IBD)
31: if mu[0] > mu[1] then
32:     swap((mu[0], sigma[0]), (mu[1], sigma[1]))
33: end if
34:
35: return (mu[0], sigma[0]), (mu[1], sigma[1])
```

### 4.4 Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Viterbi forward pass | $O(N \cdot |S|^2) = O(N)$ | $O(N \cdot |S|) = O(N)$ |
| Viterbi backtracking | $O(N)$ | $O(N)$ |
| K-means clustering | $O(N \cdot k \cdot I)$ | $O(N)$ |
| Total HMM | $O(N)$ | $O(N)$ |

Where $|S| = 2$ (constant number of states) and $I$ is the number of k-means iterations.

---

## 5. Segment Detection and Merging

### 5.1 Run-Length Encoding (RLE) Segment Detection

```
Algorithm 6: RLE-Based Segment Detection
----------------------------------------
Input:
  - Identity track T = {(window_idx, identity_score)}
  - Parameters: min_identity tau, max_gap g, min_windows m, min_length_bp l

Output:
  - List of IBD segments

1:  segments <- []
2:  current_segment <- null
3:  gap_count <- 0
4:
5:  for window_idx = 0 to N-1 do
6:      identity <- T.get(window_idx)  // May be missing
7:      is_missing <- (identity == null)
8:      is_good <- (identity != null and identity >= tau)
9:
10:     if current_segment == null then
11:         if is_good then
12:             current_segment <- new Segment(start=window_idx)
13:             current_segment.add(identity)
14:         end if
15:     else
16:         if is_good then
17:             current_segment.extend_to(window_idx)
18:             current_segment.add(identity)
19:             gap_count <- 0
20:         else if is_missing then
21:             gap_count <- gap_count + 1
22:             if gap_count > g then
23:                 finalize_and_store(current_segment, segments)
24:                 current_segment <- null
25:             else
26:                 current_segment.extend_to(window_idx)
27:             end if
28:         else  // Below threshold
29:             finalize_and_store(current_segment, segments)
30:             current_segment <- null
31:         end if
32:     end if
33: end for
34:
35: if current_segment != null then
36:     finalize_and_store(current_segment, segments)
37: end if
38:
39: return filter(segments, min_windows >= m, length_bp >= l)
```

### 5.2 Segment Merging

```
Algorithm 7: Overlapping Segment Merge
--------------------------------------
Input:
  - List of segments S, each with (chrom, start, end, hap_a, hap_b, stats)

Output:
  - Merged segment list

1:  // Sort by chromosome, haplotype pair, then position
2:  sort(S) by (chrom, hap_a, hap_b, start, end)
3:
4:  merged <- [S[0]]
5:
6:  for i = 1 to |S|-1 do
7:      current <- S[i]
8:      last <- merged.last()
9:
10:     same_context <- (current.chrom == last.chrom and
11:                      current.hap_a == last.hap_a and
12:                      current.hap_b == last.hap_b)
13:
14:     if same_context and current.start <= last.end then
15:         // Merge overlapping segments
16:         last.end <- max(last.end, current.end)
17:         last.stats <- combine_stats(last.stats, current.stats)
18:     else
19:         merged.append(current)
20:     end if
21: end for
22:
23: return merged
```

### 5.3 Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| RLE detection | $O(N)$ | $O(S)$ where $S$ = number of segments |
| Segment sorting | $O(S \log S)$ | $O(S)$ |
| Segment merging | $O(S)$ | $O(S)$ |
| Total | $O(N + S \log S)$ | $O(S)$ |

---

## 6. Online Statistics (Welford's Algorithm)

For computing running mean and variance without storing all observations:

```
Algorithm 8: Welford's Online Statistics
----------------------------------------
Data Structure:
  - n: count
  - mean: running mean
  - M2: sum of squared deviations

Initialize():
  n <- 0
  mean <- 0
  M2 <- 0

Add(x):
  n <- n + 1
  delta <- x - mean
  mean <- mean + delta / n
  delta2 <- x - mean
  M2 <- M2 + delta * delta2

GetMean():
  return mean

GetVariance():
  if n < 2 then return 0
  return M2 / (n - 1)  // Bessel's correction

GetStdDev():
  return sqrt(GetVariance())
```

**Complexity:** $O(1)$ time and space per update.

---

## 7. Complete Pipeline Complexity Summary

For a chromosome with $N$ windows and $P$ haplotype pairs:

| Stage | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| 1. Window tiling | $O(N)$ | $O(1)$ |
| 2. IBS detection | $O(N \cdot P)$ | $O(N \cdot P_{IBS})$ |
| 3. Identity state classification | $O(N)$ per pair | $O(N)$ |
| 4. HMM inference | $O(N)$ per pair | $O(N)$ |
| 5. Segment detection | $O(N)$ per pair | $O(S)$ |
| **Total per haplotype pair** | $O(N)$ | $O(N)$ |
| **Total for all pairs** | $O(N \cdot P)$ | $O(N \cdot P_{IBS})$ |

### Parallelization Strategy

The pipeline is embarrassingly parallel at multiple levels:
1. **Chromosome-level:** Each chromosome processed independently
2. **Window-level:** Windows processed in parallel batches
3. **Pair-level:** HMM inference runs independently per haplotype pair

Effective parallelization reduces wall-clock time to approximately:

$$
T_{wall} \approx \frac{O(N \cdot P)}{J}
$$

where $J$ is the number of parallel workers.

---

## 8. Parameter Sensitivity

### Critical Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| Window size ($w$) | 1,000 - 10,000 bp | Smaller = more resolution, higher noise |
| Identity threshold ($\tau$) | 0.99 - 1.0 | Higher = fewer false positives, more false negatives |
| Expected IBD length ($L$) | 10 - 1000 windows | Affects HMM transition probabilities |
| Minimum segment length | 5,000 - 100,000 bp | Filters short spurious segments |
| Maximum gap tolerance | 0 - 5 windows | Allows bridging over missing data |

### Recommended Defaults

For human pangenome analysis:
- Window size: 5,000 bp
- Identity threshold: 0.9995 (99.95%)
- Expected IBD length: 100 windows (500 kb)
- Minimum segment length: 10,000 bp
- Maximum gap: 1 window

---

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.

2. Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics*, 4(3), 419-420.

3. Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.

4. Garrison, E., et al. (2023). impg: implicit pangenome graph. *GitHub repository*.
