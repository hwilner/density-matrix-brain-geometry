# Density-matrix information geometry of brain states

**If we normalise a brain's functional-connectivity matrix to unit trace, it
becomes a formal density operator — and the entire metric toolkit of quantum
information (von Neumann entropy, purity, Uhlmann fidelity, Bures distance)
becomes a rigorous language for the geometry of cognition.**

## What this project is — and is not

This is **not** a claim that the brain is a quantum computer or that neural
coherence is quantum coherence. It is a *representation theorem*: any
positive-semidefinite (PSD) matrix C with tr(C) > 0 can be written as

```
rho = C / tr(C),      rho = rho^T,  rho >= 0,  tr rho = 1,
```

which is exactly the mathematical definition of a density matrix — the most
general description of a (possibly mixed) state in quantum mechanics. The
brain's functional-connectivity (FC) matrix is symmetric PSD with positive
trace. Therefore every theorem proved about density matrices applies to
normalised FC — as *classical covariance geometry*. We import the vocabulary,
not the physics.

Why bother? Because this vocabulary was built, over 90 years, precisely to
answer the questions neuroscience asks of FC: *how mixed/ordered is this
state?* (entropy, purity), *how similar are two states?* (fidelity, Bures,
trace distance), *how many effective degrees of freedom are active?*
(effective rank). These come with exact classical limits, tight bounds, and
interpretations that plain matrix norms lack.

## The objects, step by step

### 1. From FC to rho

Let X be the (parcellated) BOLD data matrix, regions × time. The FC matrix is
the Pearson correlation matrix C = corr(X): symmetric, ones on the diagonal,
PSD (it is a Gram matrix of standardised series). Normalise:

```
rho = C / tr(C) = C / d        (since the diagonal of C is all ones,
                                tr C = d = number of regions)
```

For a correlation matrix the normalisation is trivially the dimension — the
map is well defined for *any* FC estimate.

### 2. Von Neumann entropy — how "mixed" is the state?

```
S(rho) = -tr(rho ln rho) = -sum_i lambda_i ln lambda_i
```

where lambda_i are the eigenvalues of rho (they are non-negative and sum to
1, so the formula is exactly Shannon entropy of the eigenvalue distribution).

- **Pure state** (rank 1: one eigenvalue = 1): S = 0 — a perfectly organised
  system, all variance in a single collective mode.
- **Maximally mixed** (rho = I/d, all eigenvalues 1/d): S = ln d — maximal
  disorder, variance spread uniformly.

Derivation of the maximally mixed value: with lambda_i = 1/d for i = 1..d,

```
S = -sum_i (1/d) ln(1/d) = -d * (1/d) * (-ln d) = ln d.   QED
```

So S/ln d is a normalised disorder coordinate in [0, 1]. Hypothesis H1 says
task engagement *lowers* it: coordination concentrates variance into fewer
collective modes.

### 3. Purity and effective rank — how many modes are active?

```
purity(rho) = tr(rho^2) = sum_i lambda_i^2   in [1/d, 1]
effective rank(rho) = 1 / tr(rho^2)          in [1, d]
```

The effective rank (the "participation ratio" of the eigenmodes) counts how
many collective patterns are appreciably populated: 1 when one pattern
dominates, d when all share variance equally. It is a smoother cousin of
thresholded eigenvalue counting and needs no cutoff choice.

### 4. Renyi entropies — a whole family of disorder measures

```
S_q(rho) = ln tr(rho^q) / (1 - q),   q >= 0, q != 1
```

One-parameter generalisation: q → 1 recovers the von Neumann entropy
(proof: L'Hopital on ln tr(rho^q)/(1−q)), q = 2 gives −ln purity, q → ∞
gives −ln lambda_max (the "min-entropy", governed by the largest mode only).
Plotting S_q versus q (the "Renyi spectrum") characterises the whole
eigenvalue distribution, not just one summary.

### 5. Fidelity — how similar are two states?

Classically, the similarity of two probability vectors p, q is the
**Bhattacharyya coefficient** BC(p, q) = sum_i sqrt(p_i q_i). Uhlmann's
fidelity is its non-commuting generalisation:

```
F(rho, sigma) = ( tr sqrt( sqrt(rho) sigma sqrt(rho) ) )^2
```

When rho and sigma commute (share eigenvectors — the "classical" case), this
reduces exactly to BC² = (sum_i sqrt(p_i q_i))², with p, q the eigenvalue
vectors (verified in our tests). F = 1 iff the states are identical; F = 0
for orthogonal supports. Fidelity is the workhorse of state comparison: HCP
subjects' task states should have *higher* within-task fidelity across
subjects than across-task — that is hypothesis H2's clustering claim.

### 6. Bures and trace distances — metrics on state space

```
Bures:   d_B(rho, sigma) = sqrt(2 (1 - sqrt(F)))
Trace:   D(rho, sigma)   = (1/2) tr |rho - sigma|
```

Both are true metrics (symmetric, triangle inequality). The trace distance
has a decision-theoretic meaning: it equals the maximum single-measurement
advantage in telling the two states apart, and for commuting states it
reduces to the total-variation distance (verified in tests). The Bures
distance is Riemannian — it is the infinitesimal metric of quantum
information geometry — which lets us compute *geodesics*, *means*, and
*curvature* on the space of brain states, not just pairwise numbers.

## Hypotheses

- **H1 (state purity).** Task rho's have lower S/ln d and lower effective
  rank than rest rho's (coordination concentrates variance), after
  controlling for scan length and motion.
- **H2 (state geometry).** In the fidelity matrix of all (subject ×
  condition) states, states cluster by condition (rest vs each task) beyond
  label-shuffled and Wishart nulls; MDS of Bures distances shows an
  interpretable task organisation.
- **H3 (fingerprint).** A subject's day-2 state is best matched to their own
  day-1 state (highest root fidelity among all subjects) significantly above
  chance — an information-geometric connectome fingerprint (cf. Finn et al.,
  2015).
- **Falsifier.** If condition differences in d_B are fully explained by
  differences in scalar summaries (trace-norm, mean correlation), the metric
  geometry is redundant and the project stops at negative result.

## What is in this repository

`src/rho_geometry/density.py` — `covariance_to_rho`, `von_neumann_entropy`,
`purity`, `renyi_entropy`, `effective_rank`, `maximally_mixed`, `sqrt_psd`.
`src/rho_geometry/metrics.py` — `uhlmann_fidelity`, `bures_distance`,
`trace_distance`, `fidelity_matrix`. All functions carry Google-style
docstrings with the defining formulas. `tests/` pins 17 analytic anchors:
maximally mixed entropy = ln d, pure-state entropy = 0, purity bounds,
Renyi-2 = −ln purity, classical Bhattacharyya and total-variation limits,
the |0> vs |+> root-fidelity 1/sqrt(2), Bures of orthogonal states = sqrt(2).

## Key terms

- **Density matrix / density operator**: a PSD, unit-trace matrix rho; the
  general state of a quantum system — here, a *formal* normalised covariance.
  *Why*: it is the most structured normalisation of a PSD matrix, with a
  century of theorems attached.
- **PSD (positive semidefinite)**: all eigenvalues >= 0. Correlation matrices
  are PSD by construction (Gram matrices). *When to check*: always, before
  mapping to rho — `covariance_to_rho` validates it.
- **Von Neumann entropy**: S = −tr rho ln rho; the mixedness of the state,
  equal to Shannon entropy of the eigenvalues. *Use when*: you need one
  principled disorder number with a hard maximum (ln d).
- **Purity / effective rank**: tr rho^2 and its reciprocal; how concentrated
  the eigenmodes are. *Use when*: you want a cutoff-free count of active
  collective modes.
- **Renyi entropy S_q**: one-parameter family interpolating between the
  largest-mode (q → ∞) and uniform-weight (q = 0) summaries.
- **Uhlmann fidelity**: the non-commuting Bhattacharyya overlap; *the*
  standard state-similarity measure. *Use when*: comparing states whose
  eigenvectors may differ.
- **Bures distance**: metric induced by fidelity; turns state space into a
  Riemannian geometry (geodesics, means, curvature exist).
- **Trace distance**: optimal single-shot distinguishability; reduces to
  total variation for classical states.
- **FC (functional connectivity)**: correlation matrix of regional BOLD time
  series; our raw object before normalisation.
- **MDS (multidimensional scaling)**: embeds a distance matrix in 2–3-D for
  visualisation; applied to Bures distances for the state-space map.
- **Wishart null**: random covariance matrices from white noise with matched
  T/d; the essential baseline, since even noise rho's have non-trivial
  entropy and geometry.

## Related work and differentiation

"Quantum cognition" (Busemeyer & Bruza, 2012) uses quantum *probability* for
human judgements; quantum-brain hypotheses (Hameroff–Penrose) posit physical
quantum effects. We do neither: we use density matrices as *coordinate-free
covariance geometry*, closer in spirit to information geometry (Amari) and to
RMT analyses of correlation matrices — but where RMT studies the *spectrum's
fluctuations* (our sister repo `quantum-chaos-spectral-transitions`), this
project studies the *state space itself*: entropies, purities, and metric
distances between whole-brain configurations.

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum
  Information*. Cambridge University Press. (Ch. 2, 9: density operators,
  fidelity, trace distance.)
- Bengtsson, I., & Zyczkowski, K. (2017). *Geometry of Quantum States*.
  Cambridge University Press. (Bures geometry.)
- Jozsa, R. (1994). Fidelity for mixed quantum states. *Journal of Modern
  Optics*, 41, 2315.
- Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
- Busemeyer, J. R., & Bruza, P. D. (2012). *Quantum Models of Cognition and
  Decision*. Cambridge University Press.
- Finn, E. S., et al. (2015). Functional connectome fingerprinting.
  *Nature Neuroscience*, 18, 1664.
- Van Essen, D. C., et al. (2013). The WU-Minn Human Connectome Project.
  *NeuroImage*, 80, 62.
- Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral
  cortex. *Nature*, 536, 171.
