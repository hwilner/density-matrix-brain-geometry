# Extended introduction — brain states as points in a geometric space

*Written for curious readers with a high-school background. Every technical
word is explained, and the only math you truly need is averaging and square
roots.*

## The one-sentence version

Brain scans give us, for every moment in a mental state, a **table of
correlations** between brain regions; if we rescale that table so its numbers
add up to one, it *mathematically becomes* an object physicists call a
**density matrix** — and physicists have spent 90 years inventing beautiful
rulers (entropy, purity, fidelity, distance) for exactly such objects. We
borrow those rulers to map the geometry of mental states. No quantum magic in
the brain is claimed or needed — we are borrowing *mathematics*, not physics.

## Step 0 — What is a correlation table?

Give 200 people a fitness tracker and record each person's daily step count
for a year. If Alice and Bob always walk more on the same days (both lazy on
rainy days), their records are *correlated*. Replace people by brain regions
and steps by the fMRI blood-flow signal (see **BOLD** in the table below),
and you get the brain version: a 200×200 table where entry (i, j) says how
in-sync region i and region j are. This is the **functional connectivity**
(FC) matrix — the most-used summary of whole-brain activity.

## Step 1 — The magic rescaling

A FC matrix has 1's on its diagonal (every region is perfectly in sync with
itself), so its diagonal adds up to the number of regions, d. Divide every
entry by d. Now:

- the diagonal adds to exactly 1 (like probabilities do),
- the matrix is still symmetric and "positive" (a technical property —
  **PSD**, see table — that correlation matrices always have).

A symmetric, positive matrix whose diagonal adds to 1 is *precisely* what
quantum physics calls a **density matrix** — the ID card of a state of a
system. Ours describes a *classical* correlation pattern, but because the
math is identical, every tool invented for density matrices works on it.

## Step 2 — The eigenvalue "budget"

Every such matrix can be re-described by a list of numbers called
**eigenvalues** — think of them as a **budget of importance** split between
hidden collective patterns:

- One eigenvalue = 1, all others 0: *all* importance sits in one single
  brain-wide pattern. Perfect order. Physicists call this a **pure state**.
- Every eigenvalue = 1/d: importance spread perfectly evenly over d patterns.
  Perfect disorder. Called the **maximally mixed state**.

Real brains live between these extremes, and the whole point of this project
is to find *where* — and whether the position moves with mental state.

## Step 3 — The rulers

**Entropy (von Neumann entropy).** The formula is the same one you may have
seen for information: take each eigenvalue lambda, compute lambda·ln(lambda),
sum, negate. Result: 0 for perfect order, ln(d) for perfect disorder.
*Worked example* (d = 4, maximally mixed): each eigenvalue is 1/4 = 0.25, so
S = −4 × (0.25 × ln 0.25) = −4 × 0.25 × (−1.386) = 1.386 = ln 4. ✔
We usually report S/ln d, a disorder score between 0 and 1.
*Hypothesis H1*: a focused task **lowers** this score — coordination means a
few patterns dominate.

**Purity and effective rank.** Square each eigenvalue and add: that is the
*purity* (between 1/d and 1). Flip it upside down and you get the *effective
rank*: the honest count of "how many patterns really matter". Evenly mixed
d = 4 gives purity 1/4, so effective rank 4 — all four patterns count. One
dominant pattern gives purity 1, effective rank 1.

**Fidelity.** How similar are two states? For plain probability lists the
classic recipe is: multiply matching entries, take square roots, add — the
**Bhattacharyya coefficient**. Example: p = (0.5, 0.5), q = (0.25, 0.75):
sqrt(0.5·0.25) + sqrt(0.5·0.75) = 0.354 + 0.612 = 0.966, so the lists are
~97% similar (squared: 0.93). The **Uhlmann fidelity** is the matrix version
of the same idea, and — our code tests this — when the matrices are
"classical" (they commute), it gives *exactly* the Bhattacharyya answer.

**Distances (Bures, trace).** Once you have similarity you can build a map:
**Bures distance** turns fidelity into a proper mileage counter between
states, so we can draw 2-D maps of mental states (rest over here, memory task
over there) with multi-dimensional scaling. **Trace distance** has a game
interpretation: it is your best edge in a coin-flip guess of which of two
states you are looking at.

## Step 4 — What we will actually do

1. Download HCP brain scans (resting + 7 tasks) — public data, see links.
2. Build each person's FC matrix in each condition; rescale to a density
   matrix.
3. Compute entropy, purity, effective rank per condition — does task beat
   rest in "order"?
4. Compute fidelities between all (person × condition) pairs — do states
   cluster by condition?
5. Fingerprint check: can we pick your day-2 scan out of 1 000 people using
   day-1 geometry?
6. Throughout, compare against **null models** (fake data with the same
   amount of noise but no real coordination) — because even pure noise has
   non-trivial entropy, and we refuse to fool ourselves.

## Every keyword, explained

| Term | Plain meaning | Why here | Link |
|---|---|---|---|
| fMRI / BOLD | Scanner signal tracking blood oxygen as an activity proxy | Raw data | https://en.wikipedia.org/wiki/Functional_magnetic_resonance_imaging |
| Functional connectivity | Correlation table of regional signals | The object we re-normalise | https://en.wikipedia.org/wiki/Functional_connectivity |
| PSD matrix | A matrix whose hidden "budget numbers" are all ≥ 0 | Needed for the rescaling to be valid | https://en.wikipedia.org/wiki/Definite_matrix |
| Density matrix | PSD matrix whose diagonal sums to 1; physics' state ID card | What FC becomes after rescaling | https://en.wikipedia.org/wiki/Density_matrix |
| Eigenvalue | One "budget number" of a matrix | Entropy/purity are computed from them | https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors |
| Von Neumann entropy | Disorder score of the budget (0 … ln d) | H1: task = more order | https://en.wikipedia.org/wiki/Von_Neumann_entropy |
| Purity / effective rank | How concentrated the budget is; honest pattern count | H1's second ruler | https://en.wikipedia.org/wiki/Purity_(quantum_mechanics) |
| Renyi entropy | A dial-able family of disorder scores | Richer than one number | https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy |
| Bhattacharyya coefficient | Classic similarity of two probability lists | What fidelity reduces to | https://en.wikipedia.org/wiki/Bhattacharyya_distance |
| Uhlmann fidelity | Matrix version of that similarity | State comparisons (H2, H3) | https://en.wikipedia.org/wiki/Fidelity_of_quantum_states |
| Bures distance | Mileage counter built from fidelity | Map-making | https://en.wikipedia.org/wiki/Bures_metric |
| Trace distance | Best guessing-edge between two states | Distinguishability | https://en.wikipedia.org/wiki/Trace_distance |
| MDS | Algorithm turning distances into a 2-D map | Visualising state space | https://en.wikipedia.org/wiki/Multidimensional_scaling |
| Wishart null | Random correlation tables from pure noise | The honesty baseline | https://en.wikipedia.org/wiki/Wishart_distribution |
| HCP Young Adult | Public dataset: ~1 200 scanned volunteers | Where we test it all | https://www.humanconnectome.org/study/hcp-young-adult |
| Fingerprinting | Re-identifying a person from their brain data | H3 | https://www.nature.com/articles/nn.4135 |

## Try it yourself

```python
import numpy as np
from rho_geometry import (covariance_to_rho, von_neumann_entropy,
                          effective_rank, uhlmann_fidelity, maximally_mixed)

# Fake data: 30 regions, 1000 time points, pure noise
x = np.random.default_rng(0).normal(size=(30, 1000))
rho = covariance_to_rho(np.corrcoef(x))

print("entropy (bits):", von_neumann_entropy(rho, base=2))
print("max possible:  ", np.log2(30))
print("effective rank:", effective_rank(rho))
print("fidelity with maximally mixed:", uhlmann_fidelity(rho, maximally_mixed(30)))
```

## Common confusions, pre-empted

- *"So the brain is quantum?"* No. We borrow math, like using geometry
  invented for land surveying to measure your kitchen table.
- *"Isn't entropy just about information?"* It is — the von Neumann formula
  IS Shannon's formula applied to the budget numbers. Same idea, new object.
- *"Why not just compare correlation tables directly?"* People do, but the
  density-matrix rulers come with guarantees (exact classical limits, true
  distances, hard bounds) that ad-hoc comparisons lack — that is what this
  project stress-tests.

## Friendly resources

- Entropy intuition (Khan Academy):
  https://www.khanacademy.org/computing/computer-science/informationtheory
- 3Blue1Brown eigen-video: https://www.youtube.com/watch?v=PFDu9oVAE-g
- HCP open data: https://db.humanconnectome.org/
- Our code & tests: see `src/` and `tests/` in this repository.
