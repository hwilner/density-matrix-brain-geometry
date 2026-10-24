# Current results and discussion

## Synthetic validation (complete)

All 39 anchors pass (`python -m pytest -q`). Highlights:

- `covariance_to_rho` reproduces trace-1 PSD matrices and rejects
  asymmetric, indefinite, and zero-trace inputs. The PSD check is relative
  to the spectral scale of the matrix, so an indefinite matrix is rejected
  at every magnitude rather than only when its entries are large.
- Maximally mixed d = 8: S = ln 8 nats = 3 bits; purity = 1/8; effective
  rank = 8.
- Pure states: S = 0, purity = 1, effective rank = 1.
- Renyi q → 1 converges to von Neumann; S_2 = −ln purity exactly; S_0 is
  the log of the number of populated modes.
- Classical (commuting) limits: fidelity = squared Bhattacharyya; trace
  distance = total variation.
- Fidelity matrix of |0>, |+>, I/2 has unit diagonal, symmetry, and the
  |0>-|+> entry = 1/sqrt(2).

## What the noise floor looks like

White-noise FC does not give a maximally mixed state, but the size of the
gap depends strongly on the number of time points `T` relative to the
number of channels `d`. `analysis/noise_floor.py` regenerates the table
below from the public API with a fixed seed (`python analysis/noise_floor.py`,
seed 0):

| configuration | S (bits) | max S (bits) | purity | effective rank | max rank |
|---|---|---|---|---|---|
| d = 50, T = 2000 | 5.626 | 5.644 | 0.02048 | 48.82 | 50 |
| d = 50, T = 100 | 5.283 | 5.644 | 0.02985 | 33.50 | 50 |
| d = 50, T = 50 | 4.900 | 5.644 | 0.03989 | 25.07 | 50 |
| d = 360, T = 2000 | 8.363 | 8.492 | 0.00328 | 305.30 | 360 |
| d = 360, T = 100 | 6.433 | 8.492 | 0.01284 | 77.88 | 360 |
| d = 360, T = 30 | 4.801 | 8.492 | 0.03719 | 26.89 | 360 |

At `d = 50, T = 2000` white noise is very nearly maximally mixed
(5.626 of 5.644 bits, effective rank 48.8 of 50): with 40 samples per
channel there is too little finite-sample bias to matter. The floor bites
in the *windowed* regime that `docs/METHODS.md` actually specifies — 60 s
windows at TR = 2 s give `T = 30` time points per window, so at
`d = 360, T = 30` white noise reaches only 4.801 of 8.492 bits and
populates 26.9 of 360 modes, 7.5% of the spectrum. Consequences:

1. All empirical entropies/ranks must be reported as **differences from
   matched Wishart nulls**, never against ln d. In the windowed regime the
   null sits several bits below ln d, so an uncorrected comparison would
   attribute the finite-sample bias to the brain.
2. The null must be matched on `T` as well as `d`; the table shows the
   floor moving by 3.6 bits across the `T` values at fixed `d = 360`.
3. The interesting H1 signal is a *condition difference* of these
   null-corrected values, mirroring the design of our spectral-transition
   sister project.

**Scope of this table.** It is a software-level check of the estimator on
synthetic white-noise input, computed by a committed script over the
public API. It is not a scientific result about brain organisation: it
uses no HCP data, tests no hypothesis, and involves no label or condition
structure. It establishes only the baseline an empirical value would have
to be judged against.

## Interpretation sketch

If task engagement recruits structured coordination, eigenvalues concentrate
(few collective modes dominate): S_norm down, purity up, effective rank
down. The Bures map should then show task states displaced from rest along a
"coordination axis", with tasks arranged by their known network demands
(language vs motor vs social).

## Risks

- **PSD violations after denoising** (e.g. after global-signal regression,
  which introduces negative correlations): handled by nearest-PSD projection
  plus a with/without-GSR sensitivity arm.
- **Window-length bias**: shorter windows → noisier rho → lower entropy
  trivially. Window analyses use Wishart nulls matched per window length.
- **Interpretation discipline**: we repeatedly state (README, INTRODUCTION)
  that no quantum-brain claim is made; the density matrix is classical
  covariance geometry.

## Open questions

- Does the Bures geometry recover the same task organisation as
  representational-similarity analyses on activation patterns?
- Is the day-1 → day-2 fingerprint carried mostly by the eigenvalues
  (marginal spectrum) or eigenvectors (pattern identity)? Fidelity variants
  restricted to commuting projections will answer this.
