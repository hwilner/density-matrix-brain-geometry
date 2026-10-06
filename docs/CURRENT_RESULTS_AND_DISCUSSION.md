# Current results and discussion

## Synthetic validation (complete)

All 17 anchors pass. Highlights:

- `covariance_to_rho` reproduces trace-1 PSD matrices and rejects
  asymmetric, indefinite, and zero-trace inputs.
- Maximally mixed d = 8: S = ln 8 nats = 3 bits; purity = 1/8; effective
  rank = 8.
- Pure states: S = 0, purity = 1, effective rank = 1.
- Renyi q → 1 converges to von Neumann; S_2 = −ln purity exactly.
- Classical (commuting) limits: fidelity = squared Bhattacharyya; trace
  distance = total variation.
- Fidelity matrix of |0>, |+>, I/2 has unit diagonal, symmetry, and the
  |0>-|+> entry = 1/sqrt(2).

## What the noise floor looks like

For white-noise FC (d = 50, T = 2 000), the formal rho is *not* maximally
mixed: S ≈ 4.7 bits vs the ln₂ 50 ≈ 5.64 maximum, and effective rank ≈ 30,
not 50. Finite T alone concentrates the spectrum. Consequences:

1. All empirical entropies/ranks must be reported as **differences from
   matched Wishart nulls**, never against ln d.
2. The interesting H1 signal is a *condition difference* of these
   null-corrected values, mirroring the design of our spectral-transition
   sister project.

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
