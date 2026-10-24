"""Regenerate the white-noise noise floor quoted in the results document.

The floor is the value that a purely random functional-connectivity matrix
produces, so that an empirical entropy or effective rank can be compared
against a matched null rather than against the theoretical maximum
``log(d)``.  The numbers this script prints are the ones quoted in
``docs/CURRENT_RESULTS_AND_DISCUSSION.md``.

Every configuration is seeded, so the output is reproducible.  Run it with

    python analysis/noise_floor.py

This script is a software-level check of the estimator on synthetic input.
It does not use HCP data, it does not test any hypothesis, and it does not
produce a scientific result about brain organisation; it establishes the
baseline against which an empirical value would have to be judged.
"""

from __future__ import annotations

import argparse

import numpy as np

from rho_geometry import (
    covariance_to_rho,
    effective_rank,
    purity,
    von_neumann_entropy,
)

#: Default random seed.  Every configuration below uses it, so the printed
#: table is reproducible from a clean checkout.
DEFAULT_SEED = 0

#: Configurations reported in the results document.  Each entry is
#: ``(label, number of channels, number of time points)``.
CONFIGS = (
    ("d=50, T=2000", 50, 2000),
    ("d=50, T=100", 50, 100),
    ("d=50, T=50", 50, 50),
    ("d=360, T=2000", 360, 2000),
    ("d=360, T=100", 360, 100),
    ("d=360, T=30", 360, 30),
)


def white_noise_floor(d: int, n_timepoints: int,
                      seed: int = DEFAULT_SEED) -> dict[str, float]:
    """Summarise the density matrix of a white-noise correlation matrix.

    Args:
        d: Number of channels (the feature-space dimension).
        n_timepoints: Number of time points of the synthetic series.
        seed: Seed of the random number generator.

    Returns:
        A dict with the von Neumann entropy in bits, the maximum entropy in
        bits, the purity, the effective rank and the maximum effective rank.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(size=(d, n_timepoints))
    rho = covariance_to_rho(np.corrcoef(series))
    return {
        "entropy_bits": von_neumann_entropy(rho, base=2),
        "max_entropy_bits": float(np.log2(d)),
        "purity": purity(rho),
        "effective_rank": effective_rank(rho),
        "max_effective_rank": float(d),
    }


def main() -> None:
    """Print the noise-floor table quoted in the results document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    header = (f"{'configuration':>16}  {'S (bits)':>9}  {'max S':>7}  "
              f"{'purity':>8}  {'eff.rank':>9}  {'max rank':>9}")
    print(f"white-noise noise floor (seed {args.seed})")
    print(header)
    print("-" * len(header))
    for label, d, n_timepoints in CONFIGS:
        stats = white_noise_floor(d, n_timepoints, seed=args.seed)
        print(f"{label:>16}  {stats['entropy_bits']:9.3f}  "
              f"{stats['max_entropy_bits']:7.3f}  {stats['purity']:8.5f}  "
              f"{stats['effective_rank']:9.2f}  "
              f"{stats['max_effective_rank']:9.2f}")


if __name__ == "__main__":
    main()
