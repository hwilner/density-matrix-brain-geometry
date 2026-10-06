"""Distances and fidelities between density-matrix brain states.

These quantities turn "how similar are two neural covariance states"
into geometry: the Uhlmann fidelity generalises the overlap of
probability distributions, the Bures distance is the metric induced by
fidelity, and the trace distance is the optimal single-shot
distinguishability.  All reduce to the classical (Bhattacharyya /
Hellinger / total-variation) formulas when the states commute.
"""

from __future__ import annotations

import numpy as np

from rho_geometry.density import sqrt_psd


def uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray,
                     squared: bool = True) -> float:
    """Uhlmann-Jozsa fidelity between two density matrices.

    ``F = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2`` in the squared
    convention (default).  For commuting (classical) states it reduces
    to the squared Bhattacharyya coefficient ``(sum sqrt(p_i q_i))^2``.

    Args:
        rho: Density matrix.
        sigma: Density matrix, same shape.
        squared: If False, return the root fidelity ``sqrt(F)``.

    Returns:
        Fidelity in ``[0, 1]``; 1 iff the states are identical.
    """
    r = np.asarray(rho, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if r.shape != s.shape:
        raise ValueError("rho and sigma must have the same shape")
    sr = sqrt_psd(r)
    inner = sqrt_psd(sr @ s @ sr)
    root_f = float(np.trace(inner))
    root_f = max(root_f, 0.0)
    return root_f**2 if squared else root_f


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Bures distance ``d_B = sqrt(2 (1 - sqrt(F)))``.

    Args:
        rho: Density matrix.
        sigma: Density matrix, same shape.

    Returns:
        Bures distance in ``[0, sqrt(2)]``; 0 iff the states coincide.
    """
    root_f = uhlmann_fidelity(rho, sigma, squared=False)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - root_f))))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ``0.5 * tr |rho - sigma|``.

    Equals the maximum probability advantage in distinguishing the two
    states with a single measurement; reduces to the total-variation
    distance for commuting states.

    Args:
        rho: Density matrix.
        sigma: Density matrix, same shape.

    Returns:
        Trace distance in ``[0, 1]``.
    """
    r = np.asarray(rho, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if r.shape != s.shape:
        raise ValueError("rho and sigma must have the same shape")
    eigs = np.linalg.eigvalsh(r - s)
    return float(0.5 * np.sum(np.abs(eigs)))


def fidelity_matrix(states: list[np.ndarray]) -> np.ndarray:
    """Pairwise root-fidelity matrix of a list of density matrices.

    Args:
        states: List of density matrices of equal shape.

    Returns:
        Symmetric matrix ``M[i, j] = sqrt(F(rho_i, rho_j))`` with unit
        diagonal -- the Gram-like object used for MDS embedding and
        fingerprinting analyses.
    """
    n = len(states)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            f = uhlmann_fidelity(states[i], states[j], squared=False)
            m[i, j] = m[j, i] = f
    return m
