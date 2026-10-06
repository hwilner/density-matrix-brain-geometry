"""Density-matrix representations of neural covariance states.

Any positive-semidefinite matrix C with tr(C) > 0 defines a *formal*
density operator rho = C / tr(C): it is positive semidefinite and has
unit trace, so the full vocabulary of quantum information -- von Neumann
entropy, purity, Renyi entropies, fidelity, Bures and trace distances --
becomes available as *classical* covariance geometry.  This module makes
no claim of physical quantum coherence in the brain; it provides a
compact, well-studied representation of multivariate organization.
"""

from __future__ import annotations

import numpy as np


def covariance_to_rho(cov: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Normalise a PSD covariance/coherence matrix to a density matrix.

    Args:
        cov: Symmetric positive-semidefinite matrix with positive trace.
        tol: Symmetry tolerance.

    Returns:
        ``rho = cov / tr(cov)``, symmetrised.

    Raises:
        ValueError: If ``cov`` is not symmetric, has negative
            eigenvalues (beyond a small numerical slack), or non-positive
            trace.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.allclose(c, c.T, atol=tol):
        raise ValueError("cov must be symmetric")
    eigs = np.linalg.eigvalsh(c)
    if eigs.min() < -1e-8 * max(1.0, eigs.max()):
        raise ValueError("cov is not positive semidefinite")
    tr = float(np.trace(c))
    if tr <= 0:
        raise ValueError("cov must have positive trace")
    rho = c / tr
    return 0.5 * (rho + rho.T)


def maximally_mixed(d: int) -> np.ndarray:
    """Maximally mixed density matrix I/d of dimension d.

    Args:
        d: Hilbert-space (feature-space) dimension.

    Returns:
        ``np.eye(d) / d``.
    """
    if d < 1:
        raise ValueError("d must be positive")
    return np.eye(d) / d


def von_neumann_entropy(rho: np.ndarray, base: float = np.e) -> float:
    """Von Neumann entropy ``S = -tr(rho log rho)``.

    Computed from the eigenvalues with zeros clipped.  Equals ``log(d)``
    for the maximally mixed state and 0 for pure states.

    Args:
        rho: Density matrix (PSD, trace 1).
        base: Logarithm base (``np.e`` for nats, 2 for bits).

    Returns:
        Entropy in the chosen units.
    """
    eigs = np.linalg.eigvalsh(np.asarray(rho, dtype=float))
    eigs = eigs[eigs > 1e-15]
    return float(-np.sum(eigs * np.log(eigs)) / np.log(base))


def purity(rho: np.ndarray) -> float:
    """Purity ``tr(rho^2)``: 1 for pure states, 1/d for maximally mixed.

    Args:
        rho: Density matrix.

    Returns:
        ``tr(rho @ rho)`` in ``[1/d, 1]``.
    """
    r = np.asarray(rho, dtype=float)
    return float(np.trace(r @ r))


def renyi_entropy(rho: np.ndarray, q: float = 2.0, base: float = np.e) -> float:
    """Renyi entropy ``S_q = log tr(rho^q) / (1 - q)``.

    Args:
        rho: Density matrix.
        q: Renyi order (``q != 1``; ``q -> 1`` recovers the von Neumann
            entropy).
        base: Logarithm base.

    Returns:
        Renyi entropy in the chosen units.
    """
    if abs(q - 1.0) < 1e-12:
        return von_neumann_entropy(rho, base)
    eigs = np.linalg.eigvalsh(np.asarray(rho, dtype=float))
    eigs = eigs[eigs > 1e-15]
    return float(np.log(np.sum(eigs**q)) / (1.0 - q) / np.log(base))


def effective_rank(rho: np.ndarray) -> float:
    """Effective rank ``1 / tr(rho^2)`` (participation ratio of modes).

    Counts how many eigenmodes are appreciably populated: 1 for a pure
    state, d for the maximally mixed state.

    Args:
        rho: Density matrix.

    Returns:
        Effective number of occupied eigenmodes.
    """
    p = purity(rho)
    if p <= 0:
        raise ValueError("rho has zero purity")
    return 1.0 / p


def sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a PSD matrix via eigendecomposition.

    Args:
        matrix: Symmetric PSD matrix.

    Returns:
        Symmetric PSD matrix ``s`` with ``s @ s = matrix``.
    """
    vals, vecs = np.linalg.eigh(np.asarray(matrix, dtype=float))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T
