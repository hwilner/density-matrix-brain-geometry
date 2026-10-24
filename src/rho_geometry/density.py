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

from rho_geometry._checks import (
    PSD_RTOL,
    as_density_matrix,
    check_finite,
    check_psd,
    check_square,
    check_symmetric,
)

__all__ = [
    "covariance_to_rho",
    "effective_rank",
    "maximally_mixed",
    "project_to_psd",
    "purity",
    "renyi_entropy",
    "sqrt_psd",
    "von_neumann_entropy",
]


def covariance_to_rho(cov: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Normalise a PSD covariance/coherence matrix to a density matrix.

    Args:
        cov: Symmetric positive-semidefinite matrix with positive trace.
        tol: Relative symmetry tolerance, applied to the largest-magnitude
            entry of ``cov``.

    Returns:
        ``rho = cov / tr(cov)``, symmetrised.

    Raises:
        ValueError: If ``cov`` is not a non-empty square matrix, contains
            non-finite values, is not symmetric within ``tol``, has negative
            eigenvalues beyond a small numerical slack, or has non-positive
            trace.
    """
    c = check_square(cov, "cov")
    check_finite(c, "cov")
    check_symmetric(c, "cov", rtol=tol)
    check_psd(c, "cov")
    tr = float(np.trace(c))
    if tr <= 0:
        raise ValueError("cov must have positive trace")
    rho = c / tr
    return 0.5 * (rho + rho.T)


def project_to_psd(cov: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Project a symmetric matrix onto the nearest positive-semidefinite one.

    The guard rail described in ``docs/METHODS.md`` for FC matrices that
    fail the PSD check after denoising: the negative eigenvalues are
    clipped to zero, and the total magnitude of the clipping is returned
    so the caller can flag the matrix.

    Args:
        cov: Symmetric matrix, positive semidefinite up to numerical noise.
        tol: Relative symmetry tolerance, applied to the largest-magnitude
            entry of ``cov``.

    Returns:
        A tuple ``(projected, clipped)`` where ``projected`` is the nearest
        positive-semidefinite matrix to ``cov`` and ``clipped`` is the sum
        of the magnitudes of the negative eigenvalues that were removed.

    Raises:
        ValueError: If ``cov`` is not a non-empty square matrix, contains
            non-finite values, is not symmetric within ``tol``, or has
            non-positive trace.
    """
    c = check_square(cov, "cov")
    check_finite(c, "cov")
    check_symmetric(c, "cov", rtol=tol)
    tr = float(np.trace(c))
    if tr <= 0:
        raise ValueError("cov must have positive trace")
    eigs, vecs = np.linalg.eigh(0.5 * (c + c.T))
    negative = eigs[eigs < 0.0]
    clipped = float(-negative.sum())
    eigs = np.clip(eigs, 0.0, None)
    projected = (vecs * eigs) @ vecs.T
    return projected, clipped


def maximally_mixed(d: int) -> np.ndarray:
    """Maximally mixed density matrix I/d of dimension d.

    Args:
        d: Hilbert-space (feature-space) dimension.

    Returns:
        ``np.eye(d) / d``.

    Raises:
        ValueError: If ``d`` is not a positive integer.
    """
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
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

    Raises:
        ValueError: If ``rho`` is not a valid density matrix, or if ``base``
            is not a valid logarithm base.
    """
    _check_base(base)
    state = as_density_matrix(rho, "rho")
    eigs = np.linalg.eigvalsh(state)
    eigs = eigs[eigs > 1e-15]
    return float(-np.sum(eigs * np.log(eigs)) / np.log(base))


def purity(rho: np.ndarray) -> float:
    """Purity ``tr(rho^2)``: 1 for pure states, 1/d for maximally mixed.

    Args:
        rho: Density matrix.

    Returns:
        ``tr(rho @ rho)`` in ``[1/d, 1]``.

    Raises:
        ValueError: If ``rho`` is not a valid density matrix.
    """
    state = as_density_matrix(rho, "rho")
    return float(np.trace(state @ state))


def renyi_entropy(rho: np.ndarray, q: float = 2.0, base: float = np.e) -> float:
    """Renyi entropy ``S_q = log tr(rho^q) / (1 - q)``.

    Args:
        rho: Density matrix.
        q: Renyi order (``q >= 0``, ``q != 1``; ``q -> 1`` recovers the von
            Neumann entropy and ``q = 0`` gives the log of the number of
            populated modes).
        base: Logarithm base.

    Returns:
        Renyi entropy in the chosen units.

    Raises:
        ValueError: If ``rho`` is not a valid density matrix, if ``q`` is not
            a non-negative finite number, or if ``base`` is not a valid
            logarithm base.
    """
    _check_base(base)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("q must be a non-negative finite number")
    state = as_density_matrix(rho, "rho")
    if abs(q - 1.0) < 1e-12:
        return von_neumann_entropy(state, base)
    eigs = np.linalg.eigvalsh(state)
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

    Raises:
        ValueError: If ``rho`` is not a valid density matrix.
    """
    p = purity(rho)
    if p <= 0:
        raise ValueError("rho has zero purity")
    return 1.0 / p


def sqrt_psd(matrix: np.ndarray, rtol: float = PSD_RTOL) -> np.ndarray:
    """Matrix square root of a PSD matrix via eigendecomposition.

    Args:
        matrix: Symmetric PSD matrix.
        rtol: Relative tolerance for the positive-semidefinite check, applied
            to the largest-magnitude eigenvalue.

    Returns:
        Symmetric PSD matrix ``s`` with ``s @ s == matrix``, up to the
        numerical slack in the positive-semidefinite check.

    Raises:
        ValueError: If ``matrix`` is not a non-empty square matrix, contains
            non-finite values, is not symmetric, or is not positive
            semidefinite within ``rtol``.
    """
    m = check_square(matrix, "matrix")
    check_finite(m, "matrix")
    check_symmetric(m, "matrix")
    check_psd(m, "matrix", rtol=rtol)
    return _sqrt_psd_unchecked(m)


def _sqrt_psd_unchecked(matrix: np.ndarray) -> np.ndarray:
    """Square root of a symmetric matrix, clipping negative eigenvalues to zero.

    This is the numerical workhorse behind :func:`sqrt_psd`.  It is kept
    separate because a product of two PSD matrices is PSD in exact
    arithmetic but not necessarily in floating point: for nearly orthogonal
    states the product is dominated by rounding noise, and its eigenvalues
    are as likely to be slightly negative as slightly positive.  Clipping is
    the correct treatment there -- the quantity being computed is small
    either way -- whereas rejecting the input would be spurious.

    Args:
        matrix: Symmetric matrix, assumed PSD in exact arithmetic.

    Returns:
        Symmetric PSD matrix ``s`` with ``s @ s == matrix`` up to the
        clipped eigenvalues.
    """
    vals, vecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _check_base(base: float) -> float:
    """Validate a logarithm base and return it as a float.

    A logarithm base must be positive and different from one: a base of one
    has no inverse, and a non-positive base has no real logarithm.  Bases
    below one are legal but give the entropy a negative sign, which is the
    standard convention for a logarithm of base below one.

    Args:
        base: Candidate logarithm base.

    Returns:
        ``base`` as a float.

    Raises:
        ValueError: If ``base`` is not a positive finite number other than
            one.
    """
    value = float(base)
    if not np.isfinite(value) or value <= 0.0 or value == 1.0:
        raise ValueError(
            "base must be a positive finite number different from 1")
    return value
