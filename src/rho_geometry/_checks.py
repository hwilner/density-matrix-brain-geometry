"""Shared input validation for the public :mod:`rho_geometry` API.

Every public function in this package documents the shape of the matrix it
accepts.  These helpers turn those documented contracts into runtime checks,
so a caller who violates one gets a clear error instead of a well-formed
number that silently means something else.

The tolerances are *relative* to the scale of the matrix under test.  An
absolute tolerance is not usable here: a covariance matrix is only defined up
to its magnitude, so a check that is absolute would accept an indefinite
matrix of small magnitude and reject the same matrix scaled up, even though
the two carry the same information.
"""

from __future__ import annotations

import numpy as np

#: Relative tolerance for the symmetry check, applied to the matrix scale.
SYMMETRY_RTOL = 1e-10

#: Relative tolerance for the positive-semidefinite check, applied to the
#: largest-magnitude eigenvalue of the matrix under test.
PSD_RTOL = 1e-8

#: Absolute tolerance for the unit-trace check.  A density matrix has trace
#: one by definition, so this tolerance is stated on the trace itself.
TRACE_RTOL = 1e-8


def spectral_scale(matrix: np.ndarray) -> float:
    """Return the largest-magnitude entry of ``matrix``.

    Args:
        matrix: Array to measure.

    Returns:
        The largest absolute entry, or 0.0 for an empty array.
    """
    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(matrix)))


def check_square(matrix: np.ndarray, name: str) -> np.ndarray:
    """Return ``matrix`` as a float array after checking it is a square matrix.

    Args:
        matrix: Candidate matrix.
        name: Name of the argument, used in the error message.

    Returns:
        ``matrix`` cast to a float array.

    Raises:
        ValueError: If the matrix is not square or is empty.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def check_finite(matrix: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` unless every entry of ``matrix`` is finite.

    Args:
        matrix: Candidate matrix.
        name: Name of the argument, used in the error message.

    Raises:
        ValueError: If the matrix contains ``nan`` or an infinity.
    """
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")


def check_symmetric(matrix: np.ndarray, name: str,
                    rtol: float = SYMMETRY_RTOL) -> None:
    """Raise ``ValueError`` unless ``matrix`` is symmetric to within ``rtol``.

    The comparison is relative to the largest-magnitude entry of the matrix,
    so the same matrix passes or fails the check independently of its
    magnitude.

    Args:
        matrix: Candidate matrix.
        name: Name of the argument, used in the error message.
        rtol: Relative symmetry tolerance.

    Raises:
        ValueError: If the matrix is not symmetric within ``rtol``.
    """
    scale = spectral_scale(matrix)
    if scale == 0.0:
        return
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > rtol * scale:
        raise ValueError(
            f"{name} must be symmetric (max |A - A.T| = {asymmetry:.3e} exceeds "
            f"{rtol:.1e} times the matrix scale {scale:.3e})")


def check_psd(matrix: np.ndarray, name: str, rtol: float = PSD_RTOL) -> None:
    """Raise ``ValueError`` unless ``matrix`` is positive semidefinite.

    The slack is relative to the largest-magnitude eigenvalue, so a matrix
    whose spectrum is non-negative up to floating-point noise passes, while a
    genuinely indefinite matrix fails at any magnitude.

    Args:
        matrix: Symmetric candidate matrix.
        name: Name of the argument, used in the error message.
        rtol: Relative positive-semidefinite tolerance.

    Raises:
        ValueError: If the matrix has an eigenvalue below ``-rtol`` times the
            largest-magnitude eigenvalue.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0.0:
        return
    if float(eigenvalues.min()) < -rtol * scale:
        raise ValueError(
            f"{name} must be positive semidefinite (smallest eigenvalue "
            f"{eigenvalues.min():.3e} is below -{rtol:.1e} times the spectral "
            f"scale {scale:.3e})")


def as_density_matrix(matrix: np.ndarray, name: str = "rho") -> np.ndarray:
    """Validate ``matrix`` as a density matrix and return it as a float array.

    A density matrix is square, finite, symmetric, positive semidefinite and
    has unit trace.  The trace is checked last because it is the cheapest way
    to reject a matrix that is not a state at all.

    Args:
        matrix: Candidate density matrix.
        name: Name of the argument, used in the error message.

    Returns:
        ``matrix`` cast to a float array.

    Raises:
        ValueError: If the matrix is not a valid density matrix.
    """
    array = check_square(matrix, name)
    check_finite(array, name)
    check_symmetric(array, name)
    check_psd(array, name)
    trace = float(np.trace(array))
    if abs(trace - 1.0) > TRACE_RTOL:
        raise ValueError(f"{name} must have unit trace (got {trace!r})")
    return array
