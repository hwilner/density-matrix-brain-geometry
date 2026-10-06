"""Density-matrix information geometry of brain states.

Public API: density-matrix construction and entropies
(:mod:`rho_geometry.density`) and state distances
(:mod:`rho_geometry.metrics`).
"""

from rho_geometry.density import (
    covariance_to_rho,
    effective_rank,
    maximally_mixed,
    purity,
    renyi_entropy,
    sqrt_psd,
    von_neumann_entropy,
)
from rho_geometry.metrics import (
    bures_distance,
    fidelity_matrix,
    trace_distance,
    uhlmann_fidelity,
)

__all__ = [
    "bures_distance",
    "covariance_to_rho",
    "effective_rank",
    "fidelity_matrix",
    "maximally_mixed",
    "purity",
    "renyi_entropy",
    "sqrt_psd",
    "trace_distance",
    "uhlmann_fidelity",
    "von_neumann_entropy",
]

__version__ = "0.1.0"
