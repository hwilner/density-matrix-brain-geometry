"""Unit tests for the rho_geometry package."""

import unittest

import numpy as np

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


def pure_state(vec):
    """Density matrix of a pure state |v><v|."""
    v = np.asarray(vec, dtype=float)
    v = v / np.linalg.norm(v)
    return np.outer(v, v)


class TestConstruction(unittest.TestCase):
    """covariance -> rho normalisation and validation."""

    def test_rho_is_trace_one_psd(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(6, 6))
        cov = a @ a.T + np.eye(6)
        rho = covariance_to_rho(cov)
        self.assertAlmostEqual(np.trace(rho), 1.0, places=12)
        self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(), -1e-12)

    def test_non_symmetric_raises(self):
        with self.assertRaises(ValueError):
            covariance_to_rho(np.array([[1.0, 0.5], [0.0, 1.0]]))

    def test_negative_eigenvalue_raises(self):
        with self.assertRaises(ValueError):
            covariance_to_rho(np.diag([1.0, -0.5]))

    def test_zero_trace_raises(self):
        with self.assertRaises(ValueError):
            covariance_to_rho(np.zeros((3, 3)))


class TestEntropies(unittest.TestCase):
    """Analytic entropy anchors."""

    def test_maximally_mixed_entropy(self):
        d = 8
        rho = maximally_mixed(d)
        self.assertAlmostEqual(von_neumann_entropy(rho), np.log(d), places=10)
        self.assertAlmostEqual(von_neumann_entropy(rho, base=2), 3.0, places=10)

    def test_pure_state_entropy_zero(self):
        self.assertAlmostEqual(von_neumann_entropy(pure_state([1, 2, 3])),
                               0.0, places=10)

    def test_purity_bounds(self):
        d = 5
        self.assertAlmostEqual(purity(maximally_mixed(d)), 1 / d, places=12)
        self.assertAlmostEqual(purity(pure_state([0.3, 0.4, 0.5])), 1.0,
                               places=12)

    def test_renyi_matches_vn_near_q1(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(5, 5))
        rho = covariance_to_rho(a @ a.T + np.eye(5))
        s1 = von_neumann_entropy(rho)
        s2 = renyi_entropy(rho, q=1.0001)
        self.assertAlmostEqual(s1, s2, delta=1e-3)

    def test_renyi2_equals_minus_log_purity(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(4, 4))
        rho = covariance_to_rho(a @ a.T + np.eye(4))
        self.assertAlmostEqual(renyi_entropy(rho, q=2),
                               -np.log(purity(rho)), places=10)

    def test_effective_rank(self):
        d = 10
        self.assertAlmostEqual(effective_rank(maximally_mixed(d)), d,
                               places=10)
        self.assertAlmostEqual(effective_rank(pure_state([1, 1, 1])), 1.0,
                               places=10)


class TestDistances(unittest.TestCase):
    """Fidelity, Bures and trace distance against analytic values."""

    def test_fidelity_self_is_one(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(5, 5))
        rho = covariance_to_rho(a @ a.T + np.eye(5))
        self.assertAlmostEqual(uhlmann_fidelity(rho, rho), 1.0, places=8)

    def test_orthogonal_pure_states(self):
        rho = pure_state([1, 0, 0])
        sigma = pure_state([0, 1, 0])
        self.assertAlmostEqual(uhlmann_fidelity(rho, sigma), 0.0, places=10)
        self.assertAlmostEqual(trace_distance(rho, sigma), 1.0, places=10)
        self.assertAlmostEqual(bures_distance(rho, sigma), np.sqrt(2),
                               places=8)

    def test_classical_states_reduce_to_bhattacharyya(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        f = uhlmann_fidelity(np.diag(p), np.diag(q))
        expected = np.sum(np.sqrt(p * q)) ** 2
        self.assertAlmostEqual(f, expected, places=10)

    def test_trace_distance_classical_is_tvd(self):
        p = np.array([0.6, 0.4, 0.0])
        q = np.array([0.2, 0.5, 0.3])
        d = trace_distance(np.diag(p), np.diag(q))
        self.assertAlmostEqual(d, 0.5 * np.abs(p - q).sum(), places=10)

    def test_bures_zero_for_identical(self):
        rho = maximally_mixed(4)
        self.assertAlmostEqual(bures_distance(rho, rho), 0.0, places=8)

    def test_sqrt_psd(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(4, 4))
        m = a @ a.T
        s = sqrt_psd(m)
        np.testing.assert_allclose(s @ s, m, atol=1e-10)

    def test_fidelity_matrix_symmetric_unit_diagonal(self):
        states = [pure_state([1, 0]), pure_state([1, 1]),
                  maximally_mixed(2)]
        m = fidelity_matrix(states)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-10)
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        # |0> vs |+>: root fidelity = |<0|+>| = 1/sqrt(2)
        self.assertAlmostEqual(m[0, 1], 1 / np.sqrt(2), places=8)


if __name__ == "__main__":
    unittest.main()
