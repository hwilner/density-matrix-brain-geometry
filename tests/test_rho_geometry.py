"""Unit tests for the rho_geometry package."""

import unittest

import numpy as np

from rho_geometry.density import (
    covariance_to_rho,
    effective_rank,
    maximally_mixed,
    project_to_psd,
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

    def test_indefinite_matrix_raises_at_every_scale(self):
        """The PSD check must not depend on the magnitude of the matrix.

        A covariance matrix is defined only up to its scale, so the same
        matrix scaled up or down carries the same information and must be
        judged by the same rule.  Before the fix, a small-scale indefinite
        matrix slipped through the absolute tolerance and produced a
        "density matrix" with a large negative eigenvalue.
        """
        for scale in (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6):
            with self.subTest(scale=scale):
                cov = np.diag([2.0 * scale, -1.0 * scale])
                with self.assertRaises(ValueError):
                    covariance_to_rho(cov)

    def test_returned_rho_is_always_psd(self):
        """Every accepted input must yield a matrix with rho >= 0."""
        rng = np.random.default_rng(7)
        for scale in (1e-9, 1e-3, 1.0, 1e3, 1e9):
            with self.subTest(scale=scale):
                a = rng.normal(size=(5, 5)) * scale
                cov = a @ a.T
                rho = covariance_to_rho(cov)
                self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(), -1e-12)
                self.assertAlmostEqual(np.trace(rho), 1.0, places=12)

    def test_non_finite_raises(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                cov = np.array([[1.0, 0.0], [0.0, bad]])
                with self.assertRaises(ValueError):
                    covariance_to_rho(cov)

    def test_empty_matrix_raises(self):
        with self.assertRaises(ValueError):
            covariance_to_rho(np.zeros((0, 0)))

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            covariance_to_rho(np.ones((2, 3)))

    def test_maximally_mixed_rejects_non_integer(self):
        for bad in (0, -1, 2.5, True):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    maximally_mixed(bad)


class TestProjection(unittest.TestCase):
    """The nearest-PSD guard rail described in docs/METHODS.md."""

    def test_projection_clips_negative_eigenvalues(self):
        projected, clipped = project_to_psd(np.diag([1.0, -0.5, 2.0]))
        np.testing.assert_allclose(np.linalg.eigvalsh(projected),
                                   [0.0, 1.0, 2.0], atol=1e-12)
        self.assertAlmostEqual(clipped, 0.5, places=12)

    def test_projection_is_idempotent_on_psd_input(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(5, 5))
        cov = a @ a.T
        projected, clipped = project_to_psd(cov)
        self.assertAlmostEqual(clipped, 0.0, places=12)
        np.testing.assert_allclose(projected, cov, atol=1e-10)

    def test_projected_matrix_is_accepted_by_covariance_to_rho(self):
        """The guard rail must produce input the constructor accepts."""
        projected, _ = project_to_psd(np.diag([1.0, -0.5, 2.0]))
        rho = covariance_to_rho(projected)
        self.assertAlmostEqual(np.trace(rho), 1.0, places=12)
        self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(), -1e-12)

    def test_projection_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            project_to_psd(np.array([[1.0, 0.5], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            project_to_psd(np.array([[np.inf, 0.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            project_to_psd(np.zeros((3, 3)))


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

    def test_renyi_order_zero_is_log_support_size(self):
        """S_0 = log of the number of populated modes."""
        rho = np.diag([0.5, 0.25, 0.25, 0.0])
        self.assertAlmostEqual(renyi_entropy(rho, q=0.0), np.log(3.0),
                               places=10)

    def test_effective_rank(self):
        d = 10
        self.assertAlmostEqual(effective_rank(maximally_mixed(d)), d,
                               places=10)
        self.assertAlmostEqual(effective_rank(pure_state([1, 1, 1])), 1.0,
                               places=10)

    def test_entropy_rejects_invalid_logarithm_base(self):
        """An invalid base must raise instead of returning inf or nan.

        base = 1 divides by log(1) = 0 and returned inf; base = 0 and
        negative bases produced nan; all three were returned silently.
        """
        rho = maximally_mixed(4)
        for bad in (0.0, 1.0, -2.0, np.nan, np.inf):
            with self.subTest(base=bad):
                with self.assertRaises(ValueError):
                    von_neumann_entropy(rho, base=bad)
                with self.assertRaises(ValueError):
                    renyi_entropy(rho, q=2.0, base=bad)

    def test_entropy_rejects_invalid_renyi_order(self):
        """A negative order is undefined for a state with zero modes."""
        rho = maximally_mixed(4)
        for bad in (-1.0, np.nan, np.inf):
            with self.subTest(q=bad):
                with self.assertRaises(ValueError):
                    renyi_entropy(rho, q=bad)

    def test_entropy_rejects_non_density_matrix(self):
        """The documented input is a density matrix: trace 1 and PSD."""
        with self.assertRaises(ValueError):
            von_neumann_entropy(np.diag([2.0, 0.0]))
        with self.assertRaises(ValueError):
            von_neumann_entropy(np.diag([1.5, -0.5]))
        with self.assertRaises(ValueError):
            purity(np.diag([2.0, 0.0]))
        with self.assertRaises(ValueError):
            effective_rank(np.diag([2.0, 0.0]))


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

    def test_sqrt_psd_rejects_non_symmetric(self):
        """numpy.linalg.eigh reads only the lower triangle.

        A non-symmetric input therefore used to return the square root of a
        different matrix, with no error raised.
        """
        with self.assertRaises(ValueError):
            sqrt_psd(np.array([[1.0, 5.0], [0.0, 1.0]]))

    def test_sqrt_psd_rejects_indefinite(self):
        """The documented guarantee s @ s == matrix must hold."""
        with self.assertRaises(ValueError):
            sqrt_psd(np.diag([1.0, -0.5]))

    def test_sqrt_psd_rejects_non_finite_and_empty(self):
        with self.assertRaises(ValueError):
            sqrt_psd(np.array([[np.inf, 0.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            sqrt_psd(np.zeros((0, 0)))

    def test_trace_distance_rejects_non_symmetric(self):
        """eigvalsh reads only the lower triangle of rho - sigma."""
        with self.assertRaises(ValueError):
            trace_distance(np.array([[1.0, 0.9], [0.0, 0.0]]),
                           np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_metrics_reject_non_density_matrices(self):
        """Fidelity of a non-normalised matrix is not in [0, 1]."""
        unnormalised = np.array([[2.0, 0.0], [0.0, 0.0]])
        other = np.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            uhlmann_fidelity(unnormalised, unnormalised)
        with self.assertRaises(ValueError):
            bures_distance(unnormalised, other)
        with self.assertRaises(ValueError):
            trace_distance(unnormalised, other)

    def test_fidelity_matrix_symmetric_unit_diagonal(self):
        states = [pure_state([1, 0]), pure_state([1, 1]),
                  maximally_mixed(2)]
        m = fidelity_matrix(states)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-10)
        np.testing.assert_allclose(m, m.T, atol=1e-12)
        # |0> vs |+>: root fidelity = |<0|+>| = 1/sqrt(2)
        self.assertAlmostEqual(m[0, 1], 1 / np.sqrt(2), places=8)

    def test_fidelity_matrix_diagonal_matches_pairwise_fidelity(self):
        """The diagonal must come from the states, not from a constant.

        The old implementation hard-coded a unit diagonal, which silently
        disagreed with uhlmann_fidelity for a state that was not
        normalised.
        """
        states = [pure_state([1, 0]), maximally_mixed(2)]
        m = fidelity_matrix(states)
        for i, state in enumerate(states):
            self.assertAlmostEqual(
                m[i, i], uhlmann_fidelity(state, state, squared=False),
                places=10)

    def test_fidelity_matrix_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            fidelity_matrix([pure_state([1, 0]), pure_state([1, 1, 0])])

    def test_fidelity_matrix_rejects_non_density_matrix(self):
        with self.assertRaises(ValueError):
            fidelity_matrix([np.array([[2.0, 0.0], [0.0, 0.0]])])


if __name__ == "__main__":
    unittest.main()
