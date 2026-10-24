"""Tests for the reproducibility artefacts.

The noise-floor table quoted in ``docs/CURRENT_RESULTS_AND_DISCUSSION.md``
must be the output of ``analysis/noise_floor.py``; otherwise the published
numbers are not reproducible.  These tests check that the script runs, that
it is deterministic, and that the document quotes the values it produces.
"""

import importlib.util
import pathlib
import re
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "analysis" / "noise_floor.py"
RESULTS_PATH = REPO_ROOT / "docs" / "CURRENT_RESULTS_AND_DISCUSSION.md"


def _load_script():
    """Import analysis/noise_floor.py as a module.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location("noise_floor", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["noise_floor"] = module
    spec.loader.exec_module(module)
    return module


class TestNoiseFloorScript(unittest.TestCase):
    """The committed script must be runnable, deterministic and quotable."""

    def setUp(self):
        self.module = _load_script()

    def test_script_exists_and_runs(self):
        self.assertTrue(SCRIPT_PATH.is_file())
        self.assertTrue(hasattr(self.module, "white_noise_floor"))
        self.assertTrue(hasattr(self.module, "main"))

    def test_floor_is_deterministic(self):
        """The same seed must give the same numbers.

        The published table is a baseline an empirical value is judged
        against, so it must not move between runs.
        """
        first = self.module.white_noise_floor(50, 2000, seed=0)
        second = self.module.white_noise_floor(50, 2000, seed=0)
        self.assertEqual(first, second)

    def test_floor_is_below_the_maximum(self):
        """Finite T concentrates the spectrum, so the floor is below ln d."""
        for _label, d, n_timepoints in self.module.CONFIGS:
            with self.subTest(d=d, n_timepoints=n_timepoints):
                stats = self.module.white_noise_floor(d, n_timepoints, seed=0)
                self.assertLess(stats["entropy_bits"],
                                stats["max_entropy_bits"])
                self.assertLess(stats["effective_rank"],
                                stats["max_effective_rank"])
                self.assertGreaterEqual(stats["purity"], 0.0)
                self.assertLessEqual(stats["purity"], 1.0)

    def test_results_document_quotes_the_script_output(self):
        """Every row of the published table must match the script.

        The document previously quoted a noise floor for d = 50, T = 2000
        that the code did not produce, so the numbers are checked against
        the script rather than trusted.
        """
        text = RESULTS_PATH.read_text()
        rows = re.findall(
            r"^\|\s*d = (\d+), T = (\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|$",
            text, flags=re.MULTILINE)
        self.assertTrue(rows, "no noise-floor table found in the results doc")
        documented = {(int(d), int(t)) for d, t, *_ in rows}
        self.assertEqual(documented,
                         {(d, t) for _, d, t in self.module.CONFIGS})
        for d, t, entropy, max_entropy, purity, rank, max_rank in rows:
            with self.subTest(d=d, n_timepoints=t):
                stats = self.module.white_noise_floor(int(d), int(t), seed=0)
                self.assertAlmostEqual(float(entropy),
                                       stats["entropy_bits"], places=3)
                self.assertAlmostEqual(float(max_entropy),
                                       stats["max_entropy_bits"], places=3)
                self.assertAlmostEqual(float(purity), stats["purity"],
                                       places=5)
                self.assertAlmostEqual(float(rank),
                                       stats["effective_rank"], places=2)
                self.assertAlmostEqual(float(max_rank),
                                       stats["max_effective_rank"], places=2)


if __name__ == "__main__":
    unittest.main()
