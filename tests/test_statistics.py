from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.statistics import bootstrap_sharpe_ci, compare_return_streams


class BootstrapSharpeCiTests(unittest.TestCase):
    def test_empty_series_returns_zeros(self) -> None:
        result = bootstrap_sharpe_ci(pd.Series([], dtype=float), annualization_factor=24)
        self.assertEqual(result, {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})

    def test_ci_bounds_ordered(self) -> None:
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        result = bootstrap_sharpe_ci(returns, annualization_factor=24, iterations=100)
        self.assertLessEqual(result["ci_lower"], result["ci_upper"])

    def test_deterministic_with_same_seed(self) -> None:
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0.0, 0.01, 300))
        first = bootstrap_sharpe_ci(returns, annualization_factor=24, iterations=80, random_seed=42)
        second = bootstrap_sharpe_ci(returns, annualization_factor=24, iterations=80, random_seed=42)
        self.assertEqual(first, second)

    def test_positive_drift_gives_positive_sharpe(self) -> None:
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0.01, 0.005, 400))
        result = bootstrap_sharpe_ci(returns, annualization_factor=24, iterations=100)
        self.assertGreater(result["sharpe"], 0.0)
        self.assertGreater(result["ci_lower"], 0.0)


class CompareReturnStreamsTests(unittest.TestCase):
    def test_empty_streams_return_defaults(self) -> None:
        result = compare_return_streams(
            pd.Series([], dtype=float),
            pd.Series([0.01, 0.02]),
            annualization_factor=24,
        )
        self.assertFalse(result["significant_outperformance"])
        self.assertEqual(result["sharpe_diff"], 0.0)

    def test_clear_outperformance_is_significant(self) -> None:
        rng = np.random.default_rng(4)
        baseline = pd.Series(rng.normal(0.0, 0.01, 500))
        strategy = baseline + 0.02
        result = compare_return_streams(strategy, baseline, annualization_factor=24, iterations=100)
        self.assertTrue(result["significant_outperformance"])
        self.assertGreater(result["sharpe_diff"], 0.0)
        self.assertGreater(result["pnl_diff_mean"], 0.0)

    def test_identical_streams_not_significant(self) -> None:
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0, 0.01, 500))
        result = compare_return_streams(returns, returns.copy(), annualization_factor=24, iterations=100)
        self.assertFalse(result["significant_outperformance"])
        self.assertAlmostEqual(result["sharpe_diff"], 0.0, places=9)
        self.assertAlmostEqual(result["pnl_diff_mean"], 0.0, places=9)

    def test_deterministic_with_same_seed(self) -> None:
        rng = np.random.default_rng(6)
        strategy = pd.Series(rng.normal(0.002, 0.01, 300))
        baseline = pd.Series(rng.normal(0.0, 0.01, 300))
        first = compare_return_streams(strategy, baseline, annualization_factor=24, iterations=80, random_seed=42)
        second = compare_return_streams(strategy, baseline, annualization_factor=24, iterations=80, random_seed=42)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
