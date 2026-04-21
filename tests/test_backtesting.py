from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.backtesting.engine import (
    BacktestConfig,
    compute_max_drawdown,
    compute_sharpe,
    compute_trade_count,
    compute_turnover_summary,
    evaluate_decision_accuracy,
    run_backtest,
)


def build_scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "price_eur_mwh": [50.0, 52.0, 49.0, 55.0, 53.0, 58.0],
            "pred_price_eur_mwh": [51.0, 54.0, 50.0, 57.0, 52.0, 60.0],
            "pred_demand_kw": [10000.0, 10500.0, 9800.0, 11000.0, 10800.0, 11200.0],
            "pred_renewable_mw": [8.5, 8.0, 7.8, 9.2, 8.1, 8.7],
        }
    )


def make_workspace_temp_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1] / "artifacts" / "_test_backtesting"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class MetricHelperTests(unittest.TestCase):
    def test_sharpe_is_zero_for_zero_volatility(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        self.assertEqual(compute_sharpe(returns, annualization_factor=24), 0.0)

    def test_max_drawdown_handles_monotonic_growth(self) -> None:
        equity = pd.Series([1.0, 1.1, 1.2, 1.3])
        self.assertEqual(compute_max_drawdown(equity), 0.0)

    def test_max_drawdown_handles_drawdown_series(self) -> None:
        equity = pd.Series([1.0, 1.2, 0.9, 0.95])
        self.assertAlmostEqual(compute_max_drawdown(equity), -0.25)

    def test_turnover_and_trade_helpers_handle_edge_cases(self) -> None:
        turnover = pd.Series([0.0, 0.2, 0.0, 0.4])
        decisions = pd.Series(["HOLD", "LONG", "HOLD", "SHORT"])
        summary = compute_turnover_summary(turnover)

        self.assertEqual(compute_trade_count(decisions), 2)
        self.assertAlmostEqual(summary["mean_turnover"], 0.15)
        self.assertAlmostEqual(summary["max_turnover"], 0.4)
        self.assertAlmostEqual(summary["total_turnover"], 0.6)


class DecisionAccuracyTests(unittest.TestCase):
    def test_directional_accuracy_labels_long_short_and_hold(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
                "price_eur_mwh": [100.0, 101.0, 100.05, 100.04],
                "decision": ["LONG", "SHORT", "HOLD", "LONG"],
                "strategy_return": [0.01, -0.01, 0.0, 0.0],
                "pnl": [100.0, -100.0, 0.0, 0.0],
            }
        )

        review_df, summary = evaluate_decision_accuracy(df, horizon_steps=1, hold_tolerance_pct=0.002)

        self.assertEqual(review_df.loc[0, "accuracy_status"], "correct")
        self.assertEqual(review_df.loc[1, "accuracy_status"], "correct")
        self.assertEqual(review_df.loc[2, "accuracy_status"], "correct")
        self.assertEqual(review_df.loc[3, "accuracy_status"], "pending")
        self.assertAlmostEqual(summary["directional_accuracy"], 1.0)

    def test_horizon_switch_changes_accuracy_results(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
                "price_eur_mwh": [100.0, 101.0, 99.0, 102.0],
                "decision": ["LONG", "LONG", "LONG", "HOLD"],
                "strategy_return": [0.01, 0.01, 0.01, 0.0],
                "pnl": [100.0, 100.0, 100.0, 0.0],
            }
        )

        next_step_df, _ = evaluate_decision_accuracy(df, horizon_steps=1, hold_tolerance_pct=0.002)
        next_day_df, _ = evaluate_decision_accuracy(df, horizon_steps=2, hold_tolerance_pct=0.002)

        self.assertEqual(next_step_df.loc[1, "accuracy_status"], "incorrect")
        self.assertEqual(next_day_df.loc[1, "accuracy_status"], "correct")


class BacktestEngineTests(unittest.TestCase):
    def test_run_backtest_writes_isolated_artifacts_and_review_columns(self) -> None:
        df = build_scored_df()
        simulation_metrics = Path("artifacts/simulation/backtest_metrics.json")
        original_simulation_exists = simulation_metrics.exists()

        temp_dir = make_workspace_temp_dir()
        try:
            output_dir = temp_dir / "backtesting"
            result = run_backtest(df, BacktestConfig(output_dir=output_dir, accuracy_horizon_steps=1, hold_tolerance_pct=0.002))

            self.assertTrue((output_dir / "backtest_results.csv").exists())
            self.assertTrue((output_dir / "backtest_metrics.json").exists())
            self.assertTrue((output_dir / "backtest_analytics.json").exists())
            self.assertIn("equity_curve", result.result_df.columns)
            self.assertIn("cumulative_returns", result.result_df.columns)
            self.assertIn("future_price_change_eur_mwh", result.result_df.columns)
            self.assertIn("directional_correct", result.result_df.columns)
            self.assertIn("pnl_positive", result.result_df.columns)
            self.assertIn("directional_accuracy", result.metrics)
            self.assertIn("pnl_positive_rate", result.metrics)
            self.assertIn("accuracy_summary", result.analytics)
            self.assertEqual(simulation_metrics.exists(), original_simulation_exists)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_run_all_imports_without_new_backtesting_dependency(self) -> None:
        hf_module = types.ModuleType("huggingface_hub")
        hf_module.InferenceClient = object
        torch_module = types.ModuleType("torch")
        torch_nn_module = types.ModuleType("torch.nn")
        torch_nn_module.Module = object
        torch_module.nn = torch_nn_module
        torch_module.Tensor = object

        with patch.dict(
            sys.modules,
            {
                "huggingface_hub": hf_module,
                "torch": torch_module,
                "torch.nn": torch_nn_module,
            },
        ):
            import scripts.run_all as run_all

            self.assertTrue(callable(run_all.parse_args))
            self.assertTrue(callable(run_all.run_workflow))


class BacktestCliTests(unittest.TestCase):
    def test_cli_creates_expected_artifacts(self) -> None:
        df = build_scored_df()

        temp_path = make_workspace_temp_dir()
        try:
            input_path = temp_path / "scored.csv"
            output_dir = temp_path / "artifacts"
            df.to_csv(input_path, index=False)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_backtest.py",
                    "--input-path",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--accuracy-horizon-steps",
                    "24",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(completed.stdout)
            self.assertTrue((output_dir / "backtest_results.csv").exists())
            self.assertTrue((output_dir / "backtest_metrics.json").exists())
            self.assertTrue((output_dir / "backtest_analytics.json").exists())
            self.assertIn("metrics", payload)
            self.assertIn("analytics", payload)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
