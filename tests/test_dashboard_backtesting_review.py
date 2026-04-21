from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from dashboard.backtesting_review import build_review_dataset, filter_review_dataset, load_backtest_artifacts


def make_workspace_temp_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1] / "artifacts" / "_test_dashboard_review"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_review_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC"),
            "price_eur_mwh": [50.0 + idx for idx in range(30)],
            "pred_price_eur_mwh": [50.5 + idx for idx in range(30)],
            "pred_demand_kw": [10000.0 + idx * 10 for idx in range(30)],
            "pred_renewable_mw": [8.0 + idx * 0.01 for idx in range(30)],
            "strategy_return": [0.01 if idx % 2 == 0 else -0.005 for idx in range(30)],
            "pnl": [100.0 if idx % 2 == 0 else -50.0 for idx in range(30)],
            "decision": ["LONG" if idx % 3 == 0 else "SHORT" if idx % 3 == 1 else "HOLD" for idx in range(30)],
            "equity_curve": [1.0 + idx * 0.01 for idx in range(30)],
            "cumulative_returns": [idx * 0.01 for idx in range(30)],
            "position": [0.5 if idx % 3 == 0 else -0.5 if idx % 3 == 1 else 0.0 for idx in range(30)],
        }
    )


class DashboardBacktestingReviewTests(unittest.TestCase):
    def test_load_backtest_artifacts_reads_saved_files(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            df = build_review_source_df()
            (temp_dir / "backtest_results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
            (temp_dir / "backtest_metrics.json").write_text(json.dumps({"sharpe_ratio": 1.2}), encoding="utf-8")
            (temp_dir / "backtest_analytics.json").write_text(json.dumps({"accuracy_summary": {"directional_accuracy": 0.7}}), encoding="utf-8")

            loaded_df, metrics, analytics = load_backtest_artifacts(temp_dir)

            self.assertEqual(len(loaded_df), len(df))
            self.assertEqual(metrics["sharpe_ratio"], 1.2)
            self.assertIn("accuracy_summary", analytics)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_load_backtest_artifacts_raises_for_missing_files(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            with self.assertRaises(FileNotFoundError):
                load_backtest_artifacts(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_review_dataset_supports_horizon_switching(self) -> None:
        df = build_review_source_df()

        next_period_df, next_period_summary = build_review_dataset(df, horizon_steps=1, hold_tolerance_pct=0.002)
        next_24h_df, next_24h_summary = build_review_dataset(df, horizon_steps=24, hold_tolerance_pct=0.002)

        self.assertIn("directional_correct", next_period_df.columns)
        self.assertIn("future_price_change_eur_mwh", next_period_df.columns)
        self.assertNotEqual(next_period_summary["evaluable_rows"], next_24h_summary["evaluable_rows"])
        self.assertEqual(next_period_summary["accuracy_horizon_steps"], 1)
        self.assertEqual(next_24h_summary["accuracy_horizon_steps"], 24)

    def test_filter_review_dataset_limits_date_range(self) -> None:
        df = build_review_source_df()
        filtered = filter_review_dataset(df, start_date="2026-01-01 05:00:00+00:00", end_date="2026-01-01 10:00:00+00:00")

        self.assertEqual(len(filtered), 6)
        self.assertEqual(filtered["timestamp_utc"].min().hour, 5)
        self.assertEqual(filtered["timestamp_utc"].max().hour, 10)


if __name__ == "__main__":
    unittest.main()
