from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard.backtesting_review import (
    DEFAULT_SIMULATION_SCORED_CSV,
    build_review_dataset,
    default_scored_csv_path,
    filter_review_dataset,
    load_backtest_artifacts,
    load_model_comparison,
    run_model_comparison_workflow,
)


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

    def test_load_model_comparison_reads_saved_files(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            summary_df = pd.DataFrame(
                [{"rank": 1, "model_key": "xgboost", "directional_accuracy": 0.62, "price_mae": 4.1}]
            )
            (temp_dir / "model_comparison_summary.csv").write_text(summary_df.to_csv(index=False), encoding="utf-8")
            (temp_dir / "model_comparison_summary.json").write_text(
                json.dumps({"winner_model": "xgboost", "rows": summary_df.to_dict(orient="records")}),
                encoding="utf-8",
            )

            loaded_df, metadata = load_model_comparison(temp_dir)

            self.assertEqual(len(loaded_df), 1)
            self.assertEqual(metadata["winner_model"], "xgboost")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_default_scored_csv_path_falls_back_to_simulation_path(self) -> None:
        with patch("dashboard.backtesting_review.Path.exists", return_value=False):
            path = default_scored_csv_path()

        self.assertIsInstance(path, Path)
        self.assertEqual(path, DEFAULT_SIMULATION_SCORED_CSV)

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

    def test_filter_review_dataset_accepts_naive_dates_for_utc_timestamps(self) -> None:
        df = build_review_source_df()

        filtered = filter_review_dataset(df, start_date=pd.Timestamp("2026-01-01").date(), end_date=pd.Timestamp("2026-01-01").date())

        self.assertEqual(len(filtered), 24)
        self.assertEqual(filtered["timestamp_utc"].min().hour, 0)
        self.assertEqual(filtered["timestamp_utc"].max().hour, 23)

    def test_filter_review_dataset_end_date_includes_full_day(self) -> None:
        df = build_review_source_df()

        filtered = filter_review_dataset(df, start_date=pd.Timestamp("2026-01-01").date(), end_date=pd.Timestamp("2026-01-02").date())

        self.assertEqual(len(filtered), 30)
        self.assertEqual(filtered["timestamp_utc"].max().day, 2)
        self.assertEqual(filtered["timestamp_utc"].max().hour, 5)

    def test_run_model_comparison_workflow_returns_metadata(self) -> None:
        summary_df = pd.DataFrame([{"model_key": "xgboost", "directional_accuracy": 0.61}])
        with patch("dashboard.backtesting_review.run_data_pipeline") as mock_pipeline, patch(
            "dashboard.backtesting_review.build_features"
        ) as mock_build_features, patch("dashboard.backtesting_review.run_model_comparison") as mock_run_model_comparison:
            mock_pipeline.return_value = type(
                "PipelineOut",
                (),
                {
                    "merged_df": pd.DataFrame({"x": [1]}),
                    "energy_source": "synthetic",
                    "provenance_summary": {"synthetic_coverage_ratio": 1.0},
                    "cache_summary": {"energy": {"cache_status": "loaded"}},
                },
            )()
            mock_build_features.return_value = pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")})
            mock_run_model_comparison.return_value = type(
                "ComparisonResult",
                (),
                {"summary_df": summary_df, "metadata": {"winner_model": "xgboost", "failures": []}},
            )()

            loaded_df, metadata = run_model_comparison_workflow(zone="DE_LU", lookback_days=90, output_dir="artifacts/backtesting")

        self.assertEqual(list(loaded_df["model_key"]), ["xgboost"])
        self.assertEqual(metadata["winner_model"], "xgboost")
        self.assertEqual(metadata["energy_source"], "synthetic")
        self.assertIn("provenance_summary", metadata)
        self.assertIn("cache_summary", metadata)


if __name__ == "__main__":
    unittest.main()
