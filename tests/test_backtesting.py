from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import types
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtesting.comparison import run_model_comparison, sort_model_comparison
from src.backtesting.engine import (
    BacktestConfig,
    compute_price_return,
    compute_max_drawdown,
    compute_price_change,
    compute_sharpe,
    compute_trade_count,
    compute_turnover_summary,
    evaluate_decision_accuracy,
    generate_backtest_outputs,
    run_backtest,
)
from src.config import AppConfig
from src.features.build_features import build_features
from src.models.base import scored_predictions_path
from src.trading.backtest import run_backtest as run_pipeline_backtest
from scripts.run_all import _adjust_walk_forward_config, run_workflow


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

    def test_price_return_sanitizes_zero_price_transitions(self) -> None:
        returns = compute_price_return(pd.Series([50.0, 0.0, 25.0, 30.0]))

        self.assertTrue(pd.notna(returns).all())
        self.assertEqual(list(returns), [0.0, -1.0, 25.0, 0.2])

    def test_price_change_uses_absolute_moves(self) -> None:
        changes = compute_price_change(pd.Series([50.0, -10.0, 5.0]))

        self.assertEqual(list(changes), [0.0, -60.0, 15.0])


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
            self.assertIn("recommended_decision", result.result_df.columns)
            self.assertIn("target_position", result.result_df.columns)
            self.assertIn("position", result.result_df.columns)
            self.assertIn("decision", result.result_df.columns)
            self.assertIn("directional_accuracy", result.metrics)
            self.assertIn("pnl_positive_rate", result.metrics)
            self.assertIn("accuracy_summary", result.analytics)
            self.assertEqual(simulation_metrics.exists(), original_simulation_exists)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_price_led_recommendation_does_not_go_long_on_bearish_price_edge(self) -> None:
        df = build_scored_df()
        df["price_eur_mwh"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        df["pred_price_eur_mwh"] = df["price_eur_mwh"] - 1.0
        df["pred_demand_kw"] = 50_000_000.0
        df["pred_renewable_mw"] = 1.0

        result_df, _, _ = generate_backtest_outputs(
            df,
            BacktestConfig(
                output_dir=Path("artifacts/backtesting"),
                enable_execution_delay=False,
                long_price_edge_threshold=0.5,
                short_price_edge_threshold=-0.5,
            ),
        )

        self.assertTrue((result_df["recommended_decision"] == "SHORT").all())
        self.assertTrue((result_df["target_position"] < 0.0).all())
        self.assertFalse((result_df["recommended_decision"] == "LONG").any())

    def test_price_led_recommendation_holds_inside_price_edge_band(self) -> None:
        df = build_scored_df()
        df["price_eur_mwh"] = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        df["pred_price_eur_mwh"] = df["price_eur_mwh"] + 0.25
        df["pred_demand_kw"] = 50_000_000.0
        df["pred_renewable_mw"] = 1.0

        result_df, _, _ = generate_backtest_outputs(
            df,
            BacktestConfig(
                output_dir=Path("artifacts/backtesting"),
                enable_execution_delay=False,
                long_price_edge_threshold=0.5,
                short_price_edge_threshold=-0.5,
            ),
        )

        self.assertTrue((result_df["recommended_decision"] == "HOLD").all())
        self.assertTrue((result_df["target_position"] == 0.0).all())

    def test_run_backtest_handles_zero_prices_without_nan_metrics(self) -> None:
        df = build_scored_df()
        df.loc[1, "price_eur_mwh"] = 0.0
        df.loc[2, "price_eur_mwh"] = 10.0

        temp_dir = make_workspace_temp_dir()
        try:
            output_dir = temp_dir / "backtesting"
            result = run_backtest(df, BacktestConfig(output_dir=output_dir))

            self.assertTrue(pd.notna(result.result_df["strategy_return"]).all())
            self.assertEqual(result.metrics["total_pnl"], result.metrics["total_pnl"])
            self.assertEqual(result.analytics["final_equity"], result.analytics["final_equity"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_generate_backtest_outputs_handles_negative_prices_without_exploding(self) -> None:
        df = build_scored_df()
        df["price_eur_mwh"] = [50.0, 0.1, -0.1, 5.0, -2.0, 10.0]

        result_df, metrics, analytics = generate_backtest_outputs(df, BacktestConfig(output_dir=Path("artifacts/backtesting")))

        self.assertTrue(pd.notna(result_df["strategy_return"]).all())
        self.assertLess(result_df["strategy_return"].abs().max(), 1.0)
        self.assertTrue((result_df["equity_curve"] > 0).all())
        self.assertGreaterEqual(metrics["max_drawdown"], -1.0)
        self.assertEqual(analytics["final_equity"], analytics["final_equity"])

    def test_pipeline_backtest_matches_isolated_metric_definitions(self) -> None:
        df = build_scored_df()
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, models_dir=temp_dir / "models", simulation_dir=temp_dir / "simulation")
            cfg.simulation_dir.mkdir(parents=True, exist_ok=True)
            isolated_df, isolated_metrics, _ = generate_backtest_outputs(
                df,
                BacktestConfig(
                    output_dir=temp_dir / "isolated",
                    transaction_cost_bps=cfg.tcost_bps,
                    annualization_factor=cfg.annualization_factor,
                    notional_eur=cfg.backtest_notional_eur,
                ),
            )

            pipeline_out = run_pipeline_backtest(df, cfg)
            pipeline_metrics = json.loads(Path(pipeline_out.metrics_path).read_text(encoding="utf-8"))

            self.assertEqual(len(pipeline_out.result_df), len(isolated_df))
            self.assertAlmostEqual(pipeline_metrics["total_pnl"], isolated_metrics["total_pnl"])
            self.assertAlmostEqual(pipeline_metrics["sharpe_ratio"], isolated_metrics["sharpe_ratio"])
            self.assertAlmostEqual(pipeline_metrics["max_drawdown"], isolated_metrics["max_drawdown"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_features_tolerates_missing_optional_entsoe_context(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, data_processed_dir=temp_dir / "processed")
            cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "timestamp_utc": pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC"),
                    "price_eur_mwh": np.linspace(50.0, 60.0, 80),
                    "demand_kw": np.linspace(10000.0, 11000.0, 80),
                    "renewable_mw": np.linspace(8.0, 10.0, 80),
                    "temperature_c": np.linspace(5.0, 8.0, 80),
                    "wind_speed_mps": np.linspace(4.0, 6.0, 80),
                    "radiation_wm2": np.linspace(50.0, 150.0, 80),
                    "humidity_pct": np.linspace(60.0, 70.0, 80),
                    "day_ahead_price_eur_mwh": np.linspace(49.0, 59.0, 80),
                    "intraday_price_eur_mwh": np.linspace(50.0, 60.0, 80),
                    "imbalance_price_eur_mwh": [np.nan] * 80,
                    "intraday_renewable_forecast_mw": [np.nan] * 80,
                }
            )

            features = build_features(df, cfg)

            self.assertFalse(features.empty)
            self.assertIn("imbalance_price_eur_mwh_available", features.columns)
            self.assertIn("intraday_renewable_forecast_mw_available", features.columns)
            self.assertTrue((features["imbalance_price_eur_mwh_available"] == 0.0).all())
            self.assertNotIn("imbalance_price_buy_eur_mwh", features.columns)
            self.assertNotIn("imbalance_price_sell_eur_mwh", features.columns)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_features_bulk_interaction_columns_match_expected_values(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, data_processed_dir=temp_dir / "processed")
            cfg.data_processed_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "timestamp_utc": pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC"),
                    "price_eur_mwh": np.linspace(50.0, 60.0, 80),
                    "demand_kw": np.linspace(10000.0, 11000.0, 80),
                    "renewable_mw": np.linspace(8.0, 10.0, 80),
                    "temperature_c": np.linspace(5.0, 8.0, 80),
                    "wind_speed_mps": np.linspace(4.0, 6.0, 80),
                    "radiation_wm2": np.linspace(50.0, 150.0, 80),
                    "humidity_pct": np.linspace(60.0, 70.0, 80),
                    "day_ahead_price_eur_mwh": np.linspace(49.0, 59.0, 80),
                    "intraday_price_eur_mwh": np.linspace(50.0, 60.0, 80),
                    "imbalance_price_eur_mwh": np.linspace(51.0, 61.0, 80),
                    "intraday_renewable_forecast_mw": np.linspace(8.5, 10.5, 80),
                }
            )

            features = build_features(df, cfg)

            self.assertIn("net_load_mw_x_hour_sin", features.columns)
            self.assertIn("intraday_day_ahead_spread_eur_mwh_x_dow_cos", features.columns)
            probe = features.iloc[0]
            self.assertAlmostEqual(
                probe["net_load_mw_x_hour_sin"],
                probe["net_load_mw"] * probe["hour_sin"],
            )
            self.assertAlmostEqual(
                probe["renewable_forecast_error_mw_x_dow_cos"],
                probe["renewable_forecast_error_mw"] * probe["dow_cos"],
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_adjust_walk_forward_config_shrinks_train_window_when_features_trim_history(self) -> None:
        cfg = AppConfig(
            walk_forward_train_window_days=90,
            walk_forward_test_window_days=7,
        )

        adjusted = _adjust_walk_forward_config(cfg, feature_rows=2161)

        self.assertEqual(adjusted.walk_forward_train_window_days, 83)
        self.assertEqual(adjusted.walk_forward_test_window_days, 7)

    def test_adjust_walk_forward_config_still_fails_when_not_enough_for_one_split(self) -> None:
        cfg = AppConfig(
            walk_forward_train_window_days=90,
            walk_forward_test_window_days=7,
        )

        with self.assertRaises(ValueError):
            _adjust_walk_forward_config(cfg, feature_rows=(7 * 24 * 2) - 1)

    def test_run_workflow_defaults_skip_model_comparison_when_missing_from_namespace(self) -> None:
        args = argparse.Namespace(zone="DE_LU", lookback_days=90, simulation_horizon=24, model="xgboost")
        fake_pipeline = types.SimpleNamespace(
            merged_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}),
            energy_source="synthetic",
            provenance_summary={
                "real_coverage_ratio": 0.0,
                "partial_synthetic_coverage_ratio": 0.0,
                "synthetic_coverage_ratio": 1.0,
                "research_grade": False,
            },
            cache_summary={"energy": {"cache_status": "loaded"}, "weather": {"cache_status": "loaded"}},
        )
        fake_train = types.SimpleNamespace(
            scored_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}),
            metrics_path="metrics.json",
            diagnostics_path="diag.json",
            scored_path="scored.csv",
            model_key="xgboost",
        )
        fake_backtest = types.SimpleNamespace(result_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}), metrics={})
        fake_strategy = types.SimpleNamespace(
            strategy_metrics_df=pd.DataFrame([{"strategy_name": "upgraded_strategy", "sharpe_ratio": 1.0}]),
            significance_df=pd.DataFrame(),
        )
        fake_model_comparison = types.SimpleNamespace(summary_df=pd.DataFrame(), summary_csv_path="summary.csv", summary_json_path="summary.json")
        fake_research = {"summary": {"source": "deterministic_fallback"}, "json_path": "research.json", "note_path": "research.md"}

        with patch("scripts.run_all.ensure_directories"), patch("scripts.run_all.set_global_seed"), patch(
            "scripts.run_all.run_data_pipeline", return_value=fake_pipeline
        ), patch("scripts.run_all.build_features", return_value=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=24 * 14, freq="h", tz="UTC")})), patch(
            "scripts.run_all.run_anomaly_review", return_value={"report": {}, "path": "anomaly.json"}
        ), patch("scripts.run_all.train_with_model", return_value=fake_train), patch(
            "scripts.run_all.run_backtest", return_value=fake_backtest
        ), patch("scripts.run_all.run_strategy_comparison", return_value=fake_strategy), patch(
            "scripts.run_all.run_model_comparison", return_value=fake_model_comparison
        ) as mock_model_comparison, patch("scripts.run_all.run_realtime_simulation", return_value="sim.jsonl"), patch(
            "scripts.run_all.write_research_note", return_value=fake_research
        ):
            result = run_workflow(args)

        self.assertIn("config", result)
        self.assertFalse(result["config"]["skip_model_comparison"])
        self.assertTrue(mock_model_comparison.called)
        self.assertIn("data_provenance", result)
        self.assertIn("cache_summary", result)

    def test_run_workflow_normalizes_runtime_modes_from_provenance(self) -> None:
        args = argparse.Namespace(zone="DE_LU", lookback_days=90, simulation_horizon=24, model="xgboost")
        fake_pipeline = types.SimpleNamespace(
            merged_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}),
            energy_source="entsoe_partial_synthetic",
            provenance_summary={
                "real_rows": 24,
                "partially_synthetic_rows": 0,
                "synthetic_rows": 0,
                "real_coverage_ratio": 1.0,
                "partial_synthetic_coverage_ratio": 0.0,
                "synthetic_coverage_ratio": 0.0,
                "research_grade": True,
            },
            cache_summary={"energy": {"cache_status": "loaded"}, "weather": {"cache_status": "loaded"}},
        )
        fake_train = types.SimpleNamespace(
            scored_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}),
            metrics_path="metrics.json",
            diagnostics_path="diag.json",
            scored_path="scored.csv",
            model_key="xgboost",
        )
        fake_backtest = types.SimpleNamespace(result_df=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}), metrics={})
        fake_strategy = types.SimpleNamespace(
            strategy_metrics_df=pd.DataFrame([{"strategy_name": "upgraded_strategy", "sharpe_ratio": 1.0}]),
            significance_df=pd.DataFrame(),
        )
        fake_model_comparison = types.SimpleNamespace(summary_df=pd.DataFrame(), summary_csv_path="summary.csv", summary_json_path="summary.json")
        fake_research = {"summary": {"source": "deterministic_fallback"}, "json_path": "research.json", "note_path": "research.md"}

        with patch("scripts.run_all.ensure_directories"), patch("scripts.run_all.set_global_seed"), patch(
            "scripts.run_all.run_data_pipeline", return_value=fake_pipeline
        ), patch("scripts.run_all.build_features", return_value=pd.DataFrame({"timestamp_utc": pd.date_range("2026-01-01", periods=24 * 14, freq="h", tz="UTC")})), patch(
            "scripts.run_all.run_anomaly_review", return_value={"report": {}, "path": "anomaly.json"}
        ), patch("scripts.run_all.train_with_model", return_value=fake_train), patch(
            "scripts.run_all.run_backtest", return_value=fake_backtest
        ), patch("scripts.run_all.run_strategy_comparison", return_value=fake_strategy), patch(
            "scripts.run_all.run_model_comparison", return_value=fake_model_comparison
        ), patch("scripts.run_all.run_realtime_simulation", return_value="sim.jsonl"), patch(
            "scripts.run_all.write_research_note", return_value=fake_research
        ):
            result = run_workflow(args)

        self.assertEqual(result["runtime_modes"]["energy_source"], "entsoe")
        self.assertTrue(result["runtime_modes"]["research_grade"])

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

    def test_parse_args_supports_cache_controls(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_all.py",
                "--zone",
                "DE_LU",
                "--lookback-days",
                "90",
                "--model",
                "xgboost",
                "--force-refresh",
                "--rebuild-cache",
            ],
        ):
            import scripts.run_all as run_all

            args = run_all.parse_args()

        self.assertTrue(args.force_refresh)
        self.assertTrue(args.rebuild_cache)


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


class ModelRegistryTests(unittest.TestCase):
    def test_scored_predictions_path_uses_model_specific_filename(self) -> None:
        cfg = AppConfig()
        self.assertEqual(
            scored_predictions_path("xgboost", cfg),
            cfg.models_dir / "scored_predictions_xgboost.csv",
        )


class ModelComparisonTests(unittest.TestCase):
    def test_sort_model_comparison_uses_directional_accuracy_then_tiebreakers(self) -> None:
        summary_df = pd.DataFrame(
            [
                {"model_key": "prophet", "directional_accuracy": 0.6, "price_mae": 5.0, "pnl_positive_rate": 0.5},
                {"model_key": "lstm", "directional_accuracy": 0.6, "price_mae": 4.0, "pnl_positive_rate": 0.4},
                {"model_key": "xgboost", "directional_accuracy": 0.55, "price_mae": 1.0, "pnl_positive_rate": 0.9},
            ]
        )

        ranked = sort_model_comparison(summary_df)

        self.assertEqual(list(ranked["model_key"]), ["lstm", "prophet", "xgboost"])
        self.assertEqual(list(ranked["rank"]), [1, 2, 3])

    def test_run_model_comparison_writes_ranked_outputs_without_collisions(self) -> None:
        temp_dir = make_workspace_temp_dir()
        features_df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
                "price_eur_mwh": [50.0, 52.0, 49.0, 55.0, 53.0, 58.0],
                "demand_kw": [10000, 10100, 10200, 10300, 10400, 10500],
                "renewable_mw": [8.0, 8.1, 8.2, 8.3, 8.4, 8.5],
                "feature_one": [1, 2, 3, 4, 5, 6],
            }
        )

        def fake_trainer_factory(model_key: str, price_offset: float, price_mae: float) -> types.SimpleNamespace:
            metrics_path = temp_dir / f"metrics_{model_key}.json"
            metrics_path.write_text(json.dumps({"price": {"mae": price_mae, "rmse": price_mae + 1.0}}), encoding="utf-8")
            scored_df = features_df.copy()
            scored_df["pred_demand_kw"] = scored_df["demand_kw"]
            scored_df["pred_renewable_mw"] = scored_df["renewable_mw"]
            scored_df["pred_price_eur_mwh"] = scored_df["price_eur_mwh"] + price_offset
            return types.SimpleNamespace(
                demand_model_path=str(temp_dir / f"demand_{model_key}.bin"),
                renewable_model_path=str(temp_dir / f"renewable_{model_key}.bin"),
                price_model_path=str(temp_dir / f"price_{model_key}.bin"),
                metrics_path=str(metrics_path),
                scored_df=scored_df,
                model_key=model_key,
                scored_path=str(temp_dir / f"scored_predictions_{model_key}.csv"),
            )

        cfg = AppConfig(project_root=temp_dir, models_dir=temp_dir / "models", simulation_dir=temp_dir / "simulation")
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        cfg.simulation_dir.mkdir(parents=True, exist_ok=True)

        with patch("src.backtesting.comparison.train_with_model") as mock_train_with_model:
            mock_train_with_model.side_effect = [
                fake_trainer_factory("xgboost", 1.5, 5.0),
                fake_trainer_factory("lstm", -1.0, 3.0),
                fake_trainer_factory("prophet", 0.2, 4.0),
            ]
            result = run_model_comparison(features_df, cfg, output_dir=temp_dir / "comparison")

        self.assertEqual(set(result.summary_df["model_key"]), {"xgboost", "lstm", "prophet"})
        self.assertTrue(Path(result.summary_csv_path).exists())
        self.assertTrue(Path(result.summary_json_path).exists())
        for model_key in ["xgboost", "lstm", "prophet"]:
            self.assertTrue((temp_dir / "comparison" / model_key / "backtest_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
