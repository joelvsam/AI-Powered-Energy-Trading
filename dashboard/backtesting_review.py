"""Helpers for the isolated backtesting review dashboard page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting import BacktestConfig, run_backtest, run_model_comparison
from src.backtesting.comparison import COMPARISON_CSV_NAME, COMPARISON_JSON_NAME
from src.config import AppConfig, ensure_directories, set_global_seed
from src.data_pipeline.run_pipeline import run_data_pipeline
from src.features.build_features import build_features
from src.backtesting.engine import evaluate_decision_accuracy


DEFAULT_BACKTESTING_DIR = Path("artifacts") / "backtesting"
DEFAULT_SIMULATION_SCORED_CSV = Path("artifacts") / "simulation" / "backtest_trades.csv"


def default_scored_csv_path() -> Path:
    """Pick the most helpful default scored CSV path for isolated backtesting."""
    candidates = [
        Path("artifacts") / "models" / "scored_predictions_xgboost.csv",
        Path("artifacts") / "models" / "scored_predictions_lstm.csv",
        Path("artifacts") / "models" / "scored_predictions_prophet.csv",
        DEFAULT_SIMULATION_SCORED_CSV,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DEFAULT_SIMULATION_SCORED_CSV


def load_backtest_artifacts(output_dir: Path | str = DEFAULT_BACKTESTING_DIR) -> tuple[pd.DataFrame, dict, dict]:
    """Load a saved isolated backtest run from disk."""
    root = Path(output_dir)
    results_path = root / "backtest_results.csv"
    metrics_path = root / "backtest_metrics.json"
    analytics_path = root / "backtest_analytics.json"

    missing = [str(path) for path in [results_path, metrics_path, analytics_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing isolated backtesting artifacts: {missing}")

    result_df = pd.read_csv(results_path, parse_dates=["timestamp_utc"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    analytics = json.loads(analytics_path.read_text(encoding="utf-8"))
    return result_df, metrics, analytics


def load_model_comparison(output_dir: Path | str = DEFAULT_BACKTESTING_DIR) -> tuple[pd.DataFrame, dict]:
    """Load a saved multi-model comparison summary from disk."""
    root = Path(output_dir)
    summary_csv_path = root / COMPARISON_CSV_NAME
    summary_json_path = root / COMPARISON_JSON_NAME
    missing = [str(path) for path in [summary_csv_path, summary_json_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model comparison artifacts: {missing}")
    summary_df = pd.read_csv(summary_csv_path)
    metadata = json.loads(summary_json_path.read_text(encoding="utf-8"))
    return summary_df, metadata


def run_backtest_from_csv(
    input_path: Path | str,
    *,
    output_dir: Path | str = DEFAULT_BACKTESTING_DIR,
    transaction_cost_bps: float = 5.0,
    annualization_factor: int = 24,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    notional_eur: float = 10000.0,
    accuracy_horizon_steps: int = 1,
    hold_tolerance_pct: float = 0.002,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run the isolated backtester directly from a scored CSV."""
    scored_df = pd.read_csv(input_path, parse_dates=["timestamp_utc"])
    cfg = AppConfig(
        tcost_bps=transaction_cost_bps,
        annualization_factor=annualization_factor,
        backtest_notional_eur=notional_eur,
    )
    result = run_backtest(
        scored_df,
        BacktestConfig(
            output_dir=Path(output_dir),
            transaction_cost_bps=transaction_cost_bps,
            annualization_factor=annualization_factor,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            notional_eur=notional_eur,
            accuracy_horizon_steps=accuracy_horizon_steps,
            hold_tolerance_pct=hold_tolerance_pct,
            enable_new_signal=cfg.enable_new_signal,
            signal_volatility_window_hours=cfg.signal_volatility_window_hours,
            signal_position_scale_k=cfg.signal_position_scale_k,
            enable_volatility_scaling=cfg.enable_volatility_scaling,
            enable_execution_delay=cfg.enable_execution_delay,
        ),
    )
    return result.result_df, result.metrics, result.analytics


def run_model_comparison_workflow(
    *,
    zone: str | None = None,
    lookback_days: int | None = None,
    output_dir: Path | str = DEFAULT_BACKTESTING_DIR,
    transaction_cost_bps: float = 5.0,
    annualization_factor: int = 24,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    notional_eur: float = 10000.0,
    accuracy_horizon_steps: int = 1,
    hold_tolerance_pct: float = 0.002,
    enable_new_signal: bool = True,
    signal_volatility_window_hours: int = 24,
    signal_position_scale_k: float = 2.0,
    enable_volatility_scaling: bool = True,
    enable_execution_delay: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run the shared data pipeline once, then compare xgboost, lstm, and prophet."""
    cfg = AppConfig(
        tcost_bps=transaction_cost_bps,
        annualization_factor=annualization_factor,
        backtest_notional_eur=notional_eur,
        enable_new_signal=enable_new_signal,
        signal_volatility_window_hours=signal_volatility_window_hours,
        signal_position_scale_k=signal_position_scale_k,
        enable_volatility_scaling=enable_volatility_scaling,
        enable_execution_delay=enable_execution_delay,
    )
    ensure_directories(cfg)
    set_global_seed(cfg.random_seed)
    pipeline_out = run_data_pipeline(cfg=cfg, zone=zone, lookback_days=lookback_days)
    features_df = build_features(pipeline_out.merged_df, cfg)
    result = run_model_comparison(
        features_df,
        cfg,
        output_dir=output_dir,
        transaction_cost_bps=transaction_cost_bps,
        annualization_factor=annualization_factor,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        notional_eur=notional_eur,
        accuracy_horizon_steps=accuracy_horizon_steps,
        hold_tolerance_pct=hold_tolerance_pct,
    )
    metadata = dict(result.metadata)
    metadata["zone"] = zone or cfg.default_zone
    metadata["lookback_days"] = lookback_days or cfg.lookback_days
    metadata["energy_source"] = pipeline_out.energy_source
    return result.summary_df, metadata


def build_review_dataset(
    result_df: pd.DataFrame,
    *,
    horizon_steps: int,
    hold_tolerance_pct: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recompute decision-review columns for the selected dashboard horizon."""
    enriched_df, accuracy_summary = evaluate_decision_accuracy(
        result_df,
        horizon_steps=horizon_steps,
        hold_tolerance_pct=hold_tolerance_pct,
    )
    summary = {
        "directional_accuracy": float(accuracy_summary["directional_accuracy"]),
        "pnl_positive_rate": float(accuracy_summary["pnl_positive_rate"]),
        "accuracy_horizon_steps": int(accuracy_summary["accuracy_horizon_steps"]),
        "hold_tolerance_pct": float(accuracy_summary["hold_tolerance_pct"]),
        "evaluable_rows": int(accuracy_summary["evaluable_rows"]),
        "correct_count": int(accuracy_summary["correct_count"]),
        "incorrect_count": int(accuracy_summary["incorrect_count"]),
        "pending_count": int(accuracy_summary["pending_count"]),
        "decision_distribution": {
            key: int(value)
            for key, value in enriched_df["decision"].value_counts(dropna=False).reindex(["LONG", "SHORT", "HOLD"], fill_value=0).items()
        },
    }
    return enriched_df, summary


def filter_review_dataset(
    review_df: pd.DataFrame,
    *,
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Filter backtesting review rows by date range."""
    start_ts = pd.Timestamp(pd.to_datetime(start_date))
    end_ts = pd.Timestamp(pd.to_datetime(end_date))
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC").normalize()
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC").normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        end_ts = end_ts.tz_convert("UTC")
    timestamps = pd.to_datetime(review_df["timestamp_utc"], utc=True)
    return review_df.loc[(timestamps >= start_ts) & (timestamps <= end_ts)].copy()
