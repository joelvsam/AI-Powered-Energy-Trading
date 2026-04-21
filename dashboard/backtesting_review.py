"""Helpers for the isolated backtesting review dashboard page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtesting import BacktestConfig, run_backtest
from src.backtesting.engine import evaluate_decision_accuracy


DEFAULT_BACKTESTING_DIR = Path("artifacts") / "backtesting"


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
        ),
    )
    return result.result_df, result.metrics, result.analytics


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
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    timestamps = pd.to_datetime(review_df["timestamp_utc"])
    return review_df.loc[(timestamps >= start_ts) & (timestamps <= end_ts)].copy()
