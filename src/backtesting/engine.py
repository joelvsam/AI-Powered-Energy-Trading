"""Standalone backtesting engine for offline strategy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for isolated backtesting runs."""

    output_dir: Path
    transaction_cost_bps: float = 5.0
    annualization_factor: int = 24
    long_threshold: float = 0.1
    short_threshold: float = -0.1
    notional_eur: float = 10000.0
    accuracy_horizon_steps: int = 1
    hold_tolerance_pct: float = 0.002


@dataclass(frozen=True)
class BacktestResult:
    """Structured result for a completed backtest run."""

    result_df: pd.DataFrame
    metrics: dict[str, float | int]
    analytics: dict[str, object]
    results_path: str
    metrics_path: str
    analytics_path: str


def compute_sharpe(returns: pd.Series, annualization_factor: int) -> float:
    """Compute annualized Sharpe ratio for a returns series."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(annualization_factor))


def compute_max_drawdown(equity: pd.Series) -> float:
    """Compute the maximum drawdown from an equity curve."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def compute_trade_count(decisions: pd.Series) -> int:
    """Count non-hold trade decisions."""
    return int(decisions.isin(["LONG", "SHORT"]).sum())


def compute_turnover_summary(turnover: pd.Series) -> dict[str, float]:
    """Summarize turnover activity."""
    clean_turnover = turnover.fillna(0.0)
    return {
        "mean_turnover": float(clean_turnover.mean()),
        "max_turnover": float(clean_turnover.max()),
        "total_turnover": float(clean_turnover.sum()),
    }


def _validate_columns(scored_df: pd.DataFrame) -> None:
    required_columns = {
        "timestamp_utc",
        "pred_demand_kw",
        "pred_renewable_mw",
        "pred_price_eur_mwh",
        "price_eur_mwh",
    }
    missing = sorted(required_columns.difference(scored_df.columns))
    if missing:
        raise ValueError(f"Scored dataframe is missing required columns: {missing}")


def _decision_series(position: pd.Series, long_threshold: float, short_threshold: float) -> pd.Series:
    return pd.Series(
        np.where(position > long_threshold, "LONG", np.where(position < short_threshold, "SHORT", "HOLD")),
        index=position.index,
    )


def evaluate_decision_accuracy(
    result_df: pd.DataFrame,
    *,
    horizon_steps: int,
    hold_tolerance_pct: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add realized future move and decision-accuracy columns for a selected horizon."""
    if horizon_steps < 1:
        raise ValueError("accuracy horizon must be at least 1 step")
    if hold_tolerance_pct < 0:
        raise ValueError("hold tolerance must be non-negative")

    df = result_df.copy().sort_values("timestamp_utc")
    future_price = df["price_eur_mwh"].shift(-horizon_steps)
    df["future_price_eur_mwh"] = future_price
    df["future_price_change_eur_mwh"] = future_price - df["price_eur_mwh"]
    denominator = df["price_eur_mwh"].replace(0, np.nan)
    df["future_price_return"] = (df["future_price_change_eur_mwh"] / denominator).fillna(0.0)
    df["pnl_positive"] = df["strategy_return"] > 0

    valid_future = future_price.notna()
    abs_return = df["future_price_return"].abs()
    is_long_correct = (df["decision"] == "LONG") & (df["future_price_change_eur_mwh"] > 0)
    is_short_correct = (df["decision"] == "SHORT") & (df["future_price_change_eur_mwh"] < 0)
    is_hold_correct = (df["decision"] == "HOLD") & (abs_return <= hold_tolerance_pct)
    directional_correct = (is_long_correct | is_short_correct | is_hold_correct) & valid_future

    df["directional_correct"] = directional_correct
    df["accuracy_status"] = np.where(
        ~valid_future,
        "pending",
        np.where(directional_correct, "correct", "incorrect"),
    )
    df["accuracy_horizon_steps"] = horizon_steps
    df["hold_tolerance_pct"] = hold_tolerance_pct

    evaluable_count = int(valid_future.sum())
    directional_accuracy = float(directional_correct[valid_future].mean()) if evaluable_count else 0.0
    pnl_positive_rate = float(df.loc[valid_future, "pnl_positive"].mean()) if evaluable_count else 0.0
    summary = {
        "accuracy_horizon_steps": int(horizon_steps),
        "hold_tolerance_pct": float(hold_tolerance_pct),
        "evaluable_rows": evaluable_count,
        "directional_accuracy": directional_accuracy,
        "pnl_positive_rate": pnl_positive_rate,
        "correct_count": int((df.loc[valid_future, "accuracy_status"] == "correct").sum()),
        "incorrect_count": int((df.loc[valid_future, "accuracy_status"] == "incorrect").sum()),
        "pending_count": int((df["accuracy_status"] == "pending").sum()),
    }
    return df, summary


def run_backtest(scored_df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run an isolated offline backtest from an already-scored dataframe."""
    _validate_columns(scored_df)

    df = scored_df.copy().sort_values("timestamp_utc")
    df["imbalance_pred"] = df["pred_demand_kw"] / 1000.0 - df["pred_renewable_mw"]
    df["price_trend"] = df["price_eur_mwh"].diff().fillna(0.0)
    df["pred_price_delta"] = df["pred_price_eur_mwh"] - df["price_eur_mwh"]

    raw_signal = 0.06 * df["imbalance_pred"] + 0.4 * df["pred_price_delta"]
    df["signal_strength"] = raw_signal
    df["position"] = np.tanh(raw_signal / 10.0)
    df["decision"] = _decision_series(df["position"], config.long_threshold, config.short_threshold)

    df["price_return"] = df["price_eur_mwh"].pct_change().fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    transaction_cost = config.transaction_cost_bps / 10000.0
    df["strategy_return"] = df["position"].shift(1).fillna(0.0) * df["price_return"] - transaction_cost * df["turnover"]
    df["pnl"] = df["strategy_return"] * config.notional_eur
    df["cumulative_returns"] = (1.0 + df["strategy_return"]).cumprod() - 1.0
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()

    df, accuracy_summary = evaluate_decision_accuracy(
        df,
        horizon_steps=config.accuracy_horizon_steps,
        hold_tolerance_pct=config.hold_tolerance_pct,
    )

    trade_mask = df["decision"].isin(["LONG", "SHORT"])
    nonzero_return_mask = df["strategy_return"] != 0
    decision_distribution = {
        key: int(value)
        for key, value in df["decision"].value_counts(dropna=False).reindex(["LONG", "SHORT", "HOLD"], fill_value=0).items()
    }
    turnover_summary = compute_turnover_summary(df["turnover"])

    metrics = {
        "sharpe_ratio": compute_sharpe(df["strategy_return"], config.annualization_factor),
        "max_drawdown": compute_max_drawdown(df["equity_curve"]),
        "hit_rate": float((df["strategy_return"] > 0).sum() / max(nonzero_return_mask.sum(), 1)),
        "total_pnl": float(df["pnl"].sum()),
        "trade_count": compute_trade_count(df["decision"]),
        "average_trade_return": float(df.loc[trade_mask, "strategy_return"].mean()) if trade_mask.any() else 0.0,
        "directional_accuracy": float(accuracy_summary["directional_accuracy"]),
        "pnl_positive_rate": float(accuracy_summary["pnl_positive_rate"]),
    }
    analytics = {
        "row_count": int(len(df)),
        "long_count": decision_distribution["LONG"],
        "short_count": decision_distribution["SHORT"],
        "hold_count": decision_distribution["HOLD"],
        "decision_distribution": decision_distribution,
        "turnover_summary": turnover_summary,
        "final_equity": float(df["equity_curve"].iloc[-1]) if not df.empty else 1.0,
        "cumulative_return": float(df["cumulative_returns"].iloc[-1]) if not df.empty else 0.0,
        "accuracy_summary": accuracy_summary,
        "accuracy_horizon_steps": int(config.accuracy_horizon_steps),
        "hold_tolerance_pct": float(config.hold_tolerance_pct),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "backtest_results.csv"
    metrics_path = config.output_dir / "backtest_metrics.json"
    analytics_path = config.output_dir / "backtest_analytics.json"

    df.to_csv(results_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with analytics_path.open("w", encoding="utf-8") as handle:
        json.dump(analytics, handle, indent=2)

    return BacktestResult(
        result_df=df,
        metrics=metrics,
        analytics=analytics,
        results_path=str(results_path),
        metrics_path=str(metrics_path),
        analytics_path=str(analytics_path),
    )
