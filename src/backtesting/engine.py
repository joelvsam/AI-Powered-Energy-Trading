"""Standalone backtesting engine for offline strategy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.trading.signal import SignalConfig, compute_signal_frame, decision_from_position


MIN_EQUITY_FLOOR = 1e-6
MAX_PERIOD_RETURN = 1.0 - MIN_EQUITY_FLOOR


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
    enable_new_signal: bool = True
    signal_volatility_window_hours: int = 24
    signal_position_scale_k: float = 2.0
    enable_volatility_scaling: bool = True
    enable_execution_delay: bool = True


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
    clean_equity = pd.to_numeric(equity, errors="coerce").fillna(1.0).clip(lower=MIN_EQUITY_FLOOR)
    peak = clean_equity.cummax()
    drawdown = (clean_equity - peak) / peak
    return float(drawdown.min())


def compute_trade_count(position: pd.Series, tolerance: float = 1e-6) -> int:
    """Count rebalances based on realized position changes."""
    if not is_numeric_dtype(position):
        return int(position.isin(["LONG", "SHORT"]).sum())
    numeric = pd.to_numeric(position, errors="coerce").fillna(0.0)
    turnover = numeric.diff().abs().fillna(numeric.abs())
    return int((turnover > tolerance).sum())


def compute_price_change(prices: pd.Series) -> pd.Series:
    """Compute absolute price changes in EUR/MWh."""
    clean_prices = pd.to_numeric(prices, errors="coerce")
    return clean_prices.diff().fillna(0.0)


def compute_price_return(prices: pd.Series, scale_floor: float = 1.0) -> pd.Series:
    """Compute a stable normalized price-move series for reporting."""
    clean_prices = pd.to_numeric(prices, errors="coerce")
    previous = clean_prices.shift(1)
    denominator = previous.abs().clip(lower=scale_floor)
    returns = (clean_prices - previous) / denominator
    return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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


def _reference_price_scale(prices: pd.Series) -> float:
    clean_abs = pd.to_numeric(prices, errors="coerce").abs()
    positive = clean_abs[clean_abs > 0]
    if positive.empty:
        return 1.0
    return float(max(positive.median(), 1.0))


def _threshold_decision_series(position: pd.Series, long_threshold: float, short_threshold: float) -> pd.Series:
    return pd.Series(
        np.where(position > long_threshold, "LONG", np.where(position < short_threshold, "SHORT", "HOLD")),
        index=position.index,
    )


def _simulate_equity_curve(net_pnl_eur: pd.Series, initial_equity_eur: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    equity_values: list[float] = []
    realized_pnl_values: list[float] = []
    strategy_returns: list[float] = []

    equity = max(float(initial_equity_eur), 1.0)
    for raw_pnl in net_pnl_eur.fillna(0.0):
        normalized_return = float(raw_pnl) / equity if equity > 0 else 0.0
        clipped_return = min(max(normalized_return, -1.0 + MIN_EQUITY_FLOOR), MAX_PERIOD_RETURN)
        next_equity = max(equity * (1.0 + clipped_return), MIN_EQUITY_FLOOR)
        realized_pnl = next_equity - equity
        strategy_returns.append(clipped_return)
        realized_pnl_values.append(realized_pnl)
        equity_values.append(next_equity)
        equity = next_equity

    index = net_pnl_eur.index
    return (
        pd.Series(strategy_returns, index=index),
        pd.Series(realized_pnl_values, index=index),
        pd.Series(equity_values, index=index),
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
    denominator = pd.to_numeric(df["price_eur_mwh"], errors="coerce").abs().clip(lower=1.0)
    df["future_price_return"] = (df["future_price_change_eur_mwh"] / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["pnl_positive"] = df["pnl"] > 0

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


def _build_signal_config(config: BacktestConfig) -> SignalConfig:
    return SignalConfig(
        volatility_window_hours=config.signal_volatility_window_hours,
        position_scale_k=config.signal_position_scale_k,
        enable_new_signal=config.enable_new_signal,
        enable_volatility_scaling=config.enable_volatility_scaling,
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
    )


def generate_backtest_outputs(scored_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict[str, float | int], dict[str, object]]:
    """Generate a backtest dataframe plus summary metrics without writing files."""
    _validate_columns(scored_df)

    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df = compute_signal_frame(df, _build_signal_config(config))
    reference_price_eur_mwh = _reference_price_scale(df["price_eur_mwh"])
    exposure_mwh = config.notional_eur / reference_price_eur_mwh
    transaction_cost_rate = config.transaction_cost_bps / 10000.0

    df["price_change_eur_mwh"] = compute_price_change(df["price_eur_mwh"])
    df["price_return"] = df["price_change_eur_mwh"] / reference_price_eur_mwh
    df["forward_price_change_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce").shift(-1) - pd.to_numeric(
        df["price_eur_mwh"], errors="coerce"
    )
    df["forward_price_change_eur_mwh"] = df["forward_price_change_eur_mwh"].fillna(0.0)

    if config.enable_execution_delay:
        df["position"] = df["target_position"].shift(1).fillna(0.0)
    else:
        df["position"] = df["target_position"].fillna(0.0)
    df["position"] = pd.to_numeric(df["position"], errors="coerce").clip(-1.0, 1.0).fillna(0.0)
    if config.enable_new_signal:
        df["decision"] = decision_from_position(df["position"])
    else:
        df["decision"] = _threshold_decision_series(df["position"], config.long_threshold, config.short_threshold)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["gross_pnl_eur"] = df["position"] * df["forward_price_change_eur_mwh"] * exposure_mwh
    df["transaction_cost_eur"] = df["turnover"] * config.notional_eur * transaction_cost_rate
    df["net_pnl_eur"] = df["gross_pnl_eur"] - df["transaction_cost_eur"]

    strategy_return, realized_pnl_eur, equity_eur = _simulate_equity_curve(df["net_pnl_eur"], config.notional_eur)
    initial_equity = max(float(config.notional_eur), 1.0)
    df["strategy_return"] = strategy_return
    df["pnl"] = realized_pnl_eur
    df["equity_eur"] = equity_eur
    df["equity_curve"] = equity_eur / initial_equity
    df["cumulative_returns"] = df["equity_curve"] - 1.0
    df["reference_price_eur_mwh"] = reference_price_eur_mwh
    df["exposure_mwh"] = exposure_mwh

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
        "trade_count": compute_trade_count(df["position"]),
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
        "reference_price_eur_mwh": float(reference_price_eur_mwh),
        "initial_equity_eur": float(initial_equity),
        "exposure_mwh": float(exposure_mwh),
        "enable_new_signal": bool(config.enable_new_signal),
        "enable_execution_delay": bool(config.enable_execution_delay),
        "signal_volatility_window_hours": int(config.signal_volatility_window_hours),
        "signal_position_scale_k": float(config.signal_position_scale_k),
        "enable_volatility_scaling": bool(config.enable_volatility_scaling),
    }
    return df, metrics, analytics


def run_backtest(scored_df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run an isolated offline backtest from an already-scored dataframe."""
    df, metrics, analytics = generate_backtest_outputs(scored_df, config)

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
