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
    signal_equilibrium_window_hours: int = 72
    signal_position_scale_k: float = 2.0
    signal_imbalance_scale: float = 8.0
    enable_volatility_scaling: bool = True
    enable_execution_delay: bool = True
    enable_regime_switching: bool = True
    signal_forecast_weight: float = 0.45
    signal_mean_reversion_weight: float = 0.30
    signal_fundamental_weight: float = 0.25
    long_price_edge_threshold: float = 0.5
    short_price_edge_threshold: float = -0.5
    high_vol_regime_quantile: float = 0.7
    position_limit: float = 1.0
    max_position_change: float = 0.35
    bid_ask_spread_bps: float = 3.0
    bid_ask_spread_eur_mwh: float = 0.0
    slippage_volatility_factor: float = 0.08
    slippage_turnover_factor: float = 0.02
    delay_penalty_factor: float = 0.03
    strategy_name: str = "upgraded_strategy"
    baseline_name: str = ""
    signal_mode: str = "model"


@dataclass(frozen=True)
class BacktestResult:
    """Structured result for a completed backtest run."""

    result_df: pd.DataFrame
    metrics: dict[str, float | int | str | bool]
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


def compute_drawdown_duration(equity: pd.Series) -> int:
    clean_equity = pd.to_numeric(equity, errors="coerce").fillna(1.0).clip(lower=MIN_EQUITY_FLOOR)
    peak = clean_equity.cummax()
    underwater = clean_equity < peak
    max_duration = 0
    current = 0
    for flag in underwater.astype(int):
        current = current + 1 if flag else 0
        max_duration = max(max_duration, current)
    return int(max_duration)


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
        equilibrium_window_hours=config.signal_equilibrium_window_hours,
        position_scale_k=config.signal_position_scale_k,
        imbalance_scale=config.signal_imbalance_scale,
        enable_new_signal=config.enable_new_signal,
        enable_volatility_scaling=config.enable_volatility_scaling,
        enable_regime_switching=config.enable_regime_switching,
        forecast_weight=config.signal_forecast_weight,
        mean_reversion_weight=config.signal_mean_reversion_weight,
        fundamental_weight=config.signal_fundamental_weight,
        long_price_edge_threshold=config.long_price_edge_threshold,
        short_price_edge_threshold=config.short_price_edge_threshold,
        high_vol_regime_quantile=config.high_vol_regime_quantile,
        position_limit=config.position_limit,
        long_threshold=config.long_threshold,
        short_threshold=config.short_threshold,
    )


def _apply_position_constraints(target_position: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    capped = pd.to_numeric(target_position, errors="coerce").fillna(0.0).clip(-abs(config.position_limit), abs(config.position_limit))
    realized: list[float] = []
    clipped_flags: list[bool] = []
    prior = 0.0
    for value in capped:
        delta = float(value) - prior
        clipped = abs(delta) > config.max_position_change
        clipped_delta = float(np.clip(delta, -config.max_position_change, config.max_position_change))
        current = float(np.clip(prior + clipped_delta, -abs(config.position_limit), abs(config.position_limit)))
        realized.append(current)
        clipped_flags.append(clipped)
        prior = current
    return pd.DataFrame(
        {
            "target_position_capped": capped,
            "target_position_constrained": pd.Series(realized, index=target_position.index),
            "liquidity_clipped": pd.Series(clipped_flags, index=target_position.index),
            "position_clipped": (capped != pd.to_numeric(target_position, errors="coerce").fillna(0.0)).astype(bool),
        },
        index=target_position.index,
    )


def _resolve_execution_reference(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("intraday_price_eur_mwh", df["price_eur_mwh"]), errors="coerce").fillna(
        pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    )


def _compute_spread_eur(reference_price: pd.Series, config: BacktestConfig) -> pd.Series:
    spread_from_bps = reference_price.abs().clip(lower=1.0) * (config.bid_ask_spread_bps / 10000.0)
    spread_fixed = pd.Series(config.bid_ask_spread_eur_mwh, index=reference_price.index, dtype=float)
    return spread_from_bps + spread_fixed


def _compute_pnl_attribution(df: pd.DataFrame) -> dict[str, object]:
    attribution = {
        "signal_family_pnl": {
            "forecast": float((df["position"] * df["forward_price_change_eur_mwh"] * df["forecast_signal"].fillna(0.0).abs()).sum()),
            "mean_reversion": float((df["position"] * df["forward_price_change_eur_mwh"] * df["mean_reversion_signal"].fillna(0.0).abs()).sum()),
            "fundamental": float((df["position"] * df["forward_price_change_eur_mwh"] * df["fundamental_signal"].fillna(0.0).abs()).sum()),
        },
        "pnl_by_vol_regime": (
            df.groupby("vol_regime", dropna=False)["net_pnl_eur"].agg(["sum", "mean", "count"]).reset_index().to_dict(orient="records")
            if "vol_regime" in df.columns
            else []
        ),
        "pnl_by_market_regime": (
            df.groupby("market_regime", dropna=False)["net_pnl_eur"].agg(["sum", "mean", "count"]).reset_index().to_dict(orient="records")
            if "market_regime" in df.columns
            else []
        ),
        "pnl_by_hour": df.groupby(df["timestamp_utc"].dt.hour)["net_pnl_eur"].sum().to_dict(),
        "pnl_by_month": df.groupby(df["timestamp_utc"].dt.month)["net_pnl_eur"].sum().to_dict(),
    }
    return attribution


def _compute_sharpe_decomposition(df: pd.DataFrame, annualization_factor: int) -> dict[str, float]:
    returns = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(0.0)
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    return {
        "mean_return": float(returns.mean()),
        "return_volatility": float(returns.std()),
        "hit_rate": float((returns > 0).mean()),
        "avg_win": float(positive.mean()) if not positive.empty else 0.0,
        "avg_loss": float(negative.mean()) if not negative.empty else 0.0,
        "cost_drag_eur": float(df[["transaction_cost_eur", "spread_cost_eur", "slippage_cost_eur", "delay_cost_eur"]].sum().sum()),
        "annualized_sharpe": compute_sharpe(returns, annualization_factor),
    }


def generate_backtest_outputs(scored_df: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, dict[str, float | int | str | bool], dict[str, object]]:
    """Generate a backtest dataframe plus summary metrics without writing files."""
    _validate_columns(scored_df)

    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df = compute_signal_frame(df, _build_signal_config(config))
    if config.signal_mode == "zero_signal":
        for column in [
            "forecast_signal",
            "mean_reversion_signal",
            "fundamental_signal",
            "combined_signal",
            "signal_z_score",
            "signal_strength",
            "ensemble_signal",
            "target_position_raw",
            "target_position_capped",
            "target_position",
        ]:
            df[column] = 0.0
        df["target_decision"] = "HOLD"
        df["recommended_decision"] = "HOLD"
    reference_price_eur_mwh = _reference_price_scale(df["price_eur_mwh"])
    exposure_mwh = config.notional_eur / reference_price_eur_mwh
    transaction_cost_rate = config.transaction_cost_bps / 10000.0

    position_constraints = _apply_position_constraints(df["target_position"], config)
    for column in position_constraints.columns:
        df[column] = position_constraints[column]

    execution_reference = _resolve_execution_reference(df)
    spread_eur_mwh = _compute_spread_eur(execution_reference, config)
    df["execution_reference_price_eur_mwh"] = execution_reference
    df["spread_eur_mwh"] = spread_eur_mwh
    df["price_change_eur_mwh"] = compute_price_change(df["price_eur_mwh"])
    df["price_return"] = df["price_change_eur_mwh"] / reference_price_eur_mwh
    df["forward_price_change_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce").shift(-1) - pd.to_numeric(
        df["price_eur_mwh"], errors="coerce"
    )
    df["forward_price_change_eur_mwh"] = df["forward_price_change_eur_mwh"].fillna(0.0)

    signal_to_execute = df["target_position_constrained"]
    if config.enable_execution_delay:
        signal_to_execute = signal_to_execute.shift(1).fillna(0.0)
    df["position"] = pd.to_numeric(signal_to_execute, errors="coerce").clip(-abs(config.position_limit), abs(config.position_limit)).fillna(0.0)
    df["position_change"] = df["position"].diff().fillna(df["position"])
    df["turnover"] = df["position_change"].abs()
    trade_direction = np.sign(df["position_change"])
    df["fill_price_eur_mwh"] = execution_reference + trade_direction * (spread_eur_mwh / 2.0)
    df["spread_cost_eur"] = df["turnover"] * exposure_mwh * (spread_eur_mwh / 2.0)
    df["slippage_eur_mwh"] = (
        config.slippage_volatility_factor * df["rolling_volatility"].fillna(0.0)
        + config.slippage_turnover_factor * df["turnover"] * execution_reference.abs().clip(lower=1.0)
    )
    df["slippage_cost_eur"] = df["turnover"] * exposure_mwh * df["slippage_eur_mwh"]
    df["delay_cost_eur"] = np.where(
        config.enable_execution_delay,
        df["turnover"] * exposure_mwh * df["rolling_volatility"].fillna(0.0) * config.delay_penalty_factor,
        0.0,
    )
    df["transaction_cost_eur"] = df["turnover"] * config.notional_eur * transaction_cost_rate
    df["gross_pnl_eur"] = df["position"] * df["forward_price_change_eur_mwh"] * exposure_mwh
    df["net_pnl_eur"] = (
        df["gross_pnl_eur"]
        - df["transaction_cost_eur"]
        - df["spread_cost_eur"]
        - df["slippage_cost_eur"]
        - df["delay_cost_eur"]
    )
    df["unfilled_position_change"] = (df["target_position_capped"] - df["target_position_constrained"]).abs()
    df["opportunity_cost_proxy_eur"] = df["unfilled_position_change"] * exposure_mwh * df["forward_price_change_eur_mwh"].abs()

    if config.enable_new_signal:
        df["decision"] = decision_from_position(df["position"])
    else:
        df["decision"] = _threshold_decision_series(df["position"], config.long_threshold, config.short_threshold)
    df["recommended_decision"] = df.get("recommended_decision", df["target_decision"])

    strategy_return, realized_pnl_eur, equity_eur = _simulate_equity_curve(df["net_pnl_eur"], config.notional_eur)
    initial_equity = max(float(config.notional_eur), 1.0)
    df["strategy_return"] = strategy_return
    df["pnl"] = realized_pnl_eur
    df["equity_eur"] = equity_eur
    df["equity_curve"] = equity_eur / initial_equity
    df["cumulative_returns"] = df["equity_curve"] - 1.0
    df["reference_price_eur_mwh"] = reference_price_eur_mwh
    df["exposure_mwh"] = exposure_mwh
    df["strategy_name"] = config.strategy_name
    df["baseline_name"] = config.baseline_name
    df["signal_mode"] = config.signal_mode

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
    sharpe_decomposition = _compute_sharpe_decomposition(df, config.annualization_factor)

    metrics = {
        "strategy_name": config.strategy_name,
        "baseline_name": config.baseline_name,
        "signal_mode": config.signal_mode,
        "sharpe_ratio": compute_sharpe(df["strategy_return"], config.annualization_factor),
        "max_drawdown": compute_max_drawdown(df["equity_curve"]),
        "drawdown_duration_steps": compute_drawdown_duration(df["equity_curve"]),
        "hit_rate": float((df["strategy_return"] > 0).sum() / max(nonzero_return_mask.sum(), 1)),
        "total_pnl": float(df["pnl"].sum()),
        "trade_count": compute_trade_count(df["position"]),
        "average_trade_return": float(df.loc[trade_mask, "strategy_return"].mean()) if trade_mask.any() else 0.0,
        "directional_accuracy": float(accuracy_summary["directional_accuracy"]),
        "pnl_positive_rate": float(accuracy_summary["pnl_positive_rate"]),
        "gross_pnl_eur": float(df["gross_pnl_eur"].sum()),
        "net_pnl_eur": float(df["net_pnl_eur"].sum()),
        "spread_cost_eur": float(df["spread_cost_eur"].sum()),
        "slippage_cost_eur": float(df["slippage_cost_eur"].sum()),
        "delay_cost_eur": float(df["delay_cost_eur"].sum()),
        "opportunity_cost_proxy_eur": float(df["opportunity_cost_proxy_eur"].sum()),
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
        "signal_equilibrium_window_hours": int(config.signal_equilibrium_window_hours),
        "signal_position_scale_k": float(config.signal_position_scale_k),
        "signal_imbalance_scale": float(config.signal_imbalance_scale),
        "long_price_edge_threshold": float(config.long_price_edge_threshold),
        "short_price_edge_threshold": float(config.short_price_edge_threshold),
        "enable_volatility_scaling": bool(config.enable_volatility_scaling),
        "enable_regime_switching": bool(config.enable_regime_switching),
        "position_limit": float(config.position_limit),
        "max_position_change": float(config.max_position_change),
        "execution_costs": {
            "transaction_cost_eur": float(df["transaction_cost_eur"].sum()),
            "spread_cost_eur": float(df["spread_cost_eur"].sum()),
            "slippage_cost_eur": float(df["slippage_cost_eur"].sum()),
            "delay_cost_eur": float(df["delay_cost_eur"].sum()),
        },
        "sharpe_decomposition": sharpe_decomposition,
        "pnl_attribution": _compute_pnl_attribution(df),
        "strategy_name": config.strategy_name,
        "baseline_name": config.baseline_name,
        "signal_mode": config.signal_mode,
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
