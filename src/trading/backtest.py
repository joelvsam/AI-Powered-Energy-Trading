"""Backtesting module for trading strategy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import AppConfig


@dataclass
class BacktestOutputs:
    trades_path: str
    metrics_path: str
    result_df: pd.DataFrame


def _sharpe(returns: pd.Series, annualization_factor: int) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(annualization_factor))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def run_backtest(scored_df: pd.DataFrame, cfg: AppConfig) -> BacktestOutputs:
    df = scored_df.copy().sort_values("timestamp_utc")
    df["imbalance_pred"] = df["pred_demand_kw"] / 1000.0 - df["pred_renewable_mw"]
    df["price_trend"] = df["price_eur_mwh"].diff().fillna(0.0)
    df["pred_price_delta"] = df["pred_price_eur_mwh"] - df["price_eur_mwh"]
    raw_signal = 0.06 * df["imbalance_pred"] + 0.4 * df["pred_price_delta"]
    df["position"] = np.tanh(raw_signal / 10.0)
    df["decision"] = np.where(df["position"] > 0.1, "LONG", np.where(df["position"] < -0.1, "SHORT", "HOLD"))

    df["price_return"] = df["price_eur_mwh"].pct_change().fillna(0.0)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    tcost = cfg.tcost_bps / 10000.0
    df["strategy_return"] = df["position"].shift(1).fillna(0.0) * df["price_return"] - tcost * df["turnover"]
    df["pnl"] = df["strategy_return"] * 10000.0
    df["cumulative_returns"] = (1.0 + df["strategy_return"]).cumprod() - 1.0
    equity = (1.0 + df["strategy_return"]).cumprod()

    hit = ((df["strategy_return"] > 0).sum() / max((df["strategy_return"] != 0).sum(), 1)).item()
    metrics = {
        "sharpe_ratio": _sharpe(df["strategy_return"], cfg.annualization_factor),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": float(hit),
        "total_pnl": float(df["pnl"].sum()),
    }

    trades_path = cfg.simulation_dir / "backtest_trades.csv"
    metrics_path = cfg.simulation_dir / "backtest_metrics.json"
    df.to_csv(trades_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return BacktestOutputs(trades_path=str(trades_path), metrics_path=str(metrics_path), result_df=df)
