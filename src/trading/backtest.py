"""Backtesting module for trading strategy evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.backtesting.engine import BacktestConfig, generate_backtest_outputs
from src.config import AppConfig


@dataclass
class BacktestOutputs:
    trades_path: str
    metrics_path: str
    result_df: pd.DataFrame


def run_backtest(scored_df: pd.DataFrame, cfg: AppConfig) -> BacktestOutputs:
    config = BacktestConfig(
        output_dir=cfg.simulation_dir,
        transaction_cost_bps=cfg.tcost_bps,
        annualization_factor=cfg.annualization_factor,
        notional_eur=cfg.backtest_notional_eur,
    )
    df, metrics, _ = generate_backtest_outputs(scored_df, config)

    trades_path = cfg.simulation_dir / "backtest_trades.csv"
    metrics_path = cfg.simulation_dir / "backtest_metrics.json"
    df.to_csv(trades_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return BacktestOutputs(trades_path=str(trades_path), metrics_path=str(metrics_path), result_df=df)
