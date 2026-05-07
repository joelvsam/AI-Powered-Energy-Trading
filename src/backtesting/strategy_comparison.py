"""Baseline generation and strategy comparison workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestResult, run_backtest
from src.backtesting.statistics import bootstrap_sharpe_ci, compare_return_streams
from src.config import AppConfig


@dataclass(frozen=True)
class StrategyComparisonResult:
    strategy_metrics_df: pd.DataFrame
    significance_df: pd.DataFrame
    result_frames: dict[str, pd.DataFrame]
    metadata: dict[str, object]
    metrics_csv_path: str
    significance_csv_path: str
    metadata_json_path: str


def _base_backtest_config(cfg: AppConfig, output_dir: Path, *, strategy_name: str, baseline_name: str, signal_mode: str, enable_new_signal: bool) -> BacktestConfig:
    return BacktestConfig(
        output_dir=output_dir,
        transaction_cost_bps=cfg.tcost_bps,
        annualization_factor=cfg.annualization_factor,
        notional_eur=cfg.backtest_notional_eur,
        accuracy_horizon_steps=1,
        hold_tolerance_pct=0.002,
        enable_new_signal=enable_new_signal,
        signal_volatility_window_hours=cfg.signal_volatility_window_hours,
        signal_equilibrium_window_hours=cfg.signal_equilibrium_window_hours,
        signal_position_scale_k=cfg.signal_position_scale_k,
        signal_imbalance_scale=cfg.signal_imbalance_scale,
        enable_volatility_scaling=cfg.enable_volatility_scaling,
        enable_execution_delay=cfg.enable_execution_delay,
        enable_regime_switching=cfg.enable_regime_switching,
        signal_forecast_weight=cfg.signal_forecast_weight,
        signal_mean_reversion_weight=cfg.signal_mean_reversion_weight,
        signal_fundamental_weight=cfg.signal_fundamental_weight,
        high_vol_regime_quantile=cfg.high_vol_regime_quantile,
        position_limit=cfg.position_limit,
        max_position_change=cfg.max_position_change,
        bid_ask_spread_bps=cfg.bid_ask_spread_bps,
        bid_ask_spread_eur_mwh=cfg.bid_ask_spread_eur_mwh,
        slippage_volatility_factor=cfg.slippage_volatility_factor,
        slippage_turnover_factor=cfg.slippage_turnover_factor,
        delay_penalty_factor=cfg.delay_penalty_factor,
        strategy_name=strategy_name,
        baseline_name=baseline_name,
        signal_mode=signal_mode,
    )


def _seasonal_baseline(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df["hour"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.dayofweek
    for target in ["price_eur_mwh", "demand_kw", "renewable_mw"]:
        grouped = (
            df.groupby(["day_of_week", "hour"], dropna=False)[target]
            .expanding()
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .shift(1)
        )
        fallback = df[target].expanding().mean().shift(1)
        pred = grouped.fillna(fallback).fillna(df[target].shift(1)).fillna(df[target])
        if target == "price_eur_mwh":
            df["pred_price_eur_mwh"] = pred
        elif target == "demand_kw":
            df["pred_demand_kw"] = pred
        else:
            df["pred_renewable_mw"] = pred
    return df


def _persistence_baseline(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df["pred_price_eur_mwh"] = df["price_eur_mwh"].shift(1).fillna(df["price_eur_mwh"])
    df["pred_demand_kw"] = df.get("demand_kw", df["pred_demand_kw"]).shift(1).fillna(df.get("demand_kw", df["pred_demand_kw"]))
    df["pred_renewable_mw"] = df.get("renewable_mw", df["pred_renewable_mw"]).shift(1).fillna(df.get("renewable_mw", df["pred_renewable_mw"]))
    return df


def _spread_mean_reversion_baseline(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    spread_source = df["intraday_day_ahead_spread_eur_mwh"] if "intraday_day_ahead_spread_eur_mwh" in df.columns else pd.Series(0.0, index=df.index)
    day_ahead_source = df["day_ahead_price_eur_mwh"] if "day_ahead_price_eur_mwh" in df.columns else df["price_eur_mwh"]
    spread = pd.to_numeric(spread_source, errors="coerce").fillna(0.0)
    df["pred_price_eur_mwh"] = pd.to_numeric(day_ahead_source, errors="coerce") - 0.5 * spread
    df["pred_demand_kw"] = df.get("demand_kw", df["pred_demand_kw"]).shift(1).fillna(df.get("demand_kw", df["pred_demand_kw"]))
    df["pred_renewable_mw"] = df.get("renewable_mw", df["pred_renewable_mw"]).shift(1).fillna(df.get("renewable_mw", df["pred_renewable_mw"]))
    return df


def _zero_signal_baseline(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    df["pred_price_eur_mwh"] = df["price_eur_mwh"]
    df["pred_demand_kw"] = df.get("demand_kw", df["pred_demand_kw"])
    df["pred_renewable_mw"] = df.get("renewable_mw", df["pred_renewable_mw"])
    return df


def build_baseline_frames(scored_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "persistence": _persistence_baseline(scored_df),
        "seasonal": _seasonal_baseline(scored_df),
        "naive_mean_reversion": _spread_mean_reversion_baseline(scored_df),
        "zero_signal": _zero_signal_baseline(scored_df),
    }


def _run_strategy(name: str, frame: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return run_backtest(frame, config)


def run_strategy_comparison(
    scored_df: pd.DataFrame,
    cfg: AppConfig,
    *,
    output_dir: Path | str,
    model_key: str,
) -> StrategyComparisonResult:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_frames: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, object]] = []

    strategies: dict[str, tuple[pd.DataFrame, BacktestConfig]] = {
        "original_strategy": (
            scored_df.copy(),
            _base_backtest_config(cfg, root / "original_strategy", strategy_name="original_strategy", baseline_name="", signal_mode="legacy", enable_new_signal=False),
        ),
        "upgraded_strategy": (
            scored_df.copy(),
            _base_backtest_config(cfg, root / "upgraded_strategy", strategy_name="upgraded_strategy", baseline_name="", signal_mode="model", enable_new_signal=True),
        ),
    }

    baseline_frames = build_baseline_frames(scored_df)
    for baseline_name, baseline_df in baseline_frames.items():
        strategies[baseline_name] = (
            baseline_df,
            _base_backtest_config(
                cfg,
                root / baseline_name,
                strategy_name=baseline_name,
                baseline_name=baseline_name,
                signal_mode="zero_signal" if baseline_name == "zero_signal" else "baseline",
                enable_new_signal=baseline_name != "naive_mean_reversion",
            ),
        )

    strategy_returns: dict[str, pd.Series] = {}
    for strategy_name, (frame, backtest_config) in strategies.items():
        result = _run_strategy(strategy_name, frame, backtest_config)
        result_frames[strategy_name] = result.result_df
        strategy_returns[strategy_name] = result.result_df["strategy_return"]
        row = dict(result.metrics)
        sharpe_ci = bootstrap_sharpe_ci(
            result.result_df["strategy_return"],
            annualization_factor=cfg.annualization_factor,
            iterations=cfg.bootstrap_iterations,
            random_seed=cfg.random_seed,
        )
        row.update(
            {
                "model_key": model_key,
                "energy_source_research_grade": False,
                "sharpe_ci_lower": sharpe_ci["ci_lower"],
                "sharpe_ci_upper": sharpe_ci["ci_upper"],
                "results_path": result.results_path,
                "metrics_path": result.metrics_path,
                "analytics_path": result.analytics_path,
            }
        )
        metrics_rows.append(row)

    significance_rows: list[dict[str, object]] = []
    for strategy_name in ["upgraded_strategy", "original_strategy"]:
        for baseline_name in baseline_frames:
            stats = compare_return_streams(
                strategy_returns[strategy_name],
                strategy_returns[baseline_name],
                annualization_factor=cfg.annualization_factor,
                iterations=cfg.bootstrap_iterations,
                random_seed=cfg.random_seed,
            )
            significance_rows.append(
                {
                    "model_key": model_key,
                    "strategy_name": strategy_name,
                    "baseline_name": baseline_name,
                    **stats,
                }
            )

    strategy_metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["sharpe_ratio", "total_pnl", "directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    significance_df = pd.DataFrame(significance_rows)
    metadata = {
        "model_key": model_key,
        "strategies": list(strategies.keys()),
        "baseline_names": list(baseline_frames.keys()),
        "winner_model_metric": "sharpe_ratio",
        "rows": strategy_metrics_df.to_dict(orient="records"),
        "significance_summary": significance_df.to_dict(orient="records"),
    }

    metrics_csv_path = root / f"strategy_metrics_{model_key}.csv"
    significance_csv_path = root / f"strategy_significance_{model_key}.csv"
    metadata_json_path = root / f"strategy_comparison_{model_key}.json"
    strategy_metrics_df.to_csv(metrics_csv_path, index=False)
    significance_df.to_csv(significance_csv_path, index=False)
    metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return StrategyComparisonResult(
        strategy_metrics_df=strategy_metrics_df,
        significance_df=significance_df,
        result_frames=result_frames,
        metadata=metadata,
        metrics_csv_path=str(metrics_csv_path),
        significance_csv_path=str(significance_csv_path),
        metadata_json_path=str(metadata_json_path),
    )
