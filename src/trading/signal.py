"""Shared signal-generation helpers for trading and decision surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for transforming forecasts into tradable positions."""

    volatility_window_hours: int = 24
    equilibrium_window_hours: int = 72
    position_scale_k: float = 2.0
    imbalance_scale: float = 8.0
    enable_new_signal: bool = True
    enable_volatility_scaling: bool = True
    enable_regime_switching: bool = True
    forecast_weight: float = 0.45
    mean_reversion_weight: float = 0.30
    fundamental_weight: float = 0.25
    high_vol_regime_quantile: float = 0.7
    position_limit: float = 1.0
    long_threshold: float = 0.1
    short_threshold: float = -0.1
    long_price_edge_threshold: float = 0.5
    short_price_edge_threshold: float = -0.5


def decision_from_position(position: pd.Series, tolerance: float = 1e-6) -> pd.Series:
    """Map a continuous position into a sign-only label for reporting."""
    numeric = pd.to_numeric(position, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(numeric > tolerance, "LONG", np.where(numeric < -tolerance, "SHORT", "HOLD")),
        index=position.index,
    )


def decision_from_price_edge(
    price_edge: pd.Series,
    *,
    long_threshold: float,
    short_threshold: float,
) -> pd.Series:
    """Map forecasted price edge into the user-facing trading recommendation."""
    numeric = pd.to_numeric(price_edge, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(numeric > long_threshold, "LONG", np.where(numeric < short_threshold, "SHORT", "HOLD")),
        index=price_edge.index,
    )


def compute_signal_frame(df: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """Compute signal diagnostics and target positions from model forecasts."""
    out = df.copy()
    out["day_ahead_price_eur_mwh"] = pd.to_numeric(
        out.get("day_ahead_price_eur_mwh", out["price_eur_mwh"]),
        errors="coerce",
    )
    out["intraday_price_eur_mwh"] = pd.to_numeric(
        out.get("intraday_price_eur_mwh", out["price_eur_mwh"]),
        errors="coerce",
    )
    out["pred_price_delta"] = pd.to_numeric(out["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(
        out["price_eur_mwh"], errors="coerce"
    )
    out["imbalance_pred"] = pd.to_numeric(out["pred_demand_kw"], errors="coerce") / 1000.0 - pd.to_numeric(
        out["pred_renewable_mw"], errors="coerce"
    )
    out["net_load_mw"] = pd.to_numeric(out.get("demand_kw", out["pred_demand_kw"]), errors="coerce") / 1000.0 - pd.to_numeric(
        out.get("renewable_mw", out["pred_renewable_mw"]),
        errors="coerce",
    )
    out["intraday_day_ahead_spread_eur_mwh"] = out["intraday_price_eur_mwh"] - out["day_ahead_price_eur_mwh"]
    out["price_trend"] = pd.to_numeric(out["price_eur_mwh"], errors="coerce").diff().fillna(0.0)

    # Warmup NaNs are filled with expanding (past-only) statistics, never bfill:
    # backward fills would leak future prices into earlier signal rows.
    clean_price = pd.to_numeric(out["price_eur_mwh"], errors="coerce")
    rolling_volatility = (
        clean_price.rolling(config.volatility_window_hours, min_periods=max(2, config.volatility_window_hours // 2))
        .std()
        .ffill()
        .fillna(clean_price.expanding(min_periods=2).std())
    )
    equilibrium_price = (
        clean_price.rolling(config.equilibrium_window_hours, min_periods=max(6, config.equilibrium_window_hours // 3))
        .median()
        .ffill()
        .fillna(clean_price.expanding(min_periods=1).median())
    )
    out["rolling_volatility"] = rolling_volatility
    out["equilibrium_price_eur_mwh"] = equilibrium_price
    out["forecast_signal"] = out["pred_price_delta"] / (rolling_volatility + 1e-6)
    out["mean_reversion_signal"] = -1.0 * (
        (pd.to_numeric(out["price_eur_mwh"], errors="coerce") - equilibrium_price) / (rolling_volatility + 1e-6)
    )
    net_load_equilibrium = (
        out["net_load_mw"]
        .rolling(config.equilibrium_window_hours, min_periods=6)
        .mean()
        .ffill()
        .fillna(out["net_load_mw"].expanding(min_periods=1).mean())
    )
    out["fundamental_signal"] = (out["imbalance_pred"] - net_load_equilibrium) / max(config.imbalance_scale, 1e-6)
    spread_volatility = (
        out["intraday_day_ahead_spread_eur_mwh"]
        .rolling(config.volatility_window_hours, min_periods=4)
        .std()
        .ffill()
        .fillna(out["intraday_day_ahead_spread_eur_mwh"].expanding(min_periods=2).std())
    )
    out["spread_signal"] = out["intraday_day_ahead_spread_eur_mwh"] / (spread_volatility + 1e-6)
    # A NaN threshold during warmup compares False, defaulting to the
    # conservative low_vol regime instead of borrowing a future quantile.
    high_vol_threshold = (
        rolling_volatility
        .rolling(config.equilibrium_window_hours, min_periods=6)
        .quantile(config.high_vol_regime_quantile)
        .ffill()
    )
    out["vol_regime"] = np.where(rolling_volatility > high_vol_threshold, "high_vol", "low_vol")
    out["market_regime"] = np.where(
        out["price_trend"].rolling(config.volatility_window_hours, min_periods=4).mean().abs()
        > rolling_volatility.rolling(config.volatility_window_hours, min_periods=4).mean().fillna(0.0) * 0.3,
        "trend",
        "mean_revert",
    )
    out["price_edge_signal"] = out["pred_price_delta"]
    out["price_edge_decision"] = decision_from_price_edge(
        out["pred_price_delta"],
        long_threshold=config.long_price_edge_threshold,
        short_threshold=config.short_price_edge_threshold,
    )
    price_edge_direction = pd.Series(
        np.where(out["price_edge_decision"] == "LONG", 1.0, np.where(out["price_edge_decision"] == "SHORT", -1.0, 0.0)),
        index=out.index,
    )
    long_excess = (out["pred_price_delta"] - config.long_price_edge_threshold).clip(lower=0.0)
    short_excess = (config.short_price_edge_threshold - out["pred_price_delta"]).clip(lower=0.0)
    out["price_edge_strength"] = long_excess + short_excess

    if config.enable_new_signal:
        ensemble_signal = (
            config.forecast_weight * out["forecast_signal"]
            + config.mean_reversion_weight * out["mean_reversion_signal"]
            + config.fundamental_weight * (out["fundamental_signal"] - 0.25 * out["spread_signal"])
        )
        if config.enable_regime_switching:
            ensemble_signal = np.where(
                out["market_regime"] == "trend",
                config.forecast_weight * out["forecast_signal"] + config.fundamental_weight * out["fundamental_signal"],
                config.mean_reversion_weight * out["mean_reversion_signal"] + config.fundamental_weight * out["fundamental_signal"],
            )
            ensemble_signal = pd.Series(ensemble_signal, index=out.index)
        out["ensemble_signal"] = pd.to_numeric(ensemble_signal, errors="coerce").fillna(0.0)
        out["combined_signal"] = price_edge_direction * out["price_edge_strength"]
        out["signal_z_score"] = out["combined_signal"]
        out["signal_strength"] = out["price_edge_strength"]
        raw_position = price_edge_direction * np.tanh(out["price_edge_strength"] / max(config.position_scale_k, 1e-6))
        aligned_fundamental = price_edge_direction * out["fundamental_signal"]
        size_multiplier = 1.0 + 0.25 * np.tanh(aligned_fundamental.clip(lower=0.0))
        raw_position = raw_position * size_multiplier
        if config.enable_volatility_scaling:
            raw_position = raw_position / (1.0 + rolling_volatility)
        high_vol_mask = out["vol_regime"] == "high_vol"
        raw_position = pd.Series(raw_position, index=out.index)
        raw_position.loc[high_vol_mask] = raw_position.loc[high_vol_mask] * 0.65
        out["target_position_raw"] = raw_position
        out["target_position_capped"] = pd.to_numeric(out["target_position_raw"], errors="coerce").clip(
            -abs(config.position_limit), abs(config.position_limit)
        )
        out["target_position"] = out["target_position_capped"]
    else:
        raw_signal = out["pred_price_delta"]
        out["ensemble_signal"] = 0.06 * out["imbalance_pred"] + 0.4 * out["pred_price_delta"]
        out["forecast_signal"] = out["pred_price_delta"]
        out["mean_reversion_signal"] = 0.0
        out["fundamental_signal"] = out["imbalance_pred"] / max(config.imbalance_scale, 1e-6)
        out["combined_signal"] = raw_signal
        out["signal_z_score"] = 0.0
        out["signal_strength"] = out["price_edge_strength"]
        out["target_position_raw"] = price_edge_direction * np.tanh(out["price_edge_strength"] / max(config.position_scale_k, 1e-6))
        out["target_position_capped"] = pd.to_numeric(out["target_position_raw"], errors="coerce").clip(
            -abs(config.position_limit), abs(config.position_limit)
        )
        out["target_position"] = out["target_position_capped"]
        out["vol_regime"] = "low_vol"
        out["market_regime"] = "trend"

    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").clip(-1.0, 1.0).fillna(0.0)
    out["target_decision"] = out["price_edge_decision"]
    out["recommended_decision"] = out["target_decision"]
    return out
