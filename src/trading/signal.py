"""Shared signal-generation helpers for trading and decision surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for transforming forecasts into tradable positions."""

    volatility_window_hours: int = 24
    position_scale_k: float = 2.0
    enable_new_signal: bool = True
    enable_volatility_scaling: bool = True
    long_threshold: float = 0.1
    short_threshold: float = -0.1


def decision_from_position(position: pd.Series, tolerance: float = 1e-6) -> pd.Series:
    """Map a continuous position into a sign-only label for reporting."""
    numeric = pd.to_numeric(position, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(numeric > tolerance, "LONG", np.where(numeric < -tolerance, "SHORT", "HOLD")),
        index=position.index,
    )


def compute_signal_frame(df: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """Compute signal diagnostics and target positions from model forecasts."""
    out = df.copy()
    out["pred_price_delta"] = pd.to_numeric(out["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(
        out["price_eur_mwh"], errors="coerce"
    )
    out["imbalance_pred"] = pd.to_numeric(out["pred_demand_kw"], errors="coerce") / 1000.0 - pd.to_numeric(
        out["pred_renewable_mw"], errors="coerce"
    )
    out["price_trend"] = pd.to_numeric(out["price_eur_mwh"], errors="coerce").diff().fillna(0.0)

    if config.enable_new_signal:
        rolling_volatility = (
            pd.to_numeric(out["price_eur_mwh"], errors="coerce")
            .rolling(config.volatility_window_hours, min_periods=max(2, config.volatility_window_hours // 2))
            .std()
            .bfill()
            .ffill()
            .fillna(0.0)
        )
        out["rolling_volatility"] = rolling_volatility
        out["signal_z_score"] = out["pred_price_delta"] / (rolling_volatility + 1e-6)
        out["signal_strength"] = out["signal_z_score"]
        raw_position = (out["signal_z_score"] / max(config.position_scale_k, 1e-6)).clip(-1.0, 1.0)
        if config.enable_volatility_scaling:
            raw_position = raw_position / (1.0 + rolling_volatility)
        out["target_position"] = raw_position.clip(-1.0, 1.0)
    else:
        raw_signal = 0.06 * out["imbalance_pred"] + 0.4 * out["pred_price_delta"]
        out["rolling_volatility"] = (
            pd.to_numeric(out["price_eur_mwh"], errors="coerce")
            .rolling(config.volatility_window_hours, min_periods=2)
            .std()
            .fillna(0.0)
        )
        out["signal_z_score"] = 0.0
        out["signal_strength"] = raw_signal
        out["target_position"] = np.tanh(raw_signal / 10.0)

    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").clip(-1.0, 1.0).fillna(0.0)
    out["target_decision"] = decision_from_position(out["target_position"])
    return out
