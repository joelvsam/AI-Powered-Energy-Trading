"""Feature engineering for forecasting and trading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import AppConfig


def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def build_features(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp_utc")
    out["hour"] = out["timestamp_utc"].dt.hour
    out["day_of_week"] = out["timestamp_utc"].dt.dayofweek
    out["month"] = out["timestamp_utc"].dt.month

    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour"], 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(out["day_of_week"], 7)
    out["month_sin"], out["month_cos"] = _cyclical(out["month"], 12)

    base_cols = ["demand_kw", "renewable_mw", "price_eur_mwh"]
    for col in base_cols:
        out[f"{col}_lag_1"] = out[col].shift(1)
        out[f"{col}_lag_24"] = out[col].shift(24)
        out[f"{col}_roll_mean_24"] = out[col].rolling(24).mean()
        out[f"{col}_roll_std_24"] = out[col].rolling(24).std()

    out["renewable_penetration"] = out["renewable_mw"] / (out["demand_kw"] / 1000.0 + 1e-6)
    out["price_momentum"] = out["price_eur_mwh"].pct_change().replace([np.inf, -np.inf], 0.0)
    out["demand_delta"] = out["demand_kw"].diff()

    weather_cols = ["temperature_c", "wind_speed_mps", "radiation_wm2", "humidity_pct"]
    for col in weather_cols:
        mean = out[col].rolling(24, min_periods=12).mean()
        std = out[col].rolling(24, min_periods=12).std().replace(0, np.nan)
        out[f"{col}_anomaly"] = (out[col] - mean) / std

    out = out.dropna().reset_index(drop=True)
    out.to_csv(cfg.data_processed_dir / "features.csv", index=False)
    return out
