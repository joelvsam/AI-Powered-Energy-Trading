"""Feature engineering for forecasting and trading."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.data_pipeline.cache import PROVENANCE_COLUMNS


LOGGER = logging.getLogger(__name__)
NON_MODEL_PROVENANCE_COLUMNS = set(
    PROVENANCE_COLUMNS
    + [
        "zone",
        "weather_data_source",
        "weather_is_synthetic",
        "weather_data_quality",
        "weather_fetch_timestamp_utc",
        "weather_cache_version",
        "weather_zone",
        "weather_cache_lat",
        "weather_cache_lon",
        "weather_lat",
        "weather_lon",
    ]
)


def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _nan_fraction(series: pd.Series) -> float:
    if series is None or series.empty:
        return 1.0
    return float(series.isna().mean())


def _validate_inputs(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        raise ValueError(
            "Feature build received an empty dataframe from the data pipeline. "
            "Check ingestion/merge logs for row counts and timestamp overlap."
        )

    required_cols = [
        "timestamp_utc",
        "price_eur_mwh",
        "demand_kw",
        "renewable_mw",
        "temperature_c",
        "wind_speed_mps",
        "radiation_wm2",
        "humidity_pct",
    ]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Cannot build features: missing required columns from merged dataset: "
            f"{missing_required}. Upstream sources: ENTSO-E (price/demand/renewables) + Open-Meteo (weather)."
        )

    bad_required = []
    for col in required_cols:
        frac = _nan_fraction(df[col])
        if frac > 0.05:
            bad_required.append((col, frac))
    if bad_required:
        details = ", ".join([f"{col} NaN={frac:.1%}" for col, frac in bad_required])
        raise ValueError(
            "Cannot build features: required columns have too many missing values after cleaning/merge "
            f"({details}). Check zone/lookback and upstream API availability."
        )

    optional_context = [
        "day_ahead_price_eur_mwh",
        "intraday_price_eur_mwh",
        "imbalance_price_eur_mwh",
        "intraday_renewable_forecast_mw",
    ]
    optional_missingness: dict[str, float] = {}
    for col in optional_context:
        if col not in df.columns:
            optional_missingness[col] = 1.0
        else:
            optional_missingness[col] = _nan_fraction(df[col])
    return optional_missingness


def _fill_optional_context(out: pd.DataFrame, optional_missingness: dict[str, float]) -> pd.DataFrame:
    for col in optional_missingness:
        if col not in out.columns:
            out[col] = np.nan

    out["day_ahead_price_eur_mwh"] = pd.to_numeric(out["day_ahead_price_eur_mwh"], errors="coerce").fillna(
        pd.to_numeric(out["price_eur_mwh"], errors="coerce")
    )
    out["intraday_price_eur_mwh"] = pd.to_numeric(out["intraday_price_eur_mwh"], errors="coerce").fillna(
        out["day_ahead_price_eur_mwh"]
    )
    out["imbalance_price_eur_mwh"] = pd.to_numeric(out["imbalance_price_eur_mwh"], errors="coerce").fillna(
        out["intraday_price_eur_mwh"]
    )
    out["intraday_renewable_forecast_mw"] = pd.to_numeric(
        out["intraday_renewable_forecast_mw"], errors="coerce"
    ).fillna(pd.to_numeric(out["renewable_mw"], errors="coerce"))

    out["intraday_day_ahead_spread_eur_mwh"] = out["intraday_price_eur_mwh"] - out["day_ahead_price_eur_mwh"]
    out["imbalance_price_spread_eur_mwh"] = out["imbalance_price_eur_mwh"] - out["intraday_price_eur_mwh"]

    for col, frac in optional_missingness.items():
        availability_col = f"{col}_available"
        out[availability_col] = float(frac <= 0.50)
        if frac > 0.50:
            LOGGER.warning(
                "Optional ENTSO-E context mostly missing for %s (NaN=%.1f%%). "
                "Falling back to neutral proxy values so the research pipeline can continue.",
                col,
                frac * 100.0,
            )
    return out


def build_features(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    optional_missingness = _validate_inputs(df)
    out = df.copy().sort_values("timestamp_utc")
    out = _fill_optional_context(out, optional_missingness)
    out["hour"] = out["timestamp_utc"].dt.hour
    out["day_of_week"] = out["timestamp_utc"].dt.dayofweek
    out["month"] = out["timestamp_utc"].dt.month

    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour"], 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(out["day_of_week"], 7)
    out["month_sin"], out["month_cos"] = _cyclical(out["month"], 12)

    out["net_load_mw"] = out["demand_kw"] / 1000.0 - out["renewable_mw"]
    out["renewable_forecast_error_mw"] = out["intraday_renewable_forecast_mw"] - out["renewable_mw"]
    out["intraday_day_ahead_spread_eur_mwh"] = out["intraday_price_eur_mwh"] - out["day_ahead_price_eur_mwh"]

    base_cols = [
        "demand_kw",
        "renewable_mw",
        "price_eur_mwh",
        "day_ahead_price_eur_mwh",
        "intraday_price_eur_mwh",
        "intraday_day_ahead_spread_eur_mwh",
        "imbalance_price_eur_mwh",
        "net_load_mw",
    ]
    for col in base_cols:
        out[f"{col}_lag_1"] = out[col].shift(1)
        out[f"{col}_lag_24"] = out[col].shift(24)
        out[f"{col}_roll_mean_24"] = out[col].rolling(24).mean()
        out[f"{col}_roll_std_24"] = out[col].rolling(24).std()
        out[f"{col}_ramp_1h"] = out[col].diff(1)
        out[f"{col}_ramp_24h"] = out[col].diff(24)

    out["renewable_penetration"] = out["renewable_mw"] / (out["demand_kw"] / 1000.0 + 1e-6)
    out["price_momentum"] = out["price_eur_mwh"].pct_change().replace([np.inf, -np.inf], 0.0)
    out["demand_delta"] = out["demand_kw"].diff()
    out["demand_ramp_1h"] = out["demand_kw"].diff(1)
    out["renewable_ramp_1h"] = out["renewable_mw"].diff(1)
    out["price_ramp_1h"] = out["price_eur_mwh"].diff(1)
    out["day_ahead_price_ramp_1h"] = out["day_ahead_price_eur_mwh"].diff(1)
    out["spread_ramp_1h"] = out["intraday_day_ahead_spread_eur_mwh"].diff(1)
    out["imbalance_price_spread"] = out["imbalance_price_eur_mwh"] - out["price_eur_mwh"]
    out["imbalance_zscore_24"] = (
        (out["net_load_mw"] - out["net_load_mw"].rolling(24, min_periods=12).mean())
        / out["net_load_mw"].rolling(24, min_periods=12).std().replace(0.0, np.nan)
    )
    out["realized_volatility_24"] = out["price_eur_mwh"].diff().rolling(24, min_periods=12).std()
    out["spread_volatility_24"] = out["intraday_day_ahead_spread_eur_mwh"].rolling(24, min_periods=12).std()
    out["high_vol_regime"] = (
        out["realized_volatility_24"]
        > out["realized_volatility_24"].rolling(24 * 7, min_periods=24).quantile(0.7).bfill().ffill()
    ).astype(int)
    out["intraday_stress_proxy"] = (
        out["intraday_day_ahead_spread_eur_mwh"].abs()
        + out["renewable_forecast_error_mw"].abs().fillna(0.0)
        + out["net_load_mw"].diff().abs().fillna(0.0)
    )
    out["winter_peak_interaction"] = ((out["month"].isin([11, 12, 1, 2])).astype(int) * out["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int))
    out["solar_shape_proxy"] = out["radiation_wm2"].rolling(3, min_periods=1).mean()
    out["wind_ramp_proxy"] = out["wind_speed_mps"].diff().fillna(0.0)

    weather_cols = ["temperature_c", "wind_speed_mps", "radiation_wm2", "humidity_pct"]
    for col in weather_cols:
        mean = out[col].rolling(24, min_periods=12).mean()
        std = out[col].rolling(24, min_periods=12).std().replace(0, np.nan)
        out[f"{col}_anomaly"] = (out[col] - mean) / std

    interaction_features: dict[str, pd.Series] = {}
    for interaction_col in ["net_load_mw", "intraday_day_ahead_spread_eur_mwh", "renewable_forecast_error_mw"]:
        filled = out[interaction_col].fillna(0.0)
        interaction_features[f"{interaction_col}_x_hour_sin"] = filled * out["hour_sin"]
        interaction_features[f"{interaction_col}_x_dow_cos"] = filled * out["dow_cos"]
    out = pd.concat([out, pd.DataFrame(interaction_features, index=out.index)], axis=1)

    numeric_cols = [col for col in out.select_dtypes(include=[np.number]).columns if col not in NON_MODEL_PROVENANCE_COLUMNS]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out[numeric_cols].ffill().bfill()

    all_nan_columns = [col for col in out.columns if out[col].isna().all()]
    if all_nan_columns:
        LOGGER.warning(
            "Dropping feature columns that remain entirely NaN after fallback handling: %s",
            all_nan_columns,
        )
        out = out.drop(columns=all_nan_columns)

    required_after_engineering = ["timestamp_utc"] + [col for col in out.columns if col != "timestamp_utc"]
    out = out.dropna(subset=required_after_engineering).reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "Feature engineering resulted in 0 usable rows after applying lags/rolling windows and fallback filling. "
            "This usually indicates insufficient history length or severe missingness in required market/weather inputs."
        )
    out.to_csv(cfg.data_processed_dir / "features.csv", index=False)
    return out
