"""Data cleaning functions."""

from __future__ import annotations

import pandas as pd


def _interpolate_on_timestamp(out: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = out.dropna(subset=["timestamp_utc"])
    if out.empty:
        return out

    indexed = out.set_index("timestamp_utc")
    indexed[value_cols] = indexed[value_cols].interpolate(method="time", limit=6)
    indexed[value_cols] = indexed[value_cols].ffill(limit=3).bfill(limit=3)
    return indexed.reset_index()


def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    numeric_cols = [
        "price_eur_mwh",
        "day_ahead_price_eur_mwh",
        "intraday_price_eur_mwh",
        "intraday_day_ahead_spread_eur_mwh",
        "imbalance_price_eur_mwh",
        "imbalance_price_buy_eur_mwh",
        "imbalance_price_sell_eur_mwh",
        "demand_kw",
        "renewable_mw",
        "intraday_renewable_forecast_mw",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    out = _interpolate_on_timestamp(out, numeric_cols)
    out["day_ahead_price_eur_mwh"] = out["day_ahead_price_eur_mwh"].fillna(out["price_eur_mwh"])
    out["intraday_price_eur_mwh"] = out["intraday_price_eur_mwh"].fillna(out["price_eur_mwh"])
    out["price_eur_mwh"] = out["intraday_price_eur_mwh"].fillna(out["day_ahead_price_eur_mwh"]).fillna(out["price_eur_mwh"])
    out["intraday_day_ahead_spread_eur_mwh"] = out["intraday_price_eur_mwh"] - out["day_ahead_price_eur_mwh"]
    return out


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    weather_cols = ["temperature_c", "wind_speed_mps", "radiation_wm2", "humidity_pct"]
    for col in weather_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return _interpolate_on_timestamp(out, weather_cols)
