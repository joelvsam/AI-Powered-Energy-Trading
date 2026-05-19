"""Data cleaning functions."""

from __future__ import annotations

import pandas as pd

from src.data_pipeline.cache import ENERGY_VALUE_COLUMNS, PROVENANCE_COLUMNS, WEATHER_VALUE_COLUMNS

def _interpolate_on_timestamp(out: pd.DataFrame, value_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    out = out.dropna(subset=["timestamp_utc"])
    if out.empty:
        return out, pd.Series(dtype=bool)

    indexed = out.set_index("timestamp_utc")
    missing_before = indexed[value_cols].isna().any(axis=1)
    indexed[value_cols] = indexed[value_cols].interpolate(method="time", limit=6)
    indexed[value_cols] = indexed[value_cols].ffill(limit=3).bfill(limit=3)
    return indexed.reset_index(), missing_before.reindex(indexed.index, fill_value=False)


def _update_data_quality_for_interpolation(out: pd.DataFrame, interpolation_rows: pd.Series) -> pd.DataFrame:
    if "data_quality" not in out.columns:
        return out
    if interpolation_rows.empty:
        return out
    mask = out["timestamp_utc"].isin(interpolation_rows[interpolation_rows].index)
    out.loc[mask & out["data_quality"].eq("real"), "data_quality"] = "partially_synthetic"
    out["is_synthetic"] = out["data_quality"].ne("real")
    out["data_source"] = out["data_source"].fillna("unknown")
    out.loc[out["data_quality"].eq("synthetic"), "data_source"] = out.loc[
        out["data_quality"].eq("synthetic"), "data_source"
    ].replace({"unknown": "synthetic"})
    return out


def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    numeric_cols = ENERGY_VALUE_COLUMNS.copy()
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    out, interpolated_rows = _interpolate_on_timestamp(out, numeric_cols)
    out["day_ahead_price_eur_mwh"] = out["day_ahead_price_eur_mwh"].fillna(out["price_eur_mwh"])
    out["intraday_price_eur_mwh"] = out["intraday_price_eur_mwh"].fillna(out["price_eur_mwh"])
    out["price_eur_mwh"] = out["intraday_price_eur_mwh"].fillna(out["day_ahead_price_eur_mwh"]).fillna(out["price_eur_mwh"])
    out["intraday_day_ahead_spread_eur_mwh"] = out["intraday_price_eur_mwh"] - out["day_ahead_price_eur_mwh"]
    out = _update_data_quality_for_interpolation(out, interpolated_rows)
    return out


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    weather_cols = WEATHER_VALUE_COLUMNS.copy()
    for col in weather_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in PROVENANCE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    out, interpolated_rows = _interpolate_on_timestamp(out, weather_cols)
    out = _update_data_quality_for_interpolation(out, interpolated_rows)
    return out
