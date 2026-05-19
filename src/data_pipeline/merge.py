"""Merge pipeline outputs."""

from __future__ import annotations

import pandas as pd

from src.config import AppConfig


def merge_energy_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    weather_prefixed = weather_df.rename(
        columns={
            "data_source": "weather_data_source",
            "is_synthetic": "weather_is_synthetic",
            "data_quality": "weather_data_quality",
            "fetch_timestamp_utc": "weather_fetch_timestamp_utc",
            "cache_version": "weather_cache_version",
            "zone": "weather_zone",
            "weather_lat": "weather_cache_lat",
            "weather_lon": "weather_cache_lon",
        }
    )
    merged = energy_df.merge(weather_prefixed, on="timestamp_utc", how="inner")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged.to_csv(cfg.data_processed_dir / "energy_weather_clean.csv", index=False)
    return merged
