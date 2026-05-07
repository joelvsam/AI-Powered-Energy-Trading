"""Merge pipeline outputs."""

from __future__ import annotations

import pandas as pd

from src.config import AppConfig


def merge_energy_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    merged = energy_df.merge(weather_df, on="timestamp_utc", how="inner")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged.to_csv(cfg.data_processed_dir / "energy_weather_clean.csv", index=False)
    return merged
