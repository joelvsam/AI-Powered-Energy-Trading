"""Shared model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.data_pipeline.cache import PROVENANCE_COLUMNS


@dataclass
class TrainingOutputs:
    demand_model_path: str
    renewable_model_path: str
    price_model_path: str
    metrics_path: str
    scored_df: pd.DataFrame
    model_key: str
    scored_path: str = ""
    diagnostics_path: str = ""


def scored_predictions_path(model_key: str, cfg: AppConfig) -> Path:
    """Return the persisted scored-predictions path for a trained model."""
    return cfg.models_dir / f"scored_predictions_{model_key}.csv"


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model features while excluding targets and provenance metadata."""
    excluded = {
        "timestamp_utc",
        "demand_kw",
        "renewable_mw",
        "price_eur_mwh",
        *PROVENANCE_COLUMNS,
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
    }
    numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)
    return [column for column in df.columns if column not in excluded and column in numeric_cols]
