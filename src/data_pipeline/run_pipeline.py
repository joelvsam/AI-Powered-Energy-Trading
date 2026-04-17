"""Execute data pipeline stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import AppConfig
from src.data_pipeline.clean import clean_energy_data, clean_weather_data
from src.data_pipeline.ingest import ingest_data
from src.data_pipeline.merge import merge_energy_weather

LOGGER = logging.getLogger(__name__)


@dataclass
class DataPipelineOutputs:
    merged_df: pd.DataFrame
    energy_source: str


def run_data_pipeline(cfg: AppConfig, zone: str | None = None, lookback_days: int | None = None) -> DataPipelineOutputs:
    outputs = ingest_data(cfg=cfg, zone=zone, lookback_days=lookback_days)
    LOGGER.info("Energy source selected: %s", outputs.energy_source)
    energy_clean = clean_energy_data(outputs.energy_df)
    weather_clean = clean_weather_data(outputs.weather_df)
    merged = merge_energy_weather(energy_clean, weather_clean, cfg)
    return DataPipelineOutputs(merged_df=merged, energy_source=outputs.energy_source)
