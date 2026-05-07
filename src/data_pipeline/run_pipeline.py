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


def _log_frame_diagnostics(df: pd.DataFrame, label: str) -> None:
    if df is None:
        LOGGER.info("%s: df=None", label)
        return
    if df.empty:
        LOGGER.info("%s: rows=0 cols=%s", label, list(df.columns))
        return
    ts = pd.to_datetime(df.get("timestamp_utc", pd.Series([], dtype="datetime64[ns, UTC]")), utc=True, errors="coerce")
    LOGGER.info(
        "%s: rows=%d cols=%d ts_range=[%s, %s]",
        label,
        len(df),
        df.shape[1],
        str(ts.min()) if not ts.empty else "n/a",
        str(ts.max()) if not ts.empty else "n/a",
    )


@dataclass
class DataPipelineOutputs:
    merged_df: pd.DataFrame
    energy_source: str


def run_data_pipeline(cfg: AppConfig, zone: str | None = None, lookback_days: int | None = None) -> DataPipelineOutputs:
    outputs = ingest_data(cfg=cfg, zone=zone, lookback_days=lookback_days)
    LOGGER.info("Energy source selected: %s", outputs.energy_source)
    _log_frame_diagnostics(outputs.energy_df, "ingest.energy_raw")
    _log_frame_diagnostics(outputs.weather_df, "ingest.weather_raw")
    energy_clean = clean_energy_data(outputs.energy_df)
    weather_clean = clean_weather_data(outputs.weather_df)
    _log_frame_diagnostics(energy_clean, "clean.energy")
    _log_frame_diagnostics(weather_clean, "clean.weather")
    merged = merge_energy_weather(energy_clean, weather_clean, cfg)
    _log_frame_diagnostics(merged, "merge.energy_weather")
    if merged.empty:
        raise ValueError(
            "Data pipeline produced 0 merged rows after joining energy + weather on timestamp. "
            "This usually means no overlapping timestamps or upstream NaNs were dropped. "
            "Inspect data/raw/*.csv and adjust zone/lookback or data-source availability."
        )
    return DataPipelineOutputs(merged_df=merged, energy_source=outputs.energy_source)
