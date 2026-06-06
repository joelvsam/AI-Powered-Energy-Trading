"""Ingestion pipeline for energy and weather data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import AppConfig
from src.data_pipeline.cache import resolve_dataset_with_cache, summarize_provenance
from src.data_sources.entsoe_client import fetch_entsoe_energy_data

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionOutputs:
    energy_df: pd.DataFrame
    weather_df: pd.DataFrame
    energy_source: str
    energy_cache_diagnostics: dict[str, object]
    weather_cache_diagnostics: dict[str, object]
    provenance_summary: dict[str, object]


def _date_range(cfg: AppConfig, lookback_days: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = lookback_days or cfg.lookback_days
    end = pd.Timestamp.now("UTC").floor("h")
    start = end - pd.Timedelta(days=days)
    return start, end


def _synthetic_energy_for_index(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.DatetimeIndex(index)
    hour = ts.hour.values
    dow = ts.dayofweek.values
    doy = ts.dayofyear.values

    demand_kw = (
        42000
        + 9000 * np.sin(2 * np.pi * hour / 24.0)
        + 3500 * np.cos(2 * np.pi * dow / 7.0)
        + 2500 * np.sin(2 * np.pi * doy / 365.0)
        + rng.normal(0, 1200, len(ts))
    )
    renewable_mw = (
        10
        + 5 * np.sin(2 * np.pi * (hour - 4) / 24.0)
        + 2 * np.cos(2 * np.pi * doy / 365.0)
        + rng.normal(0, 1.5, len(ts))
    ).clip(min=0.2)
    imbalance = demand_kw / 1000.0 - renewable_mw
    day_ahead_price = (42 + 2.2 * imbalance + rng.normal(0, 3, len(ts))).clip(min=0.0)
    intraday_spread = rng.normal(0, 1.8, len(ts)) + 0.15 * np.roll(imbalance, 1)
    intraday_price = (day_ahead_price + intraday_spread).clip(min=0.0)
    imbalance_price = (intraday_price + rng.normal(0, 2.0, len(ts))).clip(min=0.0)
    renewable_forecast = (renewable_mw + rng.normal(0, 0.8, len(ts))).clip(min=0.0)

    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": intraday_price,
            "day_ahead_price_eur_mwh": day_ahead_price,
            "intraday_price_eur_mwh": intraday_price,
            "intraday_day_ahead_spread_eur_mwh": intraday_price - day_ahead_price,
            "imbalance_price_eur_mwh": imbalance_price,
            "imbalance_price_buy_eur_mwh": imbalance_price + 0.8,
            "imbalance_price_sell_eur_mwh": (imbalance_price - 0.8).clip(min=0.0),
            "demand_kw": demand_kw,
            "renewable_mw": renewable_mw,
            "intraday_renewable_forecast_mw": renewable_forecast,
        }
    )


def _synthetic_weather_for_index(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    ts = pd.DatetimeIndex(index)
    hour = ts.hour.values
    doy = ts.dayofyear.values
    temperature_c = 11 + 9 * np.sin(2 * np.pi * (doy - 30) / 365.0) + 5 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 1.2, len(ts))
    wind_speed_mps = np.clip(5.5 + 2.0 * np.cos(2 * np.pi * doy / 365.0) + rng.normal(0, 0.6, len(ts)), 0.0, None)
    daylight = np.clip(np.sin(2 * np.pi * (hour - 6) / 24.0), 0.0, None)
    radiation_wm2 = np.clip(450 * daylight * (0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365.0)), 0.0, None)
    humidity_pct = np.clip(68 - 0.7 * temperature_c + rng.normal(0, 3.0, len(ts)), 20.0, 100.0)
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "temperature_c": temperature_c,
            "wind_speed_mps": wind_speed_mps,
            "radiation_wm2": radiation_wm2,
            "humidity_pct": humidity_pct,
        }
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_openmeteo_weather(start: pd.Timestamp, end: pd.Timestamp, cfg: AppConfig, *, zone: str | None = None) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo using zone-specific coordinates."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    latitude, longitude = cfg.openmeteo_coords_for_zone(zone)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,relative_humidity_2m",
        "timezone": "UTC",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }
    response = _session().get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json().get("hourly", {})
    df = pd.DataFrame(payload)
    df = df.rename(
        columns={
            "time": "timestamp_utc",
            "temperature_2m": "temperature_c",
            "wind_speed_10m": "wind_speed_mps",
            "shortwave_radiation": "radiation_wm2",
            "relative_humidity_2m": "humidity_pct",
        }
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def _energy_mode_from_summary(summary: dict[str, object]) -> str:
    synthetic_rows = int(summary.get("synthetic_rows", 0))
    partial_rows = int(summary.get("partially_synthetic_rows", 0))
    real_rows = int(summary.get("real_rows", 0))
    if synthetic_rows > 0 and real_rows == 0 and partial_rows == 0:
        return "synthetic"
    if synthetic_rows > 0 or partial_rows > 0:
        return "entsoe_partial_synthetic"
    return "entsoe"


def ingest_data(
    cfg: AppConfig,
    zone: str | None = None,
    lookback_days: int | None = None,
    *,
    force_refresh: bool = False,
    rebuild_cache: bool = False,
) -> IngestionOutputs:
    """Ingest energy and weather data through the incremental raw-data cache."""
    start, end = _date_range(cfg, lookback_days)
    selected_zone = zone or cfg.default_zone
    should_rebuild = rebuild_cache or cfg.cache_rebuild_default

    def _energy_fetcher(range_start: pd.Timestamp, range_end: pd.Timestamp) -> pd.DataFrame:
        if not cfg.entsoe_api_key:
            raise RuntimeError("ENTSOE_API_KEY missing.")
        return fetch_entsoe_energy_data(
            api_key=cfg.entsoe_api_key,
            zone=selected_zone,
            start=range_start,
            end=range_end,
            timeout_s=cfg.entsoe_timeout_s,
            chunk_days=cfg.entsoe_chunk_days,
        )

    def _weather_fetcher(range_start: pd.Timestamp, range_end: pd.Timestamp) -> pd.DataFrame:
        return fetch_openmeteo_weather(start=range_start, end=range_end, cfg=cfg, zone=selected_zone)

    energy_df, energy_cache_diag = resolve_dataset_with_cache(
        cfg=cfg,
        dataset_name="entsoe",
        zone=selected_zone,
        start=start,
        end=end,
        fetcher=_energy_fetcher,
        synthetic_builder=lambda idx: _synthetic_energy_for_index(idx, seed=cfg.random_seed),
        source_label="entsoe",
        rebuild_cache=should_rebuild,
        force_refresh=force_refresh,
    )
    weather_df, weather_cache_diag = resolve_dataset_with_cache(
        cfg=cfg,
        dataset_name="weather",
        zone=selected_zone,
        start=start,
        end=end,
        fetcher=_weather_fetcher,
        synthetic_builder=lambda idx: _synthetic_weather_for_index(idx, seed=cfg.random_seed),
        source_label="openmeteo",
        rebuild_cache=should_rebuild,
        force_refresh=force_refresh,
    )
    provenance_summary = summarize_provenance(energy_df)
    energy_source = _energy_mode_from_summary(provenance_summary)
    energy_df.to_csv(cfg.data_raw_dir / "energy_raw.csv", index=False)
    weather_df.to_csv(cfg.data_raw_dir / "weather_raw.csv", index=False)
    return IngestionOutputs(
        energy_df=energy_df,
        weather_df=weather_df,
        energy_source=energy_source,
        energy_cache_diagnostics=energy_cache_diag.to_dict(),
        weather_cache_diagnostics=weather_cache_diag.to_dict(),
        provenance_summary=provenance_summary,
    )
