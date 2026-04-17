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
from src.data_sources.entsoe_client import fetch_entsoe_energy_data

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionOutputs:
    energy_df: pd.DataFrame
    weather_df: pd.DataFrame
    energy_source: str


def _date_range(cfg: AppConfig, lookback_days: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    days = lookback_days or cfg.lookback_days
    end = pd.Timestamp.utcnow().floor("h").tz_convert("UTC")
    start = end - pd.Timedelta(days=days)
    return start, end


def _synthetic_energy(start: pd.Timestamp, end: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, end=end, freq="h", tz="UTC")
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
    price = (45 + 2.8 * imbalance + rng.normal(0, 4, len(ts))).clip(min=0.0)

    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": price,
            "demand_kw": demand_kw,
            "renewable_mw": renewable_mw,
        }
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_openmeteo_weather(start: pd.Timestamp, end: pd.Timestamp, cfg: AppConfig) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg.openmeteo_lat,
        "longitude": cfg.openmeteo_lon,
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


def ingest_data(cfg: AppConfig, zone: str | None = None, lookback_days: int | None = None) -> IngestionOutputs:
    """Ingest energy data from ENTSO-E with synthetic fallback + weather."""
    start, end = _date_range(cfg, lookback_days)
    energy_source = "entsoe"
    selected_zone = zone or cfg.default_zone

    try:
        if not cfg.entsoe_api_key:
            raise RuntimeError("ENTSOE_API_KEY missing, switching to synthetic mode.")
        energy_df = fetch_entsoe_energy_data(
            api_key=cfg.entsoe_api_key,
            zone=selected_zone,
            start=start,
            end=end,
            timeout_s=cfg.entsoe_timeout_s,
            chunk_days=cfg.entsoe_chunk_days,
        )
    except Exception as exc:
        LOGGER.warning("ENTSO-E ingestion failed: %s", exc)
        energy_df = _synthetic_energy(start=start, end=end, seed=cfg.random_seed)
        energy_source = "synthetic"

    weather_df = fetch_openmeteo_weather(start=start, end=end, cfg=cfg)
    energy_df.to_csv(cfg.data_raw_dir / "energy_raw.csv", index=False)
    weather_df.to_csv(cfg.data_raw_dir / "weather_raw.csv", index=False)
    return IngestionOutputs(energy_df=energy_df, weather_df=weather_df, energy_source=energy_source)
