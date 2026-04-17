"""ENTSO-E data client utilities."""

from __future__ import annotations

import logging

import pandas as pd
from entsoe import EntsoePandasClient
import requests

LOGGER = logging.getLogger(__name__)

RENEWABLE_PSRTYPE = ["B16", "B18", "B19", "B10", "B11", "B12"]


def _ensure_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _series_to_frame(value: pd.Series | pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Normalize ENTSO-E series/dataframe responses."""
    if isinstance(value, pd.Series):
        frame = value.to_frame(name=column_name)
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
        if frame.shape[1] == 1:
            frame.columns = [column_name]
        elif isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                " ".join(str(part).strip() for part in col if str(part).strip())
                for col in frame.columns.to_flat_index()
            ]
    else:
        raise TypeError(f"Unsupported ENTSO-E payload type: {type(value)}")
    frame = frame.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " ".join(str(part).strip() for part in col if str(part).strip())
            for col in frame.columns.to_flat_index()
        ]
    timestamp_column = frame.columns[0]
    frame = frame.rename(columns={timestamp_column: "timestamp_utc"})
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


def _parse_renewables(raw: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _series_to_frame(raw, "renewable_mw")
    if "renewable_mw" not in frame.columns:
        numeric_cols = frame.select_dtypes(include="number").columns
        frame["renewable_mw"] = frame[numeric_cols].sum(axis=1)
    return frame[["timestamp_utc", "renewable_mw"]]


def fetch_entsoe_energy_data(
    api_key: str,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch day-ahead price, load and renewables from ENTSO-E."""
    session = requests.Session()
    session.trust_env = False
    client = EntsoePandasClient(api_key=api_key, session=session, proxies={}, timeout=30)
    start = _ensure_utc_timestamp(start)
    end = _ensure_utc_timestamp(end)

    LOGGER.info("Fetching ENTSO-E day-ahead prices for %s", zone)
    prices = client.query_day_ahead_prices(country_code=zone, start=start, end=end)
    prices_df = _series_to_frame(prices, "price_eur_mwh")

    LOGGER.info("Fetching ENTSO-E load forecast for %s", zone)
    load = client.query_load_forecast(country_code=zone, start=start, end=end)
    load_df = _series_to_frame(load, "demand_mw")

    LOGGER.info("Fetching ENTSO-E renewable generation for %s", zone)
    renewables = client.query_generation(
        country_code=zone,
        start=start,
        end=end,
        psr_type=None,
    )
    renewables_df = _parse_renewables(renewables)

    merged = prices_df.merge(load_df, on="timestamp_utc", how="outer").merge(
        renewables_df, on="timestamp_utc", how="outer"
    )
    merged["demand_kw"] = merged["demand_mw"] * 1000.0
    merged = merged.drop(columns=["demand_mw"])
    return merged.sort_values("timestamp_utc").reset_index(drop=True)
