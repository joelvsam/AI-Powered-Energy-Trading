"""ENTSO-E data client utilities."""

from __future__ import annotations

import logging

import pandas as pd
from entsoe import EntsoePandasClient
import requests
from typing import Callable

LOGGER = logging.getLogger(__name__)

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


def _parse_intraday_prices(raw: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _series_to_frame(raw, "intraday_price_eur_mwh")
    if "intraday_price_eur_mwh" not in frame.columns:
        numeric_cols = frame.select_dtypes(include="number").columns
        frame["intraday_price_eur_mwh"] = frame[numeric_cols].mean(axis=1)
    return frame[["timestamp_utc", "intraday_price_eur_mwh"]]


def _parse_imbalance_prices(raw: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _series_to_frame(raw, "imbalance_price_eur_mwh")
    numeric_cols = [col for col in frame.columns if col != "timestamp_utc" and pd.api.types.is_numeric_dtype(frame[col])]
    if "Long" in frame.columns and "Short" in frame.columns:
        frame["imbalance_price_buy_eur_mwh"] = pd.to_numeric(frame["Long"], errors="coerce")
        frame["imbalance_price_sell_eur_mwh"] = pd.to_numeric(frame["Short"], errors="coerce")
    elif len(numeric_cols) >= 2:
        frame["imbalance_price_buy_eur_mwh"] = pd.to_numeric(frame[numeric_cols[0]], errors="coerce")
        frame["imbalance_price_sell_eur_mwh"] = pd.to_numeric(frame[numeric_cols[1]], errors="coerce")
    elif numeric_cols:
        numeric = pd.to_numeric(frame[numeric_cols[0]], errors="coerce")
        frame["imbalance_price_buy_eur_mwh"] = numeric
        frame["imbalance_price_sell_eur_mwh"] = numeric
    else:
        frame["imbalance_price_buy_eur_mwh"] = pd.NA
        frame["imbalance_price_sell_eur_mwh"] = pd.NA
    frame["imbalance_price_eur_mwh"] = frame[
        ["imbalance_price_buy_eur_mwh", "imbalance_price_sell_eur_mwh"]
    ].mean(axis=1)
    return frame[
        [
            "timestamp_utc",
            "imbalance_price_eur_mwh",
            "imbalance_price_buy_eur_mwh",
            "imbalance_price_sell_eur_mwh",
        ]
    ]


def _parse_intraday_renewable_forecast(raw: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _series_to_frame(raw, "intraday_renewable_forecast_mw")
    numeric_cols = [col for col in frame.columns if col != "timestamp_utc" and pd.api.types.is_numeric_dtype(frame[col])]
    if numeric_cols:
        frame["intraday_renewable_forecast_mw"] = frame[numeric_cols].sum(axis=1)
    else:
        frame["intraday_renewable_forecast_mw"] = pd.NA
    return frame[["timestamp_utc", "intraday_renewable_forecast_mw"]]


def _fetch_in_chunks(
    fetcher: Callable[[pd.Timestamp, pd.Timestamp], pd.Series | pd.DataFrame],
    parser: Callable[[pd.Series | pd.DataFrame], pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
    label: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    chunk_start = start
    delta = pd.Timedelta(days=max(1, chunk_days))

    while chunk_start < end:
        chunk_end = min(chunk_start + delta, end)
        LOGGER.info("Fetching ENTSO-E %s chunk %s -> %s", label, chunk_start.isoformat(), chunk_end.isoformat())
        raw = fetcher(chunk_start, chunk_end)
        frames.append(parser(raw))
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return merged


def fetch_entsoe_energy_data(
    api_key: str,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout_s: int = 90,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """Fetch European power market data with day-ahead and intraday context."""
    session = requests.Session()
    session.trust_env = False
    client = EntsoePandasClient(api_key=api_key, session=session, proxies={}, timeout=timeout_s)
    start = _ensure_utc_timestamp(start)
    end = _ensure_utc_timestamp(end)

    prices_df = _fetch_in_chunks(
        fetcher=lambda chunk_start, chunk_end: client.query_day_ahead_prices(
            country_code=zone,
            start=chunk_start,
            end=chunk_end,
        ),
        parser=lambda raw: _series_to_frame(raw, "price_eur_mwh"),
        start=start,
        end=end,
        chunk_days=chunk_days,
        label="day-ahead prices",
    )

    load_df = _fetch_in_chunks(
        fetcher=lambda chunk_start, chunk_end: client.query_load_forecast(
            country_code=zone,
            start=chunk_start,
            end=chunk_end,
        ),
        parser=lambda raw: _series_to_frame(raw, "demand_mw"),
        start=start,
        end=end,
        chunk_days=chunk_days,
        label="load forecast",
    )

    renewables_df = _fetch_in_chunks(
        fetcher=lambda chunk_start, chunk_end: client.query_generation(
            country_code=zone,
            start=chunk_start,
            end=chunk_end,
            psr_type=None,
        ),
        parser=_parse_renewables,
        start=start,
        end=end,
        chunk_days=chunk_days,
        label="renewable generation",
    )

    intraday_prices_df = pd.DataFrame(columns=["timestamp_utc", "intraday_price_eur_mwh"])
    try:
        intraday_prices_df = _fetch_in_chunks(
            fetcher=lambda chunk_start, chunk_end: client.query_intraday_prices(
                country_code=zone,
                start=chunk_start,
                end=chunk_end,
                sequence=1,
            ),
            parser=_parse_intraday_prices,
            start=start,
            end=end,
            chunk_days=chunk_days,
            label="intraday prices",
        )
    except Exception as exc:
        LOGGER.warning("Intraday prices unavailable for %s: %s", zone, exc)

    imbalance_prices_df = pd.DataFrame(
        columns=[
            "timestamp_utc",
            "imbalance_price_eur_mwh",
            "imbalance_price_buy_eur_mwh",
            "imbalance_price_sell_eur_mwh",
        ]
    )
    try:
        imbalance_prices_df = _fetch_in_chunks(
            fetcher=lambda chunk_start, chunk_end: client.query_imbalance_prices(
                country_code=zone,
                start=chunk_start,
                end=chunk_end,
            ),
            parser=_parse_imbalance_prices,
            start=start,
            end=end,
            chunk_days=chunk_days,
            label="imbalance prices",
        )
    except Exception as exc:
        LOGGER.warning("Imbalance prices unavailable for %s: %s", zone, exc)

    intraday_renewable_forecast_df = pd.DataFrame(columns=["timestamp_utc", "intraday_renewable_forecast_mw"])
    try:
        intraday_renewable_forecast_df = _fetch_in_chunks(
            fetcher=lambda chunk_start, chunk_end: client.query_intraday_wind_and_solar_forecast(
                country_code=zone,
                start=chunk_start,
                end=chunk_end,
            ),
            parser=_parse_intraday_renewable_forecast,
            start=start,
            end=end,
            chunk_days=chunk_days,
            label="intraday renewable forecast",
        )
    except Exception as exc:
        LOGGER.warning("Intraday renewable forecast unavailable for %s: %s", zone, exc)

    merged = (
        prices_df.rename(columns={"price_eur_mwh": "day_ahead_price_eur_mwh"})
        .merge(load_df, on="timestamp_utc", how="outer")
        .merge(renewables_df, on="timestamp_utc", how="outer")
        .merge(intraday_prices_df, on="timestamp_utc", how="left")
        .merge(imbalance_prices_df, on="timestamp_utc", how="left")
        .merge(intraday_renewable_forecast_df, on="timestamp_utc", how="left")
    )
    merged["demand_kw"] = merged["demand_mw"] * 1000.0
    merged = merged.drop(columns=["demand_mw"])
    merged["price_eur_mwh"] = merged["intraday_price_eur_mwh"].where(
        merged["intraday_price_eur_mwh"].notna(),
        merged["day_ahead_price_eur_mwh"],
    )
    merged["intraday_day_ahead_spread_eur_mwh"] = merged["intraday_price_eur_mwh"] - merged["day_ahead_price_eur_mwh"]
    return merged.sort_values("timestamp_utc").reset_index(drop=True)
