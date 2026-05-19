"""ENTSO-E data client utilities."""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_pipeline.cache import ENERGY_VALUE_COLUMNS

LOGGER = logging.getLogger(__name__)
OPTIONAL_ENTSOE_FEED_POLICY = "skipped_by_default"


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


def _rollup_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mixed-frequency ENTSO-E payloads onto the pipeline hourly index."""
    if frame.empty:
        return frame
    out = frame.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce").dt.floor("h")
    for column in ENERGY_VALUE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    numeric_cols = [col for col in ENERGY_VALUE_COLUMNS if col in out.columns]
    if not numeric_cols:
        return (
            out[["timestamp_utc"]]
            .dropna()
            .drop_duplicates(subset=["timestamp_utc"])
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
    hourly = out.groupby("timestamp_utc", as_index=False)[numeric_cols].mean()
    return hourly.sort_values("timestamp_utc").reset_index(drop=True)


def _session_with_retries() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_chunk_with_fallback(
    fetcher: Callable[[pd.Timestamp, pd.Timestamp], pd.Series | pd.DataFrame],
    parser: Callable[[pd.Series | pd.DataFrame], pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
) -> list[pd.DataFrame]:
    LOGGER.info("Fetching ENTSO-E %s chunk %s -> %s", label, start.isoformat(), end.isoformat())
    try:
        raw = fetcher(start, end)
        return [parser(raw)]
    except Exception as exc:  # noqa: BLE001
        span = end - start
        if span <= pd.Timedelta(days=1):
            LOGGER.warning(
                "Skipping ENTSO-E %s chunk %s -> %s after fetch failure: %s",
                label,
                start.isoformat(),
                end.isoformat(),
                exc,
            )
            return []
        midpoint = start + (span / 2)
        midpoint = pd.Timestamp(midpoint).ceil("h")
        if midpoint <= start or midpoint >= end:
            LOGGER.warning(
                "Skipping ENTSO-E %s chunk %s -> %s after unsplittable fetch failure: %s",
                label,
                start.isoformat(),
                end.isoformat(),
                exc,
            )
            return []
        LOGGER.warning(
            "Retrying ENTSO-E %s chunk %s -> %s in smaller ranges after fetch failure: %s",
            label,
            start.isoformat(),
            end.isoformat(),
            exc,
        )
        left = _fetch_chunk_with_fallback(fetcher, parser, start, midpoint, label)
        right = _fetch_chunk_with_fallback(fetcher, parser, midpoint, end, label)
        return [*left, *right]


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
        frames.extend(_fetch_chunk_with_fallback(fetcher, parser, chunk_start, chunk_end, label))
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame(columns=["timestamp_utc"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    return merged


def _skipped_optional_frame(columns: list[str], *, zone: str, label: str) -> pd.DataFrame:
    LOGGER.info(
        "Skipping ENTSO-E %s for %s because optional context feeds are disabled by policy (%s).",
        label,
        zone,
        OPTIONAL_ENTSOE_FEED_POLICY,
    )
    return pd.DataFrame(columns=columns)


def fetch_entsoe_energy_data(
    api_key: str,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout_s: int = 90,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """Fetch European power market data with day-ahead and intraday context."""
    session = _session_with_retries()
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

    intraday_prices_df = _skipped_optional_frame(
        ["timestamp_utc", "intraday_price_eur_mwh"],
        zone=zone,
        label="intraday prices",
    )

    imbalance_prices_df = _skipped_optional_frame(
        [
            "timestamp_utc",
            "imbalance_price_eur_mwh",
            "imbalance_price_buy_eur_mwh",
            "imbalance_price_sell_eur_mwh",
        ],
        zone=zone,
        label="imbalance prices",
    )

    intraday_renewable_forecast_df = _skipped_optional_frame(
        ["timestamp_utc", "intraday_renewable_forecast_mw"],
        zone=zone,
        label="intraday renewable forecast",
    )

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
    return _rollup_to_hourly(merged)
