"""Persistent raw-data cache helpers for research-grade ingestion."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.config import AppConfig


LOGGER = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
PROVENANCE_COLUMNS = [
    "data_source",
    "is_synthetic",
    "data_quality",
    "fetch_timestamp_utc",
    "cache_version",
]
ENERGY_VALUE_COLUMNS = [
    "price_eur_mwh",
    "day_ahead_price_eur_mwh",
    "intraday_price_eur_mwh",
    "intraday_day_ahead_spread_eur_mwh",
    "imbalance_price_eur_mwh",
    "imbalance_price_buy_eur_mwh",
    "imbalance_price_sell_eur_mwh",
    "demand_kw",
    "renewable_mw",
    "intraday_renewable_forecast_mw",
]
WEATHER_VALUE_COLUMNS = [
    "temperature_c",
    "wind_speed_mps",
    "radiation_wm2",
    "humidity_pct",
]
ENERGY_REQUIRED_COLUMNS = [
    "price_eur_mwh",
    "day_ahead_price_eur_mwh",
    "demand_kw",
    "renewable_mw",
]
WEATHER_REQUIRED_COLUMNS = WEATHER_VALUE_COLUMNS.copy()
ENERGY_OPTIONAL_CONTEXT_COLUMNS = [
    "intraday_price_eur_mwh",
    "imbalance_price_eur_mwh",
    "imbalance_price_buy_eur_mwh",
    "imbalance_price_sell_eur_mwh",
    "intraday_renewable_forecast_mw",
]


@dataclass
class CacheDiagnostics:
    dataset_name: str
    cache_path: str
    cache_status: str
    cache_used: bool
    requested_rows: int
    cache_rows_loaded: int
    cache_rows_used: int
    cache_rows_written: int
    fetched_rows: int
    fetch_attempted_range_count: int
    fetch_failed_range_count: int
    fetched_range_count: int
    live_rows_accepted: int
    synthetic_overlay_rows: int
    synthesized_rows: int
    real_rows: int
    partially_synthetic_rows: int
    synthetic_rows: int
    cache_freshness_utc: str
    rebuild_cache: bool
    force_refresh: bool
    invalidated_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cache_file_path(cfg: AppConfig, dataset_name: str, zone: str) -> Path:
    return cfg.cache_dir / f"{dataset_name}_{zone}.parquet"


def build_hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    return pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")


def _metadata_columns(dataset_name: str) -> list[str]:
    if dataset_name == "weather":
        return ["zone", "weather_lat", "weather_lon"]
    return ["zone"]


def _schema_columns(dataset_name: str) -> list[str]:
    values = ENERGY_VALUE_COLUMNS if dataset_name == "entsoe" else WEATHER_VALUE_COLUMNS
    return ["timestamp_utc", *values, *PROVENANCE_COLUMNS, *_metadata_columns(dataset_name)]


def _quarantine_cache_file(path: Path, reason: str) -> None:
    if not path.exists():
        return
    safe_reason = reason.replace(" ", "_").replace(":", "_")
    target = Path(f"{path}.invalid_{safe_reason}_{pd.Timestamp.now('UTC').strftime('%Y%m%d%H%M%S')}")
    try:
        path.replace(target)
    except OSError:
        LOGGER.warning("Failed to quarantine invalid cache file %s", path, exc_info=True)


def _normalize_cache_frame(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    for column in _schema_columns(dataset_name):
        if column not in out.columns:
            out[column] = pd.NA
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["fetch_timestamp_utc"] = pd.to_datetime(out["fetch_timestamp_utc"], utc=True, errors="coerce")
    out["is_synthetic"] = out["is_synthetic"].fillna(False).astype(bool)
    out["cache_version"] = pd.to_numeric(out["cache_version"], errors="coerce").fillna(CACHE_SCHEMA_VERSION).astype(int)
    return out[_schema_columns(dataset_name)].sort_values("timestamp_utc").reset_index(drop=True)


def validate_cache_frame(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    zone: str,
    cfg: AppConfig,
    require_full_hourly_coverage: bool = False,
    expected_index: pd.DatetimeIndex | None = None,
) -> None:
    expected_columns = _schema_columns(dataset_name)
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing cache columns: {missing}")
    timestamps = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Cache contains invalid timestamps.")
    if timestamps.duplicated().any():
        raise ValueError("Cache contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Cache timestamps must be sorted in ascending UTC order.")
    versions = set(pd.to_numeric(df["cache_version"], errors="coerce").dropna().astype(int).unique().tolist())
    if versions and versions != {cfg.cache_schema_version}:
        raise ValueError(f"Cache schema version mismatch: {versions} vs expected {cfg.cache_schema_version}")
    if "zone" in df.columns and not df.empty:
        seen_zones = set(df["zone"].dropna().astype(str).unique().tolist())
        if seen_zones and seen_zones != {zone}:
            raise ValueError(f"Cache zone mismatch: {seen_zones} vs expected {zone}")
    if dataset_name == "weather" and not df.empty:
        latitudes = set(pd.to_numeric(df["weather_lat"], errors="coerce").dropna().round(6).tolist())
        longitudes = set(pd.to_numeric(df["weather_lon"], errors="coerce").dropna().round(6).tolist())
        expected_lat = round(float(cfg.openmeteo_lat), 6)
        expected_lon = round(float(cfg.openmeteo_lon), 6)
        if latitudes and latitudes != {expected_lat}:
            raise ValueError(f"Weather cache latitude mismatch: {latitudes} vs expected {expected_lat}")
        if longitudes and longitudes != {expected_lon}:
            raise ValueError(f"Weather cache longitude mismatch: {longitudes} vs expected {expected_lon}")

    required_cols = ENERGY_REQUIRED_COLUMNS if dataset_name == "entsoe" else WEATHER_REQUIRED_COLUMNS
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required dataset columns: {missing_required}")
    if df[required_cols].isna().any().any():
        raise ValueError("Cache contains nulls in required raw fields.")
    if require_full_hourly_coverage and expected_index is not None:
        actual_index = pd.DatetimeIndex(timestamps)
        if len(actual_index) != len(expected_index) or not actual_index.equals(expected_index):
            raise ValueError("Requested window cache output is not a complete hourly UTC index.")


def load_cache_frame(
    cfg: AppConfig,
    *,
    dataset_name: str,
    zone: str,
    rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, str]:
    path = cache_file_path(cfg, dataset_name, zone)
    if rebuild_cache or not cfg.cache_enabled or not path.exists():
        return pd.DataFrame(columns=_schema_columns(dataset_name)), "not_loaded"
    try:
        frame = pd.read_parquet(path)
        frame = _normalize_cache_frame(frame, dataset_name)
        validate_cache_frame(frame, dataset_name=dataset_name, zone=zone, cfg=cfg)
        return frame, "loaded"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Invalid %s cache at %s: %s", dataset_name, path, exc)
        _quarantine_cache_file(path, "invalid")
        return pd.DataFrame(columns=_schema_columns(dataset_name)), f"invalidated:{exc}"


def _quality_rank(series: pd.Series) -> pd.Series:
    mapping = {"synthetic": 0, "partially_synthetic": 1, "real": 2}
    return series.map(mapping).fillna(-1).astype(int)


def merge_rows_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["fetch_timestamp_utc"] = pd.to_datetime(out["fetch_timestamp_utc"], utc=True, errors="coerce")
    out["_quality_rank"] = _quality_rank(out["data_quality"])
    out = out.sort_values(["timestamp_utc", "_quality_rank", "fetch_timestamp_utc"], ascending=[True, False, False])
    out = out.drop_duplicates(subset=["timestamp_utc"], keep="first")
    return out.drop(columns=["_quality_rank"]).sort_values("timestamp_utc").reset_index(drop=True)


def normalize_fetched_frame(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    """Coerce source fetch output onto the hourly cache index."""
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    if dataset_name == "entsoe":
        out["timestamp_utc"] = out["timestamp_utc"].dt.floor("h")
        numeric_cols = [col for col in out.columns if col != "timestamp_utc" and pd.api.types.is_numeric_dtype(out[col])]
        if numeric_cols:
            out = out.groupby("timestamp_utc", as_index=False)[numeric_cols].mean()
        else:
            out = out[["timestamp_utc"]].drop_duplicates(subset=["timestamp_utc"])
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def _coverage_for_fetch(requested_df: pd.DataFrame, required_cols: list[str], *, force_refresh: bool) -> pd.Series:
    if force_refresh:
        return pd.Series(False, index=requested_df.index)
    quality_is_real = requested_df["data_quality"].fillna("").eq("real")
    required_present = requested_df[required_cols].notna().all(axis=1)
    return quality_is_real & required_present


def missing_ranges_from_requested_slice(
    requested_df: pd.DataFrame,
    *,
    required_cols: list[str],
    force_refresh: bool,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if requested_df.empty:
        return []
    coverage = _coverage_for_fetch(requested_df, required_cols, force_refresh=force_refresh)
    missing_index = pd.DatetimeIndex(requested_df.loc[~coverage, "timestamp_utc"])
    if missing_index.empty:
        return []
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = missing_index[0]
    previous = missing_index[0]
    one_hour = pd.Timedelta(hours=1)
    for timestamp in missing_index[1:]:
        if timestamp - previous > one_hour:
            ranges.append((current_start, previous + one_hour))
            current_start = timestamp
        previous = timestamp
    ranges.append((current_start, previous + one_hour))
    return ranges


def annotate_source_rows(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    zone: str,
    cfg: AppConfig,
    source_label: str,
    fetch_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    value_cols = ENERGY_VALUE_COLUMNS if dataset_name == "entsoe" else WEATHER_VALUE_COLUMNS
    required_cols = ENERGY_REQUIRED_COLUMNS if dataset_name == "entsoe" else WEATHER_REQUIRED_COLUMNS
    out = df.copy()
    for column in value_cols:
        if column not in out.columns:
            out[column] = pd.NA
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["fetch_timestamp_utc"] = fetch_timestamp
    out["cache_version"] = cfg.cache_schema_version
    out["zone"] = zone
    if dataset_name == "weather":
        out["weather_lat"] = float(cfg.openmeteo_lat)
        out["weather_lon"] = float(cfg.openmeteo_lon)
    required_present = out[required_cols].notna().all(axis=1)
    out["data_quality"] = "real"
    out.loc[~required_present, "data_quality"] = "partially_synthetic"
    out["is_synthetic"] = out["data_quality"].ne("real")
    out["data_source"] = source_label
    return _normalize_cache_frame(out, dataset_name)


def apply_synthetic_overlay(
    df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    dataset_name: str,
    zone: str,
    cfg: AppConfig,
    synthetic_source_label: str,
    fetch_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    value_cols = ENERGY_VALUE_COLUMNS if dataset_name == "entsoe" else WEATHER_VALUE_COLUMNS
    required_cols = ENERGY_REQUIRED_COLUMNS if dataset_name == "entsoe" else WEATHER_REQUIRED_COLUMNS
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.set_index("timestamp_utc")
    synth = synthetic_df.copy()
    synth["timestamp_utc"] = pd.to_datetime(synth["timestamp_utc"], utc=True, errors="coerce")
    synth = synth.set_index("timestamp_utc")
    synth = synth.reindex(out.index)

    for column in value_cols:
        if column not in out.columns:
            out[column] = pd.NA
        if column not in synth.columns:
            synth[column] = pd.NA

    missing_before = out[required_cols].isna()
    any_missing_before = missing_before.any(axis=1)
    any_real_before = out[required_cols].notna().any(axis=1)
    for column in value_cols:
        out[column] = out[column].where(out[column].notna(), synth[column])

    existing_quality = out["data_quality"].fillna("synthetic")
    out["data_quality"] = existing_quality
    out.loc[~any_real_before, "data_quality"] = "synthetic"
    out.loc[any_real_before & any_missing_before, "data_quality"] = "partially_synthetic"
    out["is_synthetic"] = out["data_quality"].ne("real")
    out["data_source"] = out["data_source"].fillna(synthetic_source_label)
    out.loc[out["data_quality"].eq("synthetic"), "data_source"] = synthetic_source_label
    partial_mask = out["data_quality"].eq("partially_synthetic")
    out.loc[partial_mask, "data_source"] = (
        out.loc[partial_mask, "data_source"]
        .astype(str)
        .replace({"": synthetic_source_label})
        .map(lambda value: value if value.endswith("+synthetic") else f"{value}+synthetic")
    )
    out["fetch_timestamp_utc"] = pd.to_datetime(out["fetch_timestamp_utc"], utc=True, errors="coerce").fillna(fetch_timestamp)
    out["cache_version"] = cfg.cache_schema_version
    out["zone"] = zone
    if dataset_name == "weather":
        out["weather_lat"] = float(cfg.openmeteo_lat)
        out["weather_lon"] = float(cfg.openmeteo_lon)
    return _normalize_cache_frame(out.reset_index(), dataset_name)


def atomic_write_parquet(df: pd.DataFrame, path: Path, *, dataset_name: str, zone: str, cfg: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    df.to_parquet(temp_path, index=False)
    written = pd.read_parquet(temp_path)
    written = _normalize_cache_frame(written, dataset_name)
    validate_cache_frame(written, dataset_name=dataset_name, zone=zone, cfg=cfg)
    if cfg.cache_atomic_write_enabled:
        os.replace(temp_path, path)
    else:
        temp_path.replace(path)


def persist_cache_frame(df: pd.DataFrame, *, dataset_name: str, zone: str, cfg: AppConfig) -> None:
    if not cfg.cache_enabled:
        return
    path = cache_file_path(cfg, dataset_name, zone)
    atomic_write_parquet(df, path, dataset_name=dataset_name, zone=zone, cfg=cfg)


def summarize_provenance(df: pd.DataFrame) -> dict[str, object]:
    if df.empty or "data_quality" not in df.columns:
        return {
            "row_count": 0,
            "real_rows": 0,
            "partially_synthetic_rows": 0,
            "synthetic_rows": 0,
            "real_coverage_ratio": 0.0,
            "partial_synthetic_coverage_ratio": 0.0,
            "synthetic_coverage_ratio": 0.0,
            "research_grade": False,
        }
    counts = df["data_quality"].value_counts(dropna=False)
    total = len(df)
    real_rows = int(counts.get("real", 0))
    partial_rows = int(counts.get("partially_synthetic", 0))
    synthetic_rows = int(counts.get("synthetic", 0))
    return {
        "row_count": total,
        "real_rows": real_rows,
        "partially_synthetic_rows": partial_rows,
        "synthetic_rows": synthetic_rows,
        "real_coverage_ratio": real_rows / total if total else 0.0,
        "partial_synthetic_coverage_ratio": partial_rows / total if total else 0.0,
        "synthetic_coverage_ratio": synthetic_rows / total if total else 0.0,
        "research_grade": partial_rows == 0 and synthetic_rows == 0,
    }


def resolve_dataset_with_cache(
    *,
    cfg: AppConfig,
    dataset_name: str,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetcher: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    synthetic_builder: Callable[[pd.DatetimeIndex], pd.DataFrame],
    source_label: str,
    rebuild_cache: bool = False,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, CacheDiagnostics]:
    value_cols = ENERGY_VALUE_COLUMNS if dataset_name == "entsoe" else WEATHER_VALUE_COLUMNS
    required_cols = ENERGY_REQUIRED_COLUMNS if dataset_name == "entsoe" else WEATHER_REQUIRED_COLUMNS
    path = cache_file_path(cfg, dataset_name, zone)
    expected_index = build_hourly_index(start, end)
    fetch_timestamp = pd.Timestamp.now("UTC")

    existing_cache, load_status = load_cache_frame(cfg, dataset_name=dataset_name, zone=zone, rebuild_cache=rebuild_cache)
    existing_cache = _normalize_cache_frame(existing_cache, dataset_name) if not existing_cache.empty else existing_cache
    requested_cache = existing_cache.loc[
        existing_cache["timestamp_utc"].between(expected_index.min(), expected_index.max(), inclusive="both")
    ].copy() if not existing_cache.empty else pd.DataFrame(columns=_schema_columns(dataset_name))
    cache_rows_used = 0 if force_refresh else len(requested_cache)

    if force_refresh:
        requested_slice = pd.DataFrame({"timestamp_utc": expected_index})
    else:
        requested_slice = pd.DataFrame({"timestamp_utc": expected_index}).merge(
            requested_cache,
            on="timestamp_utc",
            how="left",
        )
    for column in _schema_columns(dataset_name):
        if column not in requested_slice.columns:
            requested_slice[column] = pd.NA
    requested_slice = _normalize_cache_frame(requested_slice, dataset_name)
    missing_ranges = missing_ranges_from_requested_slice(requested_slice, required_cols=required_cols, force_refresh=force_refresh)

    fetched_frames: list[pd.DataFrame] = []
    fetch_failed_range_count = 0
    for range_start, range_end in missing_ranges:
        try:
            fetched = fetcher(range_start, range_end)
            if fetched is None or fetched.empty:
                LOGGER.info(
                    "Fetch returned no %s rows for %s range %s -> %s.",
                    dataset_name,
                    zone,
                    range_start.isoformat(),
                    range_end.isoformat(),
                )
                continue
            fetched = normalize_fetched_frame(fetched, dataset_name=dataset_name)
            fetched = fetched.loc[
                fetched["timestamp_utc"].between(expected_index.min(), expected_index.max(), inclusive="both")
            ].copy()
            if fetched.empty:
                LOGGER.info(
                    "Fetched %s rows for %s range %s -> %s but none overlapped the requested hourly window.",
                    dataset_name,
                    zone,
                    range_start.isoformat(),
                    range_end.isoformat(),
                )
                continue
            annotated = annotate_source_rows(
                fetched,
                dataset_name=dataset_name,
                zone=zone,
                cfg=cfg,
                source_label=source_label,
                fetch_timestamp=fetch_timestamp,
            )
            live_rows = int(annotated["data_quality"].eq("real").sum())
            degraded_rows = int(annotated["data_quality"].ne("real").sum())
            LOGGER.info(
                "Accepted %s fetched %s rows for %s; live_rows=%s degraded_rows=%s range=%s -> %s.",
                len(annotated),
                dataset_name,
                zone,
                live_rows,
                degraded_rows,
                range_start.isoformat(),
                range_end.isoformat(),
            )
            fetched_frames.append(annotated)
        except Exception as exc:  # noqa: BLE001
            fetch_failed_range_count += 1
            LOGGER.warning(
                "Failed to fetch %s data for %s range %s -> %s: %s",
                dataset_name,
                zone,
                range_start.isoformat(),
                range_end.isoformat(),
                exc,
            )

    fetched_df = merge_rows_by_quality(pd.concat(fetched_frames, ignore_index=True)) if fetched_frames else pd.DataFrame(columns=_schema_columns(dataset_name))
    merged_requested = requested_slice if not force_refresh else pd.DataFrame({"timestamp_utc": expected_index})
    if not fetched_df.empty:
        merged_requested = pd.concat([merged_requested, fetched_df], ignore_index=True, sort=False)
    merged_requested = merge_rows_by_quality(merged_requested)
    merged_requested = pd.DataFrame({"timestamp_utc": expected_index}).merge(merged_requested, on="timestamp_utc", how="left")
    merged_requested = _normalize_cache_frame(merged_requested, dataset_name)
    live_rows_accepted = int(merged_requested["data_quality"].fillna("").eq("real").sum())
    synthetic_overlay_rows = int(merged_requested[required_cols].isna().any(axis=1).sum())

    synthetic_df = synthetic_builder(expected_index)
    resolved_requested = apply_synthetic_overlay(
        merged_requested,
        synthetic_df,
        dataset_name=dataset_name,
        zone=zone,
        cfg=cfg,
        synthetic_source_label=f"{source_label}_synthetic",
        fetch_timestamp=fetch_timestamp,
    )
    validate_cache_frame(
        resolved_requested,
        dataset_name=dataset_name,
        zone=zone,
        cfg=cfg,
        require_full_hourly_coverage=True,
        expected_index=expected_index,
    )

    base_cache = existing_cache.loc[
        ~existing_cache["timestamp_utc"].between(expected_index.min(), expected_index.max(), inclusive="both")
    ].copy() if not existing_cache.empty else pd.DataFrame(columns=_schema_columns(dataset_name))
    combined_cache = merge_rows_by_quality(pd.concat([base_cache, resolved_requested], ignore_index=True, sort=False))
    persist_cache_frame(combined_cache, dataset_name=dataset_name, zone=zone, cfg=cfg)

    provenance_summary = summarize_provenance(resolved_requested)
    diagnostics = CacheDiagnostics(
        dataset_name=dataset_name,
        cache_path=str(path),
        cache_status="rebuilt" if rebuild_cache else "refreshed" if force_refresh else load_status,
        cache_used=not force_refresh and load_status == "loaded",
        requested_rows=len(resolved_requested),
        cache_rows_loaded=len(existing_cache),
        cache_rows_used=cache_rows_used,
        cache_rows_written=len(combined_cache),
        fetched_rows=len(fetched_df),
        fetch_attempted_range_count=len(missing_ranges),
        fetch_failed_range_count=fetch_failed_range_count,
        fetched_range_count=len(missing_ranges),
        live_rows_accepted=live_rows_accepted,
        synthetic_overlay_rows=synthetic_overlay_rows,
        synthesized_rows=int(provenance_summary["synthetic_rows"]),
        real_rows=int(provenance_summary["real_rows"]),
        partially_synthetic_rows=int(provenance_summary["partially_synthetic_rows"]),
        synthetic_rows=int(provenance_summary["synthetic_rows"]),
        cache_freshness_utc=str(resolved_requested["fetch_timestamp_utc"].max()) if not resolved_requested.empty else "",
        rebuild_cache=rebuild_cache,
        force_refresh=force_refresh,
        invalidated_reason="" if not load_status.startswith("invalidated:") else load_status.split(":", 1)[1],
    )
    return resolved_requested, diagnostics
