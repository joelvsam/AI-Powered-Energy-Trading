from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from src.config import AppConfig, ensure_directories
from src.data_pipeline.cache import (
    CACHE_SCHEMA_VERSION,
    atomic_write_parquet,
    cache_file_path,
    resolve_dataset_with_cache,
)


def make_workspace_temp_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1] / "artifacts" / "_test_data_cache"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_energy_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    values = list(range(len(index)))
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "price_eur_mwh": [50.0 + x for x in values],
            "day_ahead_price_eur_mwh": [49.5 + x for x in values],
            "intraday_price_eur_mwh": [50.0 + x for x in values],
            "intraday_day_ahead_spread_eur_mwh": [0.5 for _ in values],
            "imbalance_price_eur_mwh": [51.0 + x for x in values],
            "imbalance_price_buy_eur_mwh": [51.5 + x for x in values],
            "imbalance_price_sell_eur_mwh": [50.5 + x for x in values],
            "demand_kw": [10000.0 + x for x in values],
            "renewable_mw": [10.0 + x for x in values],
            "intraday_renewable_forecast_mw": [10.5 + x for x in values],
        }
    )


def build_quarter_hour_energy_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end, freq="15min", tz="UTC", inclusive="left")
    values = list(range(len(index)))
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "price_eur_mwh": [50.0 + x for x in values],
            "day_ahead_price_eur_mwh": [49.5 + x for x in values],
            "intraday_price_eur_mwh": [50.0 + x for x in values],
            "intraday_day_ahead_spread_eur_mwh": [0.5 for _ in values],
            "imbalance_price_eur_mwh": [51.0 + x for x in values],
            "imbalance_price_buy_eur_mwh": [51.5 + x for x in values],
            "imbalance_price_sell_eur_mwh": [50.5 + x for x in values],
            "demand_kw": [10000.0 + x for x in values],
            "renewable_mw": [10.0 + x for x in values],
            "intraday_renewable_forecast_mw": [10.5 + x for x in values],
        }
    )


def build_weather_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h", tz="UTC")
    values = list(range(len(index)))
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "temperature_c": [8.0 + 0.5 * x for x in values],
            "wind_speed_mps": [4.0 + 0.1 * x for x in values],
            "radiation_wm2": [120.0 + 2.0 * x for x in values],
            "humidity_pct": [65.0 + 0.2 * x for x in values],
        }
    )


class DataCacheTests(unittest.TestCase):
    def test_incremental_fetch_uses_cache_and_fetches_only_tail_gap(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

            def fetcher(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
                calls.append((start, end))
                return build_energy_frame(start, end)

            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=pd.Timestamp("2026-01-01 05:00:00+00:00"),
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )
            _, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=pd.Timestamp("2026-01-01 07:00:00+00:00"),
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][0], pd.Timestamp("2026-01-01 06:00:00+00:00"))
            self.assertEqual(calls[1][1], pd.Timestamp("2026-01-01 08:00:00+00:00"))
            self.assertTrue(diagnostics.cache_used)
            self.assertEqual(diagnostics.fetched_range_count, 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_real_rows_replace_synthetic_rows_on_later_refresh(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 03:00:00+00:00")

            def failing_fetcher(_: pd.Timestamp, __: pd.Timestamp) -> pd.DataFrame:
                raise RuntimeError("network down")

            synthetic_df, _ = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=failing_fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )
            self.assertTrue((synthetic_df["data_quality"] == "synthetic").all())

            refreshed_df, _ = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=lambda range_start, range_end: build_energy_frame(range_start, range_end),
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )
            self.assertTrue((refreshed_df["data_quality"] == "real").all())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_openmeteo_coords_for_zone_returns_zone_specific_coordinates(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            self.assertEqual(cfg.openmeteo_coords_for_zone("DE_LU"), (52.52, 13.405))
            self.assertEqual(cfg.openmeteo_coords_for_zone("FR"), (48.8566, 2.3522))
            self.assertEqual(cfg.openmeteo_coords_for_zone("NL"), (52.3676, 4.9041))
            self.assertEqual(cfg.openmeteo_coords_for_zone("unknown"), (cfg.openmeteo_lat, cfg.openmeteo_lon))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_weather_cache_annotations_use_zone_coordinates(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 03:00:00+00:00")

            resolved, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="weather",
                zone="FR",
                start=start,
                end=end,
                fetcher=lambda range_start, range_end: build_weather_frame(range_start, range_end),
                synthetic_builder=lambda idx: build_weather_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="openmeteo",
            )

            self.assertTrue((resolved["weather_lat"].astype(float).round(6) == 48.8566).all())
            self.assertTrue((resolved["weather_lon"].astype(float).round(6) == 2.3522).all())
            self.assertEqual(diagnostics.cache_used, False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_missing_optional_entsoe_context_still_counts_as_real(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 03:00:00+00:00")

            def fetcher(range_start: pd.Timestamp, range_end: pd.Timestamp) -> pd.DataFrame:
                fetched = build_energy_frame(range_start, range_end)
                fetched["intraday_price_eur_mwh"] = pd.NA
                fetched["intraday_day_ahead_spread_eur_mwh"] = pd.NA
                fetched["imbalance_price_eur_mwh"] = pd.NA
                fetched["imbalance_price_buy_eur_mwh"] = pd.NA
                fetched["imbalance_price_sell_eur_mwh"] = pd.NA
                fetched["intraday_renewable_forecast_mw"] = pd.NA
                return fetched

            resolved, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            self.assertTrue((resolved["data_quality"] == "real").all())
            self.assertEqual(diagnostics.real_rows, len(resolved))
            self.assertEqual(diagnostics.partially_synthetic_rows, 0)
            self.assertEqual(diagnostics.synthetic_overlay_rows, 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_quarter_hour_fetch_rolls_up_to_hourly_cache_rows(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 01:00:00+00:00")

            resolved, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=build_quarter_hour_energy_frame,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            self.assertEqual(len(resolved), 2)
            self.assertEqual(resolved["timestamp_utc"].tolist(), [start, end])
            self.assertTrue((resolved["data_quality"] == "real").all())
            self.assertEqual(diagnostics.live_rows_accepted, 2)
            self.assertEqual(diagnostics.synthetic_overlay_rows, 0)
            self.assertAlmostEqual(float(resolved.loc[0, "price_eur_mwh"]), 51.5)
            self.assertAlmostEqual(float(resolved.loc[1, "price_eur_mwh"]), 55.5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_schema_mismatch_invalidates_existing_cache(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            cache_path = cache_file_path(cfg, "entsoe", "DE_LU")
            frame = build_energy_frame(
                pd.Timestamp("2026-01-01 00:00:00+00:00"),
                pd.Timestamp("2026-01-01 02:00:00+00:00"),
            )
            frame["data_source"] = "entsoe"
            frame["is_synthetic"] = False
            frame["data_quality"] = "real"
            frame["fetch_timestamp_utc"] = pd.Timestamp("2026-01-02 00:00:00+00:00")
            frame["cache_version"] = CACHE_SCHEMA_VERSION + 1
            frame["zone"] = "DE_LU"
            frame.to_parquet(cache_path, index=False)

            _, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=pd.Timestamp("2026-01-01 00:00:00+00:00"),
                end=pd.Timestamp("2026-01-01 02:00:00+00:00"),
                fetcher=lambda range_start, range_end: build_energy_frame(range_start, range_end),
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            self.assertIn("mismatch", diagnostics.invalidated_reason)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_atomic_write_keeps_previous_cache_when_validation_fails(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            cache_path = cache_file_path(cfg, "entsoe", "DE_LU")
            original = build_energy_frame(
                pd.Timestamp("2026-01-01 00:00:00+00:00"),
                pd.Timestamp("2026-01-01 02:00:00+00:00"),
            )
            original["data_source"] = "entsoe"
            original["is_synthetic"] = False
            original["data_quality"] = "real"
            original["fetch_timestamp_utc"] = pd.Timestamp("2026-01-02 00:00:00+00:00")
            original["cache_version"] = CACHE_SCHEMA_VERSION
            original["zone"] = "DE_LU"
            atomic_write_parquet(original, cache_path, dataset_name="entsoe", zone="DE_LU", cfg=cfg)

            bad_frame = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-01-01 00:00:00+00:00")]})
            with self.assertRaises(ValueError):
                atomic_write_parquet(bad_frame, cache_path, dataset_name="entsoe", zone="DE_LU", cfg=cfg)

            reloaded = pd.read_parquet(cache_path)
            self.assertEqual(len(reloaded), len(original))
            self.assertEqual(set(reloaded.columns) >= set(original.columns), True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_partial_fetch_failure_synthesizes_only_failed_ranges(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            preloaded = build_energy_frame(
                pd.Timestamp("2026-01-01 03:00:00+00:00"),
                pd.Timestamp("2026-01-01 04:00:00+00:00"),
            )
            preloaded["data_source"] = "entsoe"
            preloaded["is_synthetic"] = False
            preloaded["data_quality"] = "real"
            preloaded["fetch_timestamp_utc"] = pd.Timestamp("2026-01-02 00:00:00+00:00")
            preloaded["cache_version"] = CACHE_SCHEMA_VERSION
            preloaded["zone"] = "DE_LU"
            atomic_write_parquet(preloaded, cache_file_path(cfg, "entsoe", "DE_LU"), dataset_name="entsoe", zone="DE_LU", cfg=cfg)
            calls: list[pd.Timestamp] = []

            def fetcher(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
                calls.append(start)
                if start == pd.Timestamp("2026-01-01 04:00:00+00:00"):
                    raise RuntimeError("tail fetch failed")
                return build_energy_frame(start, end)

            resolved, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=pd.Timestamp("2026-01-01 00:00:00+00:00"),
                end=pd.Timestamp("2026-01-01 05:00:00+00:00"),
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            self.assertEqual(len(calls), 2)
            synthetic_hours = resolved.loc[resolved["data_quality"] == "synthetic", "timestamp_utc"].tolist()
            self.assertEqual(synthetic_hours, [pd.Timestamp("2026-01-01 04:00:00+00:00"), pd.Timestamp("2026-01-01 05:00:00+00:00")])
            self.assertEqual(diagnostics.synthetic_rows, 2)
            self.assertEqual(diagnostics.fetch_failed_range_count, 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_missing_required_live_values_trigger_partial_synthetic_overlay(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 02:00:00+00:00")

            def fetcher(range_start: pd.Timestamp, range_end: pd.Timestamp) -> pd.DataFrame:
                fetched = build_energy_frame(range_start, range_end)
                fetched.loc[fetched["timestamp_utc"] == pd.Timestamp("2026-01-01 01:00:00+00:00"), "demand_kw"] = pd.NA
                return fetched

            resolved, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )

            quality_by_hour = dict(zip(resolved["timestamp_utc"], resolved["data_quality"], strict=False))
            self.assertEqual(quality_by_hour[pd.Timestamp("2026-01-01 00:00:00+00:00")], "real")
            self.assertEqual(quality_by_hour[pd.Timestamp("2026-01-01 01:00:00+00:00")], "partially_synthetic")
            self.assertEqual(quality_by_hour[pd.Timestamp("2026-01-01 02:00:00+00:00")], "real")
            self.assertEqual(diagnostics.partially_synthetic_rows, 1)
            self.assertEqual(diagnostics.synthetic_overlay_rows, 1)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_force_refresh_refetches_requested_window(self) -> None:
        temp_dir = make_workspace_temp_dir()
        try:
            cfg = AppConfig(project_root=temp_dir, cache_dir=temp_dir / "cache")
            ensure_directories(cfg)
            calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

            def fetcher(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
                calls.append((start, end))
                return build_energy_frame(start, end)

            start = pd.Timestamp("2026-01-01 00:00:00+00:00")
            end = pd.Timestamp("2026-01-01 03:00:00+00:00")
            resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
            )
            calls.clear()

            _, diagnostics = resolve_dataset_with_cache(
                cfg=cfg,
                dataset_name="entsoe",
                zone="DE_LU",
                start=start,
                end=end,
                fetcher=fetcher,
                synthetic_builder=lambda idx: build_energy_frame(idx.min(), idx.max() + pd.Timedelta(hours=1)),
                source_label="entsoe",
                force_refresh=True,
            )

            self.assertEqual(calls, [(start, end + pd.Timedelta(hours=1))])
            self.assertEqual(diagnostics.cache_rows_used, 0)
            self.assertTrue(diagnostics.force_refresh)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
