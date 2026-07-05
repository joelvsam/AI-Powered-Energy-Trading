from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features.build_features import build_features


def build_merged_frame(rows: int = 240, seed: int = 3, include_optional: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 70.0 + np.cumsum(rng.normal(0.0, 1.5, rows))
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "price_eur_mwh": price,
            "demand_kw": 45_000_000 + rng.normal(0.0, 2_000_000, rows),
            "renewable_mw": 18_000 + rng.normal(0.0, 1_200, rows),
            "temperature_c": rng.normal(8.0, 4.0, rows),
            "wind_speed_mps": np.abs(rng.normal(6.0, 2.0, rows)),
            "radiation_wm2": np.clip(rng.normal(150.0, 80.0, rows), 0.0, None),
            "humidity_pct": rng.uniform(35.0, 95.0, rows),
        }
    )
    if include_optional:
        frame["day_ahead_price_eur_mwh"] = price + rng.normal(0.0, 1.0, rows)
        frame["intraday_price_eur_mwh"] = price + rng.normal(0.0, 2.0, rows)
        frame["imbalance_price_eur_mwh"] = price + rng.normal(0.0, 5.0, rows)
        frame["intraday_renewable_forecast_mw"] = frame["renewable_mw"] + rng.normal(0.0, 400.0, rows)
    return frame


class BuildFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        processed_dir = Path(self._tmp.name) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = replace(AppConfig(), data_processed_dir=processed_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_features_built_with_full_context(self) -> None:
        out = build_features(build_merged_frame(), self.cfg)
        self.assertFalse(out.empty)
        for flag in [
            "day_ahead_price_eur_mwh_available",
            "intraday_price_eur_mwh_available",
            "imbalance_price_eur_mwh_available",
            "intraday_renewable_forecast_mw_available",
        ]:
            self.assertIn(flag, out.columns)
            self.assertTrue((out[flag] == 1.0).all())
        numeric = out.select_dtypes(include=[np.number])
        self.assertFalse(np.isinf(numeric.to_numpy(dtype=float)).any())
        self.assertTrue((self.cfg.data_processed_dir / "features.csv").exists())

    def test_optional_context_fallback_flags(self) -> None:
        out = build_features(build_merged_frame(include_optional=False), self.cfg)
        for flag in [
            "day_ahead_price_eur_mwh_available",
            "intraday_price_eur_mwh_available",
            "imbalance_price_eur_mwh_available",
            "intraday_renewable_forecast_mw_available",
        ]:
            self.assertTrue((out[flag] == 0.0).all())
        # With no optional context, prices fall back to the realized price proxy.
        np.testing.assert_allclose(
            out["day_ahead_price_eur_mwh"].to_numpy(dtype=float),
            out["price_eur_mwh"].to_numpy(dtype=float),
        )

    def test_missing_required_column_raises(self) -> None:
        frame = build_merged_frame().drop(columns=["temperature_c"])
        with self.assertRaises(ValueError):
            build_features(frame, self.cfg)

    def test_excessive_missingness_raises(self) -> None:
        frame = build_merged_frame()
        frame.loc[frame.index[:40], "price_eur_mwh"] = np.nan
        with self.assertRaises(ValueError):
            build_features(frame, self.cfg)

    def test_empty_frame_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_features(pd.DataFrame(), self.cfg)


if __name__ == "__main__":
    unittest.main()
