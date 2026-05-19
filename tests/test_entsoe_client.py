from __future__ import annotations

import unittest

import pandas as pd

from src.data_sources.entsoe_client import _fetch_in_chunks, _rollup_to_hourly, _skipped_optional_frame


class EntsoeClientTests(unittest.TestCase):
    def test_rollup_to_hourly_preserves_price_column_when_optional_context_is_na(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2026-01-01 00:00:00+00:00", periods=4, freq="15min"),
                "day_ahead_price_eur_mwh": [100.0, 104.0, 108.0, 112.0],
                "intraday_price_eur_mwh": [pd.NA, pd.NA, pd.NA, pd.NA],
                "imbalance_price_eur_mwh": [pd.NA, pd.NA, pd.NA, pd.NA],
                "price_eur_mwh": [100.0, 104.0, 108.0, 112.0],
                "demand_kw": [1000.0, 1004.0, 1008.0, 1012.0],
                "renewable_mw": [10.0, 14.0, 18.0, 22.0],
            }
        )

        rolled = _rollup_to_hourly(frame)

        self.assertEqual(rolled.columns.tolist(), [
            "timestamp_utc",
            "price_eur_mwh",
            "day_ahead_price_eur_mwh",
            "intraday_price_eur_mwh",
            "imbalance_price_eur_mwh",
            "demand_kw",
            "renewable_mw",
        ])
        self.assertEqual(len(rolled), 1)
        self.assertAlmostEqual(float(rolled.loc[0, "price_eur_mwh"]), 106.0)
        self.assertAlmostEqual(float(rolled.loc[0, "day_ahead_price_eur_mwh"]), 106.0)
        self.assertTrue(pd.isna(rolled.loc[0, "intraday_price_eur_mwh"]))

    def test_fetch_in_chunks_splits_failed_large_chunk_and_keeps_successful_subranges(self) -> None:
        attempted: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def fetcher(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
            attempted.append((start, end))
            if start == pd.Timestamp("2026-01-01 00:00:00+00:00") and end == pd.Timestamp("2026-01-03 00:00:00+00:00"):
                raise RuntimeError("service unavailable")
            index = pd.date_range(start=start, end=end, freq="h", inclusive="left", tz="UTC")
            return pd.Series(range(len(index)), index=index)

        frame = _fetch_in_chunks(
            fetcher=fetcher,
            parser=lambda raw: raw.to_frame(name="price_eur_mwh").reset_index().rename(columns={"index": "timestamp_utc"}),
            start=pd.Timestamp("2026-01-01 00:00:00+00:00"),
            end=pd.Timestamp("2026-01-03 00:00:00+00:00"),
            chunk_days=2,
            label="day-ahead prices",
        )

        self.assertEqual(len(frame), 48)
        self.assertEqual(
            attempted,
            [
                (pd.Timestamp("2026-01-01 00:00:00+00:00"), pd.Timestamp("2026-01-03 00:00:00+00:00")),
                (pd.Timestamp("2026-01-01 00:00:00+00:00"), pd.Timestamp("2026-01-02 00:00:00+00:00")),
                (pd.Timestamp("2026-01-02 00:00:00+00:00"), pd.Timestamp("2026-01-03 00:00:00+00:00")),
            ],
        )

    def test_skipped_optional_frame_returns_empty_schema_without_fetch_attempts(self) -> None:
        frame = _skipped_optional_frame(
            ["timestamp_utc", "intraday_price_eur_mwh"],
            zone="DE_LU",
            label="intraday prices",
        )

        self.assertEqual(frame.columns.tolist(), ["timestamp_utc", "intraday_price_eur_mwh"])
        self.assertTrue(frame.empty)


if __name__ == "__main__":
    unittest.main()
