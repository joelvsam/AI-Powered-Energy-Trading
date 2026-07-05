from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.trading.signal import (
    SignalConfig,
    compute_signal_frame,
    decision_from_position,
    decision_from_price_edge,
)


EXPECTED_SIGNAL_COLUMNS = [
    "forecast_signal",
    "mean_reversion_signal",
    "fundamental_signal",
    "combined_signal",
    "ensemble_signal",
    "market_regime",
    "vol_regime",
    "target_position_raw",
    "target_position_capped",
    "target_position",
    "target_decision",
    "recommended_decision",
]


def build_scored_frame(rows: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 80.0 + np.cumsum(rng.normal(0.0, 2.0, rows))
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "price_eur_mwh": price,
            "pred_price_eur_mwh": price + rng.normal(0.0, 1.5, rows),
            "pred_demand_kw": 50_000_000 + rng.normal(0.0, 2_000_000, rows),
            "pred_renewable_mw": 20_000 + rng.normal(0.0, 1_500, rows),
            "demand_kw": 50_000_000 + rng.normal(0.0, 2_000_000, rows),
            "renewable_mw": 20_000 + rng.normal(0.0, 1_500, rows),
        }
    )


class ComputeSignalFrameTests(unittest.TestCase):
    def test_output_columns_present(self) -> None:
        out = compute_signal_frame(build_scored_frame(), SignalConfig())
        for column in EXPECTED_SIGNAL_COLUMNS:
            self.assertIn(column, out.columns)

    def test_target_position_bounded_and_finite(self) -> None:
        out = compute_signal_frame(build_scored_frame(), SignalConfig())
        position = out["target_position"]
        self.assertFalse(position.isna().any())
        self.assertTrue(np.isfinite(position).all())
        self.assertLessEqual(float(position.max()), 1.0)
        self.assertGreaterEqual(float(position.min()), -1.0)

    def test_custom_position_limit_respected(self) -> None:
        config = SignalConfig(position_limit=0.5)
        out = compute_signal_frame(build_scored_frame(), config)
        capped = out["target_position_capped"]
        self.assertLessEqual(float(capped.abs().max()), 0.5 + 1e-9)

    def test_regime_labels_are_valid(self) -> None:
        out = compute_signal_frame(build_scored_frame(), SignalConfig())
        self.assertTrue(set(out["vol_regime"].unique()).issubset({"high_vol", "low_vol"}))
        self.assertTrue(set(out["market_regime"].unique()).issubset({"trend", "mean_revert"}))

    def test_constant_price_produces_finite_positions(self) -> None:
        df = build_scored_frame()
        df["price_eur_mwh"] = 50.0
        df["pred_price_eur_mwh"] = 50.0
        out = compute_signal_frame(df, SignalConfig())
        self.assertTrue(np.isfinite(out["target_position"]).all())
        self.assertTrue((out["target_decision"] == "HOLD").all())

    def test_legacy_signal_path(self) -> None:
        config = SignalConfig(enable_new_signal=False)
        out = compute_signal_frame(build_scored_frame(), config)
        self.assertFalse(out["target_position"].isna().any())
        self.assertLessEqual(float(out["target_position"].abs().max()), 1.0)
        self.assertTrue((out["market_regime"] == "trend").all())

    def test_high_vol_rows_scaled_down(self) -> None:
        df = build_scored_frame(rows=400, seed=11)
        out_scaled = compute_signal_frame(df, SignalConfig(enable_volatility_scaling=True))
        out_unscaled = compute_signal_frame(df, SignalConfig(enable_volatility_scaling=False))
        self.assertLessEqual(
            float(out_scaled["target_position_raw"].abs().mean()),
            float(out_unscaled["target_position_raw"].abs().mean()) + 1e-9,
        )


class DecisionMappingTests(unittest.TestCase):
    def test_decision_from_position(self) -> None:
        position = pd.Series([0.4, -0.4, 0.0, 1e-9])
        decisions = decision_from_position(position)
        self.assertEqual(decisions.tolist(), ["LONG", "SHORT", "HOLD", "HOLD"])

    def test_decision_from_price_edge_thresholds(self) -> None:
        edge = pd.Series([1.0, -1.0, 0.2, np.nan])
        decisions = decision_from_price_edge(edge, long_threshold=0.5, short_threshold=-0.5)
        self.assertEqual(decisions.tolist(), ["LONG", "SHORT", "HOLD", "HOLD"])


if __name__ == "__main__":
    unittest.main()
