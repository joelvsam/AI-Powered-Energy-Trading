"""Causality regression tests: future inputs must never influence past outputs.

These tests guard against the backward-fill (bfill) lookahead pattern, where
rolling-statistic warmup NaNs were filled from future rows. The property
checked is: perturbing the input at row k leaves all outputs at rows < k
unchanged.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.features.build_features import build_features
from src.trading.signal import SignalConfig, compute_signal_frame
from tests.test_build_features import build_merged_frame
from tests.test_signal import build_scored_frame


SIGNAL_COLUMNS_UNDER_TEST = [
    "rolling_volatility",
    "equilibrium_price_eur_mwh",
    "forecast_signal",
    "mean_reversion_signal",
    "fundamental_signal",
    "spread_signal",
    "target_position",
]

# Rows inside the warmup regions of the rolling stats (min_periods 4/6/12/24),
# where the old bfill pattern leaked future values backward.
PERTURBATION_ROWS = [12, 23, 47, 71]


class SignalCausalityTests(unittest.TestCase):
    def test_future_price_perturbation_does_not_change_past_outputs(self) -> None:
        base = build_scored_frame(rows=240, seed=21)
        config = SignalConfig()
        baseline = compute_signal_frame(base, config)

        for k in PERTURBATION_ROWS:
            with self.subTest(perturbed_row=k):
                perturbed_input = base.copy()
                perturbed_input.loc[perturbed_input.index[k], "price_eur_mwh"] += 50.0
                perturbed = compute_signal_frame(perturbed_input, config)
                pd.testing.assert_frame_equal(
                    baseline.loc[baseline.index[:k], SIGNAL_COLUMNS_UNDER_TEST],
                    perturbed.loc[perturbed.index[:k], SIGNAL_COLUMNS_UNDER_TEST],
                    check_exact=True,
                    obj=f"signal outputs before perturbed row {k}",
                )


class FeatureCausalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        processed_dir = Path(self._tmp.name) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = replace(AppConfig(), data_processed_dir=processed_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_future_price_perturbation_does_not_change_past_features(self) -> None:
        base = build_merged_frame(rows=240, seed=22)
        baseline = build_features(base, self.cfg)

        k = 60
        perturbed_input = base.copy()
        perturbed_input.loc[perturbed_input.index[k], "price_eur_mwh"] += 50.0
        perturbation_time = base.loc[base.index[k], "timestamp_utc"]
        perturbed = build_features(perturbed_input, self.cfg)

        earlier_baseline = baseline[baseline["timestamp_utc"] < perturbation_time].reset_index(drop=True)
        earlier_perturbed = perturbed[perturbed["timestamp_utc"] < perturbation_time].reset_index(drop=True)
        self.assertFalse(earlier_baseline.empty)
        pd.testing.assert_frame_equal(
            earlier_baseline,
            earlier_perturbed,
            check_exact=True,
            obj=f"feature rows before perturbed timestamp {perturbation_time}",
        )

    def test_warmup_rows_are_dropped_not_fabricated(self) -> None:
        base = build_merged_frame(rows=240, seed=23)
        out = build_features(base, self.cfg)
        input_start = base["timestamp_utc"].iloc[0]
        output_start = out["timestamp_utc"].iloc[0]
        # 24h lags/rolls mean the first ~23 hours cannot have honest values.
        self.assertGreaterEqual(output_start, input_start + pd.Timedelta(hours=23))


if __name__ == "__main__":
    unittest.main()
