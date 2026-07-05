from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.feature_selection import select_model_features
from src.models.tuning import DEFAULT_XGB_PARAMS, tune_xgb_params

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class SelectModelFeaturesTests(unittest.TestCase):
    def build_frame(self, rows: int = 200, seed: int = 9) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, rows)
        d = rng.normal(0.0, 1.0, rows)
        return pd.DataFrame(
            {
                "a": a,
                "b": a * 2.0,  # perfectly correlated with a
                "c": 5.0,  # constant
                "d": d,
            }
        )

    def test_drops_constant_and_correlated_features(self) -> None:
        frame = self.build_frame()
        result = select_model_features(frame, ["a", "b", "c", "d"], fit_rows=len(frame))
        self.assertEqual(result.kept, ["a", "d"])
        self.assertEqual(result.dropped_near_constant, ["c"])
        self.assertEqual(len(result.dropped_correlated), 1)
        self.assertEqual(result.dropped_correlated[0]["feature"], "b")
        self.assertEqual(result.dropped_correlated[0]["correlated_with"], "a")

    def test_earlier_feature_wins_ties(self) -> None:
        frame = self.build_frame()
        result = select_model_features(frame, ["b", "a", "d"], fit_rows=len(frame))
        self.assertEqual(result.kept, ["b", "d"])

    def test_only_fit_rows_are_used(self) -> None:
        rng = np.random.default_rng(10)
        rows = 200
        a = rng.normal(0.0, 1.0, rows)
        e = rng.normal(0.0, 1.0, rows)
        e[100:] = a[100:]  # correlated only in the later (future) rows
        frame = pd.DataFrame({"a": a, "e": e})
        result = select_model_features(frame, ["a", "e"], fit_rows=100)
        self.assertEqual(result.kept, ["a", "e"])

    def test_all_constant_falls_back_to_original_features(self) -> None:
        frame = pd.DataFrame({"x": [1.0] * 50, "y": [2.0] * 50})
        result = select_model_features(frame, ["x", "y"], fit_rows=50)
        self.assertEqual(result.kept, ["x", "y"])


class TuneXgbParamsTests(unittest.TestCase):
    SMALL_GRID = [
        {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9},
    ]

    def build_data(self, rows: int = 300, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(
            {
                "f1": rng.normal(0.0, 1.0, rows),
                "f2": rng.normal(0.0, 1.0, rows),
                "f3": rng.normal(0.0, 1.0, rows),
            }
        )
        y = pd.Series(3.0 * X["f1"] - 2.0 * X["f2"] + rng.normal(0.0, 0.1, rows))
        return X, y

    def test_returns_candidate_from_grid_with_search_log(self) -> None:
        X, y = self.build_data()
        best, log = tune_xgb_params(X, y, first_window_rows=200, seed=0, param_grid=self.SMALL_GRID)
        self.assertIn(best, self.SMALL_GRID)
        self.assertEqual(len(log), len(self.SMALL_GRID))
        for entry in log:
            self.assertIn("params", entry)
            self.assertTrue(np.isfinite(entry["validation_mae"]))

    def test_short_window_falls_back_to_defaults(self) -> None:
        X, y = self.build_data(rows=60)
        best, log = tune_xgb_params(X, y, first_window_rows=30, seed=0, param_grid=self.SMALL_GRID)
        self.assertEqual(best, DEFAULT_XGB_PARAMS)
        self.assertEqual(log, [])

    def test_deterministic_for_same_seed(self) -> None:
        X, y = self.build_data()
        first, _ = tune_xgb_params(X, y, first_window_rows=200, seed=7, param_grid=self.SMALL_GRID)
        second, _ = tune_xgb_params(X, y, first_window_rows=200, seed=7, param_grid=self.SMALL_GRID)
        self.assertEqual(first, second)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class LstmTrainingLoopTests(unittest.TestCase):
    def test_training_history_contract(self) -> None:
        from src.models.train_lstm import _build_model, _to_tensors, _train_model

        rng = np.random.default_rng(12)
        rows, input_dim = 80, 4
        X = rng.normal(0.0, 1.0, (rows, input_dim))
        y = X @ np.array([1.0, -0.5, 0.3, 0.0]) + rng.normal(0.0, 0.05, rows)
        x_tensor, y_tensor = _to_tensors(X, y)
        model = _build_model(input_dim=input_dim, seed=0)

        history = _train_model(model, x_tensor, y_tensor, max_epochs=12, patience=2)
        self.assertGreaterEqual(history["epochs_run"], 1)
        self.assertLessEqual(history["epochs_run"], 12)
        self.assertIsInstance(history["early_stopped"], bool)
        self.assertGreater(history["validation_rows"], 0)
        self.assertIsNotNone(history["best_val_loss"])

    def test_tiny_sample_falls_back_to_fixed_epochs(self) -> None:
        from src.models.train_lstm import _build_model, _to_tensors, _train_model

        rng = np.random.default_rng(13)
        rows, input_dim = 12, 3
        X = rng.normal(0.0, 1.0, (rows, input_dim))
        y = rng.normal(0.0, 1.0, rows)
        x_tensor, y_tensor = _to_tensors(X, y)
        model = _build_model(input_dim=input_dim, seed=0)

        history = _train_model(model, x_tensor, y_tensor, max_epochs=50, patience=3)
        self.assertEqual(history["validation_rows"], 0)
        self.assertLessEqual(history["epochs_run"], 10)


if __name__ == "__main__":
    unittest.main()
