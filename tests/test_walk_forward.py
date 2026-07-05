from __future__ import annotations

import unittest

import pandas as pd

from src.models.feature_selection import first_train_window_rows
from src.models.walk_forward import iter_walk_forward_windows


def build_hourly_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_utc": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")})


class IterWalkForwardWindowsTests(unittest.TestCase):
    def test_exact_window_layout(self) -> None:
        windows = iter_walk_forward_windows(build_hourly_frame(240), train_window_days=5, test_window_days=1)
        self.assertEqual(len(windows), 5)
        first = windows[0]
        self.assertEqual((first.train_start, first.train_end, first.test_start, first.test_end), (0, 120, 120, 144))
        last = windows[-1]
        self.assertEqual(last.test_end, 240)

    def test_windows_are_chronological_and_non_overlapping(self) -> None:
        windows = iter_walk_forward_windows(build_hourly_frame(480), train_window_days=7, test_window_days=2)
        for window in windows:
            self.assertEqual(window.train_end, window.test_start)
            self.assertLess(window.train_start, window.train_end)
            self.assertLess(window.test_start, window.test_end)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(current.test_start, previous.test_end)

    def test_train_window_shrinks_when_history_is_short(self) -> None:
        # 100 rows cannot fit the requested 240h train window; it should shrink to 76h.
        windows = iter_walk_forward_windows(build_hourly_frame(100), train_window_days=10, test_window_days=1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].train_end - windows[0].train_start, 76)
        self.assertEqual(windows[0].test_end, 100)

    def test_insufficient_rows_raise(self) -> None:
        with self.assertRaises(ValueError):
            iter_walk_forward_windows(build_hourly_frame(20), train_window_days=1, test_window_days=1)

    def test_non_positive_windows_raise(self) -> None:
        frame = build_hourly_frame(100)
        with self.assertRaises(ValueError):
            iter_walk_forward_windows(frame, train_window_days=0, test_window_days=1)
        with self.assertRaises(ValueError):
            iter_walk_forward_windows(frame, train_window_days=1, test_window_days=0)


class FirstTrainWindowRowsTests(unittest.TestCase):
    def test_full_window_when_history_is_long(self) -> None:
        self.assertEqual(first_train_window_rows(10_000, 30, 7), 720)

    def test_shrinks_to_feasible_rows(self) -> None:
        self.assertEqual(first_train_window_rows(200, 30, 7), 32)

    def test_never_below_two_rows(self) -> None:
        self.assertGreaterEqual(first_train_window_rows(1, 1, 1), 2)


if __name__ == "__main__":
    unittest.main()
