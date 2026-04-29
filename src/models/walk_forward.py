"""Walk-forward utilities shared by forecasting models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """Index slices for one rolling train/test split."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def iter_walk_forward_windows(
    df: pd.DataFrame,
    *,
    train_window_days: int,
    test_window_days: int,
    timestamp_col: str = "timestamp_utc",
) -> list[WalkForwardWindow]:
    """Build fixed-length rolling windows for hourly time-series data."""
    if train_window_days < 1 or test_window_days < 1:
        raise ValueError("walk-forward windows must be positive")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    requested_train_window = train_window_days * 24
    test_window = test_window_days * 24
    max_feasible_train_window = total_rows - test_window if (total_rows := len(ordered)) > test_window else 0
    train_window = min(requested_train_window, max_feasible_train_window)
    windows: list[WalkForwardWindow] = []

    for train_end in range(train_window, total_rows - test_window + 1, test_window):
        windows.append(
            WalkForwardWindow(
                train_start=train_end - train_window,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_window,
            )
        )

    if not windows:
        raise ValueError(
            "not enough rows for walk-forward training; "
            f"need at least {max(test_window * 2, (train_window_days + test_window_days) * 24)} hourly rows"
        )
    return windows
