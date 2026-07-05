"""Chronologically honest hyperparameter search for gradient-boosted models.

The search fits candidates on the head of the first walk-forward train window
and scores them on that window's chronological tail, so tuning never touches
data from any out-of-sample test window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


LOGGER = logging.getLogger(__name__)

DEFAULT_XGB_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

XGB_PARAM_GRID: list[dict[str, object]] = [
    dict(DEFAULT_XGB_PARAMS),
    {**DEFAULT_XGB_PARAMS, "n_estimators": 500, "max_depth": 4},
    {**DEFAULT_XGB_PARAMS, "max_depth": 4, "learning_rate": 0.1},
    {**DEFAULT_XGB_PARAMS, "n_estimators": 500, "learning_rate": 0.03},
    {**DEFAULT_XGB_PARAMS, "n_estimators": 200, "max_depth": 8},
    {**DEFAULT_XGB_PARAMS, "n_estimators": 400, "max_depth": 3, "learning_rate": 0.1},
]

MIN_TUNING_FIT_ROWS = 48
MIN_TUNING_VALIDATION_ROWS = 24


def tune_xgb_params(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    first_window_rows: int,
    seed: int,
    validation_fraction: float = 0.2,
    param_grid: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Pick XGBoost params by validating on the tail of the first train window.

    Returns the winning params plus a per-candidate search log. Falls back to
    the defaults when the first window is too short to carve out a meaningful
    validation tail.
    """
    grid = [dict(params) for params in (param_grid or XGB_PARAM_GRID)]
    rows = min(int(first_window_rows), len(X))
    validation_rows = max(int(rows * validation_fraction), MIN_TUNING_VALIDATION_ROWS)
    fit_rows = rows - validation_rows
    if fit_rows < MIN_TUNING_FIT_ROWS:
        LOGGER.info(
            "Skipping hyperparameter search: first train window has %s rows, "
            "below the %s fit + %s validation minimum. Using default params.",
            rows,
            MIN_TUNING_FIT_ROWS,
            MIN_TUNING_VALIDATION_ROWS,
        )
        return dict(DEFAULT_XGB_PARAMS), []

    X_fit, y_fit = X.iloc[:fit_rows], y.iloc[:fit_rows]
    X_val, y_val = X.iloc[fit_rows:rows], y.iloc[fit_rows:rows]

    search_log: list[dict[str, object]] = []
    best_params = dict(DEFAULT_XGB_PARAMS)
    best_mae = np.inf
    for params in grid:
        model = XGBRegressor(random_state=seed, **params)
        model.fit(X_fit, y_fit)
        mae = float(mean_absolute_error(y_val, model.predict(X_val)))
        search_log.append({"params": dict(params), "validation_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_params = dict(params)

    LOGGER.info(
        "Hyperparameter search evaluated %s candidates on %s fit / %s validation rows; best validation MAE=%.4f.",
        len(grid),
        fit_rows,
        validation_rows,
        best_mae,
    )
    return best_params, search_log
