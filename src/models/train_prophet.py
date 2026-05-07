"""Prophet-based forecasting models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import AppConfig
from src.models.base import TrainingOutputs
from src.models.diagnostics import build_common_error_analysis, prophet_regressor_effects, write_model_diagnostics
from src.models.walk_forward import iter_walk_forward_windows

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


def _prepare_prophet_frame(df: pd.DataFrame, value_col: str | None, regressor_cols: list[str]) -> pd.DataFrame:
    columns = ["timestamp_utc"] + ([] if value_col is None else [value_col]) + regressor_cols
    frame = df[columns].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    renamed = {"timestamp_utc": "ds"}
    if value_col is not None:
        renamed[value_col] = "y"
    return frame.rename(columns=renamed)


def _build_model(regressor_cols: list[str]) -> object:
    if Prophet is None:
        raise RuntimeError("Prophet is required for prophet model. Please install prophet.")
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    for col in regressor_cols:
        model.add_regressor(col)
    return model


def _fit_target(
    df: pd.DataFrame,
    y_col: str,
    regressor_cols: list[str],
    *,
    train_window_days: int,
    test_window_days: int,
) -> tuple[object, pd.Series, dict[str, float]]:
    model_input = _prepare_prophet_frame(df, y_col, regressor_cols).reset_index(drop=True)
    walk_df = model_input.rename(columns={"ds": "timestamp_utc"})
    windows = iter_walk_forward_windows(
        walk_df,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )
    predictions = pd.Series(np.nan, index=df.index, dtype=float)
    actuals: list[float] = []
    preds: list[float] = []

    for window in windows:
        train_slice = slice(window.train_start, window.train_end)
        test_slice = slice(window.test_start, window.test_end)
        model = _build_model(regressor_cols)
        train_df = model_input.iloc[train_slice]
        test_df = model_input.iloc[test_slice]
        model.fit(train_df)
        fold_pred = model.predict(test_df[["ds"] + regressor_cols])["yhat"].values
        predictions.iloc[test_slice] = fold_pred
        actuals.extend(test_df["y"].tolist())
        preds.extend(fold_pred.tolist())

    final_model = _build_model(regressor_cols)
    final_model.fit(model_input)
    metrics = {
        "mae": float(mean_absolute_error(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
    }
    return final_model, predictions, metrics


def train_prophet_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    regressor_cols = [c for c in df.columns if c not in target_cols]

    demand_model, demand_pred, demand_metrics = _fit_target(
        df,
        "demand_kw",
        regressor_cols,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    renewable_model, renewable_pred, renewable_metrics = _fit_target(
        df,
        "renewable_mw",
        regressor_cols,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    price_model, price_pred, price_metrics = _fit_target(
        df,
        "price_eur_mwh",
        regressor_cols,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )

    df["pred_demand_kw"] = demand_pred
    df["pred_renewable_mw"] = renewable_pred
    df["pred_price_eur_mwh"] = price_pred
    df = df.dropna(subset=["pred_demand_kw", "pred_renewable_mw", "pred_price_eur_mwh"]).reset_index(drop=True)

    demand_path = cfg.models_dir / "demand_prophet.joblib"
    renewable_path = cfg.models_dir / "renewable_prophet.joblib"
    price_path = cfg.models_dir / "price_prophet.joblib"
    metrics_path = cfg.models_dir / "metrics_prophet.json"
    diagnostics_path = cfg.models_dir / "diagnostics_prophet.json"
    joblib.dump(demand_model, demand_path)
    joblib.dump(renewable_model, renewable_path)
    joblib.dump(price_model, price_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"demand": demand_metrics, "renewable": renewable_metrics, "price": price_metrics}, handle, indent=2)
    diagnostics_payload = {
        "model_key": "prophet",
        "price_regressor_effects": prophet_regressor_effects(price_model, regressor_cols),
        "demand_regressor_effects": prophet_regressor_effects(demand_model, regressor_cols),
        "renewable_regressor_effects": prophet_regressor_effects(renewable_model, regressor_cols),
        "error_analysis": build_common_error_analysis(df),
    }
    write_model_diagnostics(diagnostics_payload, diagnostics_path)

    return TrainingOutputs(
        demand_model_path=str(demand_path),
        renewable_model_path=str(renewable_path),
        price_model_path=str(price_path),
        metrics_path=str(metrics_path),
        scored_df=df,
        model_key="prophet",
        diagnostics_path=str(diagnostics_path),
    )
