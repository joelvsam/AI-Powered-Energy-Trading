"""Prophet-based forecasting models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import AppConfig
from src.models.base import TrainingOutputs

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None


def _fit_target(df: pd.DataFrame, y_col: str, regressor_cols: list[str]) -> tuple[object, dict]:
    if Prophet is None:
        raise RuntimeError("Prophet is required for prophet model. Please install prophet.")
    splitter = TimeSeriesSplit(n_splits=5)
    maes: list[float] = []
    rmses: list[float] = []

    model_input = df[["timestamp_utc", y_col] + regressor_cols].rename(columns={"timestamp_utc": "ds", y_col: "y"})

    for train_idx, test_idx in splitter.split(model_input):
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        for col in regressor_cols:
            model.add_regressor(col)
        train_df = model_input.iloc[train_idx]
        test_df = model_input.iloc[test_idx]
        model.fit(train_df)
        pred = model.predict(test_df[["ds"] + regressor_cols])["yhat"].values
        maes.append(float(mean_absolute_error(test_df["y"], pred)))
        rmses.append(float(np.sqrt(mean_squared_error(test_df["y"], pred))))

    final_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    for col in regressor_cols:
        final_model.add_regressor(col)
    final_model.fit(model_input)
    metrics = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}
    return final_model, metrics


def _predict(model: object, df: pd.DataFrame, regressor_cols: list[str]) -> np.ndarray:
    pred_df = df[["timestamp_utc"] + regressor_cols].rename(columns={"timestamp_utc": "ds"})
    return model.predict(pred_df)["yhat"].values


def train_prophet_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc")
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    regressor_cols = [c for c in df.columns if c not in target_cols]

    demand_model, demand_metrics = _fit_target(df, "demand_kw", regressor_cols)
    renewable_model, renewable_metrics = _fit_target(df, "renewable_mw", regressor_cols)
    price_model, price_metrics = _fit_target(df, "price_eur_mwh", regressor_cols)

    df["pred_demand_kw"] = _predict(demand_model, df, regressor_cols)
    df["pred_renewable_mw"] = _predict(renewable_model, df, regressor_cols)
    df["pred_price_eur_mwh"] = _predict(price_model, df, regressor_cols)

    demand_path = cfg.models_dir / "demand_prophet.joblib"
    renewable_path = cfg.models_dir / "renewable_prophet.joblib"
    price_path = cfg.models_dir / "price_prophet.joblib"
    metrics_path = cfg.models_dir / "metrics_prophet.json"
    joblib.dump(demand_model, demand_path)
    joblib.dump(renewable_model, renewable_path)
    joblib.dump(price_model, price_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"demand": demand_metrics, "renewable": renewable_metrics, "price": price_metrics}, handle, indent=2)

    return TrainingOutputs(
        demand_model_path=str(demand_path),
        renewable_model_path=str(renewable_path),
        price_model_path=str(price_path),
        metrics_path=str(metrics_path),
        scored_df=df,
        model_key="prophet",
    )
