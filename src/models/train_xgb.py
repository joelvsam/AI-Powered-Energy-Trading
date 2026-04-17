"""XGBoost model training with walk-forward validation."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.config import AppConfig
from src.models.base import TrainingOutputs


def _fit_target(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[XGBRegressor, dict]:
    splitter = TimeSeriesSplit(n_splits=5)
    maes: list[float] = []
    rmses: list[float] = []

    for train_idx, test_idx in splitter.split(X):
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], pred))
        rmses.append(float(np.sqrt(mean_squared_error(y.iloc[test_idx], pred))))

    final_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
    )
    final_model.fit(X, y)
    metrics = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}
    return final_model, metrics


def train_xgb_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc")
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]

    demand_model, demand_metrics = _fit_target(X, df["demand_kw"], cfg.random_seed)
    renewable_model, renewable_metrics = _fit_target(X, df["renewable_mw"], cfg.random_seed)
    price_model, price_metrics = _fit_target(X, df["price_eur_mwh"], cfg.random_seed)

    df["pred_demand_kw"] = demand_model.predict(X)
    df["pred_renewable_mw"] = renewable_model.predict(X)
    df["pred_price_eur_mwh"] = price_model.predict(X)

    demand_path = cfg.models_dir / "demand_xgboost.joblib"
    renewable_path = cfg.models_dir / "renewable_xgboost.joblib"
    price_path = cfg.models_dir / "price_xgboost.joblib"
    metrics_path = cfg.models_dir / "metrics_xgboost.json"
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
        model_key="xgboost",
    )


def train_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    """Backward-compatible alias for existing imports."""
    return train_xgb_models(features_df, cfg)
