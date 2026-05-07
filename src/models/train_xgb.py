"""XGBoost model training with walk-forward validation."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import AppConfig
from src.models.base import TrainingOutputs
from src.models.diagnostics import build_common_error_analysis, write_model_diagnostics, xgb_feature_importance
from src.models.walk_forward import iter_walk_forward_windows


def _build_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
    )


def _fit_target(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    *,
    seed: int,
    train_window_days: int,
    test_window_days: int,
) -> tuple[XGBRegressor, pd.Series, dict[str, float]]:
    walk_df = pd.DataFrame({"timestamp_utc": timestamps}).reset_index(drop=True)
    windows = iter_walk_forward_windows(
        walk_df,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )
    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    actuals: list[float] = []
    preds: list[float] = []

    for window in windows:
        model = _build_model(seed)
        train_slice = slice(window.train_start, window.train_end)
        test_slice = slice(window.test_start, window.test_end)
        model.fit(X.iloc[train_slice], y.iloc[train_slice])
        fold_pred = model.predict(X.iloc[test_slice])
        predictions.iloc[test_slice] = fold_pred
        actuals.extend(y.iloc[test_slice].tolist())
        preds.extend(fold_pred.tolist())

    final_model = _build_model(seed)
    final_model.fit(X, y)
    metrics = {
        "mae": float(mean_absolute_error(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
    }
    return final_model, predictions, metrics


def train_xgb_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]
    timestamps = df["timestamp_utc"]

    demand_model, demand_pred, demand_metrics = _fit_target(
        X,
        df["demand_kw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    renewable_model, renewable_pred, renewable_metrics = _fit_target(
        X,
        df["renewable_mw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    price_model, price_pred, price_metrics = _fit_target(
        X,
        df["price_eur_mwh"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )

    df["pred_demand_kw"] = demand_pred
    df["pred_renewable_mw"] = renewable_pred
    df["pred_price_eur_mwh"] = price_pred
    df = df.dropna(subset=["pred_demand_kw", "pred_renewable_mw", "pred_price_eur_mwh"]).reset_index(drop=True)

    demand_path = cfg.models_dir / "demand_xgboost.joblib"
    renewable_path = cfg.models_dir / "renewable_xgboost.joblib"
    price_path = cfg.models_dir / "price_xgboost.joblib"
    metrics_path = cfg.models_dir / "metrics_xgboost.json"
    diagnostics_path = cfg.models_dir / "diagnostics_xgboost.json"
    joblib.dump(demand_model, demand_path)
    joblib.dump(renewable_model, renewable_path)
    joblib.dump(price_model, price_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"demand": demand_metrics, "renewable": renewable_metrics, "price": price_metrics}, handle, indent=2)
    diagnostics_payload = {
        "model_key": "xgboost",
        "price_feature_importance": xgb_feature_importance(price_model, feature_cols),
        "demand_feature_importance": xgb_feature_importance(demand_model, feature_cols),
        "renewable_feature_importance": xgb_feature_importance(renewable_model, feature_cols),
        "error_analysis": build_common_error_analysis(df),
    }
    write_model_diagnostics(diagnostics_payload, diagnostics_path)

    return TrainingOutputs(
        demand_model_path=str(demand_path),
        renewable_model_path=str(renewable_path),
        price_model_path=str(price_path),
        metrics_path=str(metrics_path),
        scored_df=df,
        model_key="xgboost",
        diagnostics_path=str(diagnostics_path),
    )


def train_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    """Backward-compatible alias for existing imports."""
    return train_xgb_models(features_df, cfg)
