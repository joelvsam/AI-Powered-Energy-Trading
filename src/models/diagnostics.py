"""Model diagnostics and lightweight interpretability helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _price_regime_error_breakdown(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    df = scored_df.copy()
    df["price_error"] = pd.to_numeric(df["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    median_price = pd.to_numeric(df["price_eur_mwh"], errors="coerce").median()
    df["price_regime"] = np.where(pd.to_numeric(df["price_eur_mwh"], errors="coerce") >= median_price, "high_price", "low_price")
    return df.groupby("price_regime", dropna=False)["price_error"].agg(["mean", "median", "std", "count"]).reset_index().to_dict(orient="records")


def _hourly_error_breakdown(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    df = scored_df.copy()
    df["hour"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.hour
    df["abs_price_error"] = (
        pd.to_numeric(df["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    ).abs()
    return df.groupby("hour", dropna=False)["abs_price_error"].agg(["mean", "max", "count"]).reset_index().to_dict(orient="records")


def _volatility_error_breakdown(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    df = scored_df.copy()
    realized_vol = pd.to_numeric(df["price_eur_mwh"], errors="coerce").diff().rolling(24, min_periods=6).std().ffill()
    threshold = float(realized_vol.quantile(0.7)) if len(realized_vol.dropna()) else 0.0
    df["vol_regime"] = np.where(realized_vol > threshold, "high_vol", "low_vol")
    df["abs_price_error"] = (
        pd.to_numeric(df["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    ).abs()
    return df.groupby("vol_regime", dropna=False)["abs_price_error"].agg(["mean", "median", "count"]).reset_index().to_dict(orient="records")


def write_model_diagnostics(payload: dict[str, Any], output_path: Path) -> str:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)


def build_common_error_analysis(scored_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "price_regime_error": _price_regime_error_breakdown(scored_df),
        "hourly_error": _hourly_error_breakdown(scored_df),
        "volatility_regime_error": _volatility_error_breakdown(scored_df),
    }


def xgb_feature_importance(model: Any, feature_cols: list[str]) -> list[dict[str, Any]]:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return []
    pairs = sorted(zip(feature_cols, importance), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": float(value)} for feature, value in pairs[:20]]


def lstm_feature_ablation(
    model: Any,
    scaler: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    max_features: int = 12,
) -> list[dict[str, Any]]:
    try:
        import torch
    except Exception:
        return []

    sample = X.head(min(len(X), 128)).copy()
    if sample.empty:
        return []
    scaled = scaler.transform(sample)
    baseline_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        baseline_pred = model(baseline_input).view(-1).cpu().numpy()

    scores: list[dict[str, Any]] = []
    feature_subset = feature_cols[:max_features]
    medians = sample.median(numeric_only=True)
    for feature in feature_subset:
        ablated = sample.copy()
        ablated[feature] = float(medians.get(feature, 0.0))
        ablated_scaled = scaler.transform(ablated)
        ablated_input = torch.tensor(ablated_scaled, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            ablated_pred = model(ablated_input).view(-1).cpu().numpy()
        scores.append(
            {
                "feature": feature,
                "ablation_impact": float(np.mean(np.abs(baseline_pred - ablated_pred))),
            }
        )
    return sorted(scores, key=lambda item: item["ablation_impact"], reverse=True)


def prophet_regressor_effects(model: Any, regressor_cols: list[str]) -> list[dict[str, Any]]:
    if not hasattr(model, "params") or "beta" not in model.params:
        return []
    beta = np.asarray(model.params["beta"])
    if beta.ndim > 1:
        beta = beta.mean(axis=0)
    effects = []
    for index, feature in enumerate(regressor_cols[: len(beta)]):
        effects.append({"feature": feature, "effect": float(beta[index])})
    return sorted(effects, key=lambda item: abs(item["effect"]), reverse=True)[:20]
