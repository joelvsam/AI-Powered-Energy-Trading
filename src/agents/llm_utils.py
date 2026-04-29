"""Hugging Face LLM utility functions with robust fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient

from src.config import AppConfig
from src.trading.signal import decision_from_position

LOGGER = logging.getLogger(__name__)


def _safe_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = float(numeric.std())
    if np.isnan(std) or std < 1e-6:
        return pd.Series(0.0, index=series.index)
    mean = float(numeric.mean())
    return (numeric - mean) / std


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -35.0, 35.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def deterministic_fallback(market_state: dict[str, Any], historical_df: pd.DataFrame | None = None) -> dict[str, Any]:
    history = historical_df.copy() if historical_df is not None else pd.DataFrame([market_state])
    if "pred_imbalance_mw" in history.columns:
        history["pred_imbalance_mw"] = pd.to_numeric(history["pred_imbalance_mw"], errors="coerce")
    else:
        history["pred_imbalance_mw"] = pd.to_numeric(history["pred_demand_kw"], errors="coerce") / 1000.0 - pd.to_numeric(
            history["pred_renewable_mw"], errors="coerce"
        )
    if "expected_price_change_eur_mwh" not in history.columns:
        history["expected_price_change_eur_mwh"] = pd.to_numeric(history["pred_price_eur_mwh"], errors="coerce") - pd.to_numeric(
            history["price_eur_mwh"], errors="coerce"
        )

    history["imbalance_z"] = _safe_zscore(history["pred_imbalance_mw"])
    history["expected_price_change_z"] = _safe_zscore(history["expected_price_change_eur_mwh"])
    history["future_return"] = (
        (pd.to_numeric(history["price_eur_mwh"], errors="coerce").shift(-1) - pd.to_numeric(history["price_eur_mwh"], errors="coerce"))
        / pd.to_numeric(history["price_eur_mwh"], errors="coerce").abs().clip(lower=1.0)
    )
    history["future_return"] = history["future_return"].replace([np.inf, -np.inf], np.nan)

    train_df = history.dropna(subset=["imbalance_z", "expected_price_change_z", "future_return"]).copy()
    if len(train_df) < 5:
        train_df = history.dropna(subset=["imbalance_z", "expected_price_change_z"]).copy()
        train_df["future_return"] = 0.0

    X = np.column_stack(
        [
            np.ones(len(train_df)),
            train_df["imbalance_z"].to_numpy(dtype=float),
            train_df["expected_price_change_z"].to_numpy(dtype=float),
        ]
    )
    y = train_df["future_return"].to_numpy(dtype=float)
    weights, *_ = np.linalg.lstsq(X, y, rcond=None)

    latest_row = history.iloc[-1]
    latest_features = np.array(
        [
            1.0,
            float(latest_row.get("imbalance_z", 0.0)),
            float(latest_row.get("expected_price_change_z", 0.0)),
        ]
    )
    score = float(latest_features @ weights)
    prob_up = _sigmoid(score)
    position = float(np.clip(2.0 * (prob_up - 0.5), -1.0, 1.0))
    decision = decision_from_position(pd.Series([position])).iloc[0]
    return {
        "decision": decision,
        "position": position,
        "prob_up": prob_up,
        "reasoning": (
            "Data-driven fallback due to LLM unavailability. "
            f"score={score:.4f}, imbalance_z={latest_features[1]:.4f}, expected_price_change_z={latest_features[2]:.4f}."
        ),
        "risk_assessment": "Use reduced size; fallback mode active.",
        "confidence": round(abs(position), 3),
        "source": "deterministic_fallback",
        "model_weights": {
            "intercept": float(weights[0]),
            "imbalance_z": float(weights[1]),
            "expected_price_change_z": float(weights[2]),
        },
    }


def generate_response(
    prompt: str,
    market_state: dict[str, Any],
    historical_df: pd.DataFrame | None,
    cfg: AppConfig,
) -> dict[str, Any]:
    """Generate LLM response or fallback decision."""
    if not cfg.hf_token:
        LOGGER.warning("HF_TOKEN missing. Using deterministic fallback.")
        return deterministic_fallback(market_state, historical_df)

    try:
        client = InferenceClient(token=cfg.hf_token, timeout=cfg.hf_timeout_s)
        response = client.chat_completion(
            model=cfg.hf_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an energy trading analyst. Respond with a compact JSON object only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""
        try:
            parsed = json.loads(content)
            parsed["source"] = "huggingface"
            return parsed
        except Exception:
            return {
                "decision": "HOLD",
                "position": 0.0,
                "reasoning": content.strip(),
                "risk_assessment": "Model output was non-JSON; defaulted to HOLD.",
                "confidence": 0.55,
                "source": "huggingface_non_json",
            }
    except Exception as exc:
        LOGGER.warning("Hugging Face inference failed: %s", exc)
        return deterministic_fallback(market_state, historical_df)
