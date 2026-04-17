"""Hugging Face LLM utility functions with robust fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

from huggingface_hub import InferenceClient

from src.config import AppConfig

LOGGER = logging.getLogger(__name__)


def deterministic_fallback(market_state: dict[str, Any]) -> dict[str, Any]:
    imbalance = float(market_state.get("pred_imbalance_mw", 0.0))
    expected_price_change = float(market_state.get("expected_price_change_eur_mwh", 0.0))
    score = 0.7 * imbalance + 0.3 * expected_price_change
    if score > 2:
        decision = "LONG"
    elif score < -2:
        decision = "SHORT"
    else:
        decision = "HOLD"
    confidence = min(0.95, max(0.5, abs(score) / 10.0 + 0.5))
    return {
        "decision": decision,
        "reasoning": (
            "Rule-based fallback due to LLM unavailability. "
            f"score={score:.2f}, expected_price_change_eur_mwh={expected_price_change:.2f}."
        ),
        "risk_assessment": "Use reduced size; fallback mode active.",
        "confidence": round(confidence, 3),
        "source": "deterministic_fallback",
    }


def generate_response(prompt: str, market_state: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    """Generate LLM response or fallback decision."""
    if not cfg.hf_token:
        LOGGER.warning("HF_TOKEN missing. Using deterministic fallback.")
        return deterministic_fallback(market_state)

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
        # Best-effort JSON parse; if non-JSON, wrap as textual reasoning.
        try:
            parsed = json.loads(content)
            parsed["source"] = "huggingface"
            return parsed
        except Exception:
            return {
                "decision": "HOLD",
                "reasoning": content.strip(),
                "risk_assessment": "Model output was non-JSON; defaulted to HOLD.",
                "confidence": 0.55,
                "source": "huggingface_non_json",
            }
    except Exception as exc:
        LOGGER.warning("Hugging Face inference failed: %s", exc)
        return deterministic_fallback(market_state)
