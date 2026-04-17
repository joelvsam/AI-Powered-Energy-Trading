"""Prompt templates for market decisioning."""

from __future__ import annotations

import json
from typing import Any


def build_market_prompt(market_state: dict[str, Any]) -> str:
    payload = json.dumps(market_state, indent=2)
    return (
        "You are an energy trading analyst.\n"
        "Given this market state, return a JSON object with keys:\n"
        "decision (LONG|SHORT|HOLD), reasoning, risk_assessment, confidence (0-1).\n"
        "Use the predicted price, expected price change, and supply-demand imbalance when deciding.\n"
        f"MarketState:\n{payload}\n"
    )
