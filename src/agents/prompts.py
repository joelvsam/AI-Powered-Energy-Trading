"""Prompt templates for market decisioning."""

from __future__ import annotations

import json
from typing import Any


def build_market_prompt(market_state: dict[str, Any]) -> str:
    payload = json.dumps(market_state, indent=2)
    return (
        "You are an energy trading analyst.\n"
        "Given this market state, return a JSON object with keys:\n"
        "decision (LONG|SHORT|HOLD), position (-1 to 1), reasoning, risk_assessment, confidence (0-1).\n"
        "Use the predicted price, expected price change, supply-demand imbalance, and volatility-aware signal when deciding.\n"
        f"MarketState:\n{payload}\n"
    )


def build_anomaly_prompt(summary_payload: dict[str, Any]) -> str:
    payload = json.dumps(summary_payload, indent=2)
    return (
        "Review this energy-market dataset anomaly summary and return JSON with keys: "
        "issues, cleaning_actions, severity_assessment, reasoning. "
        "Focus on spikes, missing stretches, structural breaks, and suspicious market-context fields.\n"
        f"Payload:\n{payload}\n"
    )


def build_research_prompt(summary_payload: dict[str, Any]) -> str:
    payload = json.dumps(summary_payload, indent=2)
    return (
        "You are writing a trading research note. Return JSON with keys: "
        "edge_summary, strengths, weaknesses, failure_modes, next_experiments, trading_conclusion. "
        "Be concise, evidence-based, and explicit about statistical weakness when present.\n"
        f"Payload:\n{payload}\n"
    )
