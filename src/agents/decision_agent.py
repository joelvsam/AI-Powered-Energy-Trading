"""Decision agent that converts predictions into actionable guidance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.llm_utils import generate_response
from src.agents.prompts import build_market_prompt
from src.config import AppConfig


def _latest_market_state(backtest_df: pd.DataFrame) -> dict[str, Any]:
    row = backtest_df.sort_values("timestamp_utc").iloc[-1]
    current_price = float(row["price_eur_mwh"])
    predicted_price = float(row["pred_price_eur_mwh"])
    return {
        "timestamp_utc": row["timestamp_utc"].isoformat() if hasattr(row["timestamp_utc"], "isoformat") else str(row["timestamp_utc"]),
        "pred_demand_kw": float(row["pred_demand_kw"]),
        "pred_renewable_mw": float(row["pred_renewable_mw"]),
        "pred_imbalance_mw": float(row["pred_demand_kw"] / 1000.0 - row["pred_renewable_mw"]),
        "price_eur_mwh": current_price,
        "pred_price_eur_mwh": predicted_price,
        "expected_price_change_eur_mwh": predicted_price - current_price,
        "expected_price_change_pct": float((predicted_price - current_price) / current_price) if current_price else 0.0,
        "price_trend": float(row.get("price_trend", 0.0)),
        "recommended_position": str(row.get("decision", "HOLD")),
    }


def run_decision_agent(backtest_df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    market_state = _latest_market_state(backtest_df)
    prompt = build_market_prompt(market_state)
    decision = generate_response(prompt=prompt, market_state=market_state, cfg=cfg)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_state": market_state,
        "decision_report": decision,
    }
    out_path = Path(cfg.simulation_dir) / "decision_report.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
