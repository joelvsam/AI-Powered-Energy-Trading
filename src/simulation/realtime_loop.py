"""Realtime-like batch simulation loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import AppConfig
from src.trading.signal import decision_from_position


def run_realtime_simulation(scored_df: pd.DataFrame, cfg: AppConfig, model_key: str, horizon: int = 24) -> str:
    """Append recent predictions to simulation JSONL log."""
    df = scored_df.sort_values("timestamp_utc").tail(horizon).copy()
    df["pred_imbalance_mw"] = df["pred_demand_kw"] / 1000.0 - df["pred_renewable_mw"]
    df["sim_decision"] = decision_from_position(df.get("position", pd.Series(0.0, index=df.index)))

    path = cfg.simulation_dir / "simulation_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            event = {
                "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                "timestamp_utc": row["timestamp_utc"].isoformat(),
                "pred_demand_kw": float(row["pred_demand_kw"]),
                "pred_renewable_mw": float(row["pred_renewable_mw"]),
                "pred_imbalance_mw": float(row["pred_imbalance_mw"]),
                "pred_price_eur_mwh": float(row["pred_price_eur_mwh"]),
                "target_position": float(row.get("target_position", 0.0)),
                "executed_position": float(row.get("position", 0.0)),
                "decision": str(row["sim_decision"]),
                "model_key": model_key,
            }
            handle.write(json.dumps(event) + "\n")
    return str(path)
