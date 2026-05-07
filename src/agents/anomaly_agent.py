"""Dataset anomaly review with optional LLM assistance and deterministic fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.llm_utils import generate_json_response
from src.agents.prompts import build_anomaly_prompt
from src.config import AppConfig


def _detect_anomalies(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    missing = {col: int(numeric[col].isna().sum()) for col in numeric.columns if int(numeric[col].isna().sum()) > 0}
    spikes: list[dict[str, Any]] = []
    for col in [c for c in numeric.columns if any(key in c for key in ["price", "spread", "demand", "renewable"])]:
        series = pd.to_numeric(numeric[col], errors="coerce")
        zscore = (series - series.mean()) / (series.std() + 1e-6)
        spike_count = int((zscore.abs() > 4.0).sum())
        if spike_count:
            spikes.append({"column": col, "spike_count": spike_count, "max_abs_zscore": float(zscore.abs().max())})
    structural_breaks = []
    for col in [c for c in numeric.columns if any(key in c for key in ["price", "net_load", "spread"])]:
        series = pd.to_numeric(numeric[col], errors="coerce")
        early = series.head(max(len(series) // 3, 1)).mean()
        late = series.tail(max(len(series) // 3, 1)).mean()
        if abs(late - early) > series.std() * 1.5:
            structural_breaks.append({"column": col, "early_mean": float(early), "late_mean": float(late)})
    return {
        "row_count": int(len(df)),
        "missing_fields": missing,
        "spike_summary": spikes,
        "structural_breaks": structural_breaks,
    }


def _fallback_payload(summary: dict[str, Any]) -> dict[str, Any]:
    issues = []
    cleaning_actions = []
    if summary["missing_fields"]:
        issues.append("Missing values remain in numeric columns.")
        cleaning_actions.append("Interpolate short gaps and flag longer missing runs in research outputs.")
    if summary["spike_summary"]:
        issues.append("Detected heavy-tailed spikes in market variables.")
        cleaning_actions.append("Winsorize only for diagnostics and keep raw prices for trading realism.")
    if summary["structural_breaks"]:
        issues.append("Detected potential structural breaks between early and late samples.")
        cleaning_actions.append("Tag break periods as separate regimes in evaluation slices.")
    if not issues:
        issues.append("No major anomalies detected beyond normal market volatility.")
        cleaning_actions.append("Retain current cleaning rules.")
    return {
        "issues": issues,
        "cleaning_actions": cleaning_actions,
        "severity_assessment": "high" if summary["spike_summary"] or summary["structural_breaks"] else "low",
        "reasoning": "Deterministic anomaly review based on missingness, z-score spikes, and mean shifts.",
        "anomaly_summary": summary,
        "source": "deterministic_fallback",
    }


def run_anomaly_review(df: pd.DataFrame, cfg: AppConfig) -> dict[str, Any]:
    summary = _detect_anomalies(df)
    fallback = _fallback_payload(summary)
    payload = generate_json_response(
        system_prompt="You are an energy-market data quality analyst. Respond with valid JSON only.",
        user_prompt=build_anomaly_prompt(summary),
        fallback_payload=fallback,
        cfg=cfg,
    )
    payload["anomaly_summary"] = summary
    output_path = Path(cfg.research_dir) / "anomaly_review.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"report": payload, "path": str(output_path)}
