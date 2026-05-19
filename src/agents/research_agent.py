"""Trading research summary agent with optional LLM assistance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.llm_utils import generate_json_response
from src.agents.prompts import build_research_prompt
from src.config import AppConfig


def _fallback_research_payload(summary_payload: dict[str, Any]) -> dict[str, Any]:
    significance_rows = summary_payload.get("significance", [])
    provenance_summary = summary_payload.get("provenance_summary", {})
    not_significant = [
        row["baseline_name"]
        for row in significance_rows
        if row.get("strategy_name") == "upgraded_strategy" and not row.get("significant_outperformance", False)
    ]
    edge_summary = "Edge appears strongest when imbalance and spread regimes align with the combined signal."
    if not significance_rows:
        edge_summary = "Insufficient significance evidence; treat results as exploratory."
    weaknesses = [
        "Performance may degrade when execution costs dominate forecast edge.",
        "Results with synthetic or partially synthetic market rows require explicit caution in research interpretation.",
    ]
    if provenance_summary:
        weaknesses.append(
            "Synthetic coverage: "
            f"{provenance_summary.get('synthetic_coverage_ratio', 0.0):.1%}; "
            f"partial synthetic coverage: {provenance_summary.get('partial_synthetic_coverage_ratio', 0.0):.1%}."
        )
    if not_significant:
        weaknesses.append(f"No significant outperformance versus: {', '.join(not_significant)}.")
    return {
        "edge_summary": edge_summary,
        "strengths": [
            "Evaluation is walk-forward and baseline-relative.",
            "Backtest includes spread, slippage, delay, and position/liquidity constraints.",
        ],
        "weaknesses": weaknesses,
        "failure_modes": [
            "High-volatility windows with wide spreads.",
            "Regime shifts where day-ahead and intraday relationships break down.",
        ],
        "next_experiments": [
            "Stress-test across more zones and longer windows.",
            "Refine imbalance conditioning using explicit imbalance buy/sell asymmetry.",
        ],
        "trading_conclusion": "Use only as a research candidate until significance is stable across baselines and zones.",
        "source": "deterministic_fallback",
    }


def write_research_note(
    *,
    cfg: AppConfig,
    model_summary_df: pd.DataFrame,
    strategy_metrics_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    anomaly_report: dict[str, Any],
    energy_source: str,
    provenance_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "energy_source": energy_source,
        "provenance_summary": provenance_summary,
        "model_summary": model_summary_df.to_dict(orient="records"),
        "strategy_metrics": strategy_metrics_df.to_dict(orient="records"),
        "significance": significance_df.to_dict(orient="records"),
        "anomaly_review": anomaly_report,
    }
    fallback = _fallback_research_payload(payload)
    research_summary = generate_json_response(
        system_prompt="You are a power trading research analyst. Respond with valid JSON only.",
        user_prompt=build_research_prompt(payload),
        fallback_payload=fallback,
        cfg=cfg,
    )
    research_summary["energy_source_research_grade"] = bool(provenance_summary.get("research_grade", False))
    research_summary["provenance_summary"] = provenance_summary

    note_lines = [
        "# Trading Research Note",
        "",
        f"Research grade data: {'yes' if research_summary['energy_source_research_grade'] else 'no'}",
        f"Synthetic coverage: {float(provenance_summary.get('synthetic_coverage_ratio', 0.0)):.1%}",
        f"Partial synthetic coverage: {float(provenance_summary.get('partial_synthetic_coverage_ratio', 0.0)):.1%}",
        "",
        "## Strategy Description",
        "The strategy combines forecast, mean-reversion, and fundamental imbalance signals with regime-aware sizing and realistic execution costs.",
        "",
        "## Why It Should Work",
        research_summary.get("edge_summary", ""),
        "",
        "## Strengths",
    ]
    note_lines.extend([f"- {item}" for item in research_summary.get("strengths", [])])
    note_lines.extend(["", "## Weaknesses and Failure Modes"])
    note_lines.extend([f"- {item}" for item in research_summary.get("weaknesses", [])])
    note_lines.extend([f"- {item}" for item in research_summary.get("failure_modes", [])])
    note_lines.extend(["", "## Further Improvement"])
    note_lines.extend([f"- {item}" for item in research_summary.get("next_experiments", [])])
    note_lines.extend(["", "## Trading Conclusion", research_summary.get("trading_conclusion", "")])

    json_path = Path(cfg.research_dir) / "research_summary.json"
    md_path = Path(cfg.research_dir) / "research_note.md"
    json_path.write_text(json.dumps(research_summary, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(note_lines), encoding="utf-8")
    return {"summary": research_summary, "json_path": str(json_path), "note_path": str(md_path)}
