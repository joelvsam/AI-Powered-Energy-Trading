"""Utilities for comparing multiple forecasting models via isolated backtesting."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.backtesting.strategy_comparison import run_strategy_comparison
from src.models.model_registry import MODEL_REGISTRY, train_with_model


DEFAULT_MODEL_KEYS = ["xgboost", "lstm", "prophet"]
COMPARISON_CSV_NAME = "model_comparison_summary.csv"
COMPARISON_JSON_NAME = "model_comparison_summary.json"
SUMMARY_COLUMNS = [
    "rank",
    "model_key",
    "sharpe_ratio",
    "total_pnl",
    "directional_accuracy",
    "pnl_positive_rate",
    "max_drawdown",
    "drawdown_duration_steps",
    "strategy_name",
    "significant_vs_persistence",
    "price_mae",
    "price_rmse",
    "price_sharpe_ci_lower",
    "price_sharpe_ci_upper",
    "strategy_metrics_path",
    "strategy_significance_path",
    "training_metrics_path",
    "training_diagnostics_path",
    "scored_path",
    "research_output_dir",
]


@dataclass(frozen=True)
class ModelComparisonResult:
    summary_df: pd.DataFrame
    metadata: dict[str, object]
    summary_csv_path: str
    summary_json_path: str


def sort_model_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by trading performance first, then forecast diagnostics."""
    if summary_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    ranked = summary_df.copy()
    for column in ["sharpe_ratio", "total_pnl", "price_mae", "directional_accuracy"]:
        if column not in ranked.columns:
            ranked[column] = pd.NA
    ranked["sharpe_ratio"] = pd.to_numeric(ranked["sharpe_ratio"], errors="coerce")
    ranked["total_pnl"] = pd.to_numeric(ranked["total_pnl"], errors="coerce")
    ranked["price_mae"] = pd.to_numeric(ranked["price_mae"], errors="coerce")
    ranked["directional_accuracy"] = pd.to_numeric(ranked["directional_accuracy"], errors="coerce")
    ranked = ranked.sort_values(
        by=["sharpe_ratio", "total_pnl", "directional_accuracy", "price_mae", "model_key"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    for column in SUMMARY_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = pd.NA
    return ranked[SUMMARY_COLUMNS]


def _load_training_metrics(metrics_path: str | Path) -> dict[str, object]:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


def run_model_comparison(
    features_df: pd.DataFrame,
    cfg: AppConfig,
    *,
    output_dir: Path | str,
    model_keys: list[str] | None = None,
    transaction_cost_bps: float | None = None,
    annualization_factor: int | None = None,
    long_threshold: float = 0.1,
    short_threshold: float = -0.1,
    notional_eur: float | None = None,
    accuracy_horizon_steps: int = 1,
    hold_tolerance_pct: float = 0.002,
) -> ModelComparisonResult:
    """Train and compare multiple models on the same features dataset."""
    selected_models = list(model_keys or DEFAULT_MODEL_KEYS)
    invalid_models = sorted(set(selected_models).difference(MODEL_REGISTRY))
    if invalid_models:
        raise ValueError(f"Unsupported model(s): {invalid_models}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for model_key in selected_models:
        try:
            training_outputs = train_with_model(model_key, features_df, cfg)
            training_metrics = _load_training_metrics(training_outputs.metrics_path)
            model_output_dir = root / model_key
            strategy_result = run_strategy_comparison(
                training_outputs.scored_df,
                cfg,
                output_dir=model_output_dir,
                model_key=model_key,
            )
            upgraded_dir = model_output_dir / "upgraded_strategy"
            for artifact_name in ["backtest_results.csv", "backtest_metrics.json", "backtest_analytics.json"]:
                source_path = upgraded_dir / artifact_name
                if source_path.exists():
                    shutil.copyfile(source_path, model_output_dir / artifact_name)
            upgraded_row = strategy_result.strategy_metrics_df.loc[
                strategy_result.strategy_metrics_df["strategy_name"] == "upgraded_strategy"
            ].iloc[0]
            persistence_row = strategy_result.significance_df.loc[
                (strategy_result.significance_df["strategy_name"] == "upgraded_strategy")
                & (strategy_result.significance_df["baseline_name"] == "persistence")
            ].iloc[0]
            rows.append(
                {
                    "model_key": model_key,
                    "strategy_name": str(upgraded_row.get("strategy_name", "upgraded_strategy")),
                    "directional_accuracy": float(upgraded_row.get("directional_accuracy", 0.0)),
                    "pnl_positive_rate": float(upgraded_row.get("pnl_positive_rate", 0.0)),
                    "total_pnl": float(upgraded_row.get("total_pnl", 0.0)),
                    "sharpe_ratio": float(upgraded_row.get("sharpe_ratio", 0.0)),
                    "max_drawdown": float(upgraded_row.get("max_drawdown", 0.0)),
                    "drawdown_duration_steps": int(upgraded_row.get("drawdown_duration_steps", 0)),
                    "significant_vs_persistence": bool(persistence_row.get("significant_outperformance", False)),
                    "price_mae": float(training_metrics.get("price", {}).get("mae", float("nan"))),
                    "price_rmse": float(training_metrics.get("price", {}).get("rmse", float("nan"))),
                    "price_sharpe_ci_lower": float(upgraded_row.get("sharpe_ci_lower", 0.0)),
                    "price_sharpe_ci_upper": float(upgraded_row.get("sharpe_ci_upper", 0.0)),
                    "strategy_metrics_path": strategy_result.metrics_csv_path,
                    "strategy_significance_path": strategy_result.significance_csv_path,
                    "training_metrics_path": str(training_outputs.metrics_path),
                    "training_diagnostics_path": str(getattr(training_outputs, "diagnostics_path", "")),
                    "scored_path": str(training_outputs.scored_path),
                    "research_output_dir": str(model_output_dir),
                }
            )
        except Exception as exc:
            failures.append({"model_key": model_key, "error": str(exc)})

    summary_df = sort_model_comparison(pd.DataFrame(rows))
    winner_model = str(summary_df.iloc[0]["model_key"]) if not summary_df.empty else None
    metadata = {
        "models_requested": selected_models,
        "winner_metric": "sharpe_ratio",
        "winner_model": winner_model,
        "accuracy_horizon_steps": int(accuracy_horizon_steps),
        "hold_tolerance_pct": float(hold_tolerance_pct),
        "transaction_cost_bps": float(transaction_cost_bps if transaction_cost_bps is not None else cfg.tcost_bps),
        "annualization_factor": int(annualization_factor if annualization_factor is not None else cfg.annualization_factor),
        "notional_eur": float(notional_eur if notional_eur is not None else cfg.backtest_notional_eur),
        "failures": failures,
        "rows": summary_df.to_dict(orient="records"),
    }

    summary_csv_path = root / COMPARISON_CSV_NAME
    summary_json_path = root / COMPARISON_JSON_NAME
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ModelComparisonResult(
        summary_df=summary_df,
        metadata=metadata,
        summary_csv_path=str(summary_csv_path),
        summary_json_path=str(summary_json_path),
    )
