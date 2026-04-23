"""Utilities for comparing multiple forecasting models via isolated backtesting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.backtesting.engine import BacktestConfig, run_backtest
from src.config import AppConfig
from src.models.model_registry import MODEL_REGISTRY, train_with_model


DEFAULT_MODEL_KEYS = ["xgboost", "lstm", "prophet"]
COMPARISON_CSV_NAME = "model_comparison_summary.csv"
COMPARISON_JSON_NAME = "model_comparison_summary.json"
SUMMARY_COLUMNS = [
    "rank",
    "model_key",
    "directional_accuracy",
    "correct_count",
    "incorrect_count",
    "pnl_positive_rate",
    "total_pnl",
    "sharpe_ratio",
    "max_drawdown",
    "hit_rate",
    "price_mae",
    "price_rmse",
    "results_path",
    "metrics_path",
    "analytics_path",
    "training_metrics_path",
    "scored_path",
    "backtest_output_dir",
]


@dataclass(frozen=True)
class ModelComparisonResult:
    summary_df: pd.DataFrame
    metadata: dict[str, object]
    summary_csv_path: str
    summary_json_path: str


def sort_model_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank models by directional accuracy, then forecast error and pnl quality."""
    if summary_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    ranked = summary_df.copy()
    ranked["directional_accuracy"] = pd.to_numeric(ranked["directional_accuracy"], errors="coerce")
    ranked["price_mae"] = pd.to_numeric(ranked["price_mae"], errors="coerce")
    ranked["pnl_positive_rate"] = pd.to_numeric(ranked["pnl_positive_rate"], errors="coerce")
    ranked = ranked.sort_values(
        by=["directional_accuracy", "price_mae", "pnl_positive_rate", "model_key"],
        ascending=[False, True, False, True],
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
            backtest_result = run_backtest(
                training_outputs.scored_df,
                BacktestConfig(
                    output_dir=model_output_dir,
                    transaction_cost_bps=transaction_cost_bps if transaction_cost_bps is not None else cfg.tcost_bps,
                    annualization_factor=annualization_factor if annualization_factor is not None else cfg.annualization_factor,
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    notional_eur=notional_eur if notional_eur is not None else cfg.backtest_notional_eur,
                    accuracy_horizon_steps=accuracy_horizon_steps,
                    hold_tolerance_pct=hold_tolerance_pct,
                ),
            )
            accuracy_summary = backtest_result.analytics.get("accuracy_summary", {})
            rows.append(
                {
                    "model_key": model_key,
                    "directional_accuracy": float(backtest_result.metrics.get("directional_accuracy", 0.0)),
                    "correct_count": int(accuracy_summary.get("correct_count", 0)),
                    "incorrect_count": int(accuracy_summary.get("incorrect_count", 0)),
                    "pnl_positive_rate": float(backtest_result.metrics.get("pnl_positive_rate", 0.0)),
                    "total_pnl": float(backtest_result.metrics.get("total_pnl", 0.0)),
                    "sharpe_ratio": float(backtest_result.metrics.get("sharpe_ratio", 0.0)),
                    "max_drawdown": float(backtest_result.metrics.get("max_drawdown", 0.0)),
                    "hit_rate": float(backtest_result.metrics.get("hit_rate", 0.0)),
                    "price_mae": float(training_metrics.get("price", {}).get("mae", float("nan"))),
                    "price_rmse": float(training_metrics.get("price", {}).get("rmse", float("nan"))),
                    "results_path": backtest_result.results_path,
                    "metrics_path": backtest_result.metrics_path,
                    "analytics_path": backtest_result.analytics_path,
                    "training_metrics_path": str(training_outputs.metrics_path),
                    "scored_path": str(training_outputs.scored_path),
                    "backtest_output_dir": str(model_output_dir),
                }
            )
        except Exception as exc:
            failures.append({"model_key": model_key, "error": str(exc)})

    summary_df = sort_model_comparison(pd.DataFrame(rows))
    winner_model = str(summary_df.iloc[0]["model_key"]) if not summary_df.empty else None
    metadata = {
        "models_requested": selected_models,
        "winner_metric": "directional_accuracy",
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
