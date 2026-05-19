"""Run full end-to-end energy trading research workflow."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.anomaly_agent import run_anomaly_review
from src.agents.research_agent import write_research_note
from src.backtesting.comparison import run_model_comparison
from src.backtesting.strategy_comparison import run_strategy_comparison
from src.config import AppConfig, ensure_directories, set_global_seed, setup_logging
from src.data_pipeline.run_pipeline import run_data_pipeline
from src.features.build_features import build_features
from src.models.model_registry import train_with_model
from src.simulation.realtime_loop import run_realtime_simulation
from src.trading.backtest import run_backtest


def _energy_mode_from_provenance(summary: dict[str, Any]) -> str:
    synthetic_rows = int(summary.get("synthetic_rows", 0))
    partial_rows = int(summary.get("partially_synthetic_rows", 0))
    real_rows = int(summary.get("real_rows", 0))
    if synthetic_rows > 0 and real_rows == 0 and partial_rows == 0:
        return "synthetic"
    if synthetic_rows > 0 or partial_rows > 0:
        return "entsoe_partial_synthetic"
    return "entsoe"


def _min_walk_forward_rows(cfg: AppConfig) -> int:
    test_window = int(cfg.walk_forward_test_window_days) * 24
    return int(max(test_window * 2, (int(cfg.walk_forward_train_window_days) + int(cfg.walk_forward_test_window_days)) * 24))


def _adjust_walk_forward_config(cfg: AppConfig, feature_rows: int) -> AppConfig:
    requested_train_days = int(cfg.walk_forward_train_window_days)
    requested_test_days = int(cfg.walk_forward_test_window_days)
    test_hours = requested_test_days * 24

    # We need at least one train block and one test block.
    min_feasible_rows = test_hours * 2
    if feature_rows < min_feasible_rows:
        raise ValueError(
            "Not enough hourly rows for walk-forward training after feature engineering. "
            f"Have {feature_rows} rows but need at least {min_feasible_rows} rows for "
            f"one train block and one {requested_test_days}d test block. "
            "Fix by increasing `--lookback-days` (or `LOOKBACK_DAYS`) or reducing "
            "`WALK_FORWARD_TRAIN_WINDOW_DAYS` / `WALK_FORWARD_TEST_WINDOW_DAYS`."
        )

    requested_min_rows = _min_walk_forward_rows(cfg)
    if feature_rows >= requested_min_rows:
        return cfg

    feasible_train_hours = max(feature_rows - test_hours, test_hours)
    feasible_train_days = max(1, feasible_train_hours // 24)
    adjusted_cfg = replace(cfg, walk_forward_train_window_days=feasible_train_days)
    logging.warning(
        "Feature engineering reduced the usable sample below the requested walk-forward window. "
        "Adjusting train window from %sd to %sd while keeping the %sd test window. "
        "Usable feature rows=%s, requested minimum=%s.",
        requested_train_days,
        feasible_train_days,
        requested_test_days,
        feature_rows,
        requested_min_rows,
    )
    return adjusted_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research-grade energy trading workflow.")
    parser.add_argument("--zone", default=None, help="ENTSO-E bidding zone (default from config).")
    parser.add_argument("--lookback-days", type=int, default=None, help="History window in days.")
    parser.add_argument("--simulation-horizon", type=int, default=24, help="Last N rows for simulation.")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm", "prophet"],
        default="xgboost",
        help="Primary forecasting model to train and inspect.",
    )
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Skip the full cross-model comparison pass.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch the requested raw-data window even when cache rows already exist.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore existing raw-data cache files for the selected window and rebuild them from scratch.",
    )
    return parser.parse_args()


def _persist_strategy_bundle(strategy_result: Any, research_dir: Path, model_key: str) -> dict[str, str]:
    summary_path = research_dir / f"selected_strategy_metrics_{model_key}.csv"
    significance_path = research_dir / f"selected_strategy_significance_{model_key}.csv"
    strategy_result.strategy_metrics_df.to_csv(summary_path, index=False)
    strategy_result.significance_df.to_csv(significance_path, index=False)
    return {
        "selected_strategy_metrics_path": str(summary_path),
        "selected_strategy_significance_path": str(significance_path),
    }


def _load_metrics_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    setup_logging()
    cfg = AppConfig()
    ensure_directories(cfg)
    set_global_seed(cfg.random_seed)
    skip_model_comparison = bool(getattr(args, "skip_model_comparison", False))

    logging.info("1/8 Ingest-Clean-Merge pipeline")
    pipeline_out = run_data_pipeline(
        cfg=cfg,
        zone=args.zone,
        lookback_days=args.lookback_days,
        force_refresh=bool(getattr(args, "force_refresh", False)),
        rebuild_cache=bool(getattr(args, "rebuild_cache", False)),
    )

    logging.info("2/8 Feature engineering")
    features = build_features(pipeline_out.merged_df, cfg)
    cfg = _adjust_walk_forward_config(cfg, len(features))

    logging.info("3/8 Data anomaly review")
    anomaly_review = run_anomaly_review(features, cfg)

    logging.info("4/8 Train primary model (%s)", args.model)
    train_out = train_with_model(args.model, features, cfg)

    logging.info("5/8 Realistic backtest for primary model")
    backtest_out = run_backtest(train_out.scored_df, cfg)

    logging.info("6/8 Strategy comparison vs baselines")
    selected_strategy_result = run_strategy_comparison(
        train_out.scored_df,
        cfg,
        output_dir=cfg.research_dir / args.model,
        model_key=args.model,
    )
    strategy_bundle_paths = _persist_strategy_bundle(selected_strategy_result, cfg.research_dir, args.model)

    logging.info("7/8 Cross-model comparison")
    if skip_model_comparison:
        model_comparison_result = None
        model_summary_df = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "model_key": args.model,
                    "strategy_name": "upgraded_strategy",
                    "sharpe_ratio": float(
                        selected_strategy_result.strategy_metrics_df.loc[
                            selected_strategy_result.strategy_metrics_df["strategy_name"] == "upgraded_strategy",
                            "sharpe_ratio",
                        ].iloc[0]
                    ),
                }
            ]
        )
    else:
        model_comparison_result = run_model_comparison(
            features,
            cfg,
            output_dir=cfg.research_dir / "model_comparison",
        )
        model_summary_df = model_comparison_result.summary_df

    logging.info("8/8 Research note and simulation log")
    sim_path = run_realtime_simulation(
        backtest_out.result_df,
        cfg,
        model_key=args.model,
        horizon=args.simulation_horizon,
    )
    research_note = write_research_note(
        cfg=cfg,
        model_summary_df=model_summary_df,
        strategy_metrics_df=selected_strategy_result.strategy_metrics_df,
        significance_df=selected_strategy_result.significance_df,
        anomaly_report=anomaly_review["report"],
        energy_source=pipeline_out.energy_source,
        provenance_summary=pipeline_out.provenance_summary,
    )

    normalized_provenance = dict(pipeline_out.provenance_summary)
    normalized_energy_source = _energy_mode_from_provenance(normalized_provenance)
    normalized_research_grade = bool(normalized_provenance.get("research_grade", False))

    logging.info("Research workflow complete. Note: %s", research_note["note_path"])
    return {
        "config": {
            "zone": args.zone or cfg.default_zone,
            "lookback_days": args.lookback_days or cfg.lookback_days,
            "model": args.model,
            "skip_model_comparison": skip_model_comparison,
            "force_refresh": bool(getattr(args, "force_refresh", False)),
            "rebuild_cache": bool(getattr(args, "rebuild_cache", False)),
        },
        "runtime_modes": {
            "energy_source": normalized_energy_source,
            "research_source": research_note["summary"].get("source", "unknown"),
            "llm_model": cfg.hf_model,
            "research_grade": normalized_research_grade,
        },
        "data_provenance": normalized_provenance,
        "cache_summary": pipeline_out.cache_summary,
        "features_df": features,
        "scored_df": train_out.scored_df,
        "backtest_df": backtest_out.result_df,
        "backtest_metrics": backtest_out.metrics,
        "strategy_comparison": {
            "summary_df": selected_strategy_result.strategy_metrics_df,
            "significance_df": selected_strategy_result.significance_df,
        },
        "model_comparison": model_summary_df,
        "metrics_path": train_out.metrics_path,
        "diagnostics_path": train_out.diagnostics_path,
        "scored_path": train_out.scored_path,
        "sim_path": sim_path,
        "anomaly_review": anomaly_review,
        "research_summary": research_note,
        "research_artifacts": {
            **strategy_bundle_paths,
            "research_summary_json_path": research_note["json_path"],
            "research_note_path": research_note["note_path"],
            "anomaly_review_path": anomaly_review["path"],
            "model_comparison_summary_path": model_comparison_result.summary_csv_path if model_comparison_result else "",
            "model_comparison_metadata_path": model_comparison_result.summary_json_path if model_comparison_result else "",
        },
    }


def main() -> None:
    args = parse_args()
    run_workflow(args)


if __name__ == "__main__":
    main()
