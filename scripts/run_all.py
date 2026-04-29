"""Run full end-to-end energy trading pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from src.agents.decision_agent import run_decision_agent
from src.backtesting.engine import BacktestConfig, generate_backtest_outputs
from src.config import AppConfig, ensure_directories, set_global_seed, setup_logging
from src.data_pipeline.run_pipeline import run_data_pipeline
from src.features.build_features import build_features
from src.models.model_registry import train_with_model
from src.simulation.realtime_loop import run_realtime_simulation
from src.trading.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-style energy trading workflow.")
    parser.add_argument("--zone", default=None, help="ENTSO-E bidding zone (default from config).")
    parser.add_argument("--lookback-days", type=int, default=None, help="History window in days.")
    parser.add_argument("--simulation-horizon", type=int, default=24, help="Last N rows for simulation.")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm", "prophet"],
        default="xgboost",
        help="Forecasting model to train and use.",
    )
    return parser.parse_args()


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    setup_logging()
    cfg = AppConfig()
    ensure_directories(cfg)
    set_global_seed(cfg.random_seed)

    logging.info("1/8 Ingest-Clean-Merge pipeline")
    pipeline_out = run_data_pipeline(cfg=cfg, zone=args.zone, lookback_days=args.lookback_days)

    logging.info("2/8 Feature engineering")
    features = build_features(pipeline_out.merged_df, cfg)

    logging.info("3/8 Model training (%s)", args.model)
    train_out = train_with_model(args.model, features, cfg)

    logging.info("4/8 Backtesting")
    backtest_out = run_backtest(train_out.scored_df, cfg)
    _, legacy_metrics, _ = generate_backtest_outputs(
        train_out.scored_df,
        BacktestConfig(
            output_dir=cfg.simulation_dir,
            transaction_cost_bps=cfg.tcost_bps,
            annualization_factor=cfg.annualization_factor,
            long_threshold=0.1,
            short_threshold=-0.1,
            notional_eur=cfg.backtest_notional_eur,
            accuracy_horizon_steps=1,
            hold_tolerance_pct=0.002,
            enable_new_signal=False,
            signal_volatility_window_hours=cfg.signal_volatility_window_hours,
            signal_position_scale_k=cfg.signal_position_scale_k,
            enable_volatility_scaling=False,
            enable_execution_delay=False,
        ),
    )

    logging.info("5/8 Realtime simulation")
    sim_path = run_realtime_simulation(
        train_out.scored_df, cfg, model_key=args.model, horizon=args.simulation_horizon
    )

    logging.info("6/8 Decision agent")
    report = run_decision_agent(backtest_out.result_df, cfg)

    logging.info("Pipeline complete. Simulation log: %s", sim_path)
    logging.info("Decision: %s", report["decision_report"].get("decision"))
    return {
        "config": {
            "zone": args.zone or cfg.default_zone,
            "lookback_days": args.lookback_days or cfg.lookback_days,
            "model": args.model,
        },
        "runtime_modes": {
            "energy_source": pipeline_out.energy_source,
            "decision_source": report["decision_report"].get("source", "unknown"),
            "llm_model": cfg.hf_model,
        },
        "features_df": features,
        "scored_df": train_out.scored_df,
        "backtest_df": backtest_out.result_df,
        "strategy_comparison": {
            "old": legacy_metrics,
            "new": backtest_out.metrics,
        },
        "metrics_path": train_out.metrics_path,
        "scored_path": train_out.scored_path,
        "sim_path": sim_path,
        "decision_report": report,
    }


def main() -> None:
    args = parse_args()
    run_workflow(args)


if __name__ == "__main__":
    main()
