"""Standalone CLI for isolated offline backtesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtesting import BacktestConfig, run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated backtesting on a scored dataset.")
    parser.add_argument("--input-path", required=True, help="Path to a CSV containing model-scored historical rows.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "backtesting"),
        help="Directory where backtesting artifacts will be written.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Transaction cost in basis points.")
    parser.add_argument("--annualization-factor", type=int, default=24, help="Annualization factor for Sharpe ratio.")
    parser.add_argument("--long-threshold", type=float, default=0.1, help="Position threshold for LONG decisions.")
    parser.add_argument("--short-threshold", type=float, default=-0.1, help="Position threshold for SHORT decisions.")
    parser.add_argument("--notional-eur", type=float, default=10000.0, help="Notional used to scale PnL.")
    parser.add_argument("--accuracy-horizon-steps", type=int, default=1, help="Future step horizon used for decision-accuracy scoring.")
    parser.add_argument("--hold-tolerance-pct", type=float, default=0.002, help="Absolute return band for HOLD directional accuracy.")
    parser.add_argument("--enable-new-signal", choices=["true", "false"], default="true", help="Use the volatility-normalized continuous signal.")
    parser.add_argument("--signal-volatility-window-hours", type=int, default=24, help="Rolling volatility window used by the continuous signal.")
    parser.add_argument("--signal-position-scale-k", type=float, default=2.0, help="Scale factor that maps signal z-scores into positions.")
    parser.add_argument("--long-price-edge-threshold", type=float, default=0.5, help="Minimum predicted EUR/MWh upside required for a LONG recommendation.")
    parser.add_argument("--short-price-edge-threshold", type=float, default=-0.5, help="Maximum predicted EUR/MWh downside required for a SHORT recommendation.")
    parser.add_argument("--disable-volatility-scaling", action="store_true", help="Turn off exposure reduction in high-volatility regimes.")
    parser.add_argument("--disable-execution-delay", action="store_true", help="Execute signals immediately instead of at t+1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path, parse_dates=["timestamp_utc"])
    result = run_backtest(
        df,
        BacktestConfig(
            output_dir=Path(args.output_dir),
            transaction_cost_bps=args.transaction_cost_bps,
            annualization_factor=args.annualization_factor,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            notional_eur=args.notional_eur,
            accuracy_horizon_steps=args.accuracy_horizon_steps,
            hold_tolerance_pct=args.hold_tolerance_pct,
            enable_new_signal=args.enable_new_signal == "true",
            signal_volatility_window_hours=args.signal_volatility_window_hours,
            signal_position_scale_k=args.signal_position_scale_k,
            long_price_edge_threshold=args.long_price_edge_threshold,
            short_price_edge_threshold=args.short_price_edge_threshold,
            enable_volatility_scaling=not args.disable_volatility_scaling,
            enable_execution_delay=not args.disable_execution_delay,
        ),
    )
    print(
        json.dumps(
            {
                "results_path": result.results_path,
                "metrics_path": result.metrics_path,
                "analytics_path": result.analytics_path,
                "metrics": result.metrics,
                "analytics": result.analytics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
