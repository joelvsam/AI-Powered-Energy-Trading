"""Run all supported forecasting models and compare isolated backtest results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backtesting_review import run_model_comparison_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xgboost, lstm, and prophet, then compare isolated backtests.")
    parser.add_argument("--zone", default=None, help="ENTSO-E bidding zone (default from config).")
    parser.add_argument("--lookback-days", type=int, default=None, help="History window in days.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "backtesting"),
        help="Directory where comparison artifacts will be written.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Transaction cost in basis points.")
    parser.add_argument("--annualization-factor", type=int, default=24, help="Annualization factor for Sharpe ratio.")
    parser.add_argument("--long-threshold", type=float, default=0.1, help="Position threshold for LONG decisions.")
    parser.add_argument("--short-threshold", type=float, default=-0.1, help="Position threshold for SHORT decisions.")
    parser.add_argument("--notional-eur", type=float, default=10000.0, help="Backtest notional used for pnl scaling.")
    parser.add_argument("--accuracy-horizon-steps", type=int, default=1, help="Future step horizon used for decision-accuracy scoring.")
    parser.add_argument("--hold-tolerance-pct", type=float, default=0.002, help="Absolute return band for HOLD directional accuracy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df, metadata = run_model_comparison_workflow(
        zone=args.zone,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir,
        transaction_cost_bps=args.transaction_cost_bps,
        annualization_factor=args.annualization_factor,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        notional_eur=args.notional_eur,
        accuracy_horizon_steps=args.accuracy_horizon_steps,
        hold_tolerance_pct=args.hold_tolerance_pct,
    )
    print("Model Comparison Summary")
    print(summary_df.to_string(index=False) if not summary_df.empty else "No successful model runs.")
    print()
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "winner_model": metadata.get("winner_model"),
                "accuracy_horizon_steps": metadata.get("accuracy_horizon_steps"),
                "hold_tolerance_pct": metadata.get("hold_tolerance_pct"),
                "failures": metadata.get("failures", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
