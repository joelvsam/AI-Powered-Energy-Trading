"""Standalone historical backtesting package."""

from src.backtesting.comparison import ModelComparisonResult, run_model_comparison, sort_model_comparison
from src.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    compute_price_change,
    evaluate_decision_accuracy,
    generate_backtest_outputs,
    run_backtest,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ModelComparisonResult",
    "compute_price_change",
    "evaluate_decision_accuracy",
    "generate_backtest_outputs",
    "run_model_comparison",
    "run_backtest",
    "sort_model_comparison",
]
