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
from src.backtesting.strategy_comparison import StrategyComparisonResult, run_strategy_comparison

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ModelComparisonResult",
    "StrategyComparisonResult",
    "compute_price_change",
    "evaluate_decision_accuracy",
    "generate_backtest_outputs",
    "run_model_comparison",
    "run_strategy_comparison",
    "run_backtest",
    "sort_model_comparison",
]
