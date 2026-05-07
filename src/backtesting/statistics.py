"""Statistical validation helpers for strategy comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting.engine import compute_sharpe


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    annualization_factor: int,
    iterations: int = 250,
    random_seed: int = 42,
) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy()
    if clean.size == 0:
        return {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.default_rng(random_seed)
    samples = []
    for _ in range(max(iterations, 20)):
        sample = rng.choice(clean, size=clean.size, replace=True)
        samples.append(compute_sharpe(pd.Series(sample), annualization_factor))
    return {
        "sharpe": compute_sharpe(pd.Series(clean), annualization_factor),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
    }


def compare_return_streams(
    strategy_returns: pd.Series,
    baseline_returns: pd.Series,
    *,
    annualization_factor: int,
    iterations: int = 250,
    random_seed: int = 42,
) -> dict[str, float | bool]:
    strategy = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0).to_numpy()
    baseline = pd.to_numeric(baseline_returns, errors="coerce").fillna(0.0).to_numpy()
    if strategy.size == 0 or baseline.size == 0:
        return {
            "sharpe_diff": 0.0,
            "pnl_diff_mean": 0.0,
            "sharpe_diff_ci_lower": 0.0,
            "sharpe_diff_ci_upper": 0.0,
            "significant_outperformance": False,
        }
    size = min(strategy.size, baseline.size)
    strategy = strategy[:size]
    baseline = baseline[:size]
    rng = np.random.default_rng(random_seed)
    sharpe_diffs = []
    pnl_diffs = []
    for _ in range(max(iterations, 20)):
        idx = rng.choice(np.arange(size), size=size, replace=True)
        strategy_sample = strategy[idx]
        baseline_sample = baseline[idx]
        sharpe_diffs.append(
            compute_sharpe(pd.Series(strategy_sample), annualization_factor)
            - compute_sharpe(pd.Series(baseline_sample), annualization_factor)
        )
        pnl_diffs.append(float(np.mean(strategy_sample - baseline_sample)))
    lower = float(np.quantile(sharpe_diffs, 0.025))
    upper = float(np.quantile(sharpe_diffs, 0.975))
    return {
        "sharpe_diff": float(np.mean(sharpe_diffs)),
        "pnl_diff_mean": float(np.mean(pnl_diffs)),
        "sharpe_diff_ci_lower": lower,
        "sharpe_diff_ci_upper": upper,
        "significant_outperformance": bool(lower > 0.0),
    }
