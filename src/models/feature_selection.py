"""Leakage-safe feature pruning shared by forecasting trainers.

Pruning statistics (variance and pairwise correlation) are computed only on
the rows belonging to the first walk-forward train window so that no
information from later test windows can influence which features survive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Outcome of one pruning pass over the candidate feature set."""

    kept: list[str]
    dropped_near_constant: list[str] = field(default_factory=list)
    dropped_correlated: list[dict[str, object]] = field(default_factory=list)
    fit_rows: int = 0
    correlation_threshold: float = 0.95

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "fit_rows": int(self.fit_rows),
            "correlation_threshold": float(self.correlation_threshold),
            "kept_count": len(self.kept),
            "dropped_near_constant": list(self.dropped_near_constant),
            "dropped_correlated": list(self.dropped_correlated),
            "kept_features": list(self.kept),
        }


def first_train_window_rows(total_rows: int, train_window_days: int, test_window_days: int) -> int:
    """Row count of the first walk-forward train window, mirroring iter_walk_forward_windows shrinkage."""
    test_hours = max(int(test_window_days), 1) * 24
    requested_train_hours = max(int(train_window_days), 1) * 24
    max_feasible = total_rows - test_hours if total_rows > test_hours else total_rows
    return max(min(requested_train_hours, max_feasible), 2)


def select_model_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    fit_rows: int,
    correlation_threshold: float = 0.95,
    near_constant_std: float = 1e-10,
) -> FeatureSelectionResult:
    """Drop near-constant features and greedily prune highly correlated pairs.

    Keeps the earlier feature of any correlated pair so the pruning order is
    deterministic and stable across runs with the same feature layout.
    """
    fit_rows = max(int(fit_rows), 2)
    frame = df.loc[:, feature_cols].head(fit_rows).apply(pd.to_numeric, errors="coerce")

    stds = frame.std()
    near_constant = [
        col for col in feature_cols
        if not np.isfinite(stds.get(col, np.nan)) or float(stds[col]) <= near_constant_std
    ]
    candidates = [col for col in feature_cols if col not in set(near_constant)]

    kept: list[str] = []
    dropped_correlated: list[dict[str, object]] = []
    if candidates:
        corr = frame[candidates].corr().abs()
        for col in candidates:
            partner = None
            for kept_col in kept:
                value = corr.at[col, kept_col]
                if pd.notna(value) and float(value) > correlation_threshold:
                    partner = (kept_col, float(value))
                    break
            if partner is None:
                kept.append(col)
            else:
                dropped_correlated.append(
                    {
                        "feature": col,
                        "correlated_with": partner[0],
                        "abs_correlation": partner[1],
                    }
                )

    if not kept:
        LOGGER.warning(
            "Feature pruning removed every candidate feature; keeping the original %s features unpruned.",
            len(feature_cols),
        )
        return FeatureSelectionResult(
            kept=list(feature_cols),
            fit_rows=fit_rows,
            correlation_threshold=correlation_threshold,
        )

    if near_constant or dropped_correlated:
        LOGGER.info(
            "Feature pruning kept %s of %s features (%s near-constant, %s correlated above %.2f).",
            len(kept),
            len(feature_cols),
            len(near_constant),
            len(dropped_correlated),
            correlation_threshold,
        )
    return FeatureSelectionResult(
        kept=kept,
        dropped_near_constant=near_constant,
        dropped_correlated=dropped_correlated,
        fit_rows=fit_rows,
        correlation_threshold=correlation_threshold,
    )
