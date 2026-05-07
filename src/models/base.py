"""Shared model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import AppConfig


@dataclass
class TrainingOutputs:
    demand_model_path: str
    renewable_model_path: str
    price_model_path: str
    metrics_path: str
    scored_df: pd.DataFrame
    model_key: str
    scored_path: str = ""
    diagnostics_path: str = ""


def scored_predictions_path(model_key: str, cfg: AppConfig) -> Path:
    """Return the persisted scored-predictions path for a trained model."""
    return cfg.models_dir / f"scored_predictions_{model_key}.csv"
