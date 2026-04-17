"""Shared model interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingOutputs:
    demand_model_path: str
    renewable_model_path: str
    price_model_path: str
    metrics_path: str
    scored_df: pd.DataFrame
    model_key: str
