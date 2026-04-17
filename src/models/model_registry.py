"""Model registry and shared training output contracts."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.config import AppConfig
from src.models.base import TrainingOutputs
from src.models.train_lstm import train_lstm_models
from src.models.train_prophet import train_prophet_models
from src.models.train_xgb import train_xgb_models


Trainer = Callable[[pd.DataFrame, AppConfig], TrainingOutputs]


MODEL_REGISTRY: dict[str, Trainer] = {
    "xgboost": train_xgb_models,
    "lstm": train_lstm_models,
    "prophet": train_prophet_models,
}


def train_with_model(model_key: str, features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_key}'. Valid choices: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key](features_df, cfg)
