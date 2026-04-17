"""LSTM-based forecasting models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.models.base import TrainingOutputs

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class LSTMRegressor(nn.Module):
    """Simple LSTM regressor for one-step forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _build_model(input_dim: int, seed: int) -> LSTMRegressor:
    if torch is None:
        raise RuntimeError("PyTorch is required for LSTM model. Please install torch.")
    torch.manual_seed(seed)
    return LSTMRegressor(input_dim=input_dim)


def _to_tensors(X_2d: np.ndarray, y_1d: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.tensor(X_2d, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_1d, dtype=torch.float32).view(-1, 1)
    return x_tensor, y_tensor


def _train_model(model: LSTMRegressor, x_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 10) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()


def _fit_target(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[LSTMRegressor, StandardScaler, dict]:
    splitter = TimeSeriesSplit(n_splits=5)
    maes: list[float] = []
    rmses: list[float] = []

    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)

    for train_idx, test_idx in splitter.split(X_scaled_full):
        model = _build_model(input_dim=X.shape[1], seed=seed)
        x_train, y_train = _to_tensors(X_scaled_full[train_idx], y.iloc[train_idx].values)
        x_test, y_test = _to_tensors(X_scaled_full[test_idx], y.iloc[test_idx].values)
        _train_model(model, x_train, y_train, epochs=8)
        model.eval()
        with torch.no_grad():
            pred = model(x_test).view(-1).cpu().numpy()
        y_test_np = y_test.view(-1).cpu().numpy()
        maes.append(float(mean_absolute_error(y_test_np, pred)))
        rmses.append(float(np.sqrt(mean_squared_error(y_test_np, pred))))

    final_model = _build_model(input_dim=X.shape[1], seed=seed)
    final_x, final_y = _to_tensors(X_scaled_full, y.values)
    _train_model(final_model, final_x, final_y, epochs=10)

    metrics = {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}
    return final_model, scaler, metrics


def _predict(model: LSTMRegressor, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    scaled = scaler.transform(X)
    x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).view(-1).cpu().numpy()
    return pred


def train_lstm_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc")
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]

    demand_model, demand_scaler, demand_metrics = _fit_target(X, df["demand_kw"], cfg.random_seed)
    renewable_model, renewable_scaler, renewable_metrics = _fit_target(X, df["renewable_mw"], cfg.random_seed)
    price_model, price_scaler, price_metrics = _fit_target(X, df["price_eur_mwh"], cfg.random_seed)

    df["pred_demand_kw"] = _predict(demand_model, demand_scaler, X)
    df["pred_renewable_mw"] = _predict(renewable_model, renewable_scaler, X)
    df["pred_price_eur_mwh"] = _predict(price_model, price_scaler, X)

    demand_path = cfg.models_dir / "demand_lstm.pt"
    renewable_path = cfg.models_dir / "renewable_lstm.pt"
    price_path = cfg.models_dir / "price_lstm.pt"
    demand_scaler_path = cfg.models_dir / "demand_lstm_scaler.joblib"
    renewable_scaler_path = cfg.models_dir / "renewable_lstm_scaler.joblib"
    price_scaler_path = cfg.models_dir / "price_lstm_scaler.joblib"
    metrics_path = cfg.models_dir / "metrics_lstm.json"
    torch.save(demand_model.state_dict(), demand_path)
    torch.save(renewable_model.state_dict(), renewable_path)
    torch.save(price_model.state_dict(), price_path)
    joblib.dump(demand_scaler, demand_scaler_path)
    joblib.dump(renewable_scaler, renewable_scaler_path)
    joblib.dump(price_scaler, price_scaler_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"demand": demand_metrics, "renewable": renewable_metrics, "price": price_metrics}, handle, indent=2)

    return TrainingOutputs(
        demand_model_path=str(demand_path),
        renewable_model_path=str(renewable_path),
        price_model_path=str(price_path),
        metrics_path=str(metrics_path),
        scored_df=df,
        model_key="lstm",
    )
