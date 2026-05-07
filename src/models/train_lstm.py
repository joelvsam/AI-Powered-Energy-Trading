"""LSTM-based forecasting models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.models.base import TrainingOutputs
from src.models.diagnostics import build_common_error_analysis, lstm_feature_ablation, write_model_diagnostics
from src.models.walk_forward import iter_walk_forward_windows

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


_LSTM_BASE = nn.Module if nn is not None else object


class LSTMRegressor(_LSTM_BASE):
    """Simple LSTM regressor for one-step forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if nn is None:
            raise RuntimeError("PyTorch is required for LSTM model. Please install torch.")
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


def _fit_target(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    *,
    seed: int,
    train_window_days: int,
    test_window_days: int,
) -> tuple[LSTMRegressor, StandardScaler, pd.Series, dict[str, float]]:
    walk_df = pd.DataFrame({"timestamp_utc": timestamps}).reset_index(drop=True)
    windows = iter_walk_forward_windows(
        walk_df,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )
    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    actuals: list[float] = []
    preds: list[float] = []

    for window in windows:
        scaler = StandardScaler()
        train_slice = slice(window.train_start, window.train_end)
        test_slice = slice(window.test_start, window.test_end)
        X_train = scaler.fit_transform(X.iloc[train_slice])
        X_test = scaler.transform(X.iloc[test_slice])
        model = _build_model(input_dim=X.shape[1], seed=seed)
        x_train, y_train = _to_tensors(X_train, y.iloc[train_slice].values)
        x_test, y_test = _to_tensors(X_test, y.iloc[test_slice].values)
        _train_model(model, x_train, y_train, epochs=8)
        model.eval()
        with torch.no_grad():
            fold_pred = model(x_test).view(-1).cpu().numpy()
        y_test_np = y_test.view(-1).cpu().numpy()
        predictions.iloc[test_slice] = fold_pred
        actuals.extend(y_test_np.tolist())
        preds.extend(fold_pred.tolist())

    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)
    final_model = _build_model(input_dim=X.shape[1], seed=seed)
    final_x, final_y = _to_tensors(X_scaled_full, y.values)
    _train_model(final_model, final_x, final_y, epochs=10)

    metrics = {
        "mae": float(mean_absolute_error(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
    }
    return final_model, final_scaler, predictions, metrics


def train_lstm_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    target_cols = {"timestamp_utc", "demand_kw", "renewable_mw", "price_eur_mwh"}
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]
    timestamps = df["timestamp_utc"]

    demand_model, demand_scaler, demand_pred, demand_metrics = _fit_target(
        X,
        df["demand_kw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    renewable_model, renewable_scaler, renewable_pred, renewable_metrics = _fit_target(
        X,
        df["renewable_mw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )
    price_model, price_scaler, price_pred, price_metrics = _fit_target(
        X,
        df["price_eur_mwh"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
    )

    df["pred_demand_kw"] = demand_pred
    df["pred_renewable_mw"] = renewable_pred
    df["pred_price_eur_mwh"] = price_pred
    df = df.dropna(subset=["pred_demand_kw", "pred_renewable_mw", "pred_price_eur_mwh"]).reset_index(drop=True)

    demand_path = cfg.models_dir / "demand_lstm.pt"
    renewable_path = cfg.models_dir / "renewable_lstm.pt"
    price_path = cfg.models_dir / "price_lstm.pt"
    demand_scaler_path = cfg.models_dir / "demand_lstm_scaler.joblib"
    renewable_scaler_path = cfg.models_dir / "renewable_lstm_scaler.joblib"
    price_scaler_path = cfg.models_dir / "price_lstm_scaler.joblib"
    metrics_path = cfg.models_dir / "metrics_lstm.json"
    diagnostics_path = cfg.models_dir / "diagnostics_lstm.json"
    torch.save(demand_model.state_dict(), demand_path)
    torch.save(renewable_model.state_dict(), renewable_path)
    torch.save(price_model.state_dict(), price_path)
    joblib.dump(demand_scaler, demand_scaler_path)
    joblib.dump(renewable_scaler, renewable_scaler_path)
    joblib.dump(price_scaler, price_scaler_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"demand": demand_metrics, "renewable": renewable_metrics, "price": price_metrics}, handle, indent=2)
    diagnostics_payload = {
        "model_key": "lstm",
        "price_feature_ablation": lstm_feature_ablation(price_model, price_scaler, X, feature_cols),
        "demand_feature_ablation": lstm_feature_ablation(demand_model, demand_scaler, X, feature_cols),
        "renewable_feature_ablation": lstm_feature_ablation(renewable_model, renewable_scaler, X, feature_cols),
        "error_analysis": build_common_error_analysis(df),
    }
    write_model_diagnostics(diagnostics_payload, diagnostics_path)

    return TrainingOutputs(
        demand_model_path=str(demand_path),
        renewable_model_path=str(renewable_path),
        price_model_path=str(price_path),
        metrics_path=str(metrics_path),
        scored_df=df,
        model_key="lstm",
        diagnostics_path=str(diagnostics_path),
    )
