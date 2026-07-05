"""LSTM-based forecasting models."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.models.base import TrainingOutputs, model_feature_columns
from src.models.diagnostics import build_common_error_analysis, lstm_feature_ablation, write_model_diagnostics
from src.models.feature_selection import first_train_window_rows, select_model_features
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

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        if nn is None:
            raise RuntimeError("PyTorch is required for LSTM model. Please install torch.")
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
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


MIN_VALIDATION_ROWS = 8
MIN_FIT_ROWS = 16


def _train_model(
    model: LSTMRegressor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    max_epochs: int = 60,
    patience: int = 6,
    validation_fraction: float = 0.15,
) -> dict[str, object]:
    """Train with a chronological validation tail, early stopping, and best-weight restore.

    Falls back to fixed-epoch training when the sample is too small to carve
    out a meaningful validation tail.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_rows = x_train.shape[0]
    validation_rows = max(int(total_rows * validation_fraction), MIN_VALIDATION_ROWS)
    has_validation = total_rows - validation_rows >= MIN_FIT_ROWS
    if has_validation:
        x_fit, y_fit = x_train[: total_rows - validation_rows], y_train[: total_rows - validation_rows]
        x_val, y_val = x_train[total_rows - validation_rows :], y_train[total_rows - validation_rows :]
    else:
        x_fit, y_fit = x_train, y_train
        x_val, y_val = None, None
        max_epochs = min(max_epochs, 10)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    epochs_run = 0
    early_stopped = False

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_fit)
        loss = criterion(pred, y_fit)
        loss.backward()
        optimizer.step()
        epochs_run += 1

        if x_val is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                early_stopped = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "epochs_run": epochs_run,
        "max_epochs": max_epochs,
        "early_stopped": early_stopped,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "validation_rows": validation_rows if has_validation else 0,
    }


def _fit_target(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    *,
    seed: int,
    train_window_days: int,
    test_window_days: int,
    max_epochs: int = 60,
    patience: int = 6,
) -> tuple[LSTMRegressor, StandardScaler, pd.Series, dict[str, object]]:
    walk_df = pd.DataFrame({"timestamp_utc": timestamps}).reset_index(drop=True)
    windows = iter_walk_forward_windows(
        walk_df,
        train_window_days=train_window_days,
        test_window_days=test_window_days,
    )
    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    actuals: list[float] = []
    preds: list[float] = []
    fold_histories: list[dict[str, object]] = []

    for window in windows:
        scaler = StandardScaler()
        train_slice = slice(window.train_start, window.train_end)
        test_slice = slice(window.test_start, window.test_end)
        X_train = scaler.fit_transform(X.iloc[train_slice])
        X_test = scaler.transform(X.iloc[test_slice])
        model = _build_model(input_dim=X.shape[1], seed=seed)
        x_train, y_train = _to_tensors(X_train, y.iloc[train_slice].values)
        x_test, y_test = _to_tensors(X_test, y.iloc[test_slice].values)
        fold_histories.append(_train_model(model, x_train, y_train, max_epochs=max_epochs, patience=patience))
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
    final_history = _train_model(final_model, final_x, final_y, max_epochs=max_epochs, patience=patience)

    metrics = {
        "mae": float(mean_absolute_error(actuals, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
        "training": {
            "mean_fold_epochs": float(np.mean([history["epochs_run"] for history in fold_histories])),
            "fold_early_stop_rate": float(np.mean([bool(history["early_stopped"]) for history in fold_histories])),
            "final_model": final_history,
        },
    }
    return final_model, final_scaler, predictions, metrics


def train_lstm_models(features_df: pd.DataFrame, cfg: AppConfig) -> TrainingOutputs:
    df = features_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    feature_cols = model_feature_columns(df)
    feature_selection = None
    if cfg.enable_feature_pruning:
        feature_selection = select_model_features(
            df,
            feature_cols,
            fit_rows=first_train_window_rows(len(df), cfg.walk_forward_train_window_days, cfg.walk_forward_test_window_days),
            correlation_threshold=cfg.feature_correlation_threshold,
        )
        feature_cols = feature_selection.kept
    X = df[feature_cols]
    timestamps = df["timestamp_utc"]

    demand_model, demand_scaler, demand_pred, demand_metrics = _fit_target(
        X,
        df["demand_kw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
        max_epochs=cfg.lstm_max_epochs,
        patience=cfg.lstm_early_stopping_patience,
    )
    renewable_model, renewable_scaler, renewable_pred, renewable_metrics = _fit_target(
        X,
        df["renewable_mw"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
        max_epochs=cfg.lstm_max_epochs,
        patience=cfg.lstm_early_stopping_patience,
    )
    price_model, price_scaler, price_pred, price_metrics = _fit_target(
        X,
        df["price_eur_mwh"],
        timestamps,
        seed=cfg.random_seed,
        train_window_days=cfg.walk_forward_train_window_days,
        test_window_days=cfg.walk_forward_test_window_days,
        max_epochs=cfg.lstm_max_epochs,
        patience=cfg.lstm_early_stopping_patience,
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
        "feature_selection": feature_selection.summary() if feature_selection else {"enabled": False},
        "training_history": {
            "price": price_metrics.get("training", {}),
            "demand": demand_metrics.get("training", {}),
            "renewable": renewable_metrics.get("training", {}),
        },
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
