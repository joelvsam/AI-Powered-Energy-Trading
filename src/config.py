"""Central configuration for the energy trading project."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

import numpy as np

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency
    st = None


@lru_cache(maxsize=1)
def _load_local_secrets() -> dict[str, Any]:
    """Load local Streamlit secrets for non-Streamlit entrypoints such as CLI scripts."""
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}

    with secrets_path.open("rb") as handle:
        parsed = tomllib.load(handle)

    values: dict[str, Any] = {}
    for key, value in parsed.items():
        if key == "general" and isinstance(value, dict):
            values.update(value)
        else:
            values[key] = value
    return values


def _get_setting(name: str, default: Any | None = None) -> Any | None:
    """Read a configuration value from Streamlit secrets or local secrets.toml."""
    if st is not None:
        try:
            secrets = st.secrets
            if hasattr(secrets, "get"):
                value = secrets.get(name)
                if value is not None:
                    return value
                general = secrets.get("general")
                if isinstance(general, dict) and name in general:
                    return general[name]
            if name in secrets:
                return secrets[name]
        except Exception:
            pass

    local_secrets = _load_local_secrets()
    if name in local_secrets:
        return local_secrets[name]
    return default


def _get_bool_setting(name: str, default: bool = False) -> bool:
    value = _get_setting(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_int_setting(name: str, default: int = 0) -> int:
    value = _get_setting(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float_setting(name: str, default: float = 0.0) -> float:
    value = _get_setting(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

BIDDING_ZONE_OPENMETEO_COORDS: dict[str, tuple[float, float]] = {
    "DE_LU": (52.52, 13.405),
    "FR": (48.8566, 2.3522),
    "NL": (52.3676, 4.9041),
}


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration loaded from Streamlit secrets."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    cache_dir: Path = project_root / "cache"
    models_dir: Path = project_root / "artifacts" / "models"
    simulation_dir: Path = project_root / "artifacts" / "simulation"
    research_dir: Path = project_root / "artifacts" / "research"

    default_zone: str = _get_setting("ENTSOE_BIDDING_ZONE", "DE_LU") or "DE_LU"
    lookback_days: int = _get_int_setting("LOOKBACK_DAYS", 180)
    random_seed: int = _get_int_setting("SEED", 42)
    timezone_utc: str = "UTC"

    entsoe_api_key: str | None = _get_setting("ENTSOE_API_KEY")
    entsoe_timeout_s: int = _get_int_setting("ENTSOE_TIMEOUT_S", 90)
    entsoe_chunk_days: int = _get_int_setting("ENTSOE_CHUNK_DAYS", 30)
    cache_schema_version: int = _get_int_setting("CACHE_SCHEMA_VERSION", 1)
    cache_enabled: bool = _get_bool_setting("CACHE_ENABLED", True)
    cache_atomic_write_enabled: bool = _get_bool_setting("CACHE_ATOMIC_WRITE_ENABLED", True)
    cache_rebuild_default: bool = _get_bool_setting("CACHE_REBUILD_DEFAULT", False)
    cache_max_gap_fill_hours: int = _get_int_setting("CACHE_MAX_GAP_FILL_HOURS", 720)

    hf_token: str | None = _get_setting("HF_TOKEN")
    hf_model: str = _get_setting("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct") or "Qwen/Qwen2.5-72B-Instruct"
    hf_timeout_s: int = _get_int_setting("HF_TIMEOUT_S", 30)

    openmeteo_lat: float = _get_float_setting("OPENMETEO_LAT", 52.52)
    openmeteo_lon: float = _get_float_setting("OPENMETEO_LON", 13.405)

    tcost_bps: float = _get_float_setting("TCOST_BPS", 5.0)

    def openmeteo_coords_for_zone(self, zone: str | None = None) -> tuple[float, float]:
        if zone:
            normalized_zone = zone.strip().upper()
            coords = BIDDING_ZONE_OPENMETEO_COORDS.get(normalized_zone)
            if coords is not None:
                return coords
        return self.openmeteo_lat, self.openmeteo_lon

    def openmeteo_lat_for_zone(self, zone: str | None = None) -> float:
        return self.openmeteo_coords_for_zone(zone)[0]

    def openmeteo_lon_for_zone(self, zone: str | None = None) -> float:
        return self.openmeteo_coords_for_zone(zone)[1]
    annualization_factor: int = _get_int_setting("ANNUALIZATION_FACTOR", 24)
    backtest_notional_eur: float = _get_float_setting("BACKTEST_NOTIONAL_EUR", 10000.0)
    enable_new_signal: bool = _get_bool_setting("ENABLE_NEW_SIGNAL", True)
    signal_volatility_window_hours: int = _get_int_setting("SIGNAL_VOL_WINDOW_HOURS", 24)
    signal_position_scale_k: float = _get_float_setting("SIGNAL_POSITION_SCALE_K", 2.0)
    enable_volatility_scaling: bool = _get_bool_setting("ENABLE_VOLATILITY_SCALING", True)
    enable_execution_delay: bool = _get_bool_setting("ENABLE_EXECUTION_DELAY", True)
    signal_equilibrium_window_hours: int = _get_int_setting("SIGNAL_EQUILIBRIUM_WINDOW_HOURS", 72)
    signal_imbalance_scale: float = _get_float_setting("SIGNAL_IMBALANCE_SCALE", 8.0)
    signal_forecast_weight: float = _get_float_setting("SIGNAL_FORECAST_WEIGHT", 0.45)
    signal_mean_reversion_weight: float = _get_float_setting("SIGNAL_MEAN_REVERSION_WEIGHT", 0.30)
    signal_fundamental_weight: float = _get_float_setting("SIGNAL_FUNDAMENTAL_WEIGHT", 0.25)
    long_price_edge_threshold: float = _get_float_setting("LONG_PRICE_EDGE_THRESHOLD", 0.5)
    short_price_edge_threshold: float = _get_float_setting("SHORT_PRICE_EDGE_THRESHOLD", -0.5)
    enable_regime_switching: bool = _get_bool_setting("ENABLE_REGIME_SWITCHING", True)
    high_vol_regime_quantile: float = _get_float_setting("HIGH_VOL_REGIME_QUANTILE", 0.7)
    position_limit: float = _get_float_setting("POSITION_LIMIT", 1.0)
    max_position_change: float = _get_float_setting("MAX_POSITION_CHANGE", 0.35)
    bid_ask_spread_bps: float = _get_float_setting("BID_ASK_SPREAD_BPS", 3.0)
    bid_ask_spread_eur_mwh: float = _get_float_setting("BID_ASK_SPREAD_EUR_MWH", 0.0)
    slippage_volatility_factor: float = _get_float_setting("SLIPPAGE_VOLATILITY_FACTOR", 0.08)
    slippage_turnover_factor: float = _get_float_setting("SLIPPAGE_TURNOVER_FACTOR", 0.02)
    delay_penalty_factor: float = _get_float_setting("DELAY_PENALTY_FACTOR", 0.03)
    bootstrap_iterations: int = _get_int_setting("BOOTSTRAP_ITERATIONS", 250)
    walk_forward_train_window_days: int = _get_int_setting("WALK_FORWARD_TRAIN_WINDOW_DAYS", 90)
    walk_forward_test_window_days: int = _get_int_setting("WALK_FORWARD_TEST_WINDOW_DAYS", 7)
    llm_optional_mode: bool = _get_bool_setting("LLM_OPTIONAL_MODE", True)


def ensure_directories(cfg: AppConfig) -> None:
    """Ensure output directories exist."""
    for directory in [cfg.data_raw_dir, cfg.data_processed_dir, cfg.cache_dir, cfg.models_dir, cfg.simulation_dir, cfg.research_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
