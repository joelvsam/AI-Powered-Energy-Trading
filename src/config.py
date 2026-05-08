"""Central configuration for the energy trading project."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration loaded from environment variables."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "artifacts" / "models"
    simulation_dir: Path = project_root / "artifacts" / "simulation"
    research_dir: Path = project_root / "artifacts" / "research"

    default_zone: str = os.getenv("ENTSOE_BIDDING_ZONE", "DE_LU")
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "180"))
    random_seed: int = int(os.getenv("SEED", "42"))
    timezone_utc: str = "UTC"

    entsoe_api_key: str | None = os.getenv("ENTSOE_API_KEY")
    entsoe_timeout_s: int = int(os.getenv("ENTSOE_TIMEOUT_S", "90"))
    entsoe_chunk_days: int = int(os.getenv("ENTSOE_CHUNK_DAYS", "30"))

    hf_token: str | None = os.getenv("HF_TOKEN")
    hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    hf_timeout_s: int = int(os.getenv("HF_TIMEOUT_S", "30"))

    openmeteo_lat: float = float(os.getenv("OPENMETEO_LAT", "52.52"))
    openmeteo_lon: float = float(os.getenv("OPENMETEO_LON", "13.405"))

    tcost_bps: float = float(os.getenv("TCOST_BPS", "5.0"))
    annualization_factor: int = int(os.getenv("ANNUALIZATION_FACTOR", "24"))
    backtest_notional_eur: float = float(os.getenv("BACKTEST_NOTIONAL_EUR", "10000.0"))
    enable_new_signal: bool = os.getenv("ENABLE_NEW_SIGNAL", "true").strip().lower() in {"1", "true", "yes", "on"}
    signal_volatility_window_hours: int = int(os.getenv("SIGNAL_VOL_WINDOW_HOURS", "24"))
    signal_position_scale_k: float = float(os.getenv("SIGNAL_POSITION_SCALE_K", "2.0"))
    enable_volatility_scaling: bool = os.getenv("ENABLE_VOLATILITY_SCALING", "true").strip().lower() in {"1", "true", "yes", "on"}
    enable_execution_delay: bool = os.getenv("ENABLE_EXECUTION_DELAY", "true").strip().lower() in {"1", "true", "yes", "on"}
    signal_equilibrium_window_hours: int = int(os.getenv("SIGNAL_EQUILIBRIUM_WINDOW_HOURS", "72"))
    signal_imbalance_scale: float = float(os.getenv("SIGNAL_IMBALANCE_SCALE", "8.0"))
    signal_forecast_weight: float = float(os.getenv("SIGNAL_FORECAST_WEIGHT", "0.45"))
    signal_mean_reversion_weight: float = float(os.getenv("SIGNAL_MEAN_REVERSION_WEIGHT", "0.30"))
    signal_fundamental_weight: float = float(os.getenv("SIGNAL_FUNDAMENTAL_WEIGHT", "0.25"))
    long_price_edge_threshold: float = float(os.getenv("LONG_PRICE_EDGE_THRESHOLD", "0.5"))
    short_price_edge_threshold: float = float(os.getenv("SHORT_PRICE_EDGE_THRESHOLD", "-0.5"))
    enable_regime_switching: bool = os.getenv("ENABLE_REGIME_SWITCHING", "true").strip().lower() in {"1", "true", "yes", "on"}
    high_vol_regime_quantile: float = float(os.getenv("HIGH_VOL_REGIME_QUANTILE", "0.7"))
    position_limit: float = float(os.getenv("POSITION_LIMIT", "1.0"))
    max_position_change: float = float(os.getenv("MAX_POSITION_CHANGE", "0.35"))
    bid_ask_spread_bps: float = float(os.getenv("BID_ASK_SPREAD_BPS", "3.0"))
    bid_ask_spread_eur_mwh: float = float(os.getenv("BID_ASK_SPREAD_EUR_MWH", "0.0"))
    slippage_volatility_factor: float = float(os.getenv("SLIPPAGE_VOLATILITY_FACTOR", "0.08"))
    slippage_turnover_factor: float = float(os.getenv("SLIPPAGE_TURNOVER_FACTOR", "0.02"))
    delay_penalty_factor: float = float(os.getenv("DELAY_PENALTY_FACTOR", "0.03"))
    bootstrap_iterations: int = int(os.getenv("BOOTSTRAP_ITERATIONS", "250"))
    walk_forward_train_window_days: int = int(os.getenv("WALK_FORWARD_TRAIN_WINDOW_DAYS", "90"))
    walk_forward_test_window_days: int = int(os.getenv("WALK_FORWARD_TEST_WINDOW_DAYS", "7"))
    llm_optional_mode: bool = os.getenv("LLM_OPTIONAL_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}


def ensure_directories(cfg: AppConfig) -> None:
    """Ensure output directories exist."""
    for directory in [cfg.data_raw_dir, cfg.data_processed_dir, cfg.models_dir, cfg.simulation_dir, cfg.research_dir]:
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
