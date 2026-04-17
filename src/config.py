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

    default_zone: str = os.getenv("ENTSOE_BIDDING_ZONE", "DE_LU")
    lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "180"))
    random_seed: int = int(os.getenv("SEED", "42"))
    timezone_utc: str = "UTC"

    entsoe_api_key: str | None = os.getenv("ENTSOE_API_KEY")

    hf_token: str | None = os.getenv("HF_TOKEN")
    hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    hf_timeout_s: int = int(os.getenv("HF_TIMEOUT_S", "30"))

    openmeteo_lat: float = float(os.getenv("OPENMETEO_LAT", "52.52"))
    openmeteo_lon: float = float(os.getenv("OPENMETEO_LON", "13.405"))

    tcost_bps: float = float(os.getenv("TCOST_BPS", "5.0"))
    annualization_factor: int = int(os.getenv("ANNUALIZATION_FACTOR", "24"))


def ensure_directories(cfg: AppConfig) -> None:
    """Ensure output directories exist."""
    for directory in [cfg.data_raw_dir, cfg.data_processed_dir, cfg.models_dir, cfg.simulation_dir]:
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
