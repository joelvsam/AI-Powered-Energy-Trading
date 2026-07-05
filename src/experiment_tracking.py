"""Lightweight experiment tracking: per-run JSON manifests plus an append-only runs index.

Every workflow run writes a manifest under artifacts/research/runs/ capturing
the config snapshot (secrets redacted), git revision, package versions, data
provenance, and headline metrics, and appends one summary row to
artifacts/research/runs_index.csv so runs can be compared without diffing
artifact trees by hand.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig


LOGGER = logging.getLogger(__name__)

REDACTED_CONFIG_FIELDS = {"entsoe_api_key", "hf_token"}
TRACKED_PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "torch", "prophet", "streamlit", "entsoe-py"]
RUNS_INDEX_NAME = "runs_index.csv"


def _git_revision(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _config_snapshot(cfg: AppConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if field.name in REDACTED_CONFIG_FIELDS:
            snapshot[field.name] = "***set***" if value else None
        elif isinstance(value, Path):
            snapshot[field.name] = str(value)
        else:
            snapshot[field.name] = value
    return snapshot


def _winner_row(model_comparison_df: pd.DataFrame | None) -> dict[str, Any]:
    if model_comparison_df is None or model_comparison_df.empty:
        return {}
    row = model_comparison_df.iloc[0]
    return {
        "winner_model": str(row.get("model_key", "")),
        "winner_sharpe_ratio": float(pd.to_numeric(row.get("sharpe_ratio"), errors="coerce"))
        if pd.notna(row.get("sharpe_ratio"))
        else None,
    }


def write_run_manifest(
    cfg: AppConfig,
    *,
    run_config: dict[str, Any],
    runtime_modes: dict[str, Any],
    data_provenance: dict[str, Any],
    backtest_metrics: dict[str, Any],
    model_comparison_df: pd.DataFrame | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Persist one run manifest and append a summary row to the runs index."""
    timestamp = datetime.now(timezone.utc)
    model_key = str(run_config.get("model", "unknown"))
    run_id = f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}_{model_key}"

    runs_dir = cfg.research_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"run_{run_id}.json"
    runs_index_path = cfg.research_dir / RUNS_INDEX_NAME

    manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "git_revision": _git_revision(cfg.project_root),
        "package_versions": _package_versions(),
        "run_config": run_config,
        "runtime_modes": runtime_modes,
        "config_snapshot": _config_snapshot(cfg),
        "data_provenance": data_provenance,
        "backtest_metrics": backtest_metrics,
        "model_comparison": _winner_row(model_comparison_df),
        "artifact_paths": artifact_paths or {},
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    index_row = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "git_revision": manifest["git_revision"],
        "zone": run_config.get("zone"),
        "lookback_days": run_config.get("lookback_days"),
        "model": model_key,
        "energy_source": runtime_modes.get("energy_source"),
        "research_grade": runtime_modes.get("research_grade"),
        "sharpe_ratio": backtest_metrics.get("sharpe_ratio"),
        "max_drawdown": backtest_metrics.get("max_drawdown"),
        "total_pnl": backtest_metrics.get("total_pnl"),
        "net_pnl_eur": backtest_metrics.get("net_pnl_eur"),
        "directional_accuracy": backtest_metrics.get("directional_accuracy"),
        **_winner_row(model_comparison_df),
        "manifest_path": str(manifest_path),
    }
    index_frame = pd.DataFrame([index_row])
    if runs_index_path.exists():
        try:
            existing = pd.read_csv(runs_index_path)
            index_frame = pd.concat([existing, index_frame], ignore_index=True)
        except Exception:
            LOGGER.warning("Could not read existing runs index at %s; rewriting with the current run only.", runs_index_path)
    index_frame.to_csv(runs_index_path, index=False)

    LOGGER.info("Experiment manifest written: %s (index: %s)", manifest_path, runs_index_path)
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "runs_index_path": str(runs_index_path),
    }
