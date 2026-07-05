from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import AppConfig
from src.experiment_tracking import write_run_manifest


def build_inputs() -> dict:
    return {
        "run_config": {"zone": "DE_LU", "lookback_days": 30, "model": "xgboost", "skip_model_comparison": True},
        "runtime_modes": {"energy_source": "entsoe", "research_grade": True},
        "data_provenance": {"real_rows": 720, "synthetic_rows": 0},
        "backtest_metrics": {
            "sharpe_ratio": 1.25,
            "max_drawdown": -0.08,
            "total_pnl": 420.5,
            "net_pnl_eur": 400.0,
            "directional_accuracy": 0.55,
        },
        "model_comparison_df": pd.DataFrame([{"model_key": "xgboost", "sharpe_ratio": 1.25}]),
        "artifact_paths": {"research_note_path": "artifacts/research/research_note.md"},
    }


class WriteRunManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        research_dir = Path(self._tmp.name) / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = replace(
            AppConfig(),
            research_dir=research_dir,
            entsoe_api_key="SECRET-API-KEY-123",
            hf_token="SECRET-HF-TOKEN-456",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manifest_written_with_expected_fields(self) -> None:
        tracking = write_run_manifest(self.cfg, **build_inputs())
        manifest_path = Path(tracking["manifest_path"])
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_config"]["zone"], "DE_LU")
        self.assertEqual(manifest["model_comparison"]["winner_model"], "xgboost")
        self.assertIn("package_versions", manifest)
        self.assertIn("config_snapshot", manifest)

    def test_secrets_are_redacted(self) -> None:
        tracking = write_run_manifest(self.cfg, **build_inputs())
        manifest_text = Path(tracking["manifest_path"]).read_text(encoding="utf-8")
        self.assertNotIn("SECRET-API-KEY-123", manifest_text)
        self.assertNotIn("SECRET-HF-TOKEN-456", manifest_text)
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["config_snapshot"]["entsoe_api_key"], "***set***")

    def test_runs_index_appends_across_runs(self) -> None:
        first = write_run_manifest(self.cfg, **build_inputs())
        second = write_run_manifest(self.cfg, **build_inputs())
        index = pd.read_csv(first["runs_index_path"])
        self.assertEqual(len(index), 2)
        self.assertEqual(first["runs_index_path"], second["runs_index_path"])
        self.assertIn("sharpe_ratio", index.columns)
        self.assertIn("git_revision", index.columns)

    def test_empty_model_comparison_is_tolerated(self) -> None:
        inputs = build_inputs()
        inputs["model_comparison_df"] = pd.DataFrame()
        tracking = write_run_manifest(self.cfg, **inputs)
        manifest = json.loads(Path(tracking["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_comparison"], {})


if __name__ == "__main__":
    unittest.main()
