# AI Powered Energy Forecasting & Trading Decision System:

This repository is a trading research system. The core question is:

Why should this strategy make money in real European power markets after realistic execution costs, delays, and baseline comparisons?

The pipeline answers that question by combining:

- canonical raw-data parquet caches with schema validation, provenance tracking, and partial synthetic gap filling
- walk-forward forecasting with `xgboost`, `lstm`, and `prophet`
- market-structure-aware features for day-ahead, intraday, renewables, and imbalance context
- multiple signal families instead of a single forecast delta
- realistic backtesting with spread, slippage, execution delay, position caps, and liquidity limits
- mandatory baseline comparisons and bootstrap significance tests
- model diagnostics tied to trading performance
- leakage-safe feature pruning (near-constant and high-correlation drops fit on the first train window only)
- chronologically honest hyperparameter search validated on the tail of the first train window
- LSTM early stopping with a causal validation split and best-weight restore
- experiment tracking: per-run JSON manifests plus an append-only `runs_index.csv`
- optional LLM-assisted anomaly review and research summarization with deterministic fallbacks

## Research Thesis

The upgraded strategy combines three signal families:

- `forecast_signal`: expected price edge from the forecasting model
- `mean_reversion_signal`: deviation of current price from a rolling equilibrium
- `fundamental_signal`: demand minus renewable imbalance, conditioned by market structure

These matter in European power markets because:

- day-ahead and intraday prices often diverge when short-horizon supply-demand conditions shift
- renewable intermittency can force rapid repricing when wind or solar output changes
- imbalance and spread stress can reveal when market participants must pay up for short-term flexibility
- mean reversion can dominate after temporary dislocations, while trend or forecast edge can dominate during persistent stress

The research workflow evaluates when those ideas work, when they fail, and whether they beat simple alternatives with statistical credibility.

## System Design

The repository includes:

- ENTSO-E market context:
  - day-ahead prices
  - intraday prices when available
  - imbalance prices when available
  - intraday renewable forecast when available
- incremental raw-data persistence:
  - canonical parquet caches under `cache/`
  - one file per dataset and zone such as `cache/entsoe_DE_LU.parquet`
  - fetch-only-missing-range behavior on repeated runs
  - row-level provenance fields including source, synthetic status, quality, fetch timestamp, and cache version
  - atomic cache writes with validation before replace
  - explicit cache rebuild and force-refresh controls
- expanded features:
  - intraday-day-ahead spread
  - renewable forecast error proxies
  - net load and imbalance z-scores
  - realized volatility and spread volatility
  - regime indicators and seasonal interactions
- upgraded signal framework:
  - separate signal-family diagnostics
  - regime-aware weighting
  - capped and liquidity-constrained target positions
- upgraded execution model:
  - next-step fills
  - bid/ask spread costs
  - volatility and turnover slippage
  - delay penalties
  - position and trade-size limits
- research baselines:
  - persistence
  - seasonal hour/day baseline
  - naive mean-reversion baseline
  - zero-signal baseline
- statistical validation:
  - bootstrap Sharpe confidence intervals
  - pairwise outperformance tests versus baselines
  - explicit non-significance flags
- research outputs:
  - strategy comparison tables
  - significance tables
  - anomaly review
  - model diagnostics
  - final research note

## Workflow

The main entry point is `scripts/run_all.py`.

It runs:

1. data ingestion and cleaning
2. raw-data cache validation, incremental fetch, and partial synthetic gap filling
3. feature engineering
4. anomaly review
5. primary model training
6. realistic backtest
7. strategy comparison versus baselines
8. cross-model trading comparison
9. final research-note generation

## Installation

### Prerequisites

- Python 3.11 or newer (3.14 recommended — matches CI and the pinned wheels in `requirements.txt`)
- Git
- On Linux/macOS, optionally install system build tools: `build-essential`, `libssl-dev`, `python3-dev` (packages vary by distro)
- On Windows, ensure Visual Studio Build Tools are available for some binary wheels

### Create a virtual environment and install Python dependencies

Windows (PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Unix / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Dependencies are pinned to exact versions in `requirements.txt` for reproducibility. To run the test suite, install the dev toolchain instead: `pip install -r requirements-dev.txt`.
- If you need GPU-accelerated `torch`, follow official PyTorch install instructions to pick the correct package before or instead of `pip install -r requirements.txt`.

## Environment

This project uses Streamlit secrets for both local runs and Streamlit Cloud deployments. `src/config.py` reads settings from `st.secrets` when the app is running under Streamlit, and from `.streamlit/secrets.toml` for local CLI runs.

To create a local secrets file from the example:

Unix / macOS:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Windows (PowerShell):

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` and set required secrets and configuration. Important values include:

- `ENTSOE_API_KEY` (required for real ENTSO-E data)
- `HF_TOKEN` (optional; required for HuggingFace LLM calls)
- `ENTSOE_BIDDING_ZONE` (default: `DE_LU`)
- `OPENMETEO_LAT` / `OPENMETEO_LON` (default Berlin coords; overridden automatically by selected bidding zone in the dashboard for `DE_LU`, `FR`, and `NL`)
- `SEED` (random seed, default: `42`)
- `CACHE_ENABLED` (toggle cache behavior)

The committed template at `.streamlit/secrets.toml.example` includes the full supported configuration surface from `src/config.py`.

## Running It

Run the full research workflow (example):

```bash
python -m scripts.run_all --lookback-days 180 --zone DE_LU --model xgboost
```

Refetch only a historical window while preserving the rest of the cache:

```bash
python -m scripts.run_all --lookback-days 180 --zone DE_LU --model xgboost --force-refresh
```

Rebuild the canonical raw-data cache for the selected run:

```bash
python -m scripts.run_all --lookback-days 180 --zone DE_LU --model xgboost --rebuild-cache
```

Run cross-model comparison directly:

```bash
python scripts/run_model_comparison.py --output-dir artifacts/research/model_comparison
```

Run isolated backtesting from an existing scored file:

```bash
python scripts/run_backtest.py --input-path artifacts/models/scored_predictions_xgboost.csv --output-dir artifacts/backtesting
```

Start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

### Streamlit Cloud

Deploy `dashboard/app.py` as the main app file. In the Streamlit Cloud app settings, paste the contents of your local `.streamlit/secrets.toml` into the Secrets editor so the hosted app has the same configuration and API access as your local setup.

## Verification / Smoke Tests

Quick checks after installation:

```bash
# Install the test toolchain and run the full suite
pip install -r requirements-dev.txt
python -m pytest tests -q

# Run a small smoke backtest (adjust paths/args to your local config)
python scripts/run_backtest.py --input-path artifacts/models/scored_predictions_xgboost.csv --output-dir artifacts/backtesting/smoke

# Start the dashboard and open http://localhost:8501
streamlit run dashboard/app.py
```

## Troubleshooting

- Missing secrets: confirm `.streamlit/secrets.toml` exists locally, or that Streamlit Cloud Secrets are configured for the deployed app.
- Binary wheel / compilation errors on Windows: install Visual Studio Build Tools and retry, or use prebuilt wheels where available.
- `torch` / GPU: install the correct `torch` wheel for your CUDA version following the official instructions before installing the rest of `requirements.txt`.
- If HuggingFace LLM calls time out, increase `HF_TIMEOUT_S` in `.streamlit/secrets.toml`.


## Artifact Map

Canonical raw-data caches live under `cache/`:

- `cache/entsoe_<zone>.parquet`
- `cache/weather_<zone>.parquet`

These caches store only raw market and raw weather context. They do not store engineered features, signals, predictions, or backtest outputs.

The canonical research bundle lives under `artifacts/research/`.

Important outputs include:

- `artifacts/research/anomaly_review.json`
- `artifacts/research/research_summary.json`
- `artifacts/research/research_note.md`
- `artifacts/research/selected_strategy_metrics_<model>.csv`
- `artifacts/research/selected_strategy_significance_<model>.csv`
- `artifacts/research/model_comparison/model_comparison_summary.csv`
- `artifacts/research/model_comparison/model_comparison_summary.json`

Experiment-tracking outputs (one manifest per workflow run, plus a cross-run index):

- `artifacts/research/runs/run_<id>.json` — config snapshot (secrets redacted), git revision, package versions, provenance, headline metrics
- `artifacts/research/runs_index.csv` — one summary row per run for comparing sweeps

Model-level artifacts remain under `artifacts/models/`:

- `scored_predictions_<model>.csv`
- `metrics_<model>.json`
- `diagnostics_<model>.json`

Execution-level artifacts remain available under strategy backtest folders:

- `backtest_results.csv`
- `backtest_metrics.json`
- `backtest_analytics.json`

## Reading the Results

Focus on these outputs in order:

1. `research_note.md`
2. selected strategy metrics
3. significance table versus baselines
4. model comparison summary
5. anomaly review
6. model diagnostics

The system is designed to answer:

- Does the upgraded strategy beat persistence, seasonal, naive mean-reversion, and no-trade baselines?
- Is the outperformance statistically credible?
- Which model is best for trading, not just for RMSE?
- When does the strategy work?
- When does it break?

## Research Guardrails

The implementation explicitly avoids common failure modes:

- no caching of feature-engineered or model-ready datasets
- no silent overwrite of historical real rows with synthetic rows
- no direct cache overwrite without validation and atomic replace
- walk-forward scoring instead of in-sample evaluation
- no same-bar execution lookahead
- no backward-fill of rolling statistics: warmup rows are dropped (features) or filled with expanding past-only statistics (signals), never with future values; enforced by causality regression tests in `tests/test_no_lookahead.py`
- no fake performance inflation
- no reliance on a cosmetic LLM report
- no test-set tuning of signal parameters
- no hidden synthetic contamination in research outputs

One disclosed exception: raw-data gap repair in `src/data_pipeline/clean.py` uses bounded two-sided interpolation (time interpolation up to 6h, ffill/bfill up to 3h). Every filled row is provenance-marked `partially_synthetic`, so this is visible in research outputs rather than silent.

## Important Caveat

The cache layer is a research-integrity component, not only a runtime optimization. Repeated runs reuse validated raw history, fetch only missing hourly ranges, and synthesize only unresolved gaps. Research outputs and dashboard runtime diagnostics report real, partially synthetic, and fully synthetic coverage explicitly.

If ENTSO-E or weather retrieval leaves unresolved gaps, the pipeline remains runnable by filling only those timestamps synthetically. That keeps the repo operational, but any synthetic contamination must be treated cautiously in research interpretation. The research outputs mark this explicitly.

## Repo Structure

```text
src/
  agents/
    anomaly_agent.py
    research_agent.py
    llm_utils.py
  backtesting/
    engine.py
    comparison.py
    statistics.py
    strategy_comparison.py
  data_pipeline/
  data_sources/
  features/
  models/
    diagnostics.py
    feature_selection.py
    tuning.py
  simulation/
  trading/
  experiment_tracking.py
scripts/
  run_all.py
  run_backtest.py
  run_model_comparison.py
dashboard/
  app.py
.github/
  workflows/
    tests.yml
```

## Verification

The full regression suite covers the backtest engine, comparison flow, dashboard review helpers, signal construction, bootstrap statistics, walk-forward splitting, feature engineering, feature pruning, hyperparameter tuning, experiment tracking, and lookahead/causality guarantees:

```bash
python -m pytest tests -q
```

CI (`.github/workflows/tests.yml`) runs the same compile check and test suite on every push and pull request to `main`.
