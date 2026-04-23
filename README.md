# AI-Powered Energy Trading (Beginner Friendly)

This project is a full **end-to-end data science workflow** that looks like what an energy trading team might use in production.

It takes market and weather data, builds features, trains forecasting models, backtests a trading strategy, runs a simulation loop, and finally asks an LLM for a decision with a safe fallback.
You can run everything from a Streamlit dashboard (recommended) or the terminal.

## 1) Simple Project Explanation

Think of this as a smart pipeline:

1. Get energy + weather data
2. Clean and merge data
3. Create forecasting features
4. Train a selected model (XGBoost, LSTM, or Prophet) for demand and renewables
5. Backtest a trading strategy
6. Simulate real-time predictions
7. Ask an LLM for a trade decision

## 2) Visual Pipeline Flow

```mermaid
flowchart TD
    start[run_all.py] --> ingest[Ingest data]
    ingest --> entsoe[Try ENTSO-E API]
    entsoe -->|Success| realData[Real energy data]
    entsoe -->|Fail or key missing| syntheticData[Synthetic energy data]
    realData --> weather[Fetch Open-Meteo weather]
    syntheticData --> weather
    weather --> cleanMerge[Clean and merge]
    cleanMerge --> featureBuild[Build features]
    featureBuild --> trainModels[Train selected model]
    trainModels --> backtest[Backtest strategy]
    backtest --> realtime[Run simulation loop]
    realtime --> llmDecision[LLM decision agent]
    llmDecision --> outputs[Save artifacts and reports]
```

## 3) Step-By-Step Setup

### Step A: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### Step B: Install dependencies

```bash
pip install -r requirements.txt
```


### Step C: Set environment variables

1. Copy `.env.example` to `.env`
2. Fill in:
   - `ENTSOE_API_KEY`
   - `HF_TOKEN`

If `ENTSOE_API_KEY` is missing or ENTSO-E fails, the system auto-switches to synthetic energy data.
If `HF_TOKEN` is missing or Hugging Face fails, the system auto-switches to deterministic fallback decisions.

## 4) How To Run

### Option A (Recommended): Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

In the dashboard you can select:
- Region (`DE_LU`, `FR`, `NL`)
- Training window (`90`, `180`, `365` days)
- Model (`xgboost`, `lstm`, `prophet`)
- Simulation horizon

Then click **Run Pipeline**.

The dashboard surfaces the latest model-driven action (`LONG`, `SHORT`, or `HOLD`) together with the predicted market price in `EUR/MWh`.
It also includes a separate **Backtesting Review** menu in the sidebar where you can load isolated backtest artifacts, run a new isolated backtest from a scored CSV, and compare past model decisions against realized price moves and PnL outcomes.

Inside **Backtesting Review** you can:

- Load the latest isolated backtest saved under `artifacts/backtesting/`
- Run a fresh isolated backtest from a scored CSV path
- Switch the accuracy horizon between the next period and next 24 hours
- Adjust the `HOLD` tolerance band
- Filter by date range and inspect whether each historical decision was correct

Helpful defaults:

- The main workflow now saves scored predictions under `artifacts/models/scored_predictions_<model>.csv`
- If you have already run the simulation pipeline, `artifacts/simulation/backtest_trades.csv` is also a valid scored CSV input for isolated backtesting
- `artifacts/backtesting/backtest_results.csv` is an isolated backtest output, not the recommended first-run source file

### Option B: Terminal

Run full pipeline:

```bash
python -m scripts.run_all --lookback-days 180 --zone DE_LU --model xgboost --simulation-horizon 24
```

### Option C: Isolated backtesting module

Run the standalone backtester on a scored CSV without changing the current pipeline or dashboard behavior:

```bash
python scripts/run_backtest.py --input-path path/to/scored_predictions.csv --output-dir artifacts/backtesting
```

Required input columns:

- `timestamp_utc`
- `price_eur_mwh`
- `pred_price_eur_mwh`
- `pred_demand_kw`
- `pred_renewable_mw`

The command writes isolated results to `artifacts/backtesting/` and does not touch `artifacts/simulation/`.
Those isolated outputs can then be opened in the dashboard's **Backtesting Review** page.
The full training workflow also persists scored predictions to `artifacts/models/scored_predictions_<model>.csv` so you can reuse them here later.

## 5) Project Structure

```text
src/
  config.py
  data_sources/
    entsoe_client.py
  data_pipeline/
    ingest.py
    clean.py
    merge.py
    run_pipeline.py
  features/
    build_features.py
  models/
    model_registry.py
    base.py
    train_xgb.py
    train_lstm.py
    train_prophet.py
  backtesting/
    engine.py
  trading/
    backtest.py
  simulation/
    realtime_loop.py
  agents/
    llm_utils.py
    decision_agent.py
    prompts.py
scripts/
  run_all.py
dashboard/
  app.py
  charts.py
  backtesting_review.py
data/
  raw/
  processed/
artifacts/
  models/
  simulation/
  backtesting/
```

## 6) Data Flow

- `data/raw/energy_raw.csv` and `data/raw/weather_raw.csv` are created during ingestion.
- `data/processed/energy_weather_clean.csv` is created after cleaning/merging.
- `data/processed/features.csv` is created after feature engineering.
- `artifacts/models/*` stores trained model files.
- `artifacts/models/metrics_*.json` stores MAE/RMSE by model.
- `artifacts/simulation/backtest_metrics.json` stores trading metrics.
- `artifacts/simulation/simulation_log.jsonl` stores simulated live predictions.
- `artifacts/simulation/decision_report.json` stores final LLM/fallback decision.
- `artifacts/backtesting/*` stores standalone offline backtesting results, metrics, and analytics.

## 7) Key Concepts Explained

- **Day-ahead price**: energy price for future delivery periods.
- **Demand forecast**: expected load in `kW`.
- **Renewable forecast**: expected renewable generation in `MW`.
- **Imbalance**: `predicted_demand - predicted_renewables` (converted units).
- **Walk-forward validation**: train on earlier time periods and test on later ones (time-safe).
- **Model selection**: choose one model per run (`xgboost`, `lstm`, `prophet`).
- **Backtest**: test strategy on historical data before live deployment.
- **LLM fallback**: deterministic logic keeps the system running if API fails.

## 8) Example Outputs

Example backtest metrics:

```json
{
  "sharpe_ratio": 0.82,
  "max_drawdown": -0.11,
  "hit_rate": 0.54,
  "total_pnl": 1325.44
}
```

Example decision report fields:

- `decision`: LONG / SHORT / HOLD
- `reasoning`: explanation text
- `risk_assessment`: risk note
- `confidence`: value between 0 and 1
- `source`: `huggingface` or `deterministic_fallback`

Example isolated backtesting review fields:

- `future_price_change_eur_mwh`
- `future_price_return`
- `directional_correct`
- `accuracy_status`
- `pnl_positive`

Example isolated backtesting analytics:

- `directional_accuracy`
- `pnl_positive_rate`
- `accuracy_horizon_steps`
- `hold_tolerance_pct`

## 9) Troubleshooting

- **No ENTSO-E key or API error**: this is okay; synthetic mode is automatic.
- **No HF token or timeout**: this is okay; deterministic fallback is automatic.
- **Import errors**: confirm virtual environment is active and dependencies installed.
- **`Importing plotly failed. Interactive plots will not work.`**: reinstall dependencies with `pip install -r requirements.txt` inside the active virtual environment.
- **No output files**: check logs and confirm write permissions.
- **LSTM/Prophet install issues**: run `pip install -r requirements.txt` and ensure your Python version is compatible with `torch` and `prophet`.

## 10) How To Extend

- Add more markets/zones and compare model performance.
- Add battery/storage optimization logic.
- Add probabilistic forecasting (prediction intervals).
- Add portfolio constraints and risk limits.
- Replace prompt-only LLM with structured tool-calling.

## Environment Variables Summary

Secrets (never hardcode in code):

- `ENTSOE_API_KEY`
- `HF_TOKEN`

Optional:

- `HF_MODEL` (default: `Qwen/Qwen2.5-72B-Instruct`)
- `LOOKBACK_DAYS` (default: `180`)
- `ENTSOE_BIDDING_ZONE` (default: `DE_LU`)
