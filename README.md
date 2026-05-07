# Research-Grade Energy Trading

This repository is now organized as a trading research system, not a generic ML demo. The core question is:

Why should this strategy make money in real European power markets after realistic execution costs, delays, and baseline comparisons?

The pipeline answers that question by combining:

- walk-forward forecasting with `xgboost`, `lstm`, and `prophet`
- market-structure-aware features for day-ahead, intraday, renewables, and imbalance context
- multiple signal families instead of a single forecast delta
- realistic backtesting with spread, slippage, execution delay, position caps, and liquidity limits
- mandatory baseline comparisons and bootstrap significance tests
- model diagnostics tied to trading performance
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

## What Changed

The repo now includes:

- richer ENTSO-E market context:
  - day-ahead prices
  - intraday prices when available
  - imbalance prices when available
  - intraday renewable forecast when available
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
2. feature engineering
3. anomaly review
4. primary model training
5. realistic backtest
6. strategy comparison versus baselines
7. cross-model trading comparison
8. final research-note generation

## Running It

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set optional secrets in `.env`:

- `ENTSOE_API_KEY`
- `HF_TOKEN`

Run the full research workflow:

```bash
python -m scripts.run_all --lookback-days 180 --zone DE_LU --model xgboost
```

Run cross-model comparison directly:

```bash
python scripts/run_model_comparison.py --output-dir artifacts/research/model_comparison
```

Run isolated backtesting from an existing scored file:

```bash
python scripts/run_backtest.py --input-path artifacts/models/scored_predictions_xgboost.csv --output-dir artifacts/backtesting
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Artifact Map

The canonical research bundle lives under `artifacts/research/`.

Important outputs include:

- `artifacts/research/anomaly_review.json`
- `artifacts/research/research_summary.json`
- `artifacts/research/research_note.md`
- `artifacts/research/selected_strategy_metrics_<model>.csv`
- `artifacts/research/selected_strategy_significance_<model>.csv`
- `artifacts/research/model_comparison/model_comparison_summary.csv`
- `artifacts/research/model_comparison/model_comparison_summary.json`

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

- walk-forward scoring instead of in-sample evaluation
- no same-bar execution lookahead
- no fake performance inflation
- no reliance on a cosmetic LLM report
- no test-set tuning of signal parameters

## Important Caveat

If ENTSO-E ingestion fails, the pipeline can fall back to synthetic energy data. That keeps the repo runnable, but those runs are not research-grade evidence for a real trading strategy. The research outputs mark this explicitly.

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
  simulation/
  trading/
scripts/
  run_all.py
  run_backtest.py
  run_model_comparison.py
dashboard/
  app.py
```

## Verification

Current regression coverage includes:

```bash
python -m unittest tests.test_backtesting tests.test_dashboard_backtesting_review
```

This ensures the upgraded engine, comparison flow, and dashboard review helpers remain functional while the repository evolves into a research-first trading system.
