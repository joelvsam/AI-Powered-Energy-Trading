"""Streamlit dashboard for the energy trading workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.charts import render_backtest_chart, render_history_charts, render_prediction_chart
from scripts.run_all import run_workflow

st.set_page_config(page_title="Energy Trading Dashboard", layout="wide")
st.title("Energy Trading Dashboard")
st.caption("Choose region, training window, and model; then run full pipeline.")


def _mode_label(value: str, mapping: dict[str, tuple[str, str]]) -> tuple[str, str]:
    normalized = (value or "").strip().lower()
    return mapping.get(normalized, (value or "Unknown", "secondary"))

region = st.selectbox("Region", options=["DE_LU", "FR", "NL"], index=0)
lookback_days = st.selectbox("Training Window (days)", options=[90, 180, 365], index=1)
model = st.selectbox("Model", options=["xgboost", "lstm", "prophet"], index=0)
horizon = st.slider("Simulation Horizon", min_value=12, max_value=168, value=24, step=12)

if st.button("Run Pipeline", type="primary"):
    args = argparse.Namespace(zone=region, lookback_days=lookback_days, simulation_horizon=horizon, model=model)
    with st.spinner("Running pipeline... this may take a while."):
        result = run_workflow(args)

    st.success("Pipeline completed")
    st.write(
        {
            "selected_region": result["config"]["zone"],
            "selected_training_window_days": result["config"]["lookback_days"],
            "selected_model": result["config"]["model"],
        }
    )

    runtime_modes = result.get("runtime_modes", {})
    energy_label, energy_type = _mode_label(
        runtime_modes.get("energy_source", "unknown"),
        {
            "entsoe": ("ENTSO-E live data", "primary"),
            "synthetic": ("Synthetic data fallback", "warning"),
        },
    )
    llm_label, llm_type = _mode_label(
        runtime_modes.get("decision_source", "unknown"),
        {
            "huggingface": ("LLM decision active", "primary"),
            "huggingface_non_json": ("LLM used, non-JSON response", "warning"),
            "deterministic_fallback": ("Deterministic fallback active", "warning"),
        },
    )

    st.subheader("Runtime Modes")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Energy Data Mode", energy_label)
        st.caption(f"Source key: `{runtime_modes.get('energy_source', 'unknown')}`")
        if energy_type == "primary":
            st.success("ENTSO-E API data was used for this run.")
        elif energy_type == "warning":
            st.warning("ENTSO-E was unavailable, so the pipeline switched to synthetic data.")
        else:
            st.info("Energy data mode could not be determined.")
    with m2:
        st.metric("Decision Engine Mode", llm_label)
        st.caption(f"Model: `{runtime_modes.get('llm_model', 'unknown')}`")
        if llm_type == "primary":
            st.success("The Hugging Face LLM produced the final decision.")
        elif llm_type == "warning":
            st.warning("The pipeline fell back safely because the LLM response was unusable or unavailable.")
        else:
            st.info("Decision engine mode could not be determined.")

    metrics_path = Path(result["metrics_path"])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        price_metrics = metrics.get("price", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Demand MAE", f"{metrics['demand']['mae']:.2f}")
        c2.metric("Demand RMSE", f"{metrics['demand']['rmse']:.2f}")
        c3.metric("Renewable MAE", f"{metrics['renewable']['mae']:.2f}")
        c4.metric("Renewable RMSE", f"{metrics['renewable']['rmse']:.2f}")
        c5.metric("Price MAE", f"{price_metrics.get('mae', float('nan')):.2f}")
        c6.metric("Price RMSE", f"{price_metrics.get('rmse', float('nan')):.2f}")

    backtest_metrics_path = Path("artifacts/simulation/backtest_metrics.json")
    if backtest_metrics_path.exists():
        bt = json.loads(backtest_metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", f"{bt['sharpe_ratio']:.3f}")
        c2.metric("Max Drawdown", f"{bt['max_drawdown']:.3f}")
        c3.metric("Hit Rate", f"{bt['hit_rate']:.3f}")
        c4.metric("Total PnL", f"{bt['total_pnl']:.2f}")

    features_df: pd.DataFrame = result["features_df"]
    scored_df: pd.DataFrame = result["scored_df"]
    backtest_df: pd.DataFrame = result["backtest_df"]

    latest_signal = backtest_df.sort_values("timestamp_utc").iloc[-1]
    st.subheader("Latest Trading Signal")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Recommended Action", str(latest_signal["decision"]))
    s2.metric("Predicted Price", f"{latest_signal['pred_price_eur_mwh']:.2f} EUR/MWh")
    s3.metric(
        "Current Price",
        f"{latest_signal['price_eur_mwh']:.2f} EUR/MWh",
        delta=f"{latest_signal['pred_price_eur_mwh'] - latest_signal['price_eur_mwh']:.2f} EUR/MWh",
    )
    s4.metric("Predicted Imbalance", f"{latest_signal['imbalance_pred']:.2f} MW")

    render_history_charts(features_df.tail(300))
    render_prediction_chart(scored_df.tail(300))
    render_backtest_chart(backtest_df.tail(300))

    st.subheader("Decision Report")
    st.json(result["decision_report"])
