"""Streamlit dashboard for the energy trading workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.backtesting_review import (
    DEFAULT_BACKTESTING_DIR,
    build_review_dataset,
    default_scored_csv_path,
    filter_review_dataset,
    load_backtest_artifacts,
    load_model_comparison,
    run_backtest_from_csv,
    run_model_comparison_workflow,
)
from dashboard.charts import (
    render_backtest_chart,
    render_decision_review_table,
    render_equity_chart,
    render_history_charts,
    render_prediction_chart,
    render_review_accuracy_chart,
)
from scripts.run_all import run_workflow


st.set_page_config(page_title="Energy Trading Dashboard", layout="wide")


def _mode_label(value: str, mapping: dict[str, tuple[str, str]]) -> tuple[str, str]:
    normalized = (value or "").strip().lower()
    return mapping.get(normalized, (value or "Unknown", "secondary"))


def _render_environment_diagnostics() -> None:
    with st.expander("Environment Diagnostics"):
        st.write(
            {
                "python_executable": sys.executable,
                "project_root": str(Path(__file__).parent.parent.resolve()),
                "entsoe_api_key_loaded": bool(os.getenv("ENTSOE_API_KEY")),
                "hf_token_loaded": bool(os.getenv("HF_TOKEN")),
            }
        )


def _render_pipeline_page() -> None:
    st.title("Energy Trading Dashboard")
    st.caption("Choose region, training window, and model; then run the full volatility-aware trading pipeline.")
    _render_environment_diagnostics()

    region = st.selectbox("Region", options=["DE_LU", "FR", "NL"], index=0)
    lookback_days = st.selectbox("Training Window (days)", options=[90, 180, 365], index=1)
    model = st.selectbox("Model", options=["xgboost", "lstm", "prophet"], index=0)
    horizon = st.slider("Simulation Horizon", min_value=12, max_value=168, value=24, step=12)
    skip_model_comparison = st.checkbox("Skip full model comparison", value=False, help="Run only the selected model workflow and skip the cross-model comparison pass.")

    result = st.session_state.get("pipeline_result")
    if st.button("Run Pipeline", type="primary"):
        args = argparse.Namespace(
            zone=region,
            lookback_days=lookback_days,
            simulation_horizon=horizon,
            model=model,
            skip_model_comparison=skip_model_comparison,
        )
        with st.spinner("Running pipeline... this may take a while."):
            result = run_workflow(args)
        st.session_state["pipeline_result"] = result

    if not result:
        st.info("Run the pipeline to see model metrics, signals, and charts.")
        return

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
        runtime_modes.get("research_source", "unknown"),
        {
            "huggingface": ("LLM research analysis active", "primary"),
            "deterministic_fallback": ("Deterministic research fallback active", "warning"),
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
        st.metric("Research Agent Mode", llm_label)
        st.caption(f"Model: `{runtime_modes.get('llm_model', 'unknown')}`")
        if llm_type == "primary":
            st.success("The Hugging Face LLM produced the research summary.")
        elif llm_type == "warning":
            st.warning("The pipeline fell back safely because the LLM was unavailable.")
        else:
            st.info("Research agent mode could not be determined.")

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
    recommended_decision = latest_signal.get("recommended_decision", latest_signal["decision"])
    st.subheader("Latest Trading Signal")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Recommended Action", str(recommended_decision))
    s2.metric("Predicted Price", f"{latest_signal['pred_price_eur_mwh']:.2f} EUR/MWh")
    s3.metric(
        "Current Price",
        f"{latest_signal['price_eur_mwh']:.2f} EUR/MWh",
        delta=f"{latest_signal['pred_price_eur_mwh'] - latest_signal['price_eur_mwh']:.2f} EUR/MWh",
    )
    s4.metric("Executed Position", f"{latest_signal['position']:.3f}")
    s5.metric("Predicted Imbalance", f"{latest_signal['imbalance_pred']:.2f} MW")

    strategy_comparison = result.get("strategy_comparison", {})
    if strategy_comparison and "summary_df" in strategy_comparison:
        st.subheader("Strategy Comparison")
        st.dataframe(strategy_comparison["summary_df"], use_container_width=True)
        st.dataframe(strategy_comparison["significance_df"], use_container_width=True)

    render_history_charts(features_df.tail(300))
    render_prediction_chart(scored_df.tail(300))
    render_backtest_chart(backtest_df.tail(300))

    st.subheader("Research Summary")
    st.json(result["research_summary"]["summary"])


def _render_review_details(review_df: pd.DataFrame, metrics: dict[str, object], analytics: dict[str, object], *, horizon_steps: int, hold_tolerance_decimal: float) -> None:
    review_df, review_summary = build_review_dataset(
        review_df,
        horizon_steps=horizon_steps,
        hold_tolerance_pct=hold_tolerance_decimal,
    )

    timestamps = pd.to_datetime(review_df["timestamp_utc"])
    min_date = timestamps.min().date()
    max_date = timestamps.max().date()
    date_range = st.date_input("Filter date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = filter_review_dataset(review_df, start_date=date_range[0], end_date=date_range[1])
    else:
        filtered_df = review_df

    if filtered_df.empty:
        st.warning("No rows match the selected date range.")
        return

    st.subheader("Backtesting Scorecard")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sharpe", f"{float(metrics.get('sharpe_ratio', 0.0)):.3f}")
    m2.metric("Max Drawdown", f"{float(metrics.get('max_drawdown', 0.0)):.3f}")
    m3.metric("Hit Rate", f"{float(metrics.get('hit_rate', 0.0)):.3f}")
    m4.metric("Total PnL", f"{float(metrics.get('total_pnl', 0.0)):.2f}")
    m5.metric("Trade Count", f"{int(metrics.get('trade_count', 0))}")
    m6.metric("Avg Trade Return", f"{float(metrics.get('average_trade_return', 0.0)):.4f}")

    st.subheader("Decision Accuracy Review")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Directional Accuracy", f"{review_summary['directional_accuracy']:.3f}")
    a2.metric("Positive PnL Rate", f"{review_summary['pnl_positive_rate']:.3f}")
    a3.metric("Correct Decisions", f"{review_summary['correct_count']}")
    a4.metric("Incorrect Decisions", f"{review_summary['incorrect_count']}")
    a5.metric("Evaluable Rows", f"{review_summary['evaluable_rows']}")

    distribution = review_summary["decision_distribution"]
    d1, d2, d3 = st.columns(3)
    d1.metric("LONG Count", f"{distribution['LONG']}")
    d2.metric("SHORT Count", f"{distribution['SHORT']}")
    d3.metric("HOLD Count", f"{distribution['HOLD']}")

    st.caption(
        f"Accuracy horizon: {review_summary['accuracy_horizon_steps']} step(s). HOLD tolerance band: {review_summary['hold_tolerance_pct'] * 100:.2f}%."
    )

    render_prediction_chart(filtered_df.tail(300))
    render_equity_chart(filtered_df.tail(300))
    render_review_accuracy_chart(filtered_df.tail(300))
    render_decision_review_table(filtered_df.sort_values("timestamp_utc", ascending=False))

    with st.expander("Accuracy Metadata"):
        st.json(
            {
                "metrics": metrics,
                "analytics": analytics,
                "review_summary": review_summary,
            }
        )


def _render_backtesting_review_page() -> None:
    st.title("Backtesting Review")
    st.caption("Review isolated backtest scores and compare past decisions against realized outcomes.")
    _render_environment_diagnostics()

    st.sidebar.subheader("Backtesting Review Controls")
    load_mode = st.sidebar.radio(
        "Review data source",
        options=["Latest saved artifacts", "Run from scored CSV", "Compare trained models"],
        index=0,
    )
    horizon_label = st.sidebar.selectbox("Accuracy horizon", options=["Next period", "Next 24 hours"], index=0)
    hold_tolerance_pct = st.sidebar.slider(
        "HOLD tolerance band (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
        help="A HOLD is counted as directionally correct when the realized move stays within this percent band.",
    )
    horizon_steps = 1 if horizon_label == "Next period" else 24
    hold_tolerance_decimal = hold_tolerance_pct / 100.0

    review_df: pd.DataFrame | None = None
    metrics: dict[str, object] = {}
    analytics: dict[str, object] = {}

    if load_mode == "Latest saved artifacts":
        st.write(f"Loading isolated backtesting artifacts from `{DEFAULT_BACKTESTING_DIR}`.")
        if st.button("Load Latest Backtest", type="primary"):
            try:
                review_df, metrics, analytics = load_backtest_artifacts()
                st.session_state["review_payload"] = (review_df, metrics, analytics)
                st.success("Loaded isolated backtesting artifacts.")
            except FileNotFoundError as exc:
                st.warning(str(exc))
        elif "review_payload" in st.session_state:
            review_df, metrics, analytics = st.session_state["review_payload"]
    elif load_mode == "Run from scored CSV":
        input_path = st.text_input("Scored CSV path", value=str(default_scored_csv_path()))
        st.caption(
            "Use a scored predictions CSV such as `artifacts/models/scored_predictions_<model>.csv` "
            "or `artifacts/simulation/backtest_trades.csv`. Do not use `artifacts/backtesting/backtest_results.csv` "
            "as the source unless it already exists from a previous isolated run."
        )
        output_dir = st.text_input("Output directory", value=str(DEFAULT_BACKTESTING_DIR))
        c1, c2, c3 = st.columns(3)
        with c1:
            transaction_cost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, value=5.0, step=1.0)
        with c2:
            annualization_factor = st.number_input("Annualization factor", min_value=1, value=24, step=1)
        with c3:
            notional_eur = st.number_input("Notional EUR", min_value=1000.0, value=10000.0, step=1000.0)
        if st.button("Run Isolated Backtest", type="primary"):
            try:
                review_df, metrics, analytics = run_backtest_from_csv(
                    input_path,
                    output_dir=output_dir,
                    transaction_cost_bps=float(transaction_cost_bps),
                    annualization_factor=int(annualization_factor),
                    notional_eur=float(notional_eur),
                    accuracy_horizon_steps=horizon_steps,
                    hold_tolerance_pct=hold_tolerance_decimal,
                )
                st.session_state["review_payload"] = (review_df, metrics, analytics)
                st.success("Isolated backtest completed and saved.")
            except (FileNotFoundError, ValueError, KeyError) as exc:
                st.error(f"Unable to run isolated backtest: {exc}")
        elif "review_payload" in st.session_state:
            review_df, metrics, analytics = st.session_state["review_payload"]
    else:
        st.write("Train xgboost, lstm, and prophet on the same dataset, then compare them by directional accuracy.")
        output_dir = st.text_input("Comparison output directory", value=str(DEFAULT_BACKTESTING_DIR))
        c1, c2 = st.columns(2)
        with c1:
            comparison_region = st.selectbox("Comparison region", options=["DE_LU", "FR", "NL"], index=0)
        with c2:
            comparison_lookback_days = st.selectbox("Comparison training window (days)", options=[90, 180, 365], index=1)
        c1, c2, c3 = st.columns(3)
        with c1:
            transaction_cost_bps = st.number_input("Transaction cost (bps)", min_value=0.0, value=5.0, step=1.0, key="comparison_tcost")
        with c2:
            annualization_factor = st.number_input("Annualization factor", min_value=1, value=24, step=1, key="comparison_annualization")
        with c3:
            notional_eur = st.number_input("Notional EUR", min_value=1000.0, value=10000.0, step=1000.0, key="comparison_notional")
        comparison_summary_df: pd.DataFrame | None = None
        comparison_metadata: dict[str, object] = {}

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Run Model Comparison", type="primary"):
                try:
                    comparison_summary_df, comparison_metadata = run_model_comparison_workflow(
                        zone=comparison_region,
                        lookback_days=int(comparison_lookback_days),
                        output_dir=output_dir,
                        transaction_cost_bps=float(transaction_cost_bps),
                        annualization_factor=int(annualization_factor),
                        notional_eur=float(notional_eur),
                        accuracy_horizon_steps=horizon_steps,
                        hold_tolerance_pct=hold_tolerance_decimal,
                    )
                    st.session_state["comparison_payload"] = (comparison_summary_df, comparison_metadata)
                    st.success("Model comparison completed and saved.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unable to run model comparison: {exc}")
        with c2:
            if st.button("Load Saved Comparison"):
                try:
                    comparison_summary_df, comparison_metadata = load_model_comparison(output_dir)
                    st.session_state["comparison_payload"] = (comparison_summary_df, comparison_metadata)
                    st.success("Loaded saved model comparison.")
                except FileNotFoundError as exc:
                    st.warning(str(exc))

        if comparison_summary_df is None and "comparison_payload" in st.session_state:
            comparison_summary_df, comparison_metadata = st.session_state["comparison_payload"]

        if comparison_summary_df is None or comparison_summary_df.empty:
            st.info("Run the three-model comparison or load a saved comparison to see ranked results.")
            return

        winner_model = comparison_metadata.get("winner_model")
        if winner_model:
            st.success(f"Top model by trading performance: {winner_model}")
        if comparison_metadata.get("energy_source"):
            st.caption(
                f"Region: {comparison_metadata.get('zone')} | Training window: {comparison_metadata.get('lookback_days')} days | "
                f"Energy source: {comparison_metadata.get('energy_source')}"
            )
        comparison_columns = [
            "rank",
            "model_key",
            "strategy_name",
            "sharpe_ratio",
            "total_pnl",
            "directional_accuracy",
            "pnl_positive_rate",
            "max_drawdown",
            "drawdown_duration_steps",
            "price_mae",
            "price_rmse",
            "significant_vs_persistence",
        ]
        st.subheader("Model Comparison")
        st.dataframe(
            comparison_summary_df[[column for column in comparison_columns if column in comparison_summary_df.columns]],
            use_container_width=True,
        )
        failure_rows = comparison_metadata.get("failures", [])
        if failure_rows:
            st.warning(f"Some models failed during comparison: {failure_rows}")

        selected_model = st.selectbox("Inspect model", options=list(comparison_summary_df["model_key"]))
        selected_row = comparison_summary_df.loc[comparison_summary_df["model_key"] == selected_model].iloc[0]
        upgraded_strategy_dir = Path(str(selected_row["research_output_dir"])) / "upgraded_strategy"
        review_df, metrics, analytics = load_backtest_artifacts(upgraded_strategy_dir)
        training_metrics_path = Path(str(selected_row["training_metrics_path"]))
        if training_metrics_path.exists():
            training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
            price_metrics = training_metrics.get("price", {})
            st.subheader("Forecast Metrics")
            f1, f2 = st.columns(2)
            f1.metric("Price MAE", f"{float(price_metrics.get('mae', float('nan'))):.3f}")
            f2.metric("Price RMSE", f"{float(price_metrics.get('rmse', float('nan'))):.3f}")
        _render_review_details(review_df, metrics, analytics, horizon_steps=horizon_steps, hold_tolerance_decimal=hold_tolerance_decimal)
        return

    if review_df is None:
        st.info(
            "Load the latest isolated artifacts or run an isolated backtest from a scored CSV. "
            "If this is your first isolated run, a good starting input is `artifacts/simulation/backtest_trades.csv`."
        )
        return

    _render_review_details(review_df, metrics, analytics, horizon_steps=horizon_steps, hold_tolerance_decimal=hold_tolerance_decimal)


page = st.sidebar.radio("Menu", options=["Pipeline", "Backtesting Review"], index=0)
if page == "Pipeline":
    _render_pipeline_page()
else:
    _render_backtesting_review_page()
