"""Chart helpers for Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_history_charts(df: pd.DataFrame) -> None:
    st.subheader("Market History")
    st.line_chart(df.set_index("timestamp_utc")[["price_eur_mwh", "demand_kw", "renewable_mw"]])


def render_prediction_chart(scored_df: pd.DataFrame) -> None:
    st.subheader("Actual vs Predicted")
    view = scored_df.set_index("timestamp_utc")[
        [
            "demand_kw",
            "pred_demand_kw",
            "renewable_mw",
            "pred_renewable_mw",
            "price_eur_mwh",
            "pred_price_eur_mwh",
        ]
    ]
    st.line_chart(view)


def render_backtest_chart(backtest_df: pd.DataFrame) -> None:
    st.subheader("Backtest Performance")
    view = backtest_df.set_index("timestamp_utc")[["cumulative_returns", "position"]]
    st.line_chart(view)


def render_equity_chart(review_df: pd.DataFrame) -> None:
    st.subheader("Equity and Returns")
    view = review_df.set_index("timestamp_utc")[["equity_curve", "cumulative_returns"]]
    st.line_chart(view)


def render_review_accuracy_chart(review_df: pd.DataFrame) -> None:
    st.subheader("Decision Accuracy Over Time")
    accuracy_view = review_df.copy().sort_values("timestamp_utc")
    accuracy_view["directional_correct_numeric"] = accuracy_view["directional_correct"].astype(int)
    accuracy_view["pnl_positive_numeric"] = accuracy_view["pnl_positive"].astype(int)
    rolling = accuracy_view.set_index("timestamp_utc")[["directional_correct_numeric", "pnl_positive_numeric"]].rolling(12, min_periods=1).mean()
    st.line_chart(rolling.rename(columns={"directional_correct_numeric": "directional_accuracy", "pnl_positive_numeric": "positive_pnl_rate"}))


def render_decision_review_table(review_df: pd.DataFrame) -> None:
    st.subheader("Past Decision Review")
    columns = [
        "timestamp_utc",
        "decision",
        "pred_price_eur_mwh",
        "price_eur_mwh",
        "future_price_change_eur_mwh",
        "future_price_return",
        "directional_correct",
        "accuracy_status",
        "pnl",
        "pnl_positive",
    ]
    present_columns = [column for column in columns if column in review_df.columns]
    st.dataframe(review_df[present_columns], use_container_width=True)
