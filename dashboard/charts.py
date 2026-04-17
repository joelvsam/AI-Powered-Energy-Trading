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
