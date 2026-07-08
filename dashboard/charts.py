"""Chart helpers for Streamlit dashboard.

Plotly figures themed for the dark bento layout. Series colors are the
dark-mode categorical slots of a CVD-validated reference palette, assigned
in fixed order and kept stable per entity across every chart:
price=blue, demand=aqua, renewables=yellow, position=violet.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


SERIES_COLORS = {
    "price": "#3987e5",
    "demand": "#199e70",
    "renewable": "#c98500",
    "position": "#9085e9",
    "rate_primary": "#3987e5",
    "rate_secondary": "#199e70",
}

_INK = "#c3c6d7"
_MUTED = "#8d90a0"
_GRID = "rgba(67, 70, 85, 0.35)"
_AXIS = "#434655"
_HOVER_BG = "#272a2c"

_RANGE_SELECTOR = dict(
    buttons=[
        dict(count=24, label="24h", step="hour", stepmode="backward"),
        dict(count=7, label="7d", step="day", stepmode="backward"),
        dict(count=30, label="30d", step="day", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    # Top-right so the buttons never collide with the legend at top-left.
    x=1.0,
    xanchor="right",
    y=1.02,
    yanchor="bottom",
    bgcolor="rgba(29, 32, 34, 0.9)",
    activecolor="#2563eb",
    bordercolor=_AXIS,
    borderwidth=1,
    font=dict(color=_INK, size=11),
)

_PLOTLY_CONFIG = {"displayModeBar": False}


def _apply_theme(fig: go.Figure, *, height: int, show_legend: bool = True, with_range_selector: bool = True) -> go.Figure:
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, "Segoe UI", Roboto, Arial, sans-serif', color=_INK, size=12),
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=_HOVER_BG, bordercolor=_AXIS, font=dict(color="#e0e3e5", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=_AXIS,
        tickcolor=_AXIS,
        tickfont=dict(color=_MUTED, size=11),
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor=_MUTED,
    )
    fig.update_yaxes(
        gridcolor=_GRID,
        griddash="dot",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=_MUTED, size=11),
        title_font=dict(color=_MUTED, size=11),
        title_standoff=6,
    )
    fig.update_xaxes(
        showticklabels=True,
        title_font=dict(color=_MUTED, size=11),
        title_standoff=8,
    )
    fig.update_annotations(font=dict(color=_INK, size=12))
    if with_range_selector:
        fig.update_layout(xaxis=dict(rangeselector=_RANGE_SELECTOR))
    return fig


def _line(
    x: pd.Series,
    y: pd.Series,
    name: str,
    color: str,
    *,
    dash: str | None = None,
    width: float = 2.0,
    shape: str = "linear",
    fill: str | None = None,
    fillcolor: str | None = None,
    hovertemplate: str = "%{y:,.2f}",
    showlegend: bool = True,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line=dict(color=color, width=width, dash=dash, shape=shape),
        fill=fill,
        fillcolor=fillcolor,
        hovertemplate=hovertemplate + "<extra>" + name + "</extra>",
        showlegend=showlegend,
    )


def render_history_charts(df: pd.DataFrame) -> None:
    st.subheader("Market History")
    view = df.sort_values("timestamp_utc")
    x = view["timestamp_utc"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Price", "Demand", "Renewables"),
    )
    fig.add_trace(
        _line(x, view["price_eur_mwh"], "Price", SERIES_COLORS["price"], hovertemplate="%{y:,.2f} EUR/MWh"),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(x, view["demand_kw"] / 1000.0, "Demand", SERIES_COLORS["demand"], hovertemplate="%{y:,.0f} MW"),
        row=2,
        col=1,
    )
    fig.add_trace(
        _line(x, view["renewable_mw"], "Renewables", SERIES_COLORS["renewable"], hovertemplate="%{y:,.0f} MW"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MW", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    _apply_theme(fig, height=560, show_legend=False)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


def render_prediction_chart(scored_df: pd.DataFrame) -> None:
    st.subheader("Actual vs Predicted")
    view = scored_df.sort_values("timestamp_utc")
    x = view["timestamp_utc"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Price", "Demand", "Renewables"),
    )
    pairs = [
        (1, "Price", "price_eur_mwh", "pred_price_eur_mwh", SERIES_COLORS["price"], "%{y:,.2f} EUR/MWh"),
        (2, "Demand", "demand_kw", "pred_demand_kw", SERIES_COLORS["demand"], "%{y:,.0f} MW"),
        (3, "Renewables", "renewable_mw", "pred_renewable_mw", SERIES_COLORS["renewable"], "%{y:,.0f} MW"),
    ]
    for row, metric_name, actual_col, pred_col, color, hover in pairs:
        actual = view[actual_col] / 1000.0 if actual_col == "demand_kw" else view[actual_col]
        predicted = view[pred_col] / 1000.0 if pred_col == "pred_demand_kw" else view[pred_col]
        # Solid = actual, dashed = predicted, one legend pair per metric.
        fig.add_trace(
            _line(x, actual, f"{metric_name} (actual)", color, hovertemplate=hover),
            row=row,
            col=1,
        )
        fig.add_trace(
            _line(x, predicted, f"{metric_name} (predicted)", color, dash="dash", width=1.6, hovertemplate=hover),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MW", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    _apply_theme(fig, height=620)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


def render_backtest_chart(backtest_df: pd.DataFrame) -> None:
    st.subheader("Backtest Performance")
    view = backtest_df.sort_values("timestamp_utc")
    x = view["timestamp_utc"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Cumulative Return", "Position"),
        row_heights=[0.6, 0.4],
    )
    fig.add_trace(
        _line(
            x,
            view["cumulative_returns"],
            "Cumulative return",
            SERIES_COLORS["price"],
            fill="tozeroy",
            fillcolor="rgba(57, 135, 229, 0.12)",
            hovertemplate="%{y:.2%}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(x, view["position"], "Position", SERIES_COLORS["position"], shape="hv", width=1.6, hovertemplate="%{y:.3f}"),
        row=2,
        col=1,
    )
    fig.update_yaxes(tickformat=".1%", title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(range=[-1.08, 1.08], title_text="Position (-1 to 1)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    _apply_theme(fig, height=460, show_legend=False)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


def render_equity_chart(review_df: pd.DataFrame) -> None:
    st.subheader("Equity and Returns")
    view = review_df.sort_values("timestamp_utc")
    x = view["timestamp_utc"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Equity Curve (multiple of start)", "Cumulative Return"),
    )
    fig.add_trace(
        _line(x, view["equity_curve"], "Equity", SERIES_COLORS["price"], hovertemplate="%{y:,.4f}x"),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            x,
            view["cumulative_returns"],
            "Cumulative return",
            SERIES_COLORS["price"],
            fill="tozeroy",
            fillcolor="rgba(57, 135, 229, 0.12)",
            hovertemplate="%{y:.2%}",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Multiple of start", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    _apply_theme(fig, height=460, show_legend=False)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


def render_review_accuracy_chart(review_df: pd.DataFrame) -> None:
    st.subheader("Decision Accuracy Over Time")
    accuracy_view = review_df.copy().sort_values("timestamp_utc")
    accuracy_view["directional_correct_numeric"] = accuracy_view["directional_correct"].astype(int)
    accuracy_view["pnl_positive_numeric"] = accuracy_view["pnl_positive"].astype(int)
    rolling = (
        accuracy_view.set_index("timestamp_utc")[["directional_correct_numeric", "pnl_positive_numeric"]]
        .rolling(12, min_periods=1)
        .mean()
    )
    x = rolling.index
    fig = go.Figure()
    fig.add_trace(
        _line(x, rolling["directional_correct_numeric"], "Directional accuracy", SERIES_COLORS["rate_primary"], hovertemplate="%{y:.1%}")
    )
    fig.add_trace(
        _line(x, rolling["pnl_positive_numeric"], "Positive PnL rate", SERIES_COLORS["rate_secondary"], hovertemplate="%{y:.1%}")
    )
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title_text="Rolling 12h rate (%)")
    fig.update_xaxes(title_text="Time (UTC)")
    _apply_theme(fig, height=340)
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)


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
    st.dataframe(review_df[present_columns], width="stretch")
