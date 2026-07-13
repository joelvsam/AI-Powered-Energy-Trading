"""Streamlit dashboard for the energy trading workflow."""

from __future__ import annotations

import argparse
import json
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
from src.config import AppConfig


st.set_page_config(page_title="Energy Trading Dashboard", layout="wide")


def _render_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --quant-bg: #0b0f10;
            --quant-surface: #101415;
            --quant-sidebar: #191c1e;
            --quant-card: #1d2022;
            --quant-card-high: #272a2c;
            --quant-border: #323537;
            --quant-border-soft: #434655;
            --quant-text: #e0e3e5;
            --quant-muted: #c3c6d7;
            --quant-subtle: #8d90a0;
            --quant-primary: #b4c5ff;
            --quant-primary-strong: #2563eb;
            --quant-green: #34d399;
            --quant-red: #ffb4ab;
            --quant-warning: #fbbf24;
            --quant-radius: 8px;
            --quant-mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
            --quant-sans: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        }

        /* Page background lives on body so the animation canvas (z-index -1)
           can paint above it while staying behind all app content. */
        html, body {
            background: #0b0f10;
        }

        .stApp {
            background: transparent;
            color: var(--quant-text);
            font-family: var(--quant-sans);
        }

        /* Background animation component: fixed full-viewport, behind content,
           never intercepting the pointer. The app's only iframe is this one. */
        .stApp [data-testid="stIFrame"] iframe,
        .stApp iframe[data-testid="stIFrame"],
        .stApp iframe[srcdoc] {
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: -1 !important;
            pointer-events: none !important;
            background: transparent !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: var(--quant-sidebar);
            border-right: 1px solid var(--quant-border-soft);
        }

        [data-testid="stSidebar"] * {
            color: var(--quant-text);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaptionContainer {
            color: var(--quant-muted);
        }

        h1, h2, h3 {
            color: var(--quant-text);
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.15;
        }

        h2, h3 {
            border-bottom: 1px solid var(--quant-border);
            padding-bottom: 0.35rem;
        }

        p, span, label, div {
            letter-spacing: 0;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] p {
            color: var(--quant-muted);
        }

        [data-testid="stMetric"] {
            background: var(--quant-card);
            border: 1px solid var(--quant-border);
            border-radius: var(--quant-radius);
            padding: 1rem;
            min-height: 96px;
            box-shadow: none;
        }

        [data-testid="stMetricLabel"] {
            color: var(--quant-muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--quant-text);
            font-family: var(--quant-mono);
            font-weight: 600;
        }

        [data-testid="stMetricDelta"] {
            font-family: var(--quant-mono);
        }

        div[data-testid="stAlert"] {
            background: var(--quant-card);
            border: 1px solid var(--quant-border-soft);
            border-radius: var(--quant-radius);
            color: var(--quant-text);
        }

        div[data-testid="stExpander"] {
            background: var(--quant-surface);
            border: 1px solid var(--quant-border);
            border-radius: var(--quant-radius);
        }

        div[data-testid="stExpander"] summary {
            color: var(--quant-muted);
            font-weight: 600;
        }

        .stButton > button {
            background: var(--quant-primary-strong);
            border: 1px solid var(--quant-primary-strong);
            border-radius: var(--quant-radius);
            color: #eef0ff;
            font-weight: 700;
            min-height: 2.5rem;
        }

        .stButton > button:hover {
            background: #2f6ff2;
            border-color: var(--quant-primary);
            color: #ffffff;
        }

        .stButton > button:focus,
        .stButton > button:active {
            box-shadow: 0 0 0 2px rgba(180, 197, 255, 0.28);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            background: var(--quant-surface);
            border-color: var(--quant-border-soft);
            border-radius: 6px;
            color: var(--quant-text);
        }

        div[data-baseweb="select"] span,
        input,
        textarea {
            color: var(--quant-text);
            caret-color: var(--quant-primary);
        }

        div[data-baseweb="popover"] ul {
            background: var(--quant-card);
            border: 1px solid var(--quant-border-soft);
        }

        div[data-baseweb="popover"] li {
            color: var(--quant-text);
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
            color: var(--quant-muted);
        }

        [data-testid="stSlider"] [role="slider"] {
            background: var(--quant-primary);
            border-color: var(--quant-primary);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: var(--quant-card);
            border: 1px solid var(--quant-border);
            border-radius: var(--quant-radius);
            overflow: hidden;
        }

        [data-testid="stJson"] {
            background: var(--quant-surface);
            border: 1px solid var(--quant-border);
            border-radius: var(--quant-radius);
            padding: 0.75rem;
        }

        /* Bento grid cards: bordered containers become rounded tiles. */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--quant-card);
            border: 1px solid var(--quant-border);
            border-radius: 14px;
            padding: 1.1rem 1.2rem 1.2rem 1.2rem;
            height: 100%;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--quant-border-soft);
        }

        [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {
            height: 100%;
        }

        /* Card titles rendered as h5. */
        h5 {
            color: var(--quant-subtle);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border-bottom: none;
            margin-bottom: 0.15rem;
        }

        /* Metric tiles inside cards sit on the elevated surface and grow with
           their content instead of stretching to a fixed row height. */
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetric"] {
            background: var(--quant-card-high);
            min-height: 0;
            padding: 0.8rem;
            height: auto;
        }

        /* Never clip text inside metric tiles or cards: wrap instead. */
        [data-testid="stMetric"] {
            height: auto;
            overflow: visible;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-size: 1.3rem;
            line-height: 1.3;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: break-word;
        }

        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] li,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
            overflow-wrap: anywhere;
        }

        code, pre {
            color: var(--quant-primary);
            font-family: var(--quant-mono);
        }

        a {
            color: var(--quant-primary);
        }

        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--quant-bg);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--quant-border-soft);
            border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--quant-subtle);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_background_animation() -> None:
    """Full-viewport 'transmission grid pulse' canvas behind the app content.

    A faint jittered lattice evokes a transmission network; small bright pulses
    travel along its edges and hop to a connected edge at each junction. Edges
    near the pointer light up. Rendered via st.iframe (the app's
    only iframe), which the theme CSS fixes across the viewport at z-index -1
    with pointer-events disabled; mouse coordinates are read from the parent
    document so the animation reacts without ever intercepting the pointer.
    """
    st.iframe(
        """
<canvas id="bg"></canvas>
<style>html, body { margin: 0; padding: 0; background: transparent; overflow: hidden; }</style>
<script>
(function () {
    const canvas = document.getElementById("bg");
    const ctx = canvas.getContext("2d");
    let parentDoc = null;
    let parentWin = window;
    try {
        parentDoc = window.parent.document;
        parentWin = window.parent;
    } catch (err) {
        parentDoc = document;
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0;
    let h = 0;

    const SPACING = 96;
    const JITTER = 18;
    const MOUSE_DIST = 160;
    const COLORS = ["57, 135, 229", "52, 211, 153", "144, 133, 233"];

    let nodes = [];
    let edges = [];
    let nodeEdges = [];
    let pulses = [];

    function buildGrid() {
        nodes = [];
        edges = [];
        nodeEdges = [];
        pulses = [];
        const cols = Math.ceil(w / SPACING) + 2;
        const rows = Math.ceil(h / SPACING) + 2;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                nodes.push({
                    x: (c - 0.5) * SPACING + (Math.random() - 0.5) * 2 * JITTER,
                    y: (r - 0.5) * SPACING + (Math.random() - 0.5) * 2 * JITTER,
                });
            }
        }
        nodeEdges = nodes.map(function () { return []; });
        function addEdge(a, b) {
            if (Math.random() < 0.12) return;  // drop some edges for irregularity
            const index = edges.length;
            edges.push({ a: a, b: b });
            nodeEdges[a].push(index);
            nodeEdges[b].push(index);
        }
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const i = r * cols + c;
                if (c + 1 < cols) addEdge(i, i + 1);
                if (r + 1 < rows) addEdge(i, i + cols);
                if (c + 1 < cols && r + 1 < rows && Math.random() < 0.16) addEdge(i, i + cols + 1);
            }
        }
        const pulseCount = Math.max(8, Math.min(18, Math.floor((w * h) / 130000)));
        for (let p = 0; p < pulseCount; p++) {
            const e = Math.floor(Math.random() * edges.length);
            pulses.push({
                edge: e,
                from: Math.random() < 0.5 ? edges[e].a : edges[e].b,
                t: Math.random(),
                speed: 55 + Math.random() * 50,  // px per second
                color: COLORS[p % COLORS.length],
            });
        }
    }

    function resize() {
        w = parentDoc.documentElement.clientWidth || window.innerWidth;
        h = parentDoc.documentElement.clientHeight || window.innerHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        buildGrid();
    }
    resize();
    parentWin.addEventListener("resize", resize);

    const mouse = { x: -1e4, y: -1e4 };
    parentDoc.addEventListener("mousemove", function (e) {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });
    parentDoc.addEventListener("mouseleave", function () {
        mouse.x = -1e4;
        mouse.y = -1e4;
    });

    function segmentDistance(px, py, ax, ay, bx, by) {
        const dx = bx - ax;
        const dy = by - ay;
        const lengthSq = dx * dx + dy * dy;
        let t = lengthSq ? ((px - ax) * dx + (py - ay) * dy) / lengthSq : 0;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }

    function drawLattice() {
        for (const edge of edges) {
            const a = nodes[edge.a];
            const b = nodes[edge.b];
            let alpha = 0.055;
            const d = segmentDistance(mouse.x, mouse.y, a.x, a.y, b.x, b.y);
            if (d < MOUSE_DIST) {
                alpha += (1 - d / MOUSE_DIST) * 0.22;  // edges light up near the pointer
            }
            ctx.strokeStyle = "rgba(180, 197, 255, " + alpha.toFixed(3) + ")";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
        }
        ctx.fillStyle = "rgba(180, 197, 255, 0.10)";
        for (const n of nodes) {
            ctx.beginPath();
            ctx.arc(n.x, n.y, 1.4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function advancePulse(pulse, dt) {
        const edge = edges[pulse.edge];
        const start = nodes[pulse.from];
        const endIndex = edge.a === pulse.from ? edge.b : edge.a;
        const end = nodes[endIndex];
        const length = Math.max(Math.hypot(end.x - start.x, end.y - start.y), 1);
        pulse.t += (pulse.speed * dt) / length;
        if (pulse.t >= 1) {
            // Hop to a random connected edge at the junction just reached.
            const candidates = nodeEdges[endIndex].filter(function (index) { return index !== pulse.edge; });
            pulse.edge = candidates.length
                ? candidates[Math.floor(Math.random() * candidates.length)]
                : pulse.edge;
            pulse.from = endIndex;
            pulse.t = 0;
        }
    }

    function drawPulse(pulse) {
        const edge = edges[pulse.edge];
        const start = nodes[pulse.from];
        const endIndex = edge.a === pulse.from ? edge.b : edge.a;
        const end = nodes[endIndex];
        const x = start.x + (end.x - start.x) * pulse.t;
        const y = start.y + (end.y - start.y) * pulse.t;
        const tailT = Math.max(0, pulse.t - 0.22);
        const tx = start.x + (end.x - start.x) * tailT;
        const ty = start.y + (end.y - start.y) * tailT;
        const nearMouse = Math.hypot(x - mouse.x, y - mouse.y) < MOUSE_DIST;
        ctx.strokeStyle = "rgba(" + pulse.color + ", " + (nearMouse ? 0.55 : 0.3) + ")";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(" + pulse.color + ", " + (nearMouse ? 0.95 : 0.65) + ")";
        ctx.beginPath();
        ctx.arc(x, y, nearMouse ? 2.6 : 1.9, 0, Math.PI * 2);
        ctx.fill();
    }

    let lastTime = performance.now();
    function step(now) {
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;
        ctx.clearRect(0, 0, w, h);
        drawLattice();
        for (const pulse of pulses) {
            advancePulse(pulse, dt);
            drawPulse(pulse);
        }
    }

    let reducedMotion = false;
    try {
        reducedMotion = parentWin.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (err) {
        reducedMotion = false;
    }
    if (reducedMotion) {
        ctx.clearRect(0, 0, w, h);
        drawLattice();
    } else {
        (function loop(now) {
            step(now || performance.now());
            requestAnimationFrame(loop);
        })(performance.now());
    }
})();
</script>
        """,
        height=1,
    )


_render_dashboard_theme()
_render_background_animation()


def _mode_label(value: str, mapping: dict[str, tuple[str, str]]) -> tuple[str, str]:
    normalized = (value or "").strip().lower()
    return mapping.get(normalized, (value or "Unknown", "secondary"))


def _energy_mode_from_provenance(summary: dict[str, object]) -> str:
    synthetic_rows = int(summary.get("synthetic_rows", 0))
    partial_rows = int(summary.get("partially_synthetic_rows", 0))
    real_rows = int(summary.get("real_rows", 0))
    if synthetic_rows > 0 and real_rows == 0 and partial_rows == 0:
        return "synthetic"
    if synthetic_rows > 0 or partial_rows > 0:
        return "entsoe_partial_synthetic"
    return "entsoe"


def _render_environment_diagnostics() -> None:
    """Environment status shown in the sidebar so page content stays focused."""
    cfg = AppConfig()
    with st.sidebar.expander("Environment Diagnostics"):
        st.markdown(f"**ENTSO-E API key:** {'loaded' if cfg.entsoe_api_key else 'not found'}")
        st.markdown(f"**Hugging Face token:** {'loaded' if cfg.hf_token else 'not found'}")
        st.caption(f"Python: `{sys.executable}`")
        st.caption(f"Project root: `{Path(__file__).parent.parent.resolve()}`")


MODEL_DISPLAY_NAMES = {"xgboost": "XGBoost", "lstm": "LSTM", "prophet": "Prophet"}


def _bento_card(title: str | None = None):
    """Bordered container styled as a bento-grid tile by the dashboard theme."""
    card = st.container(border=True)
    if title:
        card.markdown(f"##### {title}")
    return card


def _status_tile(label: str, value: str) -> None:
    """Metric-style tile for text values; wraps long text instead of truncating.

    st.metric truncates values that exceed the tile width, so text-valued
    tiles use this plain-div rendering that grows with its content.
    """
    st.markdown(
        f"""
        <div style="background: var(--quant-card-high); border: 1px solid var(--quant-border);
                    border-radius: var(--quant-radius); padding: 0.8rem;">
            <div style="color: var(--quant-muted); font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase;">{label}</div>
            <div style="color: var(--quant-text); font-family: var(--quant-mono); font-weight: 600;
                        font-size: 1.15rem; line-height: 1.35; margin-top: 0.25rem;
                        white-space: normal; overflow-wrap: anywhere;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_sidebar() -> tuple[argparse.Namespace, bool]:
    """Render run-configuration controls in the sidebar and return (args, run_clicked)."""
    st.sidebar.subheader("Run Configuration")
    region = st.sidebar.selectbox("Region", options=["DE_LU", "FR", "NL"], index=0)
    cfg = AppConfig()
    weather_lat, weather_lon = cfg.openmeteo_coords_for_zone(region)
    st.sidebar.caption(f"Open-Meteo coordinates: {weather_lat:.4f}, {weather_lon:.4f}")
    lookback_days = st.sidebar.selectbox("Training Window (days)", options=[90, 180, 365], index=0)
    model = st.sidebar.selectbox("Model", options=["xgboost", "lstm", "prophet"], index=0)
    horizon = st.sidebar.slider("Simulation Horizon", min_value=12, max_value=168, value=24, step=12)
    run_full_comparison = st.sidebar.checkbox(
        "Run full model comparison",
        value=False,
        help="Also train and rank XGBoost, LSTM, and Prophet on this run. Considerably slower; skipped by default.",
    )
    skip_model_comparison = not run_full_comparison
    force_refresh = st.sidebar.checkbox("Force refresh raw-data window", value=False, help="Refetch the selected history window even if cache rows already exist.")
    rebuild_cache = st.sidebar.checkbox("Rebuild cache for selected run", value=False, help="Ignore the existing raw-data cache for this run and rebuild it from fetched data plus explicit gap filling.")
    run_clicked = st.sidebar.button("Run Pipeline", type="primary", width="stretch")
    args = argparse.Namespace(
        zone=region,
        lookback_days=lookback_days,
        simulation_horizon=horizon,
        model=model,
        skip_model_comparison=skip_model_comparison,
        force_refresh=force_refresh,
        rebuild_cache=rebuild_cache,
    )
    return args, run_clicked


def _render_instructions(*, expanded: bool) -> None:
    """Instructions and disclaimers; collapsed automatically once a run has completed."""
    with st.expander("How to use this dashboard", expanded=expanded):
        st.markdown(
            """
**Getting started**

1. Choose a region, training window, and model in the **sidebar** on the left.
2. Optional toggles: *Run full model comparison*, *Force refresh raw-data window*, and *Rebuild cache*.
3. Click **Run Pipeline** in the sidebar and keep this tab open while it works.
4. When the run finishes, model metrics, the latest trading signal, charts, and the research summary appear on this page.
            """
        )
        st.warning(
            "A full pipeline run can take several minutes: it fetches market and weather history, "
            "trains models with walk-forward validation, backtests the strategy, and compares baselines. "
            "Please be patient while it completes.",
            icon="⚠️",
        )
        st.info(
            "The defaults are tuned for speed: a **90-day** training window with the cross-model comparison "
            "skipped. Tick **Run full model comparison** in the sidebar only when you want the full "
            "XGBoost / LSTM / Prophet ranking; it makes the run considerably longer.",
        )
        st.info(
            "External services can be temperamental: ENTSO-E or Hugging Face API keys may occasionally be "
            "rejected, rate-limited, or time out. When that happens the pipeline falls back to synthetic data "
            "or deterministic analysis and clearly flags it in the run output.",
        )
        st.error(
            "**Not financial advice.** This dashboard produces model-driven research forecasts, and forecasts "
            "can be wrong, sometimes badly. If you factor these outputs into any trading or investment "
            "decision, do so only after careful independent judgment, a hard look at the risks, and "
            "professional advice where appropriate.",
            icon="⚠️",
        )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div style="border:1px solid var(--quant-border); border-radius:8px; background:var(--quant-card);
                    padding:2.5rem; text-align:center; margin-top:1rem;">
            <div style="font-size:1.15rem; font-weight:700;">No research run yet</div>
            <div style="color:var(--quant-muted); margin-top:0.5rem; max-width:520px; margin-left:auto; margin-right:auto;">
                Set the region, training window, and model in the sidebar, then click
                <strong>Run Pipeline</strong>. Model metrics, trading signals, charts, and the
                research note will appear here when the run completes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_research_summary(result: dict) -> None:
    research = result.get("research_summary", {})
    summary = dict(research.get("summary", {}))
    if not summary:
        return

    card = _bento_card("Research Summary")
    with card:
        _render_research_summary_body(research, summary)


def _render_research_summary_body(research: dict, summary: dict) -> None:
    research_grade = bool(summary.get("energy_source_research_grade", False))
    llm_generated = str(summary.get("source", "")).strip().lower() == "huggingface"
    badge_col1, badge_col2 = st.columns(2)
    with badge_col1:
        if research_grade:
            st.success("Research-grade data: no synthetic contamination in this run.")
        else:
            st.warning("This run contains synthetic or partially synthetic data. Interpret with caution.", icon="⚠️")
    with badge_col2:
        if llm_generated:
            st.info("Summary produced by the LLM research analyst.")
        else:
            st.info("Summary produced by the deterministic research fallback.")

    edge_summary = str(summary.get("edge_summary", "")).strip()
    if edge_summary:
        st.markdown(f"**Edge thesis.** {edge_summary}")

    trading_conclusion = str(summary.get("trading_conclusion", "")).strip()
    if trading_conclusion:
        st.info(f"**Trading conclusion:** {trading_conclusion}")

    strengths = [str(item) for item in summary.get("strengths", [])]
    weaknesses = [str(item) for item in summary.get("weaknesses", [])]
    failure_modes = [str(item) for item in summary.get("failure_modes", [])]
    strengths_col, weaknesses_col = st.columns(2)
    with strengths_col:
        st.markdown("**Strengths**")
        for item in strengths or ["No strengths recorded."]:
            st.markdown(f"- {item}")
    with weaknesses_col:
        st.markdown("**Weaknesses & Failure Modes**")
        for item in (weaknesses + failure_modes) or ["No weaknesses recorded."]:
            st.markdown(f"- {item}")

    next_experiments = [str(item) for item in summary.get("next_experiments", [])]
    if next_experiments:
        st.markdown("**Suggested next experiments**")
        for item in next_experiments:
            st.markdown(f"- {item}")

    note_path = Path(str(research.get("note_path", "")))
    json_path = Path(str(research.get("json_path", "")))
    download_col1, download_col2 = st.columns(2)
    if note_path.is_file():
        download_col1.download_button(
            "Download research note (Markdown)",
            note_path.read_text(encoding="utf-8"),
            file_name="research_note.md",
            mime="text/markdown",
            width="stretch",
        )
    if json_path.is_file():
        download_col2.download_button(
            "Download research summary (JSON)",
            json_path.read_text(encoding="utf-8"),
            file_name="research_summary.json",
            mime="application/json",
            width="stretch",
        )


def _render_pipeline_page() -> None:
    args, run_clicked = _render_pipeline_sidebar()

    st.title("AI Powered Energy Trading Dashboard")
    st.caption(
        "This page produces an AI/ML-driven trading decision for European power markets: it pulls ENTSO-E "
        "market and weather history, trains the selected machine-learning forecasting model with walk-forward "
        "validation, and turns its predictions into a LONG / SHORT / HOLD recommendation backtested under "
        "realistic execution costs against baseline strategies. Configure the run in the sidebar and click "
        "Run Pipeline; the latest decision, results, and research note appear below."
    )

    result = st.session_state.get("pipeline_result")
    if run_clicked:
        spinner_text = (
            f"Running the {MODEL_DISPLAY_NAMES.get(args.model, args.model)} pipeline for {args.zone} "
            f"({args.lookback_days}-day window)... this may take several minutes."
        )
        with st.spinner(spinner_text):
            result = run_workflow(args)
        st.session_state["pipeline_result"] = result
        st.session_state["pipeline_completed_at"] = pd.Timestamp.now(tz="UTC")
        st.toast("Pipeline completed")

    _render_instructions(expanded=result is None)

    if not result:
        _render_empty_state()
        return

    completed_at = st.session_state.get("pipeline_completed_at")
    run_config = result["config"]
    model_key = str(run_config.get("model", "unknown"))
    summary_sentence = (
        f"This run trained the **{MODEL_DISPLAY_NAMES.get(model_key, model_key)}** model for the "
        f"**{run_config.get('zone', 'unknown')}** region using a "
        f"**{run_config.get('lookback_days', '?')}-day** training window."
    )
    if run_config.get("skip_model_comparison"):
        summary_sentence += " The cross-model comparison was skipped for a faster run."

    runtime_modes = dict(result.get("runtime_modes", {}))
    provenance = result.get("data_provenance", {})
    if provenance:
        runtime_modes["energy_source"] = _energy_mode_from_provenance(provenance)
        runtime_modes["research_grade"] = bool(provenance.get("research_grade", False))
    energy_label, energy_type = _mode_label(
        runtime_modes.get("energy_source", "unknown"),
        {
            "entsoe": ("ENTSO-E live data", "primary"),
            "entsoe_partial_synthetic": ("ENTSO-E with synthetic fill", "warning"),
            "synthetic": ("Synthetic data fallback", "warning"),
        },
    )
    llm_label, llm_type = _mode_label(
        runtime_modes.get("research_source", "unknown"),
        {
            "huggingface": ("LLM analysis active", "primary"),
            "deterministic_fallback": ("Deterministic research fallback active", "warning"),
        },
    )

    overview_col, integrity_col = st.columns([3, 2], gap="medium")
    with overview_col, _bento_card("Run Overview"):
        st.success("Pipeline completed")
        if completed_at is not None:
            st.caption(f"Last run completed at {completed_at:%Y-%m-%d %H:%M} UTC")
        st.markdown(summary_sentence)
        m1, m2 = st.columns(2)
        with m1:
            _status_tile("Energy Data Mode", energy_label)
            st.caption(f"Source key: `{runtime_modes.get('energy_source', 'unknown')}`")
            if energy_type == "primary":
                st.success("ENTSO-E API data was used for this run.")
            elif energy_type == "warning":
                if runtime_modes.get("energy_source") == "entsoe_partial_synthetic":
                    st.warning("Cached and fetched ENTSO-E data were used, with unresolved gaps filled synthetically.", icon="⚠️")
                else:
                    st.warning("ENTSO-E was unavailable, so the pipeline used fully synthetic market data.", icon="⚠️")
            else:
                st.info("Energy data mode could not be determined.")
        with m2:
            _status_tile("Research Agent Mode", llm_label)
            st.caption(f"Model: `{runtime_modes.get('llm_model', 'unknown')}`")
            if llm_type == "primary":
                st.success("The Hugging Face LLM produced the research summary.")
            elif llm_type == "warning":
                st.warning("The pipeline fell back safely because the LLM was unavailable.")
            else:
                st.info("Research agent mode could not be determined.")

    with integrity_col, _bento_card("Data Integrity"):
        total_rows = int(provenance.get("row_count", 0))
        real_rows = int(provenance.get("real_rows", 0))
        partial_rows = int(provenance.get("partially_synthetic_rows", 0))
        synthetic_rows = int(provenance.get("synthetic_rows", 0))
        g1, g2 = st.columns(2)
        g1.metric("Research Grade", "Yes" if runtime_modes.get("research_grade") else "No")
        g2.metric("Real Coverage", f"{float(provenance.get('real_coverage_ratio', 0.0)):.2%}")
        g3, g4 = st.columns(2)
        g3.metric("Partial Synthetic", f"{float(provenance.get('partial_synthetic_coverage_ratio', 0.0)):.2%}")
        g4.metric("Synthetic", f"{float(provenance.get('synthetic_coverage_ratio', 0.0)):.2%}")
        st.caption(
            f"Rows: real {real_rows:,} · partially synthetic {partial_rows:,} · "
            f"synthetic {synthetic_rows:,} · total {total_rows:,}"
        )
        cache_summary = result.get("cache_summary", {})
        if cache_summary:
            with st.expander("Cache details"):
                energy_cache = cache_summary.get("energy", {})
                weather_cache = cache_summary.get("weather", {})
                st.markdown("**Energy cache**")
                st.caption(
                    f"Status: {energy_cache.get('cache_status', 'unknown')} · "
                    f"Used: {'yes' if energy_cache.get('cache_used') else 'no'} · "
                    f"Fetched ranges: {energy_cache.get('fetched_range_count', 0)}"
                )
                st.caption(f"Freshness: {energy_cache.get('cache_freshness_utc', 'n/a')}")
                st.markdown("**Weather cache**")
                st.caption(
                    f"Status: {weather_cache.get('cache_status', 'unknown')} · "
                    f"Used: {'yes' if weather_cache.get('cache_used') else 'no'} · "
                    f"Fetched ranges: {weather_cache.get('fetched_range_count', 0)}"
                )
                st.caption(f"Freshness: {weather_cache.get('cache_freshness_utc', 'n/a')}")

    forecast_col, backtest_col = st.columns([3, 2], gap="medium")
    with forecast_col, _bento_card("Forecast Accuracy"):
        metrics_path = Path(result["metrics_path"])
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            price_metrics = metrics.get("price", {})
            f1, f2, f3 = st.columns(3)
            f1.metric("Price MAE", f"{price_metrics.get('mae', float('nan')):,.2f}")
            f2.metric("Demand MAE", f"{metrics['demand']['mae']:,.2f}")
            f3.metric("Renewable MAE", f"{metrics['renewable']['mae']:,.2f}")
            f4, f5, f6 = st.columns(3)
            f4.metric("Price RMSE", f"{price_metrics.get('rmse', float('nan')):,.2f}")
            f5.metric("Demand RMSE", f"{metrics['demand']['rmse']:,.2f}")
            f6.metric("Renewable RMSE", f"{metrics['renewable']['rmse']:,.2f}")
        else:
            st.caption("Training metrics file not found for this run.")

    with backtest_col, _bento_card("Backtest Summary"):
        backtest_metrics_path = Path("artifacts/simulation/backtest_metrics.json")
        if backtest_metrics_path.exists():
            bt = json.loads(backtest_metrics_path.read_text(encoding="utf-8"))
            b1, b2 = st.columns(2)
            b1.metric("Sharpe", f"{bt['sharpe_ratio']:.3f}")
            b2.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")
            b3, b4 = st.columns(2)
            b3.metric("Hit Rate", f"{bt['hit_rate']:.1%}")
            b4.metric("Total PnL", f"{bt['total_pnl']:,.2f} EUR")
        else:
            st.caption("Backtest metrics file not found for this run.")

    features_df: pd.DataFrame = result["features_df"]
    scored_df: pd.DataFrame = result["scored_df"]
    backtest_df: pd.DataFrame = result["backtest_df"]

    latest_signal = backtest_df.sort_values("timestamp_utc").iloc[-1]
    recommended_decision = latest_signal.get("recommended_decision", latest_signal["decision"])
    with _bento_card("Latest Trading Signal"):
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Recommended Action", str(recommended_decision))
        s2.metric("Predicted Price", f"{latest_signal['pred_price_eur_mwh']:,.2f} EUR/MWh")
        s3.metric(
            "Current Price",
            f"{latest_signal['price_eur_mwh']:,.2f} EUR/MWh",
            delta=f"{latest_signal['pred_price_eur_mwh'] - latest_signal['price_eur_mwh']:,.2f} EUR/MWh",
        )
        s4.metric("Executed Position", f"{latest_signal['position']:.3f}")
        s5.metric("Predicted Imbalance", f"{latest_signal['imbalance_pred']:,.2f} MW")

    _render_research_summary(result)

    strategy_comparison = result.get("strategy_comparison", {})
    if strategy_comparison and "summary_df" in strategy_comparison:
        with _bento_card("Strategy Comparison"):
            metrics_tab, significance_tab = st.tabs(["Strategy Metrics", "Significance vs Baselines"])
            with metrics_tab:
                st.dataframe(strategy_comparison["summary_df"], width="stretch")
            with significance_tab:
                st.dataframe(strategy_comparison["significance_df"], width="stretch")

    with _bento_card("Performance Charts"):
        history_tab, prediction_tab, backtest_tab = st.tabs(["Market History", "Predictions", "Backtest"])
        with history_tab:
            render_history_charts(features_df.tail(300))
        with prediction_tab:
            render_prediction_chart(scored_df.tail(300))
        with backtest_tab:
            render_backtest_chart(backtest_df.tail(300))


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

    scorecard_col, accuracy_col = st.columns(2, gap="medium")
    with scorecard_col, _bento_card("Backtesting Scorecard"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Sharpe", f"{float(metrics.get('sharpe_ratio', 0.0)):.3f}")
        m2.metric("Max Drawdown", f"{float(metrics.get('max_drawdown', 0.0)):.3f}")
        m3.metric("Hit Rate", f"{float(metrics.get('hit_rate', 0.0)):.3f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Total PnL", f"{float(metrics.get('total_pnl', 0.0)):,.2f}")
        m5.metric("Trade Count", f"{int(metrics.get('trade_count', 0)):,}")
        m6.metric("Avg Trade Return", f"{float(metrics.get('average_trade_return', 0.0)):.4f}")

    with accuracy_col, _bento_card("Decision Accuracy"):
        a1, a2, a3 = st.columns(3)
        a1.metric("Directional Accuracy", f"{review_summary['directional_accuracy']:.3f}")
        a2.metric("Positive PnL Rate", f"{review_summary['pnl_positive_rate']:.3f}")
        a3.metric("Evaluable Rows", f"{review_summary['evaluable_rows']:,}")
        a4, a5, a6 = st.columns(3)
        a4.metric("Correct", f"{review_summary['correct_count']:,}")
        a5.metric("Incorrect", f"{review_summary['incorrect_count']:,}")
        distribution = review_summary["decision_distribution"]
        a6.metric("Long / Short / Hold", f"{distribution['LONG']} / {distribution['SHORT']} / {distribution['HOLD']}")
        st.caption(
            f"Accuracy horizon: {review_summary['accuracy_horizon_steps']} step(s). "
            f"HOLD tolerance band: {review_summary['hold_tolerance_pct'] * 100:.2f}%."
        )

    with _bento_card("Review Charts"):
        prediction_tab, equity_tab, accuracy_tab, decisions_tab = st.tabs(
            ["Predictions", "Equity Curve", "Accuracy", "Decision Log"]
        )
        with prediction_tab:
            render_prediction_chart(filtered_df.tail(300))
        with equity_tab:
            render_equity_chart(filtered_df.tail(300))
        with accuracy_tab:
            render_review_accuracy_chart(filtered_df.tail(300))
        with decisions_tab:
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
    st.caption(
        "This page reviews completed backtests without rerunning the research pipeline: load the latest saved "
        "artifacts, run an isolated backtest from a scored CSV, or compare trained models side by side. "
        "Each decision is scored against the price move that actually followed it."
    )

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
        comparison_force_refresh = st.checkbox("Force refresh shared raw-data window", value=False, key="comparison_force_refresh")
        comparison_rebuild_cache = st.checkbox("Rebuild shared raw-data cache", value=False, key="comparison_rebuild_cache")
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
                        force_refresh=comparison_force_refresh,
                        rebuild_cache=comparison_rebuild_cache,
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
            width="stretch",
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
            with _bento_card("Forecast Metrics"):
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


st.sidebar.markdown("## Energy Trading Research")
st.sidebar.caption("Walk-forward forecasting, regime-aware signals, and realistic backtesting for European power markets.")
st.sidebar.divider()

# The menu is pinned to the bottom of the sidebar: the active page is read from
# session state first so each page can render its own controls above the menu,
# and the radio itself is drawn afterwards. Changing it triggers a rerun that
# picks up the new selection here.
_PAGE_OPTIONS = ["Pipeline", "Backtesting Review"]
_active_page = st.session_state.get("dashboard_menu", _PAGE_OPTIONS[0])
if _active_page == "Pipeline":
    _render_pipeline_page()
else:
    _render_backtesting_review_page()

st.sidebar.divider()
_render_environment_diagnostics()
st.sidebar.radio("Menu", options=_PAGE_OPTIONS, key="dashboard_menu")

st.divider()
st.caption("Research tool only. Model forecasts are not financial advice. Validate independently before acting on any output.")
