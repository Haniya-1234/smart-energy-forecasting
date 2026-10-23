import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import base64
from src.data_loader import (
    parse_tsf_metadata,
    get_available_series,
    load_series,
    ensure_dataset_available,
    DEFAULT_DATASET_PATH,
)
from src.eda import (
    compute_summary_statistics,
    plot_historical_series,
    plot_hourly_weekly_profile,
    plot_distribution,
)
from src.forecasting import (
    run_model_comparison,
    generate_future_forecast,
)
from src.anomaly import (
    combine_anomaly_detection,
)
from src.recommender import (
    generate_conservation_recommendations,
)

ASSETS_IMAGE_PATH = os.path.join("assets", "energy_banner.png")

def get_banner_image_uri() -> str:
    """Returns a data URI for the local image asset, with SVG gradient fallback."""
    if os.path.exists(ASSETS_IMAGE_PATH):
        try:
            with open(ASSETS_IMAGE_PATH, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                return f"data:image/png;base64,{b64_data}"
        except Exception:
            pass
    # Reliable inline SVG fallback representation
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 240" width="100%" height="100%">'
        '<defs>'
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#1E293B"/>'
        '<stop offset="100%" stop-color="#0F172A"/>'
        '</linearGradient>'
        '<linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#F45112"/>'
        '<stop offset="100%" stop-color="#F4BE25"/>'
        '</linearGradient>'
        '</defs>'
        '<rect width="700" height="240" fill="url(#bg)"/>'
        '<circle cx="120" cy="80" r="45" fill="#F45112" opacity="0.25"/>'
        '<circle cx="580" cy="160" r="70" fill="#F4BE25" opacity="0.15"/>'
        '<path d="M 0 160 Q 180 80 350 140 T 700 100" fill="none" stroke="url(#lineGrad)" stroke-width="4"/>'
        '<text x="50%" y="48%" text-anchor="middle" fill="#FFFFFF" font-family="sans-serif" font-weight="700" font-size="22">⚡ Smart Energy Grid</text>'
        '<text x="50%" y="64%" text-anchor="middle" fill="#94A3B8" font-family="sans-serif" font-size="14">Automated Load Analytics &amp; Forecasting</text>'
        '</svg>'
    )
    b64_svg = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_svg}"


# Page Configuration
st.set_page_config(
    page_title="Smart Energy Forecasting & Conservation DSS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- DESIGN SYSTEM & MODERN ENERGY DASHBOARD STYLING -----------------
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global application canvas */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F5F2E9 !important;
        color: #171717 !important;
    }

    /* Main container padding */
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1400px;
    }

    /* Sidebar consistent warm styling */
    section[data-testid="stSidebar"] {
        background-color: #EFECE3 !important;
        border-right: 1px solid #E3DDD0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #EFECE3 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Sidebar Brand Header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 14px;
        background: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .sidebar-brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #F45112 0%, #F4BE25 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.25rem;
        box-shadow: 0 2px 6px rgba(244, 81, 18, 0.35);
        flex-shrink: 0;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #171717;
        line-height: 1.15;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: #F45112;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* Sidebar Section Card Containers */
    .sidebar-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.74rem;
        font-weight: 800;
        color: #171717;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.4rem 0 0.5rem 0;
        padding-bottom: 4px;
        border-bottom: 2px solid #F45112;
    }
    .sidebar-card-header span.sec-icon {
        font-size: 0.9rem;
    }

    /* Polish Streamlit form controls in sidebar */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has([data-testid="stSelectbox"]),
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has([data-testid="stRadio"]),
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has([data-testid="stNumberInput"]),
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has([data-testid="stSlider"]) {
        background: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }

    section[data-testid="stSidebar"] label {
        color: #171717 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
    }

    /* Radio button pills enhancement */
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #FAF7F0;
        border: 1px solid #E8E2D5;
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        transition: all 0.15s ease;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #FFF;
        border-color: #F45112;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(244, 81, 18, 0.08) !important;
        border: 1.5px solid #F45112 !important;
        color: #C23B06 !important;
        font-weight: 700;
    }

    /* Slider accent color */
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #F45112 !important;
        border-color: #F45112 !important;
        box-shadow: 0 0 0 3px rgba(244, 81, 18, 0.2) !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #F45112;
    }

    /* Sidebar Footer */
    .sidebar-footer {
        margin-top: 1.2rem;
        padding: 10px 12px;
        background: rgba(255, 254, 250, 0.7);
        border: 1px dashed #DCD5C6;
        border-radius: 8px;
        font-size: 0.72rem;
        color: #787164;
        line-height: 1.4;
    }
    .sidebar-footer b {
        color: #171717;
    }

    /* Hero section split styling */
    .hero-container {
        background: linear-gradient(135deg, #FFFEFA 0%, #FAF6EE 60%, #F5EFE0 100%);
        border: 1px solid #E8E2D5;
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03), 0 1px 3px rgba(0,0,0,0.02);
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        overflow: hidden;
        position: relative;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #F45112 0%, #F4BE25 100%);
    }
    .hero-content {
        flex: 1.4;
        z-index: 1;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 81, 18, 0.08);
        border: 1px solid rgba(244, 81, 18, 0.25);
        color: #C23B06;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #171717;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 600;
        color: #F45112;
        margin: 0 0 6px 0;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #5C5850;
        line-height: 1.45;
        margin: 0;
        max-width: 620px;
    }
    .hero-image-wrap {
        flex: 1;
        max-width: 380px;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid #E8E2D5;
    }
    .hero-image-wrap img {
        width: 100%;
        height: 155px;
        object-fit: cover;
        display: block;
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 1.6rem;
    }
    .kpi-card {
        background-color: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(244, 81, 18, 0.08);
        border-color: #F4BE25;
    }
    .kpi-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #787164;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-icon-pill {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        background: #FAF6EE;
        border: 1px solid #E8E2D5;
    }
    .kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #171717;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #8C8476;
        font-weight: 500;
    }
    .kpi-accent-orange {
        border-top: 3px solid #F45112;
    }
    .kpi-accent-yellow {
        border-top: 3px solid #F4BE25;
    }
    .kpi-accent-neutral {
        border-top: 3px solid #C4BCAC;
    }

    /* Provenance and Disclaimer Box */
    .disclaimer-box {
        background-color: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-left: 4px solid #F4BE25;
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 1.4rem;
        font-size: 0.85rem;
        color: #38342D;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .disclaimer-box b {
        color: #171717;
    }

    /* Streamlit Tabs Navigation styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #EBE5D8;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #DFD8C8;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 6px 18px;
        background-color: transparent;
        border-radius: 8px;
        color: #5C5850 !important;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #171717 !important;
        background-color: rgba(255, 254, 250, 0.6);
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFEFA !important;
        color: #F45112 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Clean Card Containers in Tabs */
    .content-card {
        background-color: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }

    /* Model comparison metric boxes */
    .model-card {
        background-color: #FFFEFA;
        border: 1px solid #E8E2D5;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        height: 100%;
    }
    .model-card-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #171717;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .model-metric-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #EDE7D8;
        font-size: 0.86rem;
    }
    .model-metric-row:last-child {
        border-bottom: none;
    }
    .model-metric-name {
        color: #6C6659;
        font-weight: 500;
    }
    .model-metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #171717;
    }

    /* Subheadings within content */
    h2, h3, h4 {
        color: #171717 !important;
        font-weight: 700 !important;
    }

    /* Sidebar headers and widgets */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #171717 !important;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #2F2B23 !important;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Buttons and controls */
    .stButton>button {
        background-color: #F45112 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
    }
    .stButton>button:hover {
        background-color: #DD430B !important;
        box-shadow: 0 4px 12px rgba(244, 81, 18, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.abspath(os.path.join("data", "electricity_hourly_dataset.tsf"))

# Automatically ensure dataset is present (without re-downloading if already available)
@st.cache_resource(show_spinner=False)
def init_dataset(target_path: str = DATA_PATH) -> str:
    return ensure_dataset_available(target_path)

# Data Caching for snappy responsiveness
@st.cache_data(show_spinner=False)
def load_cached_metadata():
    return parse_tsf_metadata(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_cached_series_list():
    return get_available_series(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_cached_series(series_name: str):
    return load_series(DATA_PATH, series_name=series_name)

# Ensure dataset is ready before rendering
dataset_ready = False
try:
    if not os.path.exists(DATA_PATH):
        with st.status("Fetching electricity benchmark dataset from Zenodo...", expanded=True) as status_box:
            st.write("Initializing data directory and contacting Zenodo record 4656140...")
            def _ui_progress(pct: float, msg: str):
                st.write(f"[{int(pct * 100)}%] {msg}")
            ensure_dataset_available(DATA_PATH, progress_callback=_ui_progress)
            status_box.update(label="Dataset acquired successfully!", state="complete", expanded=False)
    else:
        init_dataset(DATA_PATH)
    dataset_ready = True
except Exception as exc:
    st.error(
        f"**Dataset Initialization Error**: Unable to prepare the electricity dataset.\n\n"
        f"**Details**: `{str(exc)}`\n\n"
        "**Retry Guidance**:\n"
        "- Refresh the page or click **Rerun** in the top-right menu to try downloading again.\n"
        "- If running in an air-gapped or restricted network environment, ensure access to `https://zenodo.org`.\n"
        "- Alternatively, manually download the file from [Zenodo Record 4656140](https://zenodo.org/records/4656140) "
        "and place `electricity_hourly_dataset.tsf` inside the `data/` folder."
    )
    st.stop()

# Sidebar Control Panel
with st.sidebar:
    # 1. Compact branded header
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">⚡</div>
        <div>
            <div class="sidebar-brand-title">ENERGYWISE</div>
            <div class="sidebar-brand-sub">Forecasting &amp; Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Indexing dataset series..."):
        series_list = load_cached_series_list()

    # SECTION 1: DATA CONTROLS
    st.markdown("""
    <div class="sidebar-card-header">
        <span class="sec-icon">🎛️</span> DATA CONTROLS
    </div>
    """, unsafe_allow_html=True)

    selected_series = st.selectbox(
        "Series / Client:",
        options=series_list,
        index=0,
        help="Select any of the 321 hourly electricity time series benchmark profiles."
    )

    # Date range filter
    history_view = st.radio(
        "Historical View Range:",
        options=["Last 30 Days", "Last 90 Days", "Full History (3 Years)"],
        index=0,
        help="Select lookback window: 30 days, 90 days, or full 3-year history."
    )

    # SECTION 2: FORECAST SETTINGS
    st.markdown("""
    <div class="sidebar-card-header" style="margin-top: 1rem;">
        <span class="sec-icon">📈</span> FORECAST SETTINGS
    </div>
    """, unsafe_allow_html=True)

    tariff_input = st.number_input(
        "Electricity Tariff (Per kWh):",
        min_value=0.5,
        max_value=100.0,
        value=8.0,
        step=0.5,
        format="%.2f",
        help="Local tariff rate used for estimated energy expenditure calculations."
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (Hours):",
        min_value=12,
        max_value=168,
        value=24,
        step=12,
        help="Predict forward load from 12 hours up to 7 days (168 hours)."
    )

    # Context & Benchmark Data Provenance Footer
    st.markdown("""
    <div class="sidebar-footer">
        <div><b>Data Source:</b> Public Hourly Electricity Benchmark (UCI/Monash, 321 series).</div>
        <div style="margin-top:4px; font-size:0.68rem; color:#8C8476;">College TSDA Mini-Project</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- HERO SECTION -----------------
banner_uri = get_banner_image_uri()
st.markdown(f"""
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-badge">⚡ Energy Intelligence & Forecasting</div>
        <div class="hero-title">Smart Energy Forecasting</div>
        <div class="hero-subtitle">Forecast demand. Detect anomalies. Make smarter energy decisions.</div>
        <p class="hero-desc">
            Empowering facility managers with automated time-series analytics, chronological machine learning validation, statistical anomaly surveillance, and operational peak-shifting heuristics.
        </p>
    </div>
    <div class="hero-image-wrap">
        <img src="{banner_uri}" alt="Smart Energy Infrastructure">
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Disclaimer Banner
st.markdown("""
<div class="disclaimer-box">
    <b>ℹ️ Data Source & Provenance Disclosure:</b> This system utilizes the open-source <i>Electricity Hourly Dataset</i> (UCI / Monash Time Series Repository / Zenodo, Lai et al. 2017) containing 321 electricity load profiles (kW) recorded hourly from 2012 to 2014. These are real-world public power benchmark measurements and not physical meter telemetry from our college campus. All cost figures and conservation savings are explicitly labeled heuristic estimates.
</div>
""", unsafe_allow_html=True)

# Load Selected Series
with st.spinner(f"Loading data for series {selected_series}..."):
    full_df = load_cached_series(selected_series)

# Apply view filter
if history_view == "Last 30 Days":
    active_df = full_df.iloc[-24 * 30:].copy()
elif history_view == "Last 90 Days":
    active_df = full_df.iloc[-24 * 90:].copy()
else:
    active_df = full_df.copy()

# Summary Metrics Calculation
stats = compute_summary_statistics(active_df)
est_cost = stats['total_energy_kwh'] * tariff_input
currency_prefix = "₹" if tariff_input == 8.0 else ""

# ----------------- KPI CARDS ROW (WARM, ROUNDED, SHADOWED) -----------------
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card kpi-accent-orange">
        <div class="kpi-card-top">
            <span class="kpi-label">Mean Demand</span>
            <div class="kpi-icon-pill">📊</div>
        </div>
        <div class="kpi-val">{stats['mean']:.1f} <span style="font-size:0.9rem; font-weight:600; color:#787164;">kW</span></div>
        <div class="kpi-sub">Average load across window</div>
    </div>
    <div class="kpi-card kpi-accent-orange">
        <div class="kpi-card-top">
            <span class="kpi-label">Peak Demand</span>
            <div class="kpi-icon-pill">🔥</div>
        </div>
        <div class="kpi-val">{stats['max']:.1f} <span style="font-size:0.9rem; font-weight:600; color:#787164;">kW</span></div>
        <div class="kpi-sub">Maximum observed surge</div>
    </div>
    <div class="kpi-card kpi-accent-yellow">
        <div class="kpi-card-top">
            <span class="kpi-label">Baseload (Min)</span>
            <div class="kpi-icon-pill">🌙</div>
        </div>
        <div class="kpi-val">{stats['min']:.1f} <span style="font-size:0.9rem; font-weight:600; color:#787164;">kW</span></div>
        <div class="kpi-sub">Continuous idle baseline</div>
    </div>
    <div class="kpi-card kpi-accent-neutral">
        <div class="kpi-card-top">
            <span class="kpi-label">Total Energy</span>
            <div class="kpi-icon-pill">⚡</div>
        </div>
        <div class="kpi-val">{stats['total_energy_kwh']:,.0f} <span style="font-size:0.9rem; font-weight:600; color:#787164;">kWh</span></div>
        <div class="kpi-sub">{len(active_df):,} cumulative hours</div>
    </div>
    <div class="kpi-card kpi-accent-yellow">
        <div class="kpi-card-top">
            <span class="kpi-label">Est. Energy Cost</span>
            <div class="kpi-icon-pill">💰</div>
        </div>
        <div class="kpi-val">{currency_prefix}{est_cost:,.0f}*</div>
        <div class="kpi-sub">At {currency_prefix}{tariff_input}/kWh tariff estimate</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- FIVE NAVIGATION TABS -----------------
tab_eda, tab_forecast, tab_anomaly, tab_dss, tab_about = st.tabs([
    "📊 Exploratory Data Analysis",
    "📈 Time-Series Forecasting",
    "🔍 Anomaly Detection",
    "💡 Decision Support & Conservation",
    "📖 About & Methodology"
])

# ----------------- TAB 1: EDA -----------------
with tab_eda:
    st.subheader("Historical Demand & Temporal Profiling")
    st.plotly_chart(plot_historical_series(active_df, selected_series), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hourly, fig_heat = plot_hourly_weekly_profile(active_df)
        st.plotly_chart(fig_hourly, use_container_width=True)
    with c2:
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Statistical Distribution")
    st.plotly_chart(plot_distribution(active_df), use_container_width=True)

# ----------------- TAB 2: FORECASTING -----------------
with tab_forecast:
    st.subheader("Comparative Time-Series Model Evaluation")
    st.markdown("""
    Chronological validation without data leakage. Evaluating models on the final held-out test split.
    <br><small><b>NRMSE Formulation:</b> $\\text{NRMSE} = \\frac{\\text{RMSE}}{\\bar{y}_{\\text{true}}} \\times 100\\%$, where $\\bar{y}_{\\text{true}}$ is the mean of the actual observations across the identical held-out test window.</small>
    """, unsafe_allow_html=True)

    backtest_days = st.selectbox(
        "Held-Out Backtest Test Window:",
        options=[7, 14, 28],
        index=0,
        format_func=lambda x: f"{x} Days ({x * 24} Hours)",
        help="Select evaluation horizon for chronological backtesting across models. For 28 days, ensure at least 90 days of history are selected in the sidebar."
    )
    backtest_hours = backtest_days * 24

    with st.spinner(f"Training baseline, statistical, and ML models on held-out {backtest_days}-day split..."):
        try:
            train_df, metrics, results_df = run_model_comparison(active_df, test_hours=backtest_hours)
        except ValueError as err:
            st.error(str(err))
            st.stop()

    # Redesigned Model Comparison Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="model-card" style="border-top: 3px solid #787164;">
            <div class="model-card-header">
                <span>⏱️</span> 1. Seasonal Naive Baseline (24h)
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">MAE</span>
                <span class="model-metric-val">{metrics['Seasonal Naive (24h)']['MAE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">RMSE</span>
                <span class="model-metric-val">{metrics['Seasonal Naive (24h)']['RMSE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">Normalized RMSE</span>
                <span class="model-metric-val">{metrics['Seasonal Naive (24h)']['NRMSE_pct']:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="model-card" style="border-top: 3px solid #059669;">
            <div class="model-card-header">
                <span>📈</span> 2. Exponential Smoothing (Holt-Winters)
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">MAE</span>
                <span class="model-metric-val">{metrics['Exponential Smoothing']['MAE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">RMSE</span>
                <span class="model-metric-val">{metrics['Exponential Smoothing']['RMSE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">Normalized RMSE</span>
                <span class="model-metric-val">{metrics['Exponential Smoothing']['NRMSE_pct']:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="model-card" style="border-top: 3px solid #F45112;">
            <div class="model-card-header">
                <span>🤖</span> 3. ML Lag-Feature Model (Ridge)
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">MAE</span>
                <span class="model-metric-val">{metrics['ML Lag-Feature (Ridge)']['MAE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">RMSE</span>
                <span class="model-metric-val">{metrics['ML Lag-Feature (Ridge)']['RMSE']:.2f} kW</span>
            </div>
            <div class="model-metric-row">
                <span class="model-metric-name">Normalized RMSE</span>
                <span class="model-metric-val">{metrics['ML Lag-Feature (Ridge)']['NRMSE_pct']:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # spacing

    # Best model highlight
    best_rmse_model = min(metrics.keys(), key=lambda m: metrics[m]["RMSE"])
    best_mae_model = min(metrics.keys(), key=lambda m: metrics[m]["MAE"])
    best_rmse_val = metrics[best_rmse_model]["RMSE"]
    best_nrmse_val = metrics[best_rmse_model]["NRMSE_pct"]
    best_mae_val = metrics[best_mae_model]["MAE"]

    st.markdown(f"""
    <div style="background-color: #FFFEFA; border: 1px solid #E8E2D5; border-left: 4px solid #F45112; padding: 14px 18px; border-radius: 10px; margin: 12px 0 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
        <b style="color: #171717;">🎯 Performance Summary & Best Model ({selected_series} — {backtest_days}-Day Window):</b>
        The <b style="color: #F45112;">{best_rmse_model}</b> achieves the lowest error on this test split with an RMSE of <b>{best_rmse_val:.2f} kW</b> (NRMSE <b>{best_nrmse_val:.1f}%</b>) and MAE of <b>{metrics[best_rmse_model]['MAE']:.2f} kW</b>.
        {" Both MAE and RMSE select this model as top performer." if best_rmse_model == best_mae_model else f" (Lowest MAE achieved by {best_mae_model} at {best_mae_val:.2f} kW)."}
        All metrics were computed strictly against the same {backtest_hours} held-out observations.
    </div>
    """, unsafe_allow_html=True)

    # Plot actual vs predicted on held-out test set
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(x=results_df["timestamp"], y=results_df["Actual"], mode="lines", name="Actual Demand", line=dict(color="#F45112", width=2.4)))
    fig_backtest.add_trace(go.Scatter(x=results_df["timestamp"], y=results_df["Seasonal Naive"], mode="lines", name="Seasonal Naive", line=dict(dash="dot", color="#8C8476", width=1.5)))
    fig_backtest.add_trace(go.Scatter(x=results_df["timestamp"], y=results_df["Exponential Smoothing"], mode="lines", name="Holt-Winters ETS", line=dict(color="#059669", width=2)))
    fig_backtest.add_trace(go.Scatter(x=results_df["timestamp"], y=results_df["ML Lag Model"], mode="lines", name="ML Lag Model (Ridge)", line=dict(color="#D97706", width=2)))
    fig_backtest.update_layout(
        title=dict(text=f"Held-Out Backtest Comparison: Actual vs Predicted (Last {backtest_days} Days / {backtest_hours} Hours)", font=dict(color="#171717", size=14, weight="bold")),
        xaxis=dict(
            title="Timestamp",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        yaxis=dict(
            title="Demand (kW)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#171717")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

    st.markdown("---")
    st.subheader(f"🔮 Operational Forward Forecast (Next {forecast_horizon} Hours)")
    st.caption("Operational forecast blends Holt-Winters ETS (65%) and Seasonal Naive (35%) using historical series data available up to the forecast origin.")
    with st.spinner("Generating forward predictions and uncertainty intervals..."):
        future_df = generate_future_forecast(active_df, horizon_hours=forecast_horizon)

    fig_future = go.Figure()
    # Contextual recent history (last 48h)
    recent_ctx = active_df.iloc[-48:]
    fig_future.add_trace(go.Scatter(x=recent_ctx["timestamp"], y=recent_ctx["consumption_kw"], mode="lines", name="Recent Actual Load", line=dict(color="#F45112", width=2)))
    # Uncertainty ribbon (heuristic band)
    fig_future.add_trace(go.Scatter(
        x=list(future_df["timestamp"]) + list(future_df["timestamp"][::-1]),
        y=list(future_df["upper_bound_kw"]) + list(future_df["lower_bound_kw"][::-1]),
        fill="toself",
        fillcolor="rgba(5, 150, 105, 0.16)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Heuristic Uncertainty Band (±1.96 Residual σ)",
        hoverinfo="skip"
    ))
    fig_future.add_trace(go.Scatter(x=future_df["timestamp"], y=future_df["forecast_kw"], mode="lines+markers", name="Operational Forecast (Holt-Winters + Naive)", line=dict(color="#059669", width=2.5)))
    fig_future.add_trace(go.Scatter(x=future_df["timestamp"], y=future_df["naive_baseline_kw"], mode="lines", name="Seasonal Baseline Reference", line=dict(dash="dot", color="#8C8476")))
    fig_future.update_layout(
        title=dict(text=f"Out-of-Sample Electricity Demand Forecast ({forecast_horizon} Hours)", font=dict(color="#171717", size=14, weight="bold")),
        xaxis=dict(
            title="Forecast Timeline",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        yaxis=dict(
            title="Demand (kW)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#171717")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=430
    )
    st.plotly_chart(fig_future, use_container_width=True)
    st.caption("ℹ️ **Uncertainty Band Disclosure:** The shaded range is a heuristic empirical uncertainty band (±1.96 standard deviation of recent 168h differenced residuals) representing historical volatility, rather than a statistically calibrated parametric prediction interval.")

# ----------------- TAB 3: ANOMALIES -----------------
with tab_anomaly:
    st.subheader("Statistical & Multivariate Anomaly Detection")
    st.markdown("""
    Combines **Rolling Z-Score** (deviation from dynamic local mean) and **Isolation Forest** (multivariate structural outliers).
    *Note: Anomalies highlight atypical behavior requiring inspection; they do not automatically guarantee equipment malfunction or confirmed electricity waste.*
    """)

    with st.spinner("Detecting consumption anomalies..."):
        anomaly_df = combine_anomaly_detection(active_df, z_threshold=2.5)

    anomalies_only = anomaly_df[anomaly_df["is_confirmed_anomaly"] == 1]
    st.write(f"Identified **{len(anomalies_only)} unusual observations** out of {len(active_df)} hours ({len(anomalies_only)/len(active_df)*100:.2f}% anomaly rate).")

    fig_anom = go.Figure()
    # Normal load profile in muted warm amber / orange line
    fig_anom.add_trace(go.Scatter(x=anomaly_df["timestamp"], y=anomaly_df["consumption_kw"], mode="lines", name="Normal Load Profile", line=dict(color="#FDBA74", width=1.5)))
    fig_anom.add_trace(go.Scatter(x=anomaly_df["timestamp"], y=anomaly_df["rolling_mean"], mode="lines", name="7-Day Rolling Baseline", line=dict(color="#57534E", dash="dash", width=1.8)))

    if not anomalies_only.empty:
        fig_anom.add_trace(go.Scatter(
            x=anomalies_only["timestamp"],
            y=anomalies_only["consumption_kw"],
            mode="markers",
            name="Unusual Observation",
            marker=dict(color="#DC2626", size=8, symbol="circle", line=dict(width=2, color="#991B1B")),
            hovertemplate="<b>Anomaly Detected:</b> %{x|%Y-%m-%d %H:%M}<br><b>Load:</b> %{y:.2f} kW<br><b>Z-Score:</b> %{customdata:.2f}<extra></extra>",
            customdata=anomalies_only["z_score"]
        ))

    fig_anom.update_layout(
        title=dict(text="Detected Consumption Anomalies vs Dynamic Baseline", font=dict(color="#171717", size=14, weight="bold")),
        xaxis=dict(
            title="Timeline",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        yaxis=dict(
            title="Demand (kW)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#171717")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=440
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    if not anomalies_only.empty:
        st.markdown("#### Recent Anomaly Event Log (Last 10 Events)")
        recent_log = anomalies_only[["timestamp", "consumption_kw", "rolling_mean", "z_score", "anomaly_direction"]].tail(10).copy()
        recent_log.columns = ["Timestamp", "Observed Load (kW)", "Expected Baseline (kW)", "Z-Score", "Deviation Type"]
        st.dataframe(recent_log.sort_values(by="Timestamp", ascending=False), use_container_width=True)

# ----------------- TAB 4: DECISION SUPPORT -----------------
with tab_dss:
    st.subheader("Energy Conservation & Demand Management Recommendations")
    st.markdown("Data-driven heuristics based on actual consumption ratios, off-peak idle baseloads, and upcoming forecasted peaks.")

    with st.spinner("Synthesizing decision support recommendations..."):
        future_for_dss = generate_future_forecast(active_df, horizon_hours=forecast_horizon)
        anom_for_dss = combine_anomaly_detection(active_df, z_threshold=2.5)
        recommendations = generate_conservation_recommendations(
            history_df=active_df,
            forecast_df=future_for_dss,
            anomaly_df=anom_for_dss,
            tariff_rate=tariff_input
        )

    for rec in recommendations:
        priority_cfg = {
            "High": {"bg": "#FFF1EE", "border": "#F45112", "badge_bg": "#FFECE6", "badge_color": "#C23B06"},
            "Medium": {"bg": "#FEFCE8", "border": "#F4BE25", "badge_bg": "#FEF9C3", "badge_color": "#A16207"},
            "Low": {"bg": "#F0FDF4", "border": "#10B981", "badge_bg": "#DCFCE7", "badge_color": "#15803D"}
        }.get(rec["priority"], {"bg": "#FFFEFA", "border": "#C4BCAC", "badge_bg": "#F5EFE0", "badge_color": "#5C5850"})

        st.markdown(f"""
        <div style="background-color: {priority_cfg['bg']}; border: 1px solid #E8E2D5; border-left: 5px solid {priority_cfg['border']}; padding: 18px 20px; border-radius: 12px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #171717; font-size: 1.05rem;">{rec['title']}</h4>
                <span style="font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; background: {priority_cfg['badge_bg']}; color: {priority_cfg['badge_color']}; border: 1px solid {priority_cfg['border']}; letter-spacing: 0.04em;">
                    {rec['priority'].upper()} PRIORITY
                </span>
            </div>
            <p style="margin: 6px 0 4px 0; font-size: 0.92rem; color: #474238;"><b>Observation:</b> {rec['observation']}</p>
            <p style="margin: 4px 0 4px 0; font-size: 0.92rem; color: #171717;"><b>Recommended Action:</b> {rec['action']}</p>
            <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #6C6659;"><b>Estimated Impact:</b> {rec['estimated_impact']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Tariff Cost Scenario Simulator
    st.markdown("---")
    st.subheader("💡 Interactive Tariff & Conservation Simulator")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        reduction_pct = st.slider("Target Demand Reduction (%)", min_value=1, max_value=30, value=10, step=1)
    with sim_col2:
        period_kwh = stats["total_energy_kwh"]
        saved_kwh = period_kwh * (reduction_pct / 100.0)
        saved_money = saved_kwh * tariff_input
        st.markdown(f"""
        <div style="background-color: #FFFEFA; border: 1px solid #E8E2D5; border-radius: 12px; padding: 16px; margin-top: 8px;">
            <div style="font-size: 0.95rem; margin-bottom: 6px;"><b>Energy Conserved:</b> <code style="color: #F45112; font-size: 1rem; font-weight: 700;">{saved_kwh:,.1f} kWh</code></div>
            <div style="font-size: 0.95rem;"><b>Estimated Financial Savings:</b> <code style="color: #059669; font-size: 1rem; font-weight: 700;">~{currency_prefix}{saved_money:,.0f}*</code> (at {currency_prefix}{tariff_input}/kWh)</div>
            <div style="font-size: 0.75rem; color: #8C8476; margin-top: 6px;">*Cost values are illustrative estimates based on user tariff parameter.</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- TAB 5: ABOUT -----------------
with tab_about:
    st.subheader("About the Project")
    st.markdown("""
    ### Smart Energy Forecasting and Conservation Decision Support System
    - **Academic Level:** Final Year TSDA College Mini-Project (Two-Student Team)
    - **Objective:** Bridge statistical time-series forecasting, anomaly detection, and energy engineering to provide practical energy-conservation decision support for facility managers.
    
    #### Dataset Provenance & Integrity
    - **Dataset:** Electricity Hourly Dataset (UCI / Monash Time Series Repository / Zenodo).
    - **Original Reference:** Lai et al. (2017) *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*.
    - **Records:** 321 anonymized client hourly profiles (3 years, 2012–2014, 26,304 hours per series).
    - **Disclaimer:** The data represents public benchmarks and is not physical telemetry from the college campus.
    
    #### Methodology
    1. **Data Preprocessing:** Strict .tsf format parser, time-ordered DatetimeIndex regularizer, missing token interpolation.
    2. **Exploratory Data Analysis:** Hourly diurnal profiles, percentile confidence intervals, day-of-week load heatmaps.
    3. **Time-Series Forecasting:**
       - *Baseline:* Seasonal Naive (24-hour cycle persistence).
       - *Statistical:* Holt-Winters Exponential Smoothing with additive trend and seasonality.
       - *Machine Learning:* Ridge Regression on cyclical hour encoders, lag features (1, 2, 3, 24, 48, 168), and rolling windows.
       - *Validation:* Pure chronological out-of-time split (no data leakage or shuffling), evaluated via MAE and RMSE.
    4. **Anomaly Detection:** Dual-pipeline combining dynamic 7-day rolling Z-score thresholds with unsupervised Isolation Forest.
    5. **Conservation Decision Support:** Heuristic engine detecting overnight baseload leakage, peak surges, and scheduling anomalies to provide concrete energy-saving operational steps.
    """)
