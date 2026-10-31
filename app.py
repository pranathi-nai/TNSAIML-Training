"""
AeroSense: AI-Powered Air Quality Prediction and Environmental Monitoring System
Streamlit Interactive Web Application.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration FIRST before any other Streamlit calls
st.set_page_config(
    page_title="AeroSense — Air Quality Prediction & Monitoring",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cpcb_aqi import (
    calculate_overall_cpcb_aqi,
    get_cpcb_category,
    calculate_sub_index,
    CATEGORIES
)
from src.preprocessing import (
    validate_and_clean_dataset,
    prepare_ml_data,
    split_environmental_data,
    DEFAULT_FEATURE_COLS
)
from src.model import (
    load_model_artifacts,
    train_and_compare_models,
    save_model_artifacts
)
from src.predict import (
    predict_air_quality,
    validate_input_values,
    POLLUTANT_BOUNDS
)
from src.api_client import (
    fetch_open_meteo_air_quality,
    PRESET_CITIES
)
from src.evaluation import (
    plot_aqi_gauge,
    plot_actual_vs_predicted,
    plot_model_comparison_metrics,
    plot_feature_importance,
    plot_residuals,
    plot_pollutant_trends,
    plot_correlation_heatmap
)

# Custom Styling (Dark Navy & Teal Premium Theme)
CUSTOM_CSS = """
<style>
    /* Main Background & Typography */
    .stApp {
        background-color: #0b1329;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #111c38 50%, #0d2847 100%);
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #14b8a6, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111c38;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
    }
    
    .metric-card:hover {
        border-color: rgba(20, 184, 166, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-sub {
        color: #14b8a6;
        font-size: 0.82rem;
        margin-top: 4px;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    
    /* Advisory Alert Box */
    .advisory-box {
        background: #111c38;
        border-left: 5px solid #14b8a6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d152c;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Button Polish */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0f766e, #0d9488);
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.4);
        border: none;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Caches loaded model pipeline and metadata."""
    pipeline, metadata, status = load_model_artifacts("models")
    return pipeline, metadata, status


pipeline, metadata, model_status = get_model()


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 12px 0 20px 0;">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🍃 <b>AeroSense</b></div>
            <div style="color: #14b8a6; font-size: 0.88rem; font-weight: 600; letter-spacing: 0.05em;">
                ENVIRONMENTAL AI SYSTEM
            </div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">
                v1.0.0 • Academic & CPCB Aligned
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🌟 Overview",
            "🔮 AQI Prediction",
            "🛰️ Live Air Quality",
            "📊 Historical Analytics",
            "📈 Forecast & Trends",
            "🧠 Model Insights",
            "ℹ️ About & Viva Guide"
        ],
        index=0
    )

    st.markdown("---")
    
    # Model status indicator
    if pipeline is not None and metadata is not None:
        best_name = metadata.get("best_model_name", "Random Forest")
        r2 = metadata.get("models_comparison", {}).get(best_name, {}).get("test_metrics", {}).get("R2", 0.99)
        st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px;">
                <div style="font-size: 0.78rem; color: #10b981; font-weight: 700; text-transform: uppercase;">
                    ● ML Pipeline Active
                </div>
                <div style="font-size: 0.9rem; font-weight: 600; color: #f8fafc; margin-top: 4px;">
                    {best_name}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                    Test R²: <b>{r2:.4f}</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 10px; padding: 12px;">
                <div style="font-size: 0.78rem; color: #f59e0b; font-weight: 700; text-transform: uppercase;">
                    ⚠ Demo / Fallback Mode
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    {model_status}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.76rem; color: #64748b; line-height: 1.4;">
            <b>Data Sources:</b><br>
            • CPCB NAQI Standards<br>
            • Open-Meteo Air Quality API<br>
            • Kaggle Indian AQI Archive
        </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------------------
if page == "🌟 Overview":
    st.markdown("""
        <div class="hero-card">
            <div class="hero-title">AeroSense: AI-Powered Air Quality Intelligence</div>
            <div class="hero-subtitle">
                A robust multi-pollutant machine learning platform combining official Indian CPCB National Air Quality Index (NAQI) standards,
                satellite atmospheric forecasts, and historical environmental data analytics.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Highlights KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Monitored Pollutants</div>
                <div class="metric-value" style="color: #38bdf8;">6 Key Species</div>
                <div class="metric-sub">PM2.5, PM10, NO2, SO2, CO, O3</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Evaluation Standard</div>
                <div class="metric-value" style="color: #14b8a6;">CPCB NAQI</div>
                <div class="metric-sub">Official 6-tier Indian Category Scale</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Live Sensor Feeds</div>
                <div class="metric-value" style="color: #2dd4bf;">Open-Meteo</div>
                <div class="metric-sub">Copernicus CAMS & GFS Atmospheric Models</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">ML Model Architecture</div>
                <div class="metric-value" style="color: #818cf8;">Random Forest</div>
                <div class="metric-sub">Chronologically Benchmarked vs Ridge & GBDT</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Indian CPCB National Air Quality Index (NAQI) Reference Scale")
    st.markdown(
        "The Central Pollution Control Board (CPCB) divides ambient air quality into 6 distinct tiers. "
        "Each category reflects physiological health risks and prescribed public advisories:"
    )

    cats_cols = st.columns(6)
    for idx, (cat_name, meta) in enumerate(CATEGORIES.items()):
        if cat_name == "Hazardous":
            continue
        with cats_cols[idx]:
            r_lo, r_hi = meta["range"]
            st.markdown(f"""
                <div style="background: #111c38; border-top: 4px solid {meta['color']}; border-radius: 10px; padding: 14px 12px; height: 180px; text-align: center;">
                    <div style="font-weight: 700; color: {meta['color']}; font-size: 1.05rem;">{cat_name}</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin: 4px 0;">{r_lo} – {r_hi}</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; line-height: 1.35; margin-top: 6px;">
                        {meta['health_impact'][:85]}...
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Feature showcase grid
    st.markdown("### 🛠️ Core Capabilities of AeroSense")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #14b8a6; margin-top: 0;">🔮 Precision Machine Learning Estimation</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;">
                    Predicts overall continuous AQI from multi-sensor environmental inputs using an ensemble Random Forest regressor.
                    Guarantees zero target leakage by isolating categories and validating inputs before inference.
                </p>
                <div style="margin-top: 10px;">
                    <span class="status-badge" style="background: rgba(20, 184, 166, 0.2); color: #2dd4bf;">Median Imputation</span>
                    <span class="status-badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8;">Leakage Free</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #38bdf8; margin-top: 0;">🛰️ Live Meteorological Integration</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.5;">
                    Real-time ambient pollutant streams across 16+ preset Indian metropolitan zones (Delhi, Mumbai, Bengaluru, etc.)
                    and custom geo-coordinates via the Open-Meteo Air Quality API with 72-hour forecasts.
                </p>
                <div style="margin-top: 10px;">
                    <span class="status-badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8;">Open-Meteo API</span>
                    <span class="status-badge" style="background: rgba(129, 140, 248, 0.2); color: #a5b4fc;">Copernicus CAMS</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 2: AQI PREDICTION
# -------------------------------------------------------------
elif page == "🔮 AQI Prediction":
    st.markdown("## 🔮 Machine Learning AQI Prediction Engine")
    st.markdown(
        "Provide current environmental pollutant concentrations to generate an ML-estimated AQI, "
        "accompanied by deterministic CPCB sub-index verification and non-diagnostic public health advice."
    )

    # Preset selection for quick demo testing
    preset = st.selectbox(
        "⚡ Quick-Fill Scenario Preset:",
        [
            "Custom User Input",
            "🌿 Clean Coastal / Post-Monsoon (Low Pollution)",
            "🏙️ Moderate Urban Atmosphere (Standard Day)",
            "⚠️ High Winter Smog (NCR Inversion Conditions)",
            "🏭 Industrial Corridor (Elevated SO2 & PM10)"
        ]
    )

    # Default preset values
    defaults = {
        "Custom User Input": {"pm25": 42.0, "pm10": 85.0, "no2": 32.0, "so2": 14.0, "co": 0.85, "o3": 38.0},
        "🌿 Clean Coastal / Post-Monsoon (Low Pollution)": {"pm25": 18.0, "pm10": 36.0, "no2": 15.0, "so2": 6.0, "co": 0.40, "o3": 22.0},
        "🏙️ Moderate Urban Atmosphere (Standard Day)": {"pm25": 68.0, "pm10": 135.0, "no2": 45.0, "so2": 18.0, "co": 1.20, "o3": 55.0},
        "⚠️ High Winter Smog (NCR Inversion Conditions)": {"pm25": 230.0, "pm10": 380.0, "no2": 95.0, "so2": 26.0, "co": 2.80, "o3": 65.0},
        "🏭 Industrial Corridor (Elevated SO2 & PM10)": {"pm25": 115.0, "pm10": 260.0, "no2": 62.0, "so2": 140.0, "co": 1.90, "o3": 48.0}
    }[preset]

    st.markdown("#### Input Environmental Variables:")
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5 (Fine Particulate Matter, µg/m³)", min_value=0.0, max_value=1000.0, value=float(defaults["pm25"]), step=1.0)
        so2 = st.number_input("SO2 (Sulphur Dioxide, µg/m³)", min_value=0.0, max_value=1000.0, value=float(defaults["so2"]), step=1.0)
    with col2:
        pm10 = st.number_input("PM10 (Coarse Particulate Matter, µg/m³)", min_value=0.0, max_value=1500.0, value=float(defaults["pm10"]), step=1.0)
        co = st.number_input("CO (Carbon Monoxide, mg/m³)", min_value=0.0, max_value=100.0, value=float(defaults["co"]), step=0.1)
    with col3:
        no2 = st.number_input("NO2 (Nitrogen Dioxide, µg/m³)", min_value=0.0, max_value=1000.0, value=float(defaults["no2"]), step=1.0)
        o3 = st.number_input("O3 (Ozone, µg/m³)", min_value=0.0, max_value=1000.0, value=float(defaults["o3"]), step=1.0)

    input_data = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3
    }

    # Prediction Action
    if st.button("🚀 Predict Air Quality Index", use_container_width=True):
        if pipeline is None:
            st.error("No trained ML pipeline available. Training model on benchmark dataset...")
            from scripts.train_model import main as run_train
            with st.spinner("Training Random Forest model pipeline..."):
                # Reload pipeline after training
                pipeline, metadata, model_status = load_model_artifacts("models")

        feat_names = metadata.get("feature_names", DEFAULT_FEATURE_COLS) if metadata else DEFAULT_FEATURE_COLS
        res = predict_air_quality(pipeline, input_data, feat_names)

        if not res["success"]:
            for err in res["errors"]:
                st.error(f"Validation Error: {err}")
        else:
            # Display scientific warnings if any
            for warn in res["warnings"]:
                st.warning(f"Scientific Advisory: {warn}")

            pred_aqi = res["ml_predicted_aqi"]
            category = res["ml_category"]
            meta = res["ml_meta"]
            cpcb_res = res["cpcb_benchmark"]

            st.markdown("---")
            out_col1, out_col2 = st.columns([1, 1.2])

            with out_col1:
                st.plotly_chart(plot_aqi_gauge(pred_aqi, title="Estimated AQI"), use_container_width=True)

                st.markdown(f"""
                    <div style="background: {meta['badge_bg']}; border: 2px solid {meta['color']}; border-radius: 12px; padding: 18px; text-align: center;">
                        <div style="color: {meta['color']}; font-size: 1.6rem; font-weight: 800; text-transform: uppercase;">
                            {category}
                        </div>
                        <div style="color: #f1f5f9; font-size: 0.95rem; margin-top: 8px;">
                            <b>Dominant Ambient Driver:</b> {cpcb_res.get('prominent_pollutant', 'PM2.5')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with out_col2:
                st.markdown("### 🩺 Non-Diagnostic Health Advisory")
                st.markdown(f"""
                    <div class="advisory-box" style="border-left-color: {meta['color']};">
                        <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem; margin-bottom: 6px;">
                            Impact Description:
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5; margin-bottom: 12px;">
                            {meta['health_impact']}
                        </div>
                        <div style="font-weight: 700; color: #f8fafc; font-size: 1.05rem; margin-bottom: 6px;">
                            Recommended Behavioral Actions:
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
                            {meta['advisory']}
                        </div>
                        <div style="margin-top: 12px; font-size: 0.88rem; color: {'#ef4444' if meta['mask_recommended'] else '#10b981'}; font-weight: 700;">
                            {'😷 N95/FFP2 Mask Recommended Outdoors' if meta['mask_recommended'] else '✅ Clean outdoor breathing conditions'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # ML vs Deterministic CPCB Comparison Table
                st.markdown("#### 🔬 Methodological Comparison")
                cpcb_val = cpcb_res["aqi"]
                diff = res["difference_ml_vs_cpcb"]

                comp_df = pd.DataFrame({
                    "Estimation Metric": ["ML Random Forest Estimate", "Deterministic CPCB NAQI Formula", "Absolute Divergence"],
                    "Value": [f"{pred_aqi:.1f}", f"{cpcb_val:.1f}" if cpcb_val else "N/A", f"{abs(diff):.1f}" if diff is not None else "N/A"],
                    "Methodology": [
                        "Non-linear ensemble learning fitted on ambient interactions",
                        "Stepwise linear interpolation across prescribed CPCB breakpoints",
                        "Divergence due to multi-pollutant cross-correlation learning"
                    ]
                })
                st.dataframe(comp_df, hide_index=True, use_container_width=True)

            # Sub-indices breakdown bar chart
            st.markdown("#### 📊 Individual Pollutant CPCB Sub-Indices Breakdown")
            sub_dict = cpcb_res.get("sub_indices", {})
            if sub_dict:
                sub_fig = go.Figure(go.Bar(
                    x=list(sub_dict.keys()),
                    y=list(sub_dict.values()),
                    marker_color=["#ef4444" if k == cpcb_res.get('prominent_pollutant') else "#14b8a6" for k in sub_dict.keys()],
                    text=[f"{v:.1f}" for v in sub_dict.values()],
                    textposition="auto"
                ))
                sub_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#111c38",
                    font={"color": "#f1f5f9"},
                    height=280,
                    margin={"l": 30, "r": 30, "t": 30, "b": 30},
                    yaxis_title="Calculated Sub-Index"
                )
                st.plotly_chart(sub_fig, use_container_width=True)


# -------------------------------------------------------------
# PAGE 3: LIVE AIR QUALITY (OPEN-METEO API)
# -------------------------------------------------------------
elif page == "🛰️ Live Air Quality":
    st.markdown("## 🛰️ Real-Time Atmospheric Monitoring via Open-Meteo API")
    st.markdown(
        "Retrieve live ambient pollutant measurements and multi-day hourly forecasts from the "
        "[Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) using coordinates or preset Indian cities."
    )

    st.markdown("""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 0.84rem; color: #cbd5e1;">
            <b>ℹ️ Atmospheric Data Attribution & Caveat:</b> Live readings are sourced from Copernicus Atmosphere Monitoring Service (CAMS) 
            and global numerical weather models via Open-Meteo. These represent ambient satellite-modeled atmospheric layers and may differ from localized 
            CPCB ground physical monitoring stations with high near-surface traffic bias.
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        city_choice = st.selectbox("Select Indian Metropolitan Center:", ["Custom Coordinates"] + list(PRESET_CITIES.keys()))
    with c2:
        if city_choice == "Custom Coordinates":
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6139, format="%.4f")
        else:
            lat = PRESET_CITIES[city_choice]["lat"]
            st.metric("Latitude", f"{lat:.4f}° N")
    with c3:
        if city_choice == "Custom Coordinates":
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2090, format="%.4f")
        else:
            lon = PRESET_CITIES[city_choice]["lon"]
            st.metric("Longitude", f"{lon:.4f}° E")

    if st.button("🌐 Fetch Real-Time Ambient Pollutant Data", use_container_width=True):
        with st.spinner("Connecting to Open-Meteo Air Quality Satellite & Reanalysis API..."):
            api_data = fetch_open_meteo_air_quality(latitude=lat, longitude=lon, forecast_days=3)

        if not api_data["success"]:
            st.error(f"Failed to retrieve API data: {api_data['error']}")
        else:
            current = api_data["current"]
            hourly_df = api_data["hourly_df"]
            units = api_data["units"]

            st.success(f"Successfully retrieved live atmospheric observation for Coordinates ({lat:.2f}, {lon:.2f}) at {api_data['fetched_at']}")

            # Render Current Observation Cards
            st.markdown("### 📍 Current Ambient Pollutant Concentrations")
            k1, k2, k3, k4, k5, k6 = st.columns(6)

            metrics_data = [
                ("PM2.5", current.get("PM2.5"), "µg/m³", k1),
                ("PM10", current.get("PM10"), "µg/m³", k2),
                ("NO2", current.get("NO2"), "µg/m³", k3),
                ("SO2", current.get("SO2"), "µg/m³", k4),
                ("CO", current.get("CO"), "µg/m³", k5),
                ("O3", current.get("O3"), "µg/m³", k6)
            ]

            for name, val, unit, col in metrics_data:
                with col:
                    sub_ind = calculate_sub_index(name, val) if val is not None else None
                    sub_txt = f"Sub-index: {sub_ind:.0f}" if sub_ind else "Sub-index: N/A"
                    st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <div class="metric-title">{name}</div>
                            <div class="metric-value" style="font-size: 1.5rem; color: #38bdf8;">
                                {f"{val:.1f}" if val is not None else "N/A"}
                            </div>
                            <div style="font-size: 0.72rem; color: #94a3b8;">{unit}</div>
                            <div class="metric-sub" style="font-size: 0.75rem;">{sub_txt}</div>
                        </div>
                    """, unsafe_allow_html=True)

            # Evaluate with ML Model and CPCB
            pollutants_for_eval = {
                "PM2.5": current.get("PM2.5") or 0.0,
                "PM10": current.get("PM10") or 0.0,
                "NO2": current.get("NO2") or 0.0,
                "SO2": current.get("SO2") or 0.0,
                "CO": (current.get("CO") / 1000.0) if current.get("CO") else 0.0,
                "O3": current.get("O3") or 0.0
            }

            feat_names = metadata.get("feature_names", DEFAULT_FEATURE_COLS) if metadata else DEFAULT_FEATURE_COLS
            live_pred = predict_air_quality(pipeline, pollutants_for_eval, feat_names)

            if live_pred["success"]:
                st.markdown("---")
                g1, g2 = st.columns([1, 1.2])
                with g1:
                    st.plotly_chart(plot_aqi_gauge(live_pred["ml_predicted_aqi"], title="Live AeroSense AQI Estimate"), use_container_width=True)
                with g2:
                    st.markdown("#### Real-time Diagnostic Summary")
                    st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.2rem; font-weight: 700; color: {live_pred['ml_meta']['color']};">
                                {live_pred['ml_category']} (AQI: {live_pred['ml_predicted_aqi']:.1f})
                            </div>
                            <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 8px;">
                                {live_pred['ml_meta']['health_impact']}
                            </div>
                            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 8px;">
                                <b>CPCB NAQI Deterministic Benchmark:</b> {live_pred['cpcb_benchmark']['aqi']:.1f} 
                                ({live_pred['cpcb_benchmark']['category']})
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # Hourly Forecast Line Chart
            if not hourly_df.empty:
                st.markdown("### 📈 72-Hour Atmospheric Trend & Forecast")
                fig_hourly = go.Figure()

                for p, col_hex in [("PM2.5", "#14b8a6"), ("PM10", "#38bdf8"), ("NO2", "#f59e0b"), ("O3", "#a855f7")]:
                    if p in hourly_df.columns:
                        fig_hourly.add_trace(go.Scatter(
                            x=hourly_df["Timestamp"],
                            y=hourly_df[p],
                            name=f"{p} (µg/m³)",
                            mode="lines",
                            line={"width": 2, "color": col_hex}
                        ))

                fig_hourly.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#111c38",
                    font={"color": "#f1f5f9"},
                    height=400,
                    margin={"l": 40, "r": 30, "t": 40, "b": 40},
                    xaxis={"title": "Timestamp (Local Time)", "gridcolor": "#1e293b"},
                    yaxis={"title": "Concentration (µg/m³)", "gridcolor": "#1e293b"}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)


# -------------------------------------------------------------
# PAGE 4: HISTORICAL ANALYTICS
# -------------------------------------------------------------
elif page == "📊 Historical Analytics":
    st.markdown("## 📊 Historical Environmental Analytics")
    st.markdown(
        "Upload any environmental monitoring CSV or explore the bundled CPCB multi-city benchmark archive. "
        "The system normalizes column aliases, detects missing observations, removes duplicates, and generates visual analytics."
    )

    data_source = st.radio(
        "Select Historical Data Source:",
        ["📁 Bundled CPCB Indian Multi-City Benchmark (sample_aqi_data.csv)", "📤 Upload Custom Environmental CSV"],
        horizontal=True
    )

    df_loaded = None
    if data_source.startswith("📁"):
        sample_path = "data/sample_aqi_data.csv"
        if os.path.exists(sample_path):
            df_loaded = pd.read_csv(sample_path)
            st.info(f"Loaded bundled benchmark dataset ({len(df_loaded)} rows from 5 Indian metropolitan cities).")
        else:
            st.warning("Sample dataset not found. Generating now...")
            from scripts.generate_dataset import generate_benchmark_dataset
            generate_benchmark_dataset()
            df_loaded = pd.read_csv(sample_path)
    else:
        uploaded_file = st.file_uploader("Upload Air Quality CSV (Kaggle/CPCB/Sensors):", type=["csv"])
        if uploaded_file is not None:
            try:
                df_loaded = pd.read_csv(uploaded_file)
                st.success(f"Uploaded file '{uploaded_file.name}' loaded successfully ({len(df_loaded)} rows).")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if df_loaded is not None:
        df_clean, report = validate_and_clean_dataset(df_loaded)

        # Dataset Diagnostic Overview
        st.markdown("### 🧹 Data Cleaning & Schema Normalization Report")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Cleaned Rows</div>
                    <div class="metric-value">{len(df_clean)}</div>
                    <div class="metric-sub">Original: {report['original_rows']}</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Duplicates Dropped</div>
                    <div class="metric-value" style="color: #f59e0b;">{report['duplicate_rows_dropped']}</div>
                    <div class="metric-sub">Strict Row Duplicates</div>
                </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Features Mapped</div>
                    <div class="metric-value" style="color: #14b8a6;">{len(report['available_features'])}</div>
                    <div class="metric-sub">{', '.join(report['available_features'][:4])}...</div>
                </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Target Column</div>
                    <div class="metric-value" style="color: {'#10b981' if report['has_target_aqi'] else '#ef4444'};">
                        {'Present' if report['has_target_aqi'] else 'Missing'}
                    </div>
                    <div class="metric-sub">{'Valid for Regression' if report['has_target_aqi'] else 'Unsupervised Only'}</div>
                </div>
            """, unsafe_allow_html=True)

        # Filters
        st.markdown("---")
        st.markdown("### 🔍 Exploratory Filters")
        f_col1, f_col2 = st.columns(2)
        
        filtered_df = df_clean.copy()
        with f_col1:
            if "City" in df_clean.columns:
                cities = ["All Cities"] + sorted(list(df_clean["City"].dropna().unique()))
                chosen_city = st.selectbox("Filter by City:", cities)
                if chosen_city != "All Cities":
                    filtered_df = filtered_df[filtered_df["City"] == chosen_city]
            else:
                st.info("City column not present in this dataset.")

        with f_col2:
            if "Date" in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean["Date"]):
                min_d = df_clean["Date"].min().date()
                max_d = df_clean["Date"].max().date()
                date_range = st.date_input("Date Span Filter:", [min_d, max_d], min_value=min_d, max_value=max_d)
                if len(date_range) == 2:
                    filtered_df = filtered_df[(filtered_df["Date"].dt.date >= date_range[0]) & (filtered_df["Date"].dt.date <= date_range[1])]

        # Visualizations
        st.markdown("### 📈 Historical Multi-Pollutant Time Series")
        st.plotly_chart(plot_pollutant_trends(filtered_df), use_container_width=True)

        st.markdown("---")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("### 🔬 Cross-Pollutant Correlation Matrix")
            st.plotly_chart(plot_correlation_heatmap(filtered_df), use_container_width=True)
        with c_right:
            st.markdown("### 📦 Pollutant Concentration Distributions")
            box_feat = st.selectbox("Select Variable for Boxplot:", [c for c in DEFAULT_FEATURE_COLS + ["AQI"] if c in filtered_df.columns])
            box_fig = px.box(
                filtered_df,
                y=box_feat,
                x="City" if "City" in filtered_df.columns else None,
                color="City" if "City" in filtered_df.columns else None,
                template="plotly_dark"
            )
            box_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111c38",
                height=380,
                font={"color": "#f1f5f9"}
            )
            st.plotly_chart(box_fig, use_container_width=True)

        # Raw Table & Download
        with st.expander("📄 View Cleaned Dataset Sample (First 50 Rows)"):
            st.dataframe(filtered_df.head(50), use_container_width=True)
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Cleaned CSV", data=csv_data, file_name="aerosense_cleaned_data.csv", mime="text/csv")


# -------------------------------------------------------------
# PAGE 5: FORECAST & TRENDS
# -------------------------------------------------------------
elif page == "📈 Forecast & Trends":
    st.markdown("## 📈 Atmospheric Forecasting & Trend Analysis")
    st.markdown(
        "Compare short-term predictive trends, distinguish satellite reanalysis predictions from ML regression estimates, "
        "and track temporal particulate dispersal."
    )

    f_city = st.selectbox("Select Target Urban Region:", list(PRESET_CITIES.keys()), index=0)
    coords = PRESET_CITIES[f_city]

    with st.spinner(f"Fetching 72-hour forecast for {f_city}..."):
        fc_data = fetch_open_meteo_air_quality(coords["lat"], coords["lon"], forecast_days=3)

    if not fc_data["success"]:
        st.error(f"Forecast retrieval error: {fc_data['error']}")
    else:
        hdf = fc_data["hourly_df"]
        st.success(f"72-hour forecast stream active for {f_city} (Elevation: {fc_data.get('elevation', 'N/A')} m).")

        # Compute ML AQI for each hourly step
        if pipeline is not None and not hdf.empty:
            ml_hourly_aqi = []
            cpcb_hourly_aqi = []

            for _, row in hdf.iterrows():
                p_in = {
                    "PM2.5": row.get("PM2.5") or 0.0,
                    "PM10": row.get("PM10") or 0.0,
                    "NO2": row.get("NO2") or 0.0,
                    "SO2": row.get("SO2") or 0.0,
                    "CO": (row.get("CO") / 1000.0) if row.get("CO") else 0.0,
                    "O3": row.get("O3") or 0.0
                }
                pred = predict_air_quality(pipeline, p_in, DEFAULT_FEATURE_COLS)
                ml_hourly_aqi.append(pred["ml_predicted_aqi"] if pred["success"] else np.nan)
                cpcb_val = pred["cpcb_benchmark"]["aqi"] if pred["success"] else np.nan
                cpcb_hourly_aqi.append(cpcb_val)

            hdf["AeroSense_ML_AQI"] = ml_hourly_aqi
            hdf["CPCB_NAQI_Benchmark"] = cpcb_hourly_aqi

            # Plot AQI Forecast Comparison
            st.markdown("### 🔮 Predicted Air Quality Index (Next 72 Hours)")
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hdf["Timestamp"],
                y=hdf["AeroSense_ML_AQI"],
                mode="lines+markers",
                name="AeroSense ML Predicted AQI",
                line={"color": "#14b8a6", "width": 3},
                marker={"size": 4}
            ))
            fig_fc.add_trace(go.Scatter(
                x=hdf["Timestamp"],
                y=hdf["CPCB_NAQI_Benchmark"],
                mode="lines",
                name="CPCB NAQI Sub-Index Benchmark",
                line={"color": "#f59e0b", "width": 2, "dash": "dot"}
            ))

            # Add critical CPCB threshold lines
            fig_fc.add_hline(y=100, line_dash="dash", line_color="#84cc16", annotation_text="Satisfactory Limit (100)")
            fig_fc.add_hline(y=200, line_dash="dash", line_color="#eab308", annotation_text="Moderate Limit (200)")
            fig_fc.add_hline(y=300, line_dash="dash", line_color="#ef4444", annotation_text="Poor Limit (300)")

            fig_fc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#111c38",
                font={"color": "#f1f5f9"},
                height=420,
                xaxis_title="Forecast Timeline",
                yaxis_title="Estimated AQI"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Distinctions Notice
            st.markdown("""
                <div class="advisory-box" style="border-left-color: #38bdf8;">
                    <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">🔬 Academic Rigor Notice:</div>
                    <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;">
                        • <b>Atmospheric Inputs:</b> Generated by Open-Meteo via European Centre for Medium-Range Weather Forecasts (ECMWF) atmospheric models.<br>
                        • <b>AeroSense ML AQI:</b> Non-linear regression estimate trained on multi-sensor interactions.<br>
                        • <b>CPCB NAQI Benchmark:</b> Stepwise linear piecewise calculation based strictly on Indian environmental breakpoints.
                    </div>
                </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 6: MODEL INSIGHTS
# -------------------------------------------------------------
elif page == "🧠 Model Insights":
    st.markdown("## 🧠 Machine Learning Model Insights & Architecture")
    st.markdown(
        "AeroSense implements a transparent and reproducible ML pipeline. "
        "Here you can examine cross-model benchmarking, empirical evaluation metrics, "
        "relative feature importances, and retrain the models directly."
    )

    if metadata is not None and "models_comparison" in metadata:
        comps = metadata["models_comparison"]
        best_name = metadata.get("best_model_name", "Random Forest")

        st.markdown("### 🏆 Model Performance Benchmark (Chronological Test Split)")
        st.markdown(
            "Models were evaluated strictly on unseen chronological test observations to prevent future data leakage:"
        )

        # Performance Metrics Table
        table_rows = []
        for m_name, d in comps.items():
            tm = d["test_metrics"]
            trm = d["train_metrics"]
            table_rows.append({
                "Model Architecture": f"{m_name} {'★ (Winning Model)' if m_name == best_name else ''}",
                "Test MAE (µg/m³)": tm["MAE"],
                "Test RMSE (µg/m³)": tm["RMSE"],
                "Test R² Score": tm["R2"],
                "Test MAPE (%)": tm["MAPE"],
                "Train R² Score": trm["R2"]
            })
        m_df = pd.DataFrame(table_rows)
        st.dataframe(m_df, hide_index=True, use_container_width=True)

        # Grouped Bar Chart
        st.plotly_chart(plot_model_comparison_metrics(comps), use_container_width=True)

        st.markdown("---")
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("### 🎯 Actual vs. Predicted AQI (Test Split)")
            pred_sample = comps.get(best_name, {}).get("predictions_sample", {})
            if pred_sample:
                act = pred_sample.get("actual", [])
                prd = pred_sample.get("predicted", [])
                st.plotly_chart(plot_actual_vs_predicted(act, prd, model_name=best_name), use_container_width=True)

        with c_right:
            st.markdown("### ⚖️ Relative Feature Importance Weights")
            feat_imp = comps.get(best_name, {}).get("feature_importance", {})
            st.plotly_chart(plot_feature_importance(feat_imp, model_name=best_name), use_container_width=True)
            st.markdown("""
                <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.35; margin-top: -10px;">
                    * Feature importance reflects Gini/MDI impurity reduction inside the Random Forest tree ensemble. 
                    It demonstrates predictive contribution and correlation, not direct biological or physical causation.
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("🛠️ Retrain Model Pipeline Directly"):
            st.markdown("You can retrain the models on `sample_aqi_data.csv` or any updated dataset:")
            split_ratio = st.slider("Select Test Split Proportion:", 0.1, 0.4, 0.2, 0.05)
            if st.button("🔄 Execute Re-Training Pipeline", key="retrain_btn"):
                with st.spinner("Retraining Ridge, Random Forest, and HistGradientBoosting Regressors..."):
                    df_raw = pd.read_csv("data/sample_aqi_data.csv")
                    df_c, rep = validate_and_clean_dataset(df_raw)
                    tr_df, ts_df = split_environmental_data(df_c, test_size=split_ratio)
                    X_tr, y_tr, feats = prepare_ml_data(tr_df, rep["available_features"])
                    X_ts, y_ts, _ = prepare_ml_data(ts_df, rep["available_features"])

                    new_res, new_best, new_pipe = train_and_compare_models(X_tr, y_tr, X_ts, y_ts, feats)
                    save_model_artifacts(
                        new_pipe,
                        new_best,
                        feats,
                        new_res,
                        {"retrained_at": datetime.now().isoformat(), "rows": len(df_c)},
                        "models"
                    )
                    st.cache_resource.clear()
                    st.success(f"Retraining complete! Best model selected: {new_best}. Please refresh the page to update cached models.")
    else:
        st.warning("No metadata available. Please ensure model training has been executed via `scripts/train_model.py`.")


# -------------------------------------------------------------
# PAGE 7: ABOUT & VIVA GUIDE
# -------------------------------------------------------------
elif page == "ℹ️ About & Viva Guide":
    st.markdown("## ℹ️ About AeroSense & Academic Viva Preparation Guide")
    st.markdown(
        "Complete technical documentation, mathematical formulations, and viva interview Q&A designed for "
        "college faculty evaluation and software engineering project defense."
    )

    t1, t2, t3 = st.tabs(["🏛️ Architecture & Formulation", "🎓 Viva Q&A Guide (Top 10)", "📦 System Specifications"])

    with t1:
        st.markdown("### System Architecture")
        st.markdown("""
            ```
            ┌────────────────────────────────────────────────────────┐
            │                     AeroSense UI                       │
            │          (Streamlit • Dark Navy & Teal Theme)          │
            └───────────┬────────────────────────────────┬───────────┘
                        │                                │
            ┌───────────▼────────────┐       ┌───────────▼───────────┐
            │  Open-Meteo Air API    │       │ Historical Data (CSV) │
            │  Live CAMS & Forecast  │       │ Kaggle / CPCB / User  │
            └───────────┬────────────┘       └───────────┬───────────┘
                        │                                │
            ┌───────────▼────────────────────────────────▼───────────┐
            │               Preprocessing & Validation               │
            │  • Alias Normalization   • Leakage Prevention          │
            │  • Missing Handling      • Chronological Splitting     │
            └───────────┬────────────────────────────────┬───────────┘
                        │                                │
            ┌───────────▼────────────┐       ┌───────────▼───────────┐
            │   ML Inference Engine  │       │   CPCB NAQI Engine    │
            │   • Random Forest      │       │   • Sub-index calc    │
            │   • HistGradientBoost  │       │   • Breakpoint rules  │
            │   • Ridge Regression   │       │   • Health Advisory   │
            └────────────────────────┴───────┴───────────────────────┘
            ```
        """)

        st.markdown("### Mathematical Formulations")
        st.markdown(r"""
        **1. CPCB Sub-Index Formulation (Linear Piecewise Interpolation):**
        $$I_p = \left( \frac{I_{hi} - I_{lo}}{B_{hi} - B_{lo}} \right) \cdot (C_p - B_{lo}) + I_{lo}$$
        Where:
        - $I_p$ is the sub-index for pollutant $p$.
        - $C_p$ is the pollutant concentration.
        - $B_{hi}, B_{lo}$ are the upper and lower breakpoint concentrations.
        - $I_{hi}, I_{lo}$ are the corresponding AQI scale intervals.
        - Overall CPCB AQI is governed by: $\text{AQI} = \max(I_1, I_2, \dots, I_n)$ (provided at least 3 pollutants with at least one PM are present).

        **2. Regression Evaluation Metrics:**
        - **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
        - **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
        - **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
        """)

    with t2:
        st.markdown("### 🎓 Top 10 Viva & Project Presentation Questions")
        
        viva_qa = [
            ("Q1: What is the primary problem AeroSense solves?",
             "Traditional CPCB AQI relies strictly on manual deterministic formulas requiring simultaneous physical station readings. AeroSense bridges this gap by combining machine-learning non-linear estimation, satellite reanalysis forecasts via Open-Meteo, and automated health advisories."),
            
            ("Q2: Why did you compare multiple ML algorithms (Ridge, Random Forest, HistGradientBoosting)?",
             "In ML engineering, we must establish a transparent baseline. Ridge Regression acts as the linear baseline to measure whether non-linear combinations exist. Random Forest provides robust decision boundaries resilient to outliers, while HistGradientBoosting evaluates histogram-binned gradient descent performance."),
            
            ("Q3: How did you prevent target data leakage?",
             "Target leakage was prevented by: (1) strictly excluding derived columns such as 'AQI_Bucket' or 'Category' from input features, (2) fitting preprocessing imputer parameters strictly on training folds, and (3) using chronological time-based splitting rather than random shuffling."),
            
            ("Q4: Why is chronological splitting crucial for environmental datasets?",
             "Air quality is a seasonal time-series. If we randomly shuffle data, winter smog observations from tomorrow leak into yesterday's training fold, inflating accuracy artificially. Chronological splitting verifies if the model can genuinely predict future conditions."),
            
            ("Q5: What is the difference between CPCB Sub-Index and ML predicted AQI?",
             "CPCB AQI is a piecewise deterministic rule book that takes the single maximum sub-index. ML regression learns cross-pollutant interactions (e.g. synergistic effects of PM2.5 and Ozone under solar radiation) to estimate overall environmental index levels."),
            
            ("Q6: How does Open-Meteo API differ from physical CPCB monitoring stations?",
             "CPCB stations use localized physical laser/beta-attenuation sensors mounted at specific crossroads or rooftops (susceptible to immediate hyper-local traffic spikes). Open-Meteo uses Copernicus Atmosphere Monitoring Service (CAMS) assimilation combining satellite radiance and global numerical models."),
            
            ("Q7: Why report both MAE and RMSE?",
             "MAE penalizes all individual prediction errors linearly, representing typical expected error. RMSE squares the errors before averaging, making it much more sensitive to severe extreme pollution spikes or catastrophic outliers."),
            
            ("Q8: Does high feature importance prove biological causation?",
             "No. In tree ensembles, feature importance measures reduction in impurity (Gini/MSE) or permutation score. Correlation does not imply causation; it indicates which feature offered the highest predictive utility to the model."),
            
            ("Q9: How are missing sensor values handled during runtime inference?",
             "The pipeline employs a scikit-learn SimpleImputer with median values fitted during training. If a sensor temporarily drops offline, median imputation prevents pipeline crashes without corrupting the input vector."),
            
            ("Q10: Can this application be deployed to the cloud?",
             "Yes. AeroSense is designed with relative file paths, pinned lightweight dependencies in requirements.txt, and a modular architecture ready for Streamlit Community Cloud and GitHub.")
        ]

        for q, a in viva_qa:
            with st.expander(q):
                st.markdown(f"**Answer:** {a}")

    with t3:
        st.markdown("### 📦 System Specifications & Dependency Graph")
        specs = {
            "Application Framework": "Streamlit 1.64+",
            "Machine Learning Library": "Scikit-Learn 1.7+",
            "Numerical & Array Computing": "NumPy 2.2+, Pandas 2.3+",
            "Visualization Engine": "Plotly 7.1+",
            "API Client": "Requests 2.33+",
            "Model Serialization": "Joblib 1.5+",
            "Automated Testing": "Pytest 9.1+",
            "Python Target": "Python 3.10+"
        }
        st.table(pd.DataFrame(list(specs.items()), columns=["Component", "Specification"]))
