"""
Plotly visualization, evaluation analytics, and interactive charts for AeroSense.
Styled with a modern dark navy and teal palette.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Theme styling constants
THEME_BG = "#0b1329"
CARD_BG = "#111c38"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"
TEAL = "#14b8a6"
CYAN = "#38bdf8"
AMBER = "#f59e0b"
ROSE = "#f43f5e"
EMERALD = "#10b981"

# Standard layout generator
def apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
        xaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "tickfont": {"color": MUTED_TEXT}
        },
        yaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": GRID_COLOR,
            "tickfont": {"color": MUTED_TEXT}
        },
        legend={
            "bgcolor": "rgba(17, 28, 56, 0.7)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
            "font": {"color": TEXT_COLOR}
        }
    )
    return fig


def plot_aqi_gauge(aqi_value: float, title: str = "Estimated AQI (CPCB Scale)") -> go.Figure:
    """
    Renders an interactive semicircular gauge showing AQI with standard CPCB color zones.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": TEXT_COLOR}},
        number={"font": {"size": 44, "color": TEXT_COLOR, "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 2, "tickcolor": MUTED_TEXT},
            "bar": {"color": "#ffffff", "thickness": 0.22},
            "bgcolor": CARD_BG,
            "borderwidth": 1,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0, 50], "color": "#10b981"},       # Good
                {"range": [50, 100], "color": "#84cc16"},     # Satisfactory
                {"range": [100, 200], "color": "#eab308"},    # Moderate
                {"range": [200, 300], "color": "#f97316"},    # Poor
                {"range": [300, 400], "color": "#ef4444"},    # Very Poor
                {"range": [400, 500], "color": "#991b1b"},    # Severe
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.75,
                "value": min(aqi_value, 500)
            }
        }
    ))
    return apply_theme(fig, "", height=280)


def plot_actual_vs_predicted(actual: List[float], predicted: List[float], model_name: str = "Trained Model") -> go.Figure:
    """
    Renders a scatter plot comparing actual vs. predicted values with an identity y = x diagonal.
    """
    fig = go.Figure()

    # Identity reference line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Prediction (y = x)",
        line={"color": ROSE, "dash": "dash", "width": 2}
    ))

    # Actual vs Predicted points
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode="markers",
        name="Test Observations",
        marker={
            "color": TEAL,
            "size": 8,
            "opacity": 0.75,
            "line": {"color": "#042f2e", "width": 1}
        },
        hovertemplate="Actual AQI: %{x:.1f}<br>Predicted AQI: %{y:.1f}<extra></extra>"
    ))

    fig.update_xaxes(title_text="Actual Observed AQI")
    fig.update_yaxes(title_text="Model Predicted AQI")
    return apply_theme(fig, f"Actual vs. Predicted AQI — {model_name}")


def plot_model_comparison_metrics(comparison_dict: Dict[str, Any]) -> go.Figure:
    """
    Grouped bar chart comparing MAE, RMSE, and R² score across trained algorithms.
    """
    models = list(comparison_dict.keys())
    mae_vals = [comparison_dict[m]["test_metrics"]["MAE"] for m in models]
    rmse_vals = [comparison_dict[m]["test_metrics"]["RMSE"] for m in models]
    r2_vals = [comparison_dict[m]["test_metrics"]["R2"] for m in models]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="MAE (Lower is Better)",
        x=models,
        y=mae_vals,
        marker_color=TEAL,
        text=mae_vals,
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="RMSE (Lower is Better)",
        x=models,
        y=rmse_vals,
        marker_color=CYAN,
        text=rmse_vals,
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        name="R² Score (Higher is Better)",
        x=models,
        y=r2_vals,
        marker_color=AMBER,
        text=r2_vals,
        textposition="auto"
    ))

    fig.update_layout(barmode="group")
    return apply_theme(fig, "Cross-Model Performance Benchmarking (Test Split)")


def plot_feature_importance(importance_dict: Dict[str, float], model_name: str = "Random Forest") -> go.Figure:
    """
    Horizontal bar chart showing relative feature importance percentages.
    """
    features = list(importance_dict.keys())
    values = list(importance_dict.values())

    # Reverse for top-down display in horizontal bar
    features.reverse()
    values.reverse()

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker={
            "color": values,
            "colorscale": [[0, "#0e7490"], [1, "#2dd4bf"]],
            "line": {"color": "#0f766e", "width": 1}
        },
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))

    fig.update_xaxes(title_text="Relative Importance Contribution (%)")
    fig.update_yaxes(title_text="Pollutant Feature")
    return apply_theme(fig, f"Feature Importance Weights ({model_name})")


def plot_residuals(actual: List[float], predicted: List[float]) -> go.Figure:
    """
    Renders residual distribution (errors = actual - predicted).
    """
    residuals = np.array(actual) - np.array(predicted)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=25,
        marker_color=CYAN,
        opacity=0.85,
        name="Residuals (Error)"
    ))

    fig.add_vline(x=0, line_dash="dash", line_color=ROSE, annotation_text="Zero Error")
    fig.update_xaxes(title_text="Residual Error (Actual - Predicted)")
    fig.update_yaxes(title_text="Frequency Count")
    return apply_theme(fig, "Model Residuals Distribution")


def plot_pollutant_trends(df: pd.DataFrame, pollutants: Optional[List[str]] = None) -> go.Figure:
    """
    Interactive time-series line chart of pollutant trends over time.
    """
    if pollutants is None:
        pollutants = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "O3", "AQI"] if c in df.columns]

    date_col = "Date" if "Date" in df.columns else None

    fig = go.Figure()
    palette = [TEAL, CYAN, AMBER, ROSE, EMERALD, "#a855f7"]

    x_axis = df[date_col] if date_col else df.index

    for idx, pol in enumerate(pollutants):
        if pol in df.columns:
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=df[pol],
                mode="lines",
                name=pol,
                line={"width": 2, "color": palette[idx % len(palette)]}
            ))

    fig.update_xaxes(
        title_text="Timeline",
        rangeslider_visible=True if date_col else False
    )
    fig.update_yaxes(title_text="Concentration (µg/m³)")
    return apply_theme(fig, "Environmental Pollutant Trends Over Time", height=450)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> go.Figure:
    """
    Interactive heatmap of Pearson correlation coefficients between pollutants.
    """
    if numeric_cols is None:
        numeric_cols = [c for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"] if c in df.columns]

    corr = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[[0, "#1e1b4b"], [0.5, "#0e7490"], [1, "#2dd4bf"]],
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": TEXT_COLOR}
    ))

    return apply_theme(fig, "Cross-Pollutant Pearson Correlation Matrix", height=420)
