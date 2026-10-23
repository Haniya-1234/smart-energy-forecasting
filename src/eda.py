import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple

def compute_summary_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes key descriptive statistics for electricity consumption.
    """
    series = df["consumption_kw"]
    stats = {
        "count": float(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "total_energy_kwh": float(series.sum()) # Since hourly, kW * 1h = kWh
    }
    return stats

def plot_historical_series(df: pd.DataFrame, series_name: str) -> go.Figure:
    """
    Creates an interactive historical consumption time series chart.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["consumption_kw"],
            mode="lines",
            name="Electricity Load (kW)",
            line=dict(color="#F45112", width=1.6),
            hovertemplate="<b>Date/Time:</b> %{x|%Y-%m-%d %H:%M}<br><b>Load:</b> %{y:.2f} kW<extra></extra>"
        )
    )
    # Add 7-day rolling mean
    rolling_7d = df["consumption_kw"].rolling(window=24 * 7, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=rolling_7d,
            mode="lines",
            name="7-Day Rolling Trend",
            line=dict(color="#2563EB", width=2, dash="dash"),
            hovertemplate="<b>7-Day Trend:</b> %{y:.2f} kW<extra></extra>"
        )
    )
    fig.update_layout(
        title=dict(text=f"Historical Electricity Demand Profile ({series_name})", font=dict(color="#171717", size=15, weight="bold")),
        xaxis=dict(
            title="Timeline",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Consumption (kW)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True,
            zeroline=False
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#171717")),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    return fig

def plot_hourly_weekly_profile(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Generates:
    1. Average daily profile by hour of day (with 10th-90th percentile ribbon)
    2. Weekly heatmap (Hour of Day vs Day of Week)
    """
    temp_df = df.copy()
    temp_df["hour"] = temp_df["timestamp"].dt.hour
    temp_df["day_name"] = temp_df["timestamp"].dt.day_name()
    temp_df["day_of_week"] = temp_df["timestamp"].dt.dayofweek

    # Hourly Profile Aggregations
    hourly_stats = temp_df.groupby("hour")["consumption_kw"].agg(
        mean="mean",
        median="median",
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90)
    ).reset_index()

    fig_hourly = go.Figure()
    # Shaded confidence band (10th-90th percentile)
    fig_hourly.add_trace(
        go.Scatter(
            x=list(hourly_stats["hour"]) + list(hourly_stats["hour"][::-1]),
            y=list(hourly_stats["p90"]) + list(hourly_stats["p10"][::-1]),
            fill="toself",
            fillcolor="rgba(244, 81, 18, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="10th-90th Percentile Range"
        )
    )
    fig_hourly.add_trace(
        go.Scatter(
            x=hourly_stats["hour"],
            y=hourly_stats["mean"],
            mode="lines+markers",
            name="Mean Diurnal Load",
            line=dict(color="#F45112", width=2.5),
            marker=dict(size=6, color="#C23B06"),
            hovertemplate="<b>Hour:</b> %{x}:00<br><b>Mean:</b> %{y:.2f} kW<extra></extra>"
        )
    )
    fig_hourly.update_layout(
        title=dict(text="24-Hour Diurnal Demand Pattern", font=dict(color="#171717", size=14, weight="bold")),
        xaxis=dict(
            title="Hour of Day (0-23)",
            tickmode="linear",
            tick0=0,
            dtick=2,
            title_font=dict(color="#171717", size=11),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        yaxis=dict(
            title="Consumption (kW)",
            title_font=dict(color="#171717", size=11),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#171717")),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )

    # Weekly Heatmap
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_heatmap = temp_df.pivot_table(
        index="day_name",
        columns="hour",
        values="consumption_kw",
        aggfunc="mean"
    ).reindex(days_order)

    fig_heatmap = px.imshow(
        pivot_heatmap,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg kW"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=days_order,
        color_continuous_scale=[
            [0.0, "#FEF3C7"],
            [0.3, "#FDE68A"],
            [0.6, "#F59E0B"],
            [0.85, "#F45112"],
            [1.0, "#991B1B"]
        ],
        aspect="auto",
        title="Weekly Load Heatmap (Peak vs Off-Peak Hours)"
    )
    fig_heatmap.update_layout(
        title=dict(text="Weekly Load Heatmap (Peak vs Off-Peak Hours)", font=dict(color="#171717", size=14, weight="bold")),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickfont=dict(color="#4A4A4A"), title_font=dict(color="#171717")),
        yaxis=dict(tickfont=dict(color="#4A4A4A"), title_font=dict(color="#171717")),
        height=380
    )

    return fig_hourly, fig_heatmap

def plot_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plots the consumption histogram and boxplot.
    """
    fig = px.histogram(
        df,
        x="consumption_kw",
        nbins=60,
        marginal="box",
        color_discrete_sequence=["#F45112"],
        title="Demand Distribution and Quartiles"
    )
    fig.update_layout(
        title=dict(text="Demand Distribution and Quartiles", font=dict(color="#171717", size=14, weight="bold")),
        xaxis=dict(
            title="Electricity Consumption (kW)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        yaxis=dict(
            title="Frequency (Hours)",
            title_font=dict(color="#171717", size=12),
            tickfont=dict(color="#4A4A4A"),
            gridcolor="#E7E2D6",
            showgrid=True
        ),
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig
