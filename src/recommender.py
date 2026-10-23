import pandas as pd
import numpy as np
from typing import List, Dict, Any

def generate_conservation_recommendations(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    tariff_rate: float = 8.0 # User entered tariff per kWh (e.g. ₹ or $), purely as reference
) -> List[Dict[str, Any]]:
    """
    Translates consumption patterns, forecasts, and detected anomalies into
    practical, realistic conservation and scheduling recommendations.
    
    CRITICAL: Does not invent true campus billing or confirmed wastage.
    Estimates are explicitly flagged as approximate heuristic indicators.
    """
    recommendations = []

    # 1. Overnight / Baseloader Baselines
    # Hours 00:00 to 05:00
    history_copy = history_df.copy()
    history_copy["hour"] = history_copy["timestamp"].dt.hour
    overnight_mask = history_copy["hour"].between(0, 5)
    daytime_mask = history_copy["hour"].between(9, 18)

    avg_overnight = history_copy[overnight_mask]["consumption_kw"].mean()
    avg_daytime = history_copy[daytime_mask]["consumption_kw"].mean()

    ratio = (avg_overnight / avg_daytime) if avg_daytime > 0 else 0.0

    if ratio > 0.40:
        recommendations.append({
            "category": "Baseload & Off-Hours Audit",
            "priority": "High",
            "title": "Substantial Inactive Hours Consumption Detected",
            "observation": f"Average nighttime load (00:00-05:00) is {avg_overnight:.1f} kW, which is {ratio*100:.1f}% of active daytime demand ({avg_daytime:.1f} kW).",
            "action": "Conduct an audit of continuous baseload equipment (e.g., HVAC chillers, server cooling, outdoor architectural lighting, unmanaged lab machinery) to identify systems running unnecessarily overnight.",
            "estimated_impact": f"Targeting a 15% reduction in off-hours idle baseload could conserve approximately {avg_overnight * 0.15 * 6:.1f} kWh each night (~{avg_overnight * 0.15 * 6 * tariff_rate:,.0f} currency units estimated per night)."
        })
    else:
        recommendations.append({
            "category": "Baseload & Off-Hours Audit",
            "priority": "Low",
            "title": "Healthy Overnight Demand Attenuation",
            "observation": f"Off-hours load is {avg_overnight:.1f} kW ({ratio*100:.1f}% of daytime load), demonstrating appropriate nighttime setback scheduling.",
            "action": "Maintain current automated timer setbacks and occupancy sensor calibrations.",
            "estimated_impact": "Operational protocols are performing effectively."
        })

    # 2. Peak Demand Shifting from Forecasts
    if forecast_df is not None and not forecast_df.empty:
        peak_forecast_row = forecast_df.loc[forecast_df["forecast_kw"].idxmax()]
        peak_time = peak_forecast_row["timestamp"]
        peak_val = peak_forecast_row["forecast_kw"]
        avg_forecast_val = forecast_df["forecast_kw"].mean()

        if peak_val > avg_forecast_val * 1.3:
            recommendations.append({
                "category": "Peak Shifting & Demand Response",
                "priority": "Medium",
                "title": f"Forecasted Peak Surge on {peak_time.strftime('%b %d, %H:00')}",
                "observation": f"Demand is projected to surge to {peak_val:.1f} kW at {peak_time.strftime('%H:00')}, exceeding the period average ({avg_forecast_val:.1f} kW) by {(peak_val/avg_forecast_val - 1)*100:.1f}%.",
                "action": "Pre-cool facility thermal masses 1-2 hours prior, stagger major motor/pump startups, and defer discretionary high-draw loads (battery charging, water heating) to off-peak periods.",
                "estimated_impact": "Minimizes utility maximum-demand penalties and reduces transformer thermal stress."
            })

    # 3. Anomaly-Driven Maintenance Trigger
    if anomaly_df is not None and "is_confirmed_anomaly" in anomaly_df.columns:
        recent_anomalies = anomaly_df.iloc[-168:] # past week
        high_spikes = recent_anomalies[(recent_anomalies["is_confirmed_anomaly"] == 1) & (recent_anomalies["z_score"] > 2.5)]
        spike_count = len(high_spikes)

        if spike_count > 0:
            recommendations.append({
                "category": "Equipment Health & Maintenance",
                "priority": "High" if spike_count >= 3 else "Medium",
                "title": f"{spike_count} Statistical Consumption Spike(s) in Last 7 Days",
                "observation": f"Detected {spike_count} instance(s) where hourly consumption exceeded standard statistical boundaries (Z > 2.5). Note: Not all anomalies indicate energy waste; some reflect legitimate surge events.",
                "action": "Inspect electrical logging at the local sub-distribution panel for short-cycling compressors, malfunctioning power factor correction capacitors, or uncoordinated batch operations.",
                "estimated_impact": "Prevents potential equipment insulation degradation and sudden breaker tripping."
            })

    # 4. Weekend vs Weekday Optimization
    history_copy["is_weekend"] = history_copy["timestamp"].dt.dayofweek >= 5
    avg_weekday = history_copy[~history_copy["is_weekend"]]["consumption_kw"].mean()
    avg_weekend = history_copy[history_copy["is_weekend"]]["consumption_kw"].mean()

    if avg_weekend > 0.75 * avg_weekday:
        recommendations.append({
            "category": "Weekend Conservation Protocol",
            "priority": "Medium",
            "title": "Elevated Weekend Consumption Pattern",
            "observation": f"Average weekend consumption ({avg_weekend:.1f} kW) remains at {(avg_weekend/avg_weekday)*100:.1f}% of weekday load ({avg_weekday:.1f} kW).",
            "action": "Verify building management system (BMS) calendar overrides, ensure weekend classroom/office HVAC setpoints are relaxed (+2°C in summer, -2°C in winter), and confirm lighting timers are active.",
            "estimated_impact": f"A 10% reduction during non-operational weekend hours can yield ~{(avg_weekend * 0.10 * 48):.1f} kWh savings per weekend."
        })

    return recommendations
