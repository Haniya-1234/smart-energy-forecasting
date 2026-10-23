import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Tuple

def detect_rolling_statistical_anomalies(
    df: pd.DataFrame,
    window_hours: int = 24 * 7, # 7-day rolling window
    z_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Detects unusual observations using a rolling statistical mean and standard deviation.
    Threshold: |x - rolling_mean| > z_threshold * rolling_std.
    """
    result = df.copy()
    rolling_mean = result["consumption_kw"].rolling(window=window_hours, min_periods=24).mean()
    rolling_std = result["consumption_kw"].rolling(window=window_hours, min_periods=24).std()

    # Fill initial period with global mean/std
    rolling_mean = rolling_mean.fillna(result["consumption_kw"].mean())
    rolling_std = rolling_std.fillna(result["consumption_kw"].std()).replace(0, 1e-5)

    z_scores = (result["consumption_kw"] - rolling_mean) / rolling_std

    result["rolling_mean"] = rolling_mean
    result["rolling_std"] = rolling_std
    result["z_score"] = z_scores
    result["is_anomaly_rolling"] = (z_scores.abs() > z_threshold).astype(int)
    result["anomaly_direction"] = np.where(z_scores > z_threshold, "High Spike",
                                  np.where(z_scores < -z_threshold, "Low Drop", "Normal"))
    return result

def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02
) -> pd.DataFrame:
    """
    Detects multivariate structural anomalies using Isolation Forest on
    consumption, rolling mean, and diurnal temporal features.
    """
    result = df.copy()
    features = pd.DataFrame({
        "consumption": result["consumption_kw"],
        "hour": result["timestamp"].dt.hour,
        "dayofweek": result["timestamp"].dt.dayofweek,
        "rolling_diff": result["consumption_kw"].diff().fillna(0)
    })

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    preds = iso.fit_predict(features)
    # IsolationForest returns -1 for anomaly, 1 for normal
    result["is_anomaly_iso"] = (preds == -1).astype(int)
    result["anomaly_score_iso"] = iso.decision_function(features)
    return result

def combine_anomaly_detection(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    contamination: float = 0.02
) -> pd.DataFrame:
    """
    Combines rolling statistical filter and Isolation Forest for comprehensive inspection.
    """
    df_rolling = detect_rolling_statistical_anomalies(df, z_threshold=z_threshold)
    df_combined = detect_isolation_forest_anomalies(df_rolling, contamination=contamination)

    # Flag as high-priority anomaly if either method flags it, or consensus
    df_combined["is_confirmed_anomaly"] = (
        (df_combined["is_anomaly_rolling"] == 1) | (df_combined["is_anomaly_iso"] == 1)
    ).astype(int)

    return df_combined
