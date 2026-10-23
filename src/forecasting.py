import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def chronological_train_test_split(
    df: pd.DataFrame,
    test_hours: int = 168 # 7 days default
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological train/test split without random shuffling.
    Default test window is 168 hours (7 days).
    """
    if len(df) <= test_hours:
        raise ValueError(f"Data length ({len(df)}) must exceed test window ({test_hours}).")

    train_df = df.iloc[:-test_hours].copy()
    test_df = df.iloc[-test_hours:].copy()
    return train_df, test_df

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates predictions with MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Normalized RMSE (relative to mean) for percentage interpretation
    mean_val = np.mean(y_true)
    nrmse = (rmse / mean_val * 100.0) if mean_val > 0 else 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "NRMSE_pct": float(nrmse)
    }

def seasonal_naive_forecast(
    train_series: pd.Series,
    forecast_horizon: int,
    seasonality: int = 24
) -> np.ndarray:
    """
    Seasonal Naive baseline:
    y_hat(t+h) = y(t + h - k*seasonality)
    Where seasonality = 24 (daily cycle) or 168 (weekly cycle).
    """
    if len(train_series) < seasonality:
        raise ValueError("Training series too short for specified seasonality.")

    # Get the last seasonal cycle from train
    last_cycle = train_series.values[-seasonality:]
    # Repeat the cycle to cover the forecast horizon
    repetitions = int(np.ceil(forecast_horizon / seasonality))
    forecast = np.tile(last_cycle, repetitions)[:forecast_horizon]
    return np.maximum(0.0, forecast)

def exponential_smoothing_forecast(
    train_df: pd.DataFrame,
    forecast_horizon: int,
    seasonal_periods: int = 24,
    recent_training_days: int = 60 # Fit on recent 60 days for speed and relevance
) -> np.ndarray:
    """
    Holt-Winters Exponential Smoothing (additive trend and seasonality).
    """
    recent_points = min(len(train_df), recent_training_days * 24)
    y_train = train_df["consumption_kw"].iloc[-recent_points:].values

    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        ).fit(optimized=True)
        preds = model.forecast(forecast_horizon)
        return np.maximum(0.0, preds)
    except Exception:
        # Fallback to simple exponential smoothing if Holt-Winters fails numerical stability
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        model = SimpleExpSmoothing(y_train).fit()
        preds = model.forecast(forecast_horizon)
        return np.maximum(0.0, preds)

def create_ml_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 24, 48, 168],
    rolling_windows: List[int] = [6, 24]
) -> pd.DataFrame:
    """
    Engineers temporal, lag, and rolling features for machine learning forecasting.
    """
    data = df.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["dayofweek"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Cyclical hour encoding
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)

    # Lag features
    for lag in lags:
        data[f"lag_{lag}"] = data["consumption_kw"].shift(lag)

    # Rolling statistics
    for win in rolling_windows:
        data[f"rolling_mean_{win}"] = data["consumption_kw"].shift(1).rolling(win).mean()
        data[f"rolling_std_{win}"] = data["consumption_kw"].shift(1).rolling(win).std().fillna(0)

    return data

def ml_lag_forecast(
    train_df: pd.DataFrame,
    test_or_horizon_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, Ridge]:
    """
    Trains a Ridge regression model with chronological lag features.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df["consumption_kw"].values

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    X_test = test_or_horizon_df[feature_cols].values
    preds = model.predict(X_test)
    return np.maximum(0.0, preds), model

def run_model_comparison(
    df: pd.DataFrame,
    test_hours: int = 168 # 7 days
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Backtests and compares Seasonal Naive, Holt-Winters, and ML Lag models
    on the held-out historical test set.
    """
    train_df, test_df = chronological_train_test_split(df, test_hours=test_hours)

    # Validate that train_df has enough history for ML lag features (max lag 168h + rolling 24h = ~192h minimum)
    if len(train_df) <= 192:
        raise ValueError(
            f"Insufficient training observations ({len(train_df)} hours) for backtest window of {test_hours} hours. "
            f"Please select 'Last 90 Days' or 'Full History' under Historical View Range to test longer horizons."
        )

    # 1. Seasonal Naive (24h period)
    naive_preds = seasonal_naive_forecast(
        train_df["consumption_kw"],
        forecast_horizon=test_hours,
        seasonality=24
    )

    # 2. Exponential Smoothing (Holt-Winters)
    ets_preds = exponential_smoothing_forecast(
        train_df,
        forecast_horizon=test_hours,
        seasonal_periods=24
    )

    # 3. ML Lag-based Model
    full_feat_df = create_ml_features(df)
    full_feat_df = full_feat_df.dropna() # Drop NaN from initial lags

    train_idx = full_feat_df["timestamp"] <= train_df["timestamp"].max()
    test_idx = full_feat_df["timestamp"] > train_df["timestamp"].max()

    feat_train = full_feat_df[train_idx]
    feat_test = full_feat_df[test_idx]

    feature_cols = [c for c in full_feat_df.columns if c not in ["timestamp", "consumption_kw"]]
    ml_preds, ml_model = ml_lag_forecast(feat_train, feat_test, feature_cols)

    # Align predictions with ground truth
    y_true = test_df["consumption_kw"].values
    eval_naive = evaluate_predictions(y_true, naive_preds)
    eval_ets = evaluate_predictions(y_true, ets_preds)
    eval_ml = evaluate_predictions(y_true, ml_preds)

    metrics = {
        "Seasonal Naive (24h)": eval_naive,
        "Exponential Smoothing": eval_ets,
        "ML Lag-Feature (Ridge)": eval_ml
    }

    results_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "Actual": y_true,
        "Seasonal Naive": naive_preds,
        "Exponential Smoothing": ets_preds,
        "ML Lag Model": ml_preds
    })

    return train_df, metrics, results_df

def generate_future_forecast(
    df: pd.DataFrame,
    horizon_hours: int = 24
) -> pd.DataFrame:
    """
    Produces out-of-sample future forecasts for the specified horizon (e.g. 24h to 168h).
    Combines Holt-Winters and Seasonal Naive into an operational forecast.
    """
    last_ts = df["timestamp"].max()
    future_dates = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

    # Predict using Holt-Winters
    hw_forecast = exponential_smoothing_forecast(df, forecast_horizon=horizon_hours, seasonal_periods=24)
    # Predict using Seasonal Naive
    naive_forecast = seasonal_naive_forecast(df["consumption_kw"], forecast_horizon=horizon_hours, seasonality=24)

    # Operational ensemble / primary prediction
    combined = 0.65 * hw_forecast + 0.35 * naive_forecast

    # Confidence intervals estimate based on recent residual std
    recent_residuals = df["consumption_kw"].iloc[-168:].diff().dropna()
    res_std = recent_residuals.std() if len(recent_residuals) > 0 else 5.0

    lower_bound = np.maximum(0.0, combined - 1.96 * res_std)
    upper_bound = combined + 1.96 * res_std

    future_df = pd.DataFrame({
        "timestamp": future_dates,
        "forecast_kw": combined,
        "lower_bound_kw": lower_bound,
        "upper_bound_kw": upper_bound,
        "naive_baseline_kw": naive_forecast
    })
    return future_df
