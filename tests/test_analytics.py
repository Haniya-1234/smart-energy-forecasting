import os
import unittest
import pandas as pd
import numpy as np
from src.data_loader import load_series
from src.forecasting import (
    chronological_train_test_split,
    seasonal_naive_forecast,
    exponential_smoothing_forecast,
    run_model_comparison,
    evaluate_predictions
)
from src.anomaly import detect_rolling_statistical_anomalies, combine_anomaly_detection
from src.recommender import generate_conservation_recommendations

DATA_PATH = os.path.join("data", "electricity_hourly_dataset.tsf")

class TestAnalyticsModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load sample data (first 500 hours to keep test fast)
        cls.full_df = load_series(DATA_PATH, series_name="T1")
        cls.sample_df = cls.full_df.iloc[:500].copy()

    def test_chronological_split(self):
        train, test = chronological_train_test_split(self.sample_df, test_hours=48)
        self.assertEqual(len(test), 48)
        self.assertEqual(len(train), len(self.sample_df) - 48)
        # Verify chronological order
        self.assertTrue(train["timestamp"].max() < test["timestamp"].min())

    def test_seasonal_naive_forecast(self):
        train, test = chronological_train_test_split(self.sample_df, test_hours=24)
        preds = seasonal_naive_forecast(train["consumption_kw"], forecast_horizon=24, seasonality=24)
        self.assertEqual(len(preds), 24)
        self.assertTrue((preds >= 0).all())
        # Metric evaluation
        metrics = evaluate_predictions(test["consumption_kw"].values, preds)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertGreater(metrics["RMSE"], 0.0)

    def test_exponential_smoothing_forecast(self):
        train, test = chronological_train_test_split(self.sample_df, test_hours=24)
        preds = exponential_smoothing_forecast(train, forecast_horizon=24, seasonal_periods=24)
        self.assertEqual(len(preds), 24)
        self.assertTrue((preds >= 0).all())

    def test_anomaly_detection(self):
        anomalies_df = combine_anomaly_detection(self.sample_df, z_threshold=2.5)
        self.assertIn("is_anomaly_rolling", anomalies_df.columns)
        self.assertIn("is_anomaly_iso", anomalies_df.columns)
        self.assertIn("is_confirmed_anomaly", anomalies_df.columns)
        self.assertEqual(len(anomalies_df), len(self.sample_df))

    def test_rolling_statistics_causality_and_properties(self):
        # Verify rolling statistics use past observations only (no lookahead / center=False)
        series_kw = self.sample_df["consumption_kw"].values
        rolling_df = detect_rolling_statistical_anomalies(self.sample_df, window_hours=48, z_threshold=2.5)
        
        # Test causality: mutating a future observation (e.g. index 100) must not change rolling stats at index 50
        mutated_df = self.sample_df.copy()
        mutated_df.loc[100, "consumption_kw"] = 99999.0
        rolling_mutated = detect_rolling_statistical_anomalies(mutated_df, window_hours=48, z_threshold=2.5)
        
        self.assertAlmostEqual(
            rolling_df.loc[50, "rolling_mean"],
            rolling_mutated.loc[50, "rolling_mean"],
            places=5
        )
        self.assertAlmostEqual(
            rolling_df.loc[50, "rolling_std"],
            rolling_mutated.loc[50, "rolling_std"],
            places=5
        )
        self.assertAlmostEqual(
            rolling_df.loc[50, "z_score"],
            rolling_mutated.loc[50, "z_score"],
            places=5
        )

        # Check explicit rolling calculation at index 47 (48 observations: 0 to 47)
        expected_mean = float(np.mean(series_kw[0:48]))
        expected_std = float(np.std(series_kw[0:48], ddof=1))
        self.assertAlmostEqual(rolling_df.loc[47, "rolling_mean"], expected_mean, places=4)
        self.assertAlmostEqual(rolling_df.loc[47, "rolling_std"], expected_std, places=4)

    def test_isolation_forest_preprocessing_and_flags(self):
        anomalies_df = combine_anomaly_detection(self.sample_df, z_threshold=2.5, contamination=0.03)
        # Check flag values are strictly binary 0 or 1
        self.assertTrue(set(anomalies_df["is_anomaly_iso"].unique()).issubset({0, 1}))
        self.assertTrue(set(anomalies_df["is_anomaly_rolling"].unique()).issubset({0, 1}))
        self.assertTrue(set(anomalies_df["is_confirmed_anomaly"].unique()).issubset({0, 1}))
        # Anomaly score from decision_function exists and has valid float values
        self.assertIn("anomaly_score_iso", anomalies_df.columns)
        self.assertFalse(anomalies_df["anomaly_score_iso"].isna().any())

    def test_anomaly_timestamp_alignment(self):
        anomalies_df = combine_anomaly_detection(self.sample_df, z_threshold=2.5)
        # Verify timestamps are identical and monotonic
        self.assertTrue((anomalies_df["timestamp"] == self.sample_df["timestamp"]).all())
        anomalies_only = anomalies_df[anomalies_df["is_confirmed_anomaly"] == 1]
        if not anomalies_only.empty:
            # Subset of anomalies retains exact original timestamps
            sample_lookup = self.sample_df.set_index("timestamp")
            for ts in anomalies_only["timestamp"]:
                self.assertIn(ts, sample_lookup.index)

    def test_recommendations_and_energy_units(self):
        anomalies_df = combine_anomaly_detection(self.sample_df, z_threshold=2.5)
        forecast_df = pd.DataFrame({
            "timestamp": pd.date_range("2012-01-22", periods=24, freq="h"),
            "forecast_kw": [10.0] * 23 + [100.0] # Peak surge at the 24th hour
        })
        tariff = 8.0
        recs = generate_conservation_recommendations(
            history_df=self.sample_df,
            forecast_df=forecast_df,
            anomaly_df=anomalies_df,
            tariff_rate=tariff
        )
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        
        # Check all expected fields are present
        categories = [r["category"] for r in recs]
        self.assertIn("Peak Shifting & Demand Response", categories)

        for r in recs:
            self.assertIn("category", r)
            self.assertIn("priority", r)
            self.assertIn("title", r)
            self.assertIn("observation", r)
            self.assertIn("action", r)
            self.assertIn("estimated_impact", r)
            # Ensure estimated_impact is either operational note or clearly framed with estimate / units
            impact_text = r["estimated_impact"].lower()
            if "currency" in impact_text or "kwh" in impact_text:
                self.assertTrue(
                    "estimate" in impact_text or "approximately" in impact_text or "~" in impact_text,
                    f"Impact must be framed as heuristic estimate: {r['estimated_impact']}"
                )

    def test_recommendations(self):
        anomalies_df = combine_anomaly_detection(self.sample_df, z_threshold=2.5)
        recs = generate_conservation_recommendations(
            history_df=self.sample_df,
            forecast_df=pd.DataFrame({
                "timestamp": pd.date_range("2012-01-22", periods=24, freq="h"),
                "forecast_kw": [50.0] * 24
            }),
            anomaly_df=anomalies_df,
            tariff_rate=8.5
        )
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertIn("action", recs[0])
        self.assertIn("title", recs[0])

    def test_run_model_comparison(self):
        train, metrics, results = run_model_comparison(self.sample_df, test_hours=24)
        self.assertEqual(len(results), 24)
        self.assertIn("Seasonal Naive (24h)", metrics)
        self.assertIn("Exponential Smoothing", metrics)
        self.assertIn("ML Lag-Feature (Ridge)", metrics)
        for model_name, model_metrics in metrics.items():
            self.assertIn("MAE", model_metrics)
            self.assertIn("RMSE", model_metrics)
            self.assertGreater(model_metrics["MAE"], 0.0)
            self.assertGreater(model_metrics["RMSE"], 0.0)

    def test_cost_calculation(self):
        # 100 kW constant over 10 hours = 1,000 kWh
        test_df = pd.DataFrame({"consumption_kw": [100.0] * 10})
        total_kwh = float(test_df["consumption_kw"].sum())
        tariff = 8.0
        cost = total_kwh * tariff
        self.assertEqual(total_kwh, 1000.0)
        self.assertEqual(cost, 8000.0)

if __name__ == "__main__":
    unittest.main()

