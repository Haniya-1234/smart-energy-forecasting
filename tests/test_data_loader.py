import os
import unittest
import pandas as pd
import numpy as np
from src.data_loader import parse_tsf_metadata, get_available_series, load_series

DATA_PATH = os.path.join("data", "electricity_hourly_dataset.tsf")

class TestDataLoader(unittest.TestCase):
    def test_parse_tsf_metadata(self):
        self.assertTrue(os.path.exists(DATA_PATH), "Dataset file not found at expected path")
        metadata = parse_tsf_metadata(DATA_PATH)
        self.assertEqual(metadata["relation"], "Electricity")
        self.assertEqual(metadata["frequency"], "hourly")
        self.assertGreater(metadata["data_line_start"], 0)
        self.assertGreaterEqual(len(metadata["attributes"]), 2)

    def test_get_available_series(self):
        series_list = get_available_series(DATA_PATH)
        self.assertEqual(len(series_list), 321)
        self.assertIn("T1", series_list)
        self.assertIn("T321", series_list)

    def test_load_series(self):
        df = load_series(DATA_PATH, series_name="T1")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("timestamp", df.columns)
        self.assertIn("consumption_kw", df.columns)
        self.assertEqual(len(df), 26304)
        self.assertFalse(df["consumption_kw"].isna().any())
        self.assertEqual(df["consumption_kw"].iloc[0], 14.0)
        # Check 1 hour spacing
        diffs = df["timestamp"].diff().dropna()
        self.assertTrue((diffs == pd.Timedelta(hours=1)).all())

if __name__ == "__main__":
    unittest.main()
