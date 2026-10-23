import os
import io
import json
import zipfile
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data_loader import (
    parse_tsf_metadata,
    get_available_series,
    load_series,
    validate_tsf_file,
    get_zenodo_file_metadata,
    ensure_dataset_available,
    DEFAULT_DATASET_PATH,
)

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

    def test_validate_tsf_file(self):
        # Local real dataset should be valid
        self.assertTrue(validate_tsf_file(DATA_PATH))

        # Nonexistent file should return False
        self.assertFalse(validate_tsf_file("nonexistent_file_path.tsf"))

        # Empty or small file should return False
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("tiny content")
            tname = tf.name
        try:
            self.assertFalse(validate_tsf_file(tname))
        finally:
            if os.path.exists(tname):
                os.remove(tname)

    def test_ensure_dataset_available_when_already_present(self):
        # Should return target path immediately without any network calls
        with patch("urllib.request.urlopen") as mock_url:
            path = ensure_dataset_available(DATA_PATH)
            self.assertEqual(path, os.path.abspath(DATA_PATH))
            mock_url.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_get_zenodo_file_metadata(self, mock_urlopen):
        fake_api_response = {
            "files": [
                {
                    "key": "electricity_hourly_dataset.zip",
                    "size": 11823931,
                    "links": {
                        "content": "https://zenodo.org/api/records/4656140/files/electricity_hourly_dataset.zip/content"
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(fake_api_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        meta = get_zenodo_file_metadata()
        self.assertEqual(meta["filename"], "electricity_hourly_dataset.zip")
        self.assertEqual(meta["url"], "https://zenodo.org/api/records/4656140/files/electricity_hourly_dataset.zip/content")
        self.assertEqual(meta["size"], 11823931)

    @patch("src.data_loader.get_zenodo_file_metadata")
    @patch("urllib.request.urlopen")
    def test_ensure_dataset_available_download_and_extract_mock(self, mock_urlopen, mock_meta):
        # Prepare a mock zip with a valid mini tsf file inside
        mini_tsf_content = (
            "@relation Electricity\n"
            "@attribute series_name string\n"
            "@attribute start_timestamp date\n"
            "@data\n"
            "T1:2012-01-01 00-00-01:10,20,30\n"
        ).encode("utf-8")
        # Pad to exceed min size requirement
        padding = ("# padding line\n" * 100).encode("utf-8")
        full_mini_tsf = padding + mini_tsf_content

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("electricity_hourly_dataset.tsf", full_mini_tsf)
        zip_bytes = zip_buf.getvalue()

        mock_meta.return_value = {
            "filename": "electricity_hourly_dataset.zip",
            "url": "https://fake.zenodo/download.zip",
            "size": len(zip_bytes)
        }

        # Mock download response chunking
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(zip_bytes))}
        # read chunks then empty bytes
        mock_resp.read.side_effect = [zip_bytes[:2048], zip_bytes[2048:], b""]
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_tsf = os.path.join(tmp_dir, "test_dataset.tsf")
            progress_events = []

            def _track_progress(p, msg):
                progress_events.append((p, msg))

            res_path = ensure_dataset_available(
                target_path=target_tsf,
                progress_callback=_track_progress
            )
            self.assertTrue(os.path.exists(res_path))
            self.assertTrue(validate_tsf_file(res_path))
            self.assertGreater(len(progress_events), 0)

            # Check that series can be read from it
            names = get_available_series(res_path)
            self.assertEqual(names, ["T1"])


if __name__ == "__main__":
    unittest.main()

