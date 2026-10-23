import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def parse_tsf_metadata(file_path: str) -> Dict:
    """
    Parses the header metadata of a Monash/Zenodo .tsf time series file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    metadata = {
        "relation": "Unknown",
        "attributes": [],
        "frequency": "hourly",
        "missing": False,
        "equallength": True,
        "series_names": [],
        "data_line_start": 0,
    }

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        line_num = 0
        for line in f:
            line_num += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@relation"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    metadata["relation"] = parts[1].strip()
            elif line.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                if len(parts) >= 3:
                    metadata["attributes"].append((parts[1].strip(), parts[2].strip()))
            elif line.startswith("@frequency"):
                parts = line.split(maxsplit=1)
                if len(parts) > 1:
                    metadata["frequency"] = parts[1].strip()
            elif line.startswith("@missing"):
                metadata["missing"] = "true" in line.lower()
            elif line.startswith("@equallength"):
                metadata["equallength"] = "true" in line.lower()
            elif line.startswith("@data"):
                metadata["data_line_start"] = line_num
                break

    return metadata


def get_available_series(file_path: str) -> List[str]:
    """
    Quickly scans and returns all available series IDs (e.g., T1, T2, ... T321) from the .tsf file.
    """
    series_names = []
    metadata = parse_tsf_metadata(file_path)
    start_line = metadata["data_line_start"]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            if idx <= start_line:
                continue
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 1)
            if parts:
                series_names.append(parts[0].strip())
    return series_names


def load_series(
    file_path: str,
    series_name: str = "T1"
) -> pd.DataFrame:
    """
    Loads a specific series from the .tsf file into a clean pandas DataFrame
    with a regular hourly DatetimeIndex and handles any missing tokens or nulls.

    Returns:
        pd.DataFrame with columns: ['timestamp', 'consumption_kw']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    target_line = None
    metadata = parse_tsf_metadata(file_path)
    start_line = metadata["data_line_start"]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            if idx <= start_line:
                continue
            line = line.strip()
            if line.startswith(f"{series_name}:"):
                target_line = line
                break

    if target_line is None:
        raise ValueError(f"Series '{series_name}' not found in {file_path}")

    # Format of data lines: series_name:start_timestamp:val1,val2,val3...
    # Notice timestamp format: "2012-01-01 00-00-01" or "2012-01-01 00:00:00"
    parts = target_line.split(":")
    if len(parts) < 3:
        raise ValueError(f"Malformed line for series {series_name}: {target_line[:80]}...")

    name = parts[0].strip()
    raw_start = parts[1].strip()
    # The remainder is comma-separated series values (parts[2] onwards)
    raw_values = ":".join(parts[2:]).split(",")

    # Parse start timestamp safely
    # In this dataset, raw_start looks like '2012-01-01 00-00-01'
    clean_ts_str = raw_start.replace("-00-01", " 00:00:00")
    try:
        start_dt = pd.to_datetime(clean_ts_str)
    except Exception:
        # Fallback regex parser
        dt_match = re.search(r"(\d{4}-\d{2}-\d{2})", raw_start)
        if dt_match:
            start_dt = pd.to_datetime(dt_match.group(1))
        else:
            start_dt = pd.to_datetime("2012-01-01 00:00:00")

    # Round/floor to nearest clean hour
    start_dt = start_dt.floor("h")

    # Parse numeric values and replace '?' with np.nan
    numeric_vals = []
    for v in raw_values:
        v = v.strip()
        if v == "?" or v == "" or v == "NA" or v == "null":
            numeric_vals.append(np.nan)
        else:
            try:
                numeric_vals.append(float(v))
            except ValueError:
                numeric_vals.append(np.nan)

    # Generate regular hourly timestamp sequence
    n_points = len(numeric_vals)
    timestamps = pd.date_range(start=start_dt, periods=n_points, freq="h")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kw": numeric_vals
    })

    # Linear interpolation for any occasional missing observation, then ffill/bfill
    if df["consumption_kw"].isna().any():
        df["consumption_kw"] = df["consumption_kw"].interpolate(method="linear").bfill().ffill()

    return df
