import os
import re
import json
import zipfile
import tempfile
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

ZENODO_RECORD_API_URL = "https://zenodo.org/api/records/4656140"
DEFAULT_DATA_DIR = "data"
EXPECTED_DATASET_FILENAME = "electricity_hourly_dataset.tsf"
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_DATA_DIR, EXPECTED_DATASET_FILENAME)


def get_zenodo_file_metadata(
    record_url: str = ZENODO_RECORD_API_URL,
    timeout: int = 30
) -> Dict[str, str]:
    """
    Queries Zenodo's public record metadata API to locate the download URL
    and filename for the dataset.
    """
    req = urllib.request.Request(
        record_url,
        headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Zenodo API responded with status {response.status}")
        data = json.loads(response.read().decode("utf-8"))

    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"No files found in Zenodo record: {record_url}")

    # Find the matching file entry (either direct .tsf or the archive containing it)
    chosen_file = None
    for f in files:
        name = f.get("key", "")
        if "electricity" in name.lower():
            chosen_file = f
            break

    if not chosen_file:
        chosen_file = files[0]

    filename = chosen_file.get("key", "")
    links = chosen_file.get("links", {})
    file_url = links.get("content") or links.get("self")
    if not file_url:
        raise RuntimeError(f"Could not extract direct download URL from Zenodo record for file: {filename}")

    return {
        "filename": filename,
        "url": file_url,
        "size": chosen_file.get("size", 0),
        "checksum": chosen_file.get("checksum", "")
    }


def validate_tsf_file(file_path: str, min_size_bytes: int = 1000) -> bool:
    """
    Validates that the file exists, is non-empty, and contains valid TSF header tokens.
    """
    if not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < min_size_bytes:
        return False

    has_data_tag = False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("@data"):
                has_data_tag = True
                break

    return has_data_tag


def ensure_dataset_available(
    target_path: str = DEFAULT_DATASET_PATH,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    timeout: int = 60
) -> str:
    """
    Ensures that the required .tsf dataset is available locally.
    1. If the file exists and is valid, returns immediately.
    2. If missing or invalid, queries Zenodo's metadata API for record 4656140.
    3. Streams the download to disk in chunks to avoid high memory consumption.
    4. Extracts/places the .tsf file atomically, cleaning up temp files.
    5. Validates the resulting .tsf file before returning.
    """
    target_path = os.path.abspath(target_path)
    if os.path.exists(target_path) and validate_tsf_file(target_path):
        return target_path

    # Ensure parent directory exists
    parent_dir = os.path.dirname(target_path)
    os.makedirs(parent_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Contacting Zenodo metadata repository for dataset details...")

    metadata = get_zenodo_file_metadata(timeout=timeout)
    file_url = metadata["url"]
    remote_filename = metadata["filename"]
    total_size = metadata.get("size", 0)

    if progress_callback:
        progress_callback(0.15, f"Downloading {remote_filename} from Zenodo...")

    # Download to a temporary file
    temp_download = tempfile.NamedTemporaryFile(delete=False, dir=parent_dir, suffix="_download")
    temp_download_path = temp_download.name

    try:
        req = urllib.request.Request(file_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    total_size = int(content_length)
                except ValueError:
                    pass

            downloaded = 0
            chunk_size = 64 * 1024  # 64 KB chunks for streaming
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                temp_download.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    pct = min(0.15 + 0.70 * (downloaded / total_size), 0.85)
                    progress_callback(pct, f"Downloaded {downloaded / (1024 * 1024):.1f} MB of {total_size / (1024 * 1024):.1f} MB...")

        temp_download.close()

        if progress_callback:
            progress_callback(0.88, "Verifying and preparing dataset file...")

        # If the downloaded file is a zip archive, extract the target .tsf file
        if zipfile.is_zipfile(temp_download_path):
            with zipfile.ZipFile(temp_download_path, "r") as zf:
                tsf_names = [n for n in zf.namelist() if n.endswith(".tsf")]
                if not tsf_names:
                    raise ValueError(f"Zip archive from Zenodo does not contain a .tsf file: {zf.namelist()}")
                chosen_tsf = tsf_names[0]
                # Extract directly to atomic temp file in target directory
                atomic_temp_tsf = tempfile.NamedTemporaryFile(delete=False, dir=parent_dir, suffix=".tsf_temp")
                atomic_temp_path = atomic_temp_tsf.name
                with zf.open(chosen_tsf) as source:
                    while True:
                        buf = source.read(64 * 1024)
                        if not buf:
                            break
                        atomic_temp_tsf.write(buf)
                atomic_temp_tsf.close()
                # Atomically replace target
                os.replace(atomic_temp_path, target_path)
        else:
            # Direct file download
            os.replace(temp_download_path, target_path)

        if not validate_tsf_file(target_path):
            if os.path.exists(target_path):
                os.remove(target_path)
            raise ValueError(f"Downloaded file at {target_path} failed TSF format validation.")

        if progress_callback:
            progress_callback(1.0, "Dataset ready!")

        return target_path

    except Exception:
        # Clean up any leftover temp files on error
        if os.path.exists(temp_download_path):
            try:
                os.remove(temp_download_path)
            except OSError:
                pass
        raise
    finally:
        # Guarantee temp download file is cleaned up if it still exists
        if os.path.exists(temp_download_path):
            try:
                os.remove(temp_download_path)
            except OSError:
                pass


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
