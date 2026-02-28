from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    target_lower = target.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return col
    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_latlong(raw_value: object) -> Tuple[Optional[float], Optional[float]]:
    if raw_value is None:
        return None, None
    text = str(raw_value).strip()
    if not text:
        return None, None
    text = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    for sep in [",", ";", " "]:
        if sep in text:
            parts = [p for p in text.split(sep) if p]
            if len(parts) >= 2:
                return _safe_float(parts[0]), _safe_float(parts[1])
    return None, None


def load_conase_csv(csv_path: str | Path, sep: str = "|", max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load and normalize CoNASE-like CSV schema into canonical columns."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=sep, dtype=str, nrows=max_rows, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    canonical_cols = ["video_id", "channel_id", "country", "state", "location", "address", "text_pos"]
    for canonical in canonical_cols:
        source_col = _find_column(df, canonical)
        if source_col and source_col != canonical:
            df[canonical] = df[source_col]
        elif source_col is None and canonical not in df.columns:
            df[canonical] = ""

    if not df["text_pos"].astype(bool).any():
        fallback_text_col = _find_column(df, "transcript_pos") or _find_column(df, "transcript")
        if fallback_text_col:
            df["text_pos"] = df[fallback_text_col]
            LOGGER.warning("Using '%s' as fallback for text_pos.", fallback_text_col)

    lat_col = _find_column(df, "lat")
    lon_col = _find_column(df, "lon")
    latlong_col = _find_column(df, "latlong")
    if lat_col and lon_col:
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        lat_values = []
        lon_values = []
        source = df[latlong_col] if latlong_col else [None] * len(df)
        for raw in source:
            lat, lon = _parse_latlong(raw)
            lat_values.append(lat)
            lon_values.append(lon)
        df["lat"] = lat_values
        df["lon"] = lon_values

    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["channel_id"] = df["channel_id"].astype(str).str.strip()
    df = df[df["video_id"] != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df
