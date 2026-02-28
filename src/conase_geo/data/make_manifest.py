from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from conase_geo.config import LabelType
from conase_geo.data.load_conase import load_conase_csv
from conase_geo.data.parse_tokens import parse_text_pos, serialize_tokens_for_window, tokens_in_window, window_text

LOGGER = logging.getLogger(__name__)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


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


def _geo_grid_label(lat: Optional[float], lon: Optional[float], cell_size_deg: float) -> Optional[str]:
    if lat is None or lon is None:
        return None
    lat_idx = int(np.floor(lat / cell_size_deg))
    lon_idx = int(np.floor(lon / cell_size_deg))
    return f"geo_{lat_idx}_{lon_idx}"


def _pick_label(
    row: pd.Series,
    label_type: LabelType,
    top_locations: Optional[set[str]],
    geo_cell_size_deg: float,
) -> Optional[str]:
    if label_type == "state":
        label = _safe_text(row.get("state"))
        return label or None
    if label_type == "location_topk":
        location = _safe_text(row.get("location"))
        if not location:
            return None
        if top_locations is not None and location not in top_locations:
            return None
        return location
    if label_type == "geo_grid":
        lat = _safe_float(row.get("lat"))
        lon = _safe_float(row.get("lon"))
        return _geo_grid_label(lat=lat, lon=lon, cell_size_deg=geo_cell_size_deg)
    raise ValueError(f"Unsupported label_type: {label_type}")


def _clip_starts(min_time: float, max_time: float, clip_seconds: float, clips_per_video: int) -> List[float]:
    min_time = max(0.0, min_time)
    clips_per_video = max(1, clips_per_video)
    max_start = max(min_time, max_time - clip_seconds)
    if max_start <= min_time:
        return [min_time]
    if clips_per_video == 1:
        return [min_time + (max_start - min_time) * 0.5]
    starts = np.linspace(min_time, max_start, num=clips_per_video)
    return [float(x) for x in starts]


def build_manifest(
    csv_path: Path,
    out_manifest: Path,
    label_type: LabelType = "state",
    clip_seconds: float = 10.0,
    clips_per_video: int = 2,
    max_videos: int = 2_000,
    min_label_examples: int = 200,
    top_k_locations: int = 200,
    geo_cell_size_deg: float = 1.5,
    sep: str = "|",
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    df = load_conase_csv(csv_path=csv_path, sep=sep)
    if df.empty:
        raise ValueError("Input CSV produced no rows.")

    if max_videos > 0:
        unique_videos = df["video_id"].dropna().astype(str).unique().tolist()
        if len(unique_videos) > max_videos:
            picked = set(random.sample(unique_videos, max_videos))
            df = df[df["video_id"].isin(picked)].copy()

    top_locations: Optional[set[str]] = None
    if label_type == "location_topk":
        top_locations = set(
            df["location"]
            .astype(str)
            .str.strip()
            .replace({"nan": ""})
            .value_counts()
            .head(top_k_locations)
            .index.tolist()
        )

    required_cols = [
        "video_id",
        "channel_id",
        "country",
        "state",
        "location",
        "lat",
        "lon",
        "clip_start",
        "clip_end",
        "label",
        "text_window",
        "token_times_json",
    ]
    rows: List[Dict[str, object]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building manifest"):
        text_pos = _safe_text(row.get("text_pos"))
        tokens = parse_text_pos(text_pos)
        if len(tokens) < 2:
            continue
        label = _pick_label(
            row=row,
            label_type=label_type,
            top_locations=top_locations,
            geo_cell_size_deg=geo_cell_size_deg,
        )
        if not label:
            continue

        times = [t.time for t in tokens]
        starts = _clip_starts(min(times), max(times), clip_seconds=clip_seconds, clips_per_video=clips_per_video)
        for start in starts:
            end = start + clip_seconds
            token_window = tokens_in_window(tokens=tokens, clip_start=start, clip_end=end)
            if len(token_window) < 2:
                continue
            rows.append(
                {
                    "video_id": _safe_text(row.get("video_id")),
                    "channel_id": _safe_text(row.get("channel_id")),
                    "country": _safe_text(row.get("country")),
                    "state": _safe_text(row.get("state")),
                    "location": _safe_text(row.get("location")),
                    "lat": _safe_float(row.get("lat")),
                    "lon": _safe_float(row.get("lon")),
                    "clip_start": float(start),
                    "clip_end": float(end),
                    "label": label,
                    "text_window": window_text(token_window),
                    "token_times_json": serialize_tokens_for_window(token_window, clip_start=start),
                }
            )

    if not rows:
        raise ValueError("No valid manifest rows were generated. Check text_pos and labels.")

    manifest = pd.DataFrame(rows)
    label_counts = manifest["label"].value_counts()
    keep_labels = label_counts[label_counts >= min_label_examples].index
    manifest = manifest[manifest["label"].isin(keep_labels)].copy()
    manifest.reset_index(drop=True, inplace=True)

    if manifest.empty:
        raise ValueError("All labels were filtered out by min_label_examples.")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest[required_cols]
    manifest.to_csv(out_manifest, index=False)
    LOGGER.info(
        "Saved manifest to %s with %s rows across %s labels.",
        out_manifest,
        len(manifest),
        manifest["label"].nunique(),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CoNASE clip-level training manifest.")
    parser.add_argument("--csv_path", required=True, type=Path, help="Path to CoNASE CSV (pipe-delimited by default).")
    parser.add_argument("--out_manifest", required=True, type=Path, help="Output CSV manifest path.")
    parser.add_argument("--label_type", default="state", choices=["state", "location_topk", "geo_grid"])
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--clips_per_video", type=int, default=2)
    parser.add_argument("--max_videos", type=int, default=2_000)
    parser.add_argument("--min_label_examples", type=int, default=200)
    parser.add_argument("--top_k_locations", type=int, default=200)
    parser.add_argument("--geo_cell_size_deg", type=float, default=1.5)
    parser.add_argument("--sep", default="|", help="Input CSV delimiter.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()
    build_manifest(
        csv_path=args.csv_path,
        out_manifest=args.out_manifest,
        label_type=args.label_type,
        clip_seconds=args.clip_seconds,
        clips_per_video=args.clips_per_video,
        max_videos=args.max_videos,
        min_label_examples=args.min_label_examples,
        top_k_locations=args.top_k_locations,
        geo_cell_size_deg=args.geo_cell_size_deg,
        sep=args.sep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
