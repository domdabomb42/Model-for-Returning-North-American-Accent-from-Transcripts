from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd

from conase_geo.models.text_mpsa import TextMPSADenseNetClassifier
from conase_geo.text_utils import encode_text_to_token_ids, scores_to_probs, split_from_group

try:
    import torch
except Exception:  # pragma: no cover - optional runtime path
    torch = None

LOGGER = logging.getLogger(__name__)
MODEL_TYPE_MPSA_TOKEN = "text_mpsa_token_map"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Timed evaluation for text model with label-diverse sampling from holdout splits."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to text_model.joblib")
    parser.add_argument("--csv_path", type=Path, required=True, help="Path to source CSV.")
    parser.add_argument("--minutes", type=float, default=5.0, help="Total wall-clock budget for evaluation.")
    parser.add_argument("--out_json", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--sep", type=str, default="|")
    parser.add_argument("--chunksize", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--splits", type=str, default="val,test", help="Comma-separated subset of val,test.")
    parser.add_argument("--per_label_cap", type=int, default=250, help="Max sampled rows per label per split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_col", type=str, default="")
    parser.add_argument("--label_col", type=str, default="")
    parser.add_argument("--group_col", type=str, default="")
    parser.add_argument("--min_text_chars", type=int, default=-1)
    parser.add_argument("--val_fraction", type=float, default=-1.0)
    parser.add_argument("--test_fraction", type=float, default=-1.0)
    parser.add_argument("--lat_col", type=str, default="")
    parser.add_argument("--lon_col", type=str, default="")
    parser.add_argument("--latlong_col", type=str, default="")
    return parser


def _parse_splits(raw: str) -> List[str]:
    out: List[str] = []
    for token in raw.split(","):
        s = token.strip().lower()
        if not s:
            continue
        if s not in {"val", "test"}:
            raise ValueError(f"Unsupported split: {s}")
        out.append(s)
    if not out:
        raise ValueError("At least one split must be selected in --splits")
    return out


def _resolve_args_with_checkpoint(args: argparse.Namespace, config: Dict[str, object]) -> argparse.Namespace:
    if not args.text_col:
        args.text_col = str(config.get("text_col", "text"))
    if not args.label_col:
        args.label_col = str(config.get("label_col", "state"))
    if not args.group_col:
        args.group_col = str(config.get("group_col", "channel_id"))
    if not args.lat_col:
        args.lat_col = str(config.get("lat_col", "lat"))
    if not args.lon_col:
        args.lon_col = str(config.get("lon_col", "lon"))
    if not args.latlong_col:
        args.latlong_col = str(config.get("latlong_col", "latlong"))
    if args.min_text_chars < 0:
        args.min_text_chars = int(config.get("min_text_chars", 20))
    if args.val_fraction < 0:
        args.val_fraction = float(config.get("val_fraction", 0.1))
    if args.test_fraction < 0:
        args.test_fraction = float(config.get("test_fraction", 0.1))
    return args


def _normalize_text_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.replace("@", " ", regex=False)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def _available_columns(csv_path: Path, sep: str) -> Set[str]:
    header = pd.read_csv(
        csv_path,
        sep=sep,
        dtype=str,
        nrows=0,
        keep_default_na=False,
        na_filter=False,
    )
    return {str(c).strip() for c in header.columns}


def _extract_coords_from_chunk(
    *,
    chunk: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    latlong_col: str,
    available_cols: Set[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(chunk)
    lat = np.full(n, np.nan, dtype=np.float64)
    lon = np.full(n, np.nan, dtype=np.float64)

    if lat_col in available_cols and lon_col in available_cols and lat_col in chunk.columns and lon_col in chunk.columns:
        lat = pd.to_numeric(chunk[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(chunk[lon_col], errors="coerce").to_numpy(dtype=np.float64)

    if latlong_col in available_cols and latlong_col in chunk.columns:
        ll = chunk[latlong_col].astype(str).str.replace(r"[\(\)\[\]]", "", regex=True).str.strip()
        parts = ll.str.extract(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$")
        lat_ll = pd.to_numeric(parts[0], errors="coerce").to_numpy(dtype=np.float64)
        lon_ll = pd.to_numeric(parts[1], errors="coerce").to_numpy(dtype=np.float64)
        lat_mask = ~np.isfinite(lat)
        lon_mask = ~np.isfinite(lon)
        lat[lat_mask] = lat_ll[lat_mask]
        lon[lon_mask] = lon_ll[lon_mask]

    return lat, lon


def _add_sample_with_reservoir(
    *,
    sample: Tuple[str, float, float],
    label_idx: int,
    split: str,
    buckets: Dict[str, Dict[int, List[Tuple[str, float, float]]]],
    seen_counts: Dict[str, Dict[int, int]],
    cap: int,
    rng: random.Random,
) -> None:
    split_buckets = buckets[split]
    split_seen = seen_counts[split]
    bucket = split_buckets.setdefault(label_idx, [])
    seen = split_seen.get(label_idx, 0) + 1
    split_seen[label_idx] = seen

    if len(bucket) < cap:
        bucket.append(sample)
        return

    # Reservoir replacement so early rows do not dominate.
    j = rng.randint(0, seen - 1)
    if j < cap:
        bucket[j] = sample


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12))))


def _build_mpsa_eval_bundle(artifact: Dict[str, object]) -> Tuple[object, Dict[str, int], int, "torch.device"]:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate token-map MPSA checkpoints.")
    state_dict = artifact.get("model_state_dict")
    vocab = artifact.get("vocab")
    idx_to_label = artifact.get("idx_to_label")
    config = artifact.get("config", {})
    if not isinstance(state_dict, dict):
        raise ValueError("Token-map MPSA checkpoint missing model_state_dict.")
    if not isinstance(vocab, dict):
        raise ValueError("Token-map MPSA checkpoint missing vocab.")
    if not isinstance(idx_to_label, list):
        raise ValueError("Token-map MPSA checkpoint missing idx_to_label.")
    if not isinstance(config, dict):
        config = {}

    block_cfg_raw = config.get("mpsa_block_config", [6, 12, 24, 16])
    if not isinstance(block_cfg_raw, Sequence) or len(block_cfg_raw) != 4:
        block_cfg_raw = [6, 12, 24, 16]
    block_cfg = tuple(int(v) for v in block_cfg_raw)
    max_len = int(config.get("chunk_len", config.get("max_len", 384)))

    model = TextMPSADenseNetClassifier(
        num_classes=len(idx_to_label),
        vocab_size=len(vocab),
        embed_dim=int(config.get("embed_dim", 128)),
        max_len=max_len,
        block_config=block_cfg,  # type: ignore[arg-type]
        growth_rate=int(config.get("mpsa_growth_rate", 32)),
        num_init_features=int(config.get("mpsa_num_init_features", 64)),
        bn_size=int(config.get("mpsa_bn_size", 4)),
        drop_rate=float(config.get("mpsa_drop_rate", 0.0)),
        embed_dropout=float(config.get("mpsa_embed_dropout", 0.0)),
        head_hidden_dim=int(config.get("head_hidden_dim", 512)),
        head_dropout=float(config.get("head_dropout", 0.2)),
    )
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, vocab, max_len, device


def _predict_probs_mpsa(
    *,
    model: object,
    vocab: Dict[str, int],
    max_len: int,
    device: "torch.device",
    texts: Sequence[str],
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate token-map MPSA checkpoints.")
    ids = np.stack([encode_text_to_token_ids(text, vocab=vocab, max_len=max_len) for text in texts], axis=0)
    x = torch.as_tensor(ids, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x)
        logits = out["accent"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs


def _evaluate_split(
    *,
    texts: List[str],
    y: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    vectorizer,
    model,
    batch_size: int,
    n_classes: int,
    centroids: np.ndarray,
) -> Dict[str, float]:
    if len(texts) == 0:
        return {
            "n": 0.0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
            "log_loss": 0.0,
            "distance_top1_km_mean": 0.0,
            "distance_top1_km_median": 0.0,
            "distance_top3min_km_mean": 0.0,
            "distance_top3min_km_median": 0.0,
            "within_100km_top1": 0.0,
            "within_300km_top1": 0.0,
            "within_800km_top1": 0.0,
            "within_100km_top3min": 0.0,
            "within_300km_top3min": 0.0,
            "within_800km_top3min": 0.0,
            "distance_coverage_top1_n": 0.0,
            "distance_coverage_top3min_n": 0.0,
        }

    n = len(texts)
    correct1 = 0
    correct3 = 0
    nll_sum = 0.0
    k = min(3, n_classes)
    d1_values: List[float] = []
    d3_values: List[float] = []

    for i in range(0, n, batch_size):
        bt = texts[i : i + batch_size]
        by = y[i : i + batch_size]
        bl_lat = true_lat[i : i + batch_size]
        bl_lon = true_lon[i : i + batch_size]
        X = vectorizer.transform(bt)
        scores = np.asarray(model.decision_function(X))
        if n_classes > 2 and scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = scores_to_probs(scores)

        pred = np.argmax(probs, axis=1)
        correct1 += int(np.sum(pred == by))

        topk = np.argpartition(probs, -k, axis=1)[:, -k:]
        correct3 += int(np.sum((topk == by[:, None]).any(axis=1)))

        nll_sum += float((-np.log(probs[np.arange(len(by)), by] + 1e-12)).sum())

        true_ok = np.isfinite(bl_lat) & np.isfinite(bl_lon)
        pred_lat = centroids[pred, 0]
        pred_lon = centroids[pred, 1]
        top1_ok = true_ok & np.isfinite(pred_lat) & np.isfinite(pred_lon)
        if np.any(top1_ok):
            d1 = _haversine_km(bl_lat[top1_ok], bl_lon[top1_ok], pred_lat[top1_ok], pred_lon[top1_ok])
            d1_values.extend(d1.tolist())

        # Distance to nearest class among top-k predictions.
        cand_lat = centroids[topk, 0]
        cand_lon = centroids[topk, 1]
        cand_ok = np.isfinite(cand_lat) & np.isfinite(cand_lon)
        if np.any(true_ok):
            lat1 = np.radians(bl_lat)[:, None]
            lon1 = np.radians(bl_lon)[:, None]
            lat2 = np.radians(cand_lat)
            lon2 = np.radians(cand_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
            a = np.clip(a, 0.0, 1.0)
            dist = 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12))))
            dist[~cand_ok] = np.inf
            d3 = np.min(dist, axis=1)
            d3_ok = true_ok & np.isfinite(d3) & (d3 < np.inf)
            if np.any(d3_ok):
                d3_values.extend(d3[d3_ok].tolist())

    d1_arr = np.asarray(d1_values, dtype=np.float64)
    d3_arr = np.asarray(d3_values, dtype=np.float64)

    return {
        "n": float(n),
        "accuracy": float(correct1 / max(n, 1)),
        "top3_accuracy": float(correct3 / max(n, 1)),
        "log_loss": float(nll_sum / max(n, 1)),
        "distance_top1_km_mean": float(d1_arr.mean()) if d1_arr.size else 0.0,
        "distance_top1_km_median": float(np.median(d1_arr)) if d1_arr.size else 0.0,
        "distance_top3min_km_mean": float(d3_arr.mean()) if d3_arr.size else 0.0,
        "distance_top3min_km_median": float(np.median(d3_arr)) if d3_arr.size else 0.0,
        "within_100km_top1": float((d1_arr <= 100.0).mean()) if d1_arr.size else 0.0,
        "within_300km_top1": float((d1_arr <= 300.0).mean()) if d1_arr.size else 0.0,
        "within_800km_top1": float((d1_arr <= 800.0).mean()) if d1_arr.size else 0.0,
        "within_100km_top3min": float((d3_arr <= 100.0).mean()) if d3_arr.size else 0.0,
        "within_300km_top3min": float((d3_arr <= 300.0).mean()) if d3_arr.size else 0.0,
        "within_800km_top3min": float((d3_arr <= 800.0).mean()) if d3_arr.size else 0.0,
        "distance_coverage_top1_n": float(d1_arr.size),
        "distance_coverage_top3min_n": float(d3_arr.size),
    }


def _evaluate_split_mpsa(
    *,
    texts: List[str],
    y: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    model: object,
    vocab: Dict[str, int],
    max_len: int,
    device: "torch.device",
    batch_size: int,
    n_classes: int,
    centroids: np.ndarray,
) -> Dict[str, float]:
    if len(texts) == 0:
        return {
            "n": 0.0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
            "log_loss": 0.0,
            "distance_top1_km_mean": 0.0,
            "distance_top1_km_median": 0.0,
            "distance_top3min_km_mean": 0.0,
            "distance_top3min_km_median": 0.0,
            "within_100km_top1": 0.0,
            "within_300km_top1": 0.0,
            "within_800km_top1": 0.0,
            "within_100km_top3min": 0.0,
            "within_300km_top3min": 0.0,
            "within_800km_top3min": 0.0,
            "distance_coverage_top1_n": 0.0,
            "distance_coverage_top3min_n": 0.0,
        }

    n = len(texts)
    correct1 = 0
    correct3 = 0
    nll_sum = 0.0
    k = min(3, n_classes)
    d1_values: List[float] = []
    d3_values: List[float] = []

    for i in range(0, n, batch_size):
        bt = texts[i : i + batch_size]
        by = y[i : i + batch_size]
        bl_lat = true_lat[i : i + batch_size]
        bl_lon = true_lon[i : i + batch_size]
        probs = _predict_probs_mpsa(
            model=model,
            vocab=vocab,
            max_len=max_len,
            device=device,
            texts=bt,
        )

        pred = np.argmax(probs, axis=1)
        correct1 += int(np.sum(pred == by))

        topk = np.argpartition(probs, -k, axis=1)[:, -k:]
        correct3 += int(np.sum((topk == by[:, None]).any(axis=1)))

        nll_sum += float((-np.log(probs[np.arange(len(by)), by] + 1e-12)).sum())

        true_ok = np.isfinite(bl_lat) & np.isfinite(bl_lon)
        pred_lat = centroids[pred, 0]
        pred_lon = centroids[pred, 1]
        top1_ok = true_ok & np.isfinite(pred_lat) & np.isfinite(pred_lon)
        if np.any(top1_ok):
            d1 = _haversine_km(bl_lat[top1_ok], bl_lon[top1_ok], pred_lat[top1_ok], pred_lon[top1_ok])
            d1_values.extend(d1.tolist())

        cand_lat = centroids[topk, 0]
        cand_lon = centroids[topk, 1]
        cand_ok = np.isfinite(cand_lat) & np.isfinite(cand_lon)
        if np.any(true_ok):
            lat1 = np.radians(bl_lat)[:, None]
            lon1 = np.radians(bl_lon)[:, None]
            lat2 = np.radians(cand_lat)
            lon2 = np.radians(cand_lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
            a = np.clip(a, 0.0, 1.0)
            dist = 6371.0 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12))))
            dist[~cand_ok] = np.inf
            d3 = np.min(dist, axis=1)
            d3_ok = true_ok & np.isfinite(d3) & (d3 < np.inf)
            if np.any(d3_ok):
                d3_values.extend(d3[d3_ok].tolist())

    d1_arr = np.asarray(d1_values, dtype=np.float64)
    d3_arr = np.asarray(d3_values, dtype=np.float64)
    return {
        "n": float(n),
        "accuracy": float(correct1 / max(n, 1)),
        "top3_accuracy": float(correct3 / max(n, 1)),
        "log_loss": float(nll_sum / max(n, 1)),
        "distance_top1_km_mean": float(d1_arr.mean()) if d1_arr.size else 0.0,
        "distance_top1_km_median": float(np.median(d1_arr)) if d1_arr.size else 0.0,
        "distance_top3min_km_mean": float(d3_arr.mean()) if d3_arr.size else 0.0,
        "distance_top3min_km_median": float(np.median(d3_arr)) if d3_arr.size else 0.0,
        "within_100km_top1": float((d1_arr <= 100.0).mean()) if d1_arr.size else 0.0,
        "within_300km_top1": float((d1_arr <= 300.0).mean()) if d1_arr.size else 0.0,
        "within_800km_top1": float((d1_arr <= 800.0).mean()) if d1_arr.size else 0.0,
        "within_100km_top3min": float((d3_arr <= 100.0).mean()) if d3_arr.size else 0.0,
        "within_300km_top3min": float((d3_arr <= 300.0).mean()) if d3_arr.size else 0.0,
        "within_800km_top3min": float((d3_arr <= 800.0).mean()) if d3_arr.size else 0.0,
        "distance_coverage_top1_n": float(d1_arr.size),
        "distance_coverage_top3min_n": float(d3_arr.size),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_arg_parser().parse_args()
    split_names = _parse_splits(args.splits)

    artifact = joblib.load(args.checkpoint)
    model_type = str(artifact.get("model_type", ""))
    label_to_idx = artifact["label_to_idx"]
    idx_to_label = artifact["idx_to_label"]
    cfg = artifact.get("config", {})
    args = _resolve_args_with_checkpoint(args, cfg if isinstance(cfg, dict) else {})
    vectorizer = artifact.get("vectorizer")
    model = artifact.get("model")
    mpsa_bundle: Optional[Tuple[object, Dict[str, int], int, "torch.device"]] = None
    if model_type == MODEL_TYPE_MPSA_TOKEN:
        mpsa_bundle = _build_mpsa_eval_bundle(artifact)
    elif vectorizer is None or model is None:
        raise ValueError("Checkpoint does not contain a usable linear text model.")

    if len(idx_to_label) < 2:
        raise ValueError("Checkpoint has fewer than 2 labels.")

    if args.minutes <= 0:
        raise ValueError("--minutes must be > 0")
    if args.per_label_cap <= 0:
        raise ValueError("--per_label_cap must be > 0")

    out_json = args.out_json or args.checkpoint.with_name("timed_eval_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    available_cols = _available_columns(args.csv_path, args.sep)
    coord_available = (
        (args.lat_col in available_cols and args.lon_col in available_cols)
        or (args.latlong_col in available_cols)
    )

    rng = random.Random(args.seed)
    start_t = time.time()
    total_budget_sec = float(args.minutes) * 60.0
    reserve_eval_sec = max(15.0, min(90.0, total_budget_sec * 0.25))
    read_deadline = start_t + max(10.0, total_budget_sec - reserve_eval_sec)

    buckets: Dict[str, Dict[int, List[Tuple[str, float, float]]]] = {s: {} for s in split_names}
    seen_counts: Dict[str, Dict[int, int]] = {s: {} for s in split_names}
    split_cache: Dict[str, str] = {}

    n_classes = len(idx_to_label)
    lat_sum = np.zeros(n_classes, dtype=np.float64)
    lon_sum = np.zeros(n_classes, dtype=np.float64)
    lat_lon_count = np.zeros(n_classes, dtype=np.int64)

    rows_seen = 0
    rows_eligible = 0
    rows_sampled = 0
    rows_with_coords = 0

    usecols = [args.text_col, args.label_col, args.group_col]
    if args.lat_col in available_cols:
        usecols.append(args.lat_col)
    if args.lon_col in available_cols and args.lon_col not in usecols:
        usecols.append(args.lon_col)
    if args.latlong_col in available_cols and args.latlong_col not in usecols:
        usecols.append(args.latlong_col)

    reader = pd.read_csv(
        args.csv_path,
        sep=args.sep,
        dtype=str,
        chunksize=args.chunksize,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
        usecols=usecols,
    )

    for chunk in reader:
        if args.max_rows > 0:
            remaining = args.max_rows - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        rows_seen += len(chunk)

        texts = _normalize_text_series(chunk[args.text_col])
        labels = chunk[args.label_col].astype(str).str.strip()
        groups = chunk[args.group_col].astype(str)
        lat_arr, lon_arr = _extract_coords_from_chunk(
            chunk=chunk,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            latlong_col=args.latlong_col,
            available_cols=available_cols,
        )

        mask = (texts.str.len() >= args.min_text_chars) & (labels != "")
        if not bool(mask.any()):
            if time.time() >= read_deadline:
                break
            continue

        sel_texts = texts[mask].tolist()
        sel_labels = labels[mask].tolist()
        sel_groups = groups[mask].tolist()
        sel_lat = lat_arr[mask.to_numpy()]
        sel_lon = lon_arr[mask.to_numpy()]

        for text, label, group, lat, lon in zip(sel_texts, sel_labels, sel_groups, sel_lat, sel_lon):
            label_idx = label_to_idx.get(label)
            if label_idx is None:
                continue
            rows_eligible += 1

            if np.isfinite(lat) and np.isfinite(lon):
                lat_sum[int(label_idx)] += float(lat)
                lon_sum[int(label_idx)] += float(lon)
                lat_lon_count[int(label_idx)] += 1
                rows_with_coords += 1

            group_key = str(group or "")
            split = split_cache.get(group_key)
            if split is None:
                split = split_from_group(
                    group_value=group_key,
                    seed=args.seed,
                    val_fraction=args.val_fraction,
                    test_fraction=args.test_fraction,
                )
                split_cache[group_key] = split

            if split not in buckets:
                continue

            pre_count = len(buckets[split].get(int(label_idx), []))
            _add_sample_with_reservoir(
                sample=(text, float(lat), float(lon)),
                label_idx=int(label_idx),
                split=split,
                buckets=buckets,
                seen_counts=seen_counts,
                cap=args.per_label_cap,
                rng=rng,
            )
            post_count = len(buckets[split].get(int(label_idx), []))
            if post_count > pre_count:
                rows_sampled += 1

        if time.time() >= read_deadline:
            break

    centroids = np.full((n_classes, 2), np.nan, dtype=np.float64)
    valid_centroid = lat_lon_count > 0
    centroids[valid_centroid, 0] = lat_sum[valid_centroid] / lat_lon_count[valid_centroid]
    centroids[valid_centroid, 1] = lon_sum[valid_centroid] / lat_lon_count[valid_centroid]

    # Flatten sampled sets for each split.
    flat_texts: Dict[str, List[str]] = {s: [] for s in split_names}
    flat_y: Dict[str, np.ndarray] = {}
    flat_lat: Dict[str, np.ndarray] = {}
    flat_lon: Dict[str, np.ndarray] = {}
    sampled_label_counts: Dict[str, Dict[str, int]] = {}

    for split in split_names:
        ys: List[int] = []
        lats: List[float] = []
        lons: List[float] = []
        label_counts: Dict[str, int] = {}
        split_buckets = buckets[split]
        for label_idx, samples in split_buckets.items():
            if not samples:
                continue
            label_name = str(idx_to_label[label_idx])
            label_counts[label_name] = len(samples)
            for text, lat, lon in samples:
                flat_texts[split].append(text)
                ys.append(int(label_idx))
                lats.append(float(lat))
                lons.append(float(lon))
        flat_y[split] = np.asarray(ys, dtype=np.int64)
        flat_lat[split] = np.asarray(lats, dtype=np.float64)
        flat_lon[split] = np.asarray(lons, dtype=np.float64)
        sampled_label_counts[split] = label_counts

    metrics: Dict[str, Dict[str, float]] = {}
    for split in split_names:
        if model_type == MODEL_TYPE_MPSA_TOKEN:
            if mpsa_bundle is None:
                raise RuntimeError("Token-map MPSA evaluation bundle is missing.")
            mpsa_model, mpsa_vocab, mpsa_max_len, mpsa_device = mpsa_bundle
            metrics[split] = _evaluate_split_mpsa(
                texts=flat_texts[split],
                y=flat_y[split],
                true_lat=flat_lat[split],
                true_lon=flat_lon[split],
                model=mpsa_model,
                vocab=mpsa_vocab,
                max_len=mpsa_max_len,
                device=mpsa_device,
                batch_size=args.batch_size,
                n_classes=n_classes,
                centroids=centroids,
            )
        else:
            metrics[split] = _evaluate_split(
                texts=flat_texts[split],
                y=flat_y[split],
                true_lat=flat_lat[split],
                true_lon=flat_lon[split],
                vectorizer=vectorizer,
                model=model,
                batch_size=args.batch_size,
                n_classes=n_classes,
                centroids=centroids,
            )

    elapsed = time.time() - start_t
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "csv_path": str(args.csv_path),
        "minutes_requested": float(args.minutes),
        "elapsed_sec": float(round(elapsed, 3)),
        "rows_seen": int(rows_seen),
        "rows_eligible": int(rows_eligible),
        "rows_with_coords": int(rows_with_coords),
        "rows_sampled_total": int(sum(len(v) for v in flat_texts.values())),
        "rows_sampled_added_during_stream": int(rows_sampled),
        "splits": split_names,
        "metrics": metrics,
        "sampled_label_counts": sampled_label_counts,
        "sampling": {
            "per_label_cap": int(args.per_label_cap),
            "read_deadline_sec_from_start": float(round(read_deadline - start_t, 3)),
            "batch_size": int(args.batch_size),
            "chunksize": int(args.chunksize),
        },
        "closeness": {
            "enabled": bool(coord_available),
            "labels_with_centroid": int(np.sum(valid_centroid)),
            "labels_total": int(n_classes),
            "lat_col": args.lat_col,
            "lon_col": args.lon_col,
            "latlong_col": args.latlong_col,
        },
    }

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    LOGGER.info("Timed evaluation complete in %.2fs", elapsed)
    for split in split_names:
        m = metrics.get(split, {})
        LOGGER.info(
            "%s | n=%s acc=%.4f top3=%.4f loss=%.4f dist1_mean=%.1fkm dist3_mean=%.1fkm labels=%s",
            split,
            int(m.get("n", 0.0)),
            float(m.get("accuracy", 0.0)),
            float(m.get("top3_accuracy", 0.0)),
            float(m.get("log_loss", 0.0)),
            float(m.get("distance_top1_km_mean", 0.0)),
            float(m.get("distance_top3min_km_mean", 0.0)),
            len(sampled_label_counts.get(split, {})),
        )
    LOGGER.info("Saved: %s", out_json)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
