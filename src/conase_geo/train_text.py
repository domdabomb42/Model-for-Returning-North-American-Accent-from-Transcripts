from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from conase_geo.models.text_linear import TextLinearDecisionModel
from conase_geo.models.text_mpsa import TextMPSADenseNetClassifier
from conase_geo.text_utils import (
    LocationMasker,
    encode_text_to_token_ids,
    encode_text_to_token_ids_with_length,
    keep_labels_from_counts,
    load_location_terms,
    read_csv_in_chunks,
    scores_to_probs,
    split_from_group,
    tokenize_text,
    topk_accuracy,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception:  # pragma: no cover - optional runtime dependency path
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None

LOGGER = logging.getLogger(__name__)
MODEL_TYPE_SKLEARN = "text_hashing_sgd"
MODEL_TYPE_TORCH = "text_hashing_torch_linear"
MODEL_TYPE_MPSA_TOKEN = "text_mpsa_token_map"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train writing-only location classifier from a large CSV.")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sep", type=str, default="|")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="state")
    parser.add_argument("--group_col", type=str, default="channel_id")
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min_text_chars", type=int, default=20)
    parser.add_argument("--min_label_examples", type=int, default=500)
    parser.add_argument("--top_k_labels", type=int, default=0, help="0 means keep all labels above min count.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_hash",
        type=str,
        choices=["blake2", "crc32"],
        default="blake2",
        help="Hashing method for deterministic group split assignment.",
    )
    parser.add_argument("--n_features", type=int, default=2**20)
    parser.add_argument("--analyzer", type=str, choices=["word", "char", "char_wb"], default="char_wb")
    parser.add_argument("--ngram_min", type=int, default=3)
    parser.add_argument("--ngram_max", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--trainer", type=str, choices=["sklearn", "torch", "mpsa"], default="sklearn")
    parser.add_argument("--device", type=str, default="auto", help="Used for torch and mpsa trainers.")
    parser.add_argument("--torch_optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--torch_momentum", type=float, default=0.9, help="Used only for --torch_optimizer sgd.")
    parser.add_argument("--torch_lr", type=float, default=3e-2, help="Used only when --trainer torch.")
    parser.add_argument("--torch_weight_decay", type=float, default=1e-6, help="Used only when --trainer torch.")
    parser.add_argument("--geo_loss_mode", type=str, choices=["none", "centroid"], default="none")
    parser.add_argument("--geo_loss_mix", type=float, default=0.35, help="0..1 mix: 0 hard CE only, 1 geo-soft only.")
    parser.add_argument("--geo_sigma_km", type=float, default=850.0, help="Distance temperature for centroid soft targets.")
    parser.add_argument("--lat_col", type=str, default="lat")
    parser.add_argument("--lon_col", type=str, default="lon")
    parser.add_argument("--latlong_col", type=str, default="latlong")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--input_repr", type=str, choices=["hashing", "token_map"], default="hashing")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Feature cache dir (default: <output_dir>/feature_cache).")
    parser.add_argument("--cache_rebuild", action="store_true", help="Ignore cached tokenized/compressed features and rebuild.")
    parser.add_argument("--rebuild_cache", action="store_true", help="Alias for --cache_rebuild.")
    parser.add_argument("--cache_shard_size", type=int, default=10000, help="Rows per on-disk cache shard.")
    parser.add_argument("--max_len", type=int, default=384, help="Deprecated for mpsa chunked flow; kept for backward compatibility.")
    parser.add_argument("--max_tokens_cap", type=int, default=1024, help="Cache full token ids up to this cap for chunk slicing.")
    parser.add_argument("--chunk_len", type=int, default=256, help="Token window length fed to MPSA model.")
    parser.add_argument("--train_chunks_per_sample", type=int, default=3, help="Number of cached windows per sample during training.")
    parser.add_argument("--eval_chunks_per_sample", type=int, default=3, help="Number of deterministic windows per sample during eval.")
    parser.add_argument(
        "--eval_chunk_agg",
        type=str,
        choices=["mean", "max", "logsumexp"],
        default="mean",
        help="How to aggregate chunk logits per sample during eval.",
    )
    parser.add_argument("--mask_locations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--location_list_path", type=Path, default=None, help="Optional newline list of extra location terms to mask.")
    parser.add_argument("--mask_prob", type=float, default=1.0, help="Location masking probability before tokenization.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Token embedding dim for token_map compact inputs.")
    parser.add_argument("--max_vocab_size", type=int, default=50000, help="Max tokenizer vocab size including special tokens.")
    parser.add_argument("--min_token_freq", type=int, default=2, help="Minimum train-split token count for vocab inclusion.")
    parser.add_argument("--mpsa_growth_rate", type=int, default=32)
    parser.add_argument("--mpsa_block_config", type=str, default="6,12,24,16")
    parser.add_argument("--mpsa_bn_size", type=int, default=4)
    parser.add_argument("--mpsa_drop_rate", type=float, default=0.0)
    parser.add_argument("--mpsa_num_init_features", type=int, default=64)
    parser.add_argument("--mpsa_embed_dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for mpsa trainer.")
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", type=str, choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="max-autotune")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--class_weighted_loss",
        action="store_true",
        help="For --trainer mpsa, use class-frequency weighted CrossEntropyLoss on train batches.",
    )
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=1.0,
        help="Weight exponent for class-weighted loss: weight ~= 1 / count^power.",
    )
    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=0.0,
        help="Optional cap for class-weighted loss weights (0 disables cap).",
    )
    parser.add_argument(
        "--weighted_sampling",
        action="store_true",
        help="For --trainer mpsa, use WeightedRandomSampler on train split.",
    )
    parser.add_argument(
        "--sampling_weight_power",
        type=float,
        default=1.0,
        help="Sampling exponent: per-class sample weight ~= 1 / count^power.",
    )
    parser.add_argument(
        "--sampling_replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replacement mode for WeightedRandomSampler.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, choices=["none", "cosine", "onecycle"], default="onecycle")
    parser.add_argument("--freeze_embedding_epochs", type=int, default=0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)
    parser.add_argument("--head_lr", type=float, default=0.0, help="Classifier head LR (0 means use --torch_lr).")
    parser.add_argument("--backbone_lr", type=float, default=0.0, help="Backbone LR (0 means scaled from --torch_lr).")
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze_strategy", type=str, choices=["all", "gradual"], default="all")
    parser.add_argument("--smoke_steps", type=int, default=0, help="Stop training after this many optimizer steps (0 disables).")
    parser.add_argument("--benchmark_mode", action="store_true", help="Measure first-step timings and throughput during training.")
    parser.add_argument("--benchmark_warmup_steps", type=int, default=20)
    parser.add_argument("--benchmark_steps", type=int, default=100)
    parser.add_argument(
        "--log_every_batches",
        type=int,
        default=0,
        help="If >0, log training progress every N batches (epoch batch index + loss).",
    )
    parser.add_argument(
        "--startup_trace_steps",
        type=int,
        default=2,
        help="Log detailed stage markers for the first N train batches to pinpoint startup crashes.",
    )
    parser.add_argument(
        "--startup_trace_sync",
        action="store_true",
        help="Call CUDA synchronize at traced startup stages for precise failure localization.",
    )
    parser.add_argument("--fast_mode", action="store_true", help="Enable speed-first defaults for quick convergence checks.")
    parser.add_argument("--autosave_every_batches", type=int, default=0)
    parser.add_argument("--autosave_every_minutes", type=float, default=30.0)
    parser.add_argument("--max_train_minutes", type=float, default=0.0, help="0 disables wall-clock stop.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip pass-3 evaluation to maximize training time.")
    parser.add_argument(
        "--eval_every_epoch",
        action="store_true",
        help="Run validation after each epoch and store per-epoch metrics.",
    )
    parser.add_argument(
        "--epoch_history_path",
        type=Path,
        default=None,
        help="Path for JSONL per-epoch history. Defaults to <output_dir>/epoch_history.jsonl.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
        help="Resume from an autosave or model artifact created by this script.",
    )
    return parser


def _normalize_text_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.replace("@", " ", regex=False)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _iter_prepared_chunks(
    csv_path: Path,
    sep: str,
    chunksize: int,
    max_rows: int,
    text_col: str,
    label_col: str,
    group_col: str,
    min_text_chars: int,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_hash: str,
    split_cache: Optional[Dict[str, str]] = None,
    text_transform: Optional[Callable[[str], str]] = None,
):
    local_split_cache: Dict[str, str] = split_cache if split_cache is not None else {}
    needed_cols = {text_col, label_col, group_col, "video_id"}
    usecols = lambda col: str(col).strip() in needed_cols
    for chunk in read_csv_in_chunks(
        str(csv_path),
        sep=sep,
        chunksize=chunksize,
        max_rows=max_rows,
        usecols=usecols,
    ):
        if text_col not in chunk.columns:
            raise ValueError(f"Missing text column: {text_col}")
        if label_col not in chunk.columns:
            raise ValueError(f"Missing label column: {label_col}")

        has_group = group_col in chunk.columns
        texts = _normalize_text_series(chunk[text_col])
        labels = chunk[label_col].astype(str).str.strip()

        if has_group:
            groups = chunk[group_col].astype(str)
        elif "video_id" in chunk.columns:
            groups = chunk["video_id"].astype(str)
        else:
            groups = labels

        mask = (texts.str.len() >= min_text_chars) & (labels != "")
        if not bool(mask.any()):
            continue

        sel_texts = texts[mask].to_numpy(dtype=str, copy=False)
        if text_transform is not None and sel_texts.size > 0:
            sel_texts = np.asarray([text_transform(t) for t in sel_texts], dtype=str)
        sel_labels = labels[mask].to_numpy(dtype=str, copy=False)
        sel_groups = groups[mask].astype(str).to_numpy(dtype=str, copy=False)

        unique_groups = pd.unique(sel_groups)
        for group_value in unique_groups:
            group_key = str(group_value or "")
            if group_key in local_split_cache:
                continue
            local_split_cache[group_key] = split_from_group(
                group_value=group_key,
                seed=seed,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                hash_method=split_hash,
            )
        splits = np.fromiter((local_split_cache[str(g or "")] for g in sel_groups), dtype="<U5", count=len(sel_groups))
        yield splits, sel_texts, sel_labels


def _iter_records(
    csv_path: Path,
    sep: str,
    chunksize: int,
    max_rows: int,
    text_col: str,
    label_col: str,
    group_col: str,
    min_text_chars: int,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_hash: str,
    split_cache: Optional[Dict[str, str]] = None,
):
    for splits, texts, labels in _iter_prepared_chunks(
        csv_path=csv_path,
        sep=sep,
        chunksize=chunksize,
        max_rows=max_rows,
        text_col=text_col,
        label_col=label_col,
        group_col=group_col,
        min_text_chars=min_text_chars,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_hash=split_hash,
        split_cache=split_cache,
    ):
        for split, text, label in zip(splits, texts, labels):
            yield split, text, label


def _pass_label_counts(args: argparse.Namespace) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total_rows = 0
    needed_cols = {args.text_col, args.label_col}
    usecols = lambda col: str(col).strip() in needed_cols
    reader = pd.read_csv(
        args.csv_path,
        sep=args.sep,
        dtype=str,
        chunksize=args.chunksize,
        usecols=usecols,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
    )
    with tqdm(desc="Pass 1/3 - counting labels", unit="rows") as pbar:
        for chunk in reader:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if args.text_col not in chunk.columns:
                raise ValueError(f"Missing text column: {args.text_col}")
            if args.label_col not in chunk.columns:
                raise ValueError(f"Missing label column: {args.label_col}")

            if args.max_rows > 0:
                remaining = args.max_rows - total_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            total_rows += len(chunk)
            pbar.update(len(chunk))

            texts = _normalize_text_series(chunk[args.text_col])
            labels = chunk[args.label_col].astype(str).str.strip()
            mask = (texts.str.len() >= args.min_text_chars) & (labels != "")
            if not bool(mask.any()):
                continue

            vc = labels[mask].value_counts()
            for label, c in vc.items():
                label_s = str(label)
                counts[label_s] = counts.get(label_s, 0) + int(c)
    return counts


def _extract_lat_lon_from_chunk(
    chunk: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    latlong_col: str,
) -> Tuple[pd.Series, pd.Series]:
    lat: pd.Series
    lon: pd.Series
    if lat_col in chunk.columns and lon_col in chunk.columns:
        lat = pd.to_numeric(chunk[lat_col], errors="coerce")
        lon = pd.to_numeric(chunk[lon_col], errors="coerce")
    else:
        lat = pd.Series(np.nan, index=chunk.index, dtype=float)
        lon = pd.Series(np.nan, index=chunk.index, dtype=float)

    if latlong_col in chunk.columns:
        ll = chunk[latlong_col].astype(str)
        ll = ll.str.replace(r"[\(\)\[\]]", "", regex=True).str.strip()
        parts = ll.str.extract(r"^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$")
        lat_ll = pd.to_numeric(parts[0], errors="coerce")
        lon_ll = pd.to_numeric(parts[1], errors="coerce")
        lat = lat.where(lat.notna(), lat_ll)
        lon = lon.where(lon.notna(), lon_ll)
    return lat, lon


def _haversine_pairwise_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + np.cos(lat)[:, None] * np.cos(lat)[None, :] * sin_dlon * sin_dlon
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 1e-12)))
    return 6371.0 * c


def _build_geo_target_matrix(
    args: argparse.Namespace,
    keep_labels: List[str],
) -> Optional[np.ndarray]:
    if args.geo_loss_mode != "centroid":
        return None
    if args.geo_sigma_km <= 0:
        LOGGER.warning("geo_loss_mode=centroid requested but geo_sigma_km<=0. Disabling geo loss.")
        return None
    if len(keep_labels) < 2:
        return None
    if len(keep_labels) > 1200:
        LOGGER.warning(
            "geo_loss_mode=centroid disabled because class count (%s) is too high for dense pairwise distances.",
            len(keep_labels),
        )
        return None

    keep_set = set(keep_labels)
    sums: Dict[str, List[float]] = {label: [0.0, 0.0, 0.0] for label in keep_labels}

    needed_cols = {args.text_col, args.label_col, args.lat_col, args.lon_col, args.latlong_col}
    usecols = lambda col: str(col).strip() in needed_cols
    total = 0
    iterator = read_csv_in_chunks(
        str(args.csv_path),
        sep=args.sep,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        usecols=usecols,
    )
    for chunk in tqdm(iterator, desc="Geo pass - label centroids"):
        if args.text_col not in chunk.columns or args.label_col not in chunk.columns:
            LOGGER.warning("Missing text/label columns for geo centroid pass. Disabling geo loss.")
            return None

        texts = _normalize_text_series(chunk[args.text_col])
        labels = chunk[args.label_col].astype(str).str.strip()
        lat_s, lon_s = _extract_lat_lon_from_chunk(
            chunk,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            latlong_col=args.latlong_col,
        )
        mask = (texts.str.len() >= args.min_text_chars) & labels.isin(keep_set) & lat_s.notna() & lon_s.notna()
        if not bool(mask.any()):
            continue

        tmp = pd.DataFrame(
            {
                "label": labels[mask].values,
                "lat": lat_s[mask].values,
                "lon": lon_s[mask].values,
            }
        )
        agg = tmp.groupby("label", sort=False).agg(lat_sum=("lat", "sum"), lon_sum=("lon", "sum"), n=("lat", "size"))
        for label, row in agg.iterrows():
            stats = sums[str(label)]
            stats[0] += float(row["lat_sum"])
            stats[1] += float(row["lon_sum"])
            stats[2] += float(row["n"])
        total += int(mask.sum())

    label_to_idx = {label: i for i, label in enumerate(keep_labels)}
    centroids = np.full((len(keep_labels), 2), np.nan, dtype=np.float64)
    for label, (lat_sum, lon_sum, n) in sums.items():
        if n <= 0:
            continue
        idx = label_to_idx[label]
        centroids[idx, 0] = lat_sum / n
        centroids[idx, 1] = lon_sum / n

    known = np.isfinite(centroids[:, 0]) & np.isfinite(centroids[:, 1])
    known_idx = np.where(known)[0]
    if known_idx.size < 2:
        LOGGER.warning("Not enough labels with valid coordinates for geo loss. Disabling geo loss.")
        return None

    targets = np.zeros((len(keep_labels), len(keep_labels)), dtype=np.float32)
    for i in range(len(keep_labels)):
        targets[i, i] = 1.0

    known_lat = centroids[known_idx, 0]
    known_lon = centroids[known_idx, 1]
    dist_km = _haversine_pairwise_km(known_lat, known_lon)
    weights = np.exp(-dist_km / float(args.geo_sigma_km))
    denom = np.sum(weights, axis=1, keepdims=True) + 1e-12
    weights = weights / denom
    for row_local, class_i in enumerate(known_idx):
        targets[class_i, :] = 0.0
        targets[class_i, known_idx] = weights[row_local].astype(np.float32)

    LOGGER.info(
        "Geo loss enabled: centroid coverage=%s/%s labels from %s rows with valid coords, sigma_km=%s mix=%s",
        int(known_idx.size),
        int(len(keep_labels)),
        total,
        args.geo_sigma_km,
        args.geo_loss_mix,
    )
    return targets


def _build_model_and_vectorizer(args: argparse.Namespace) -> Tuple[HashingVectorizer, SGDClassifier]:
    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        alternate_sign=False,
        analyzer=args.analyzer,
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=True,
        norm="l2",
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=args.alpha,
        max_iter=1,
        tol=None,
        random_state=args.seed,
    )
    return vectorizer, model


def _parse_block_config(raw: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--mpsa_block_config must have 4 comma-separated integers (e.g. 6,12,24,16).")
    values = tuple(int(p) for p in parts)
    if any(v <= 0 for v in values):
        raise ValueError("--mpsa_block_config values must be positive.")
    return values  # type: ignore[return-value]


def _resolve_feature_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir
    return args.output_dir / "feature_cache"


def _build_text_masker(args: argparse.Namespace) -> Optional[Callable[[str], str]]:
    if not bool(args.mask_locations):
        return None
    extra_terms = load_location_terms(args.location_list_path)
    return LocationMasker(
        extra_terms=extra_terms,
        mask_prob=float(args.mask_prob),
        seed=int(args.seed),
    )


def _token_cache_path(args: argparse.Namespace, keep_labels: Sequence[str]) -> Path:
    cache_dir = _resolve_feature_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_stat = args.csv_path.stat()
    extra_terms = load_location_terms(args.location_list_path)
    location_list_digest = hashlib.blake2b(
        "\n".join(sorted(extra_terms)).encode("utf-8"), digest_size=8
    ).hexdigest()
    payload = {
        "tokenizer_version": 2,
        "csv_path": str(args.csv_path.resolve()),
        "csv_size": int(csv_stat.st_size),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "sep": args.sep,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "max_rows": args.max_rows,
        "chunksize": args.chunksize,
        "min_text_chars": args.min_text_chars,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "split_hash": args.split_hash,
        "max_len": args.max_len,
        "max_tokens_cap": args.max_tokens_cap,
        "chunk_len": args.chunk_len,
        "train_chunks_per_sample": args.train_chunks_per_sample,
        "eval_chunks_per_sample": args.eval_chunks_per_sample,
        "mask_locations": bool(args.mask_locations),
        "mask_prob": float(args.mask_prob),
        "location_list_digest": location_list_digest,
        "embed_dim": args.embed_dim,
        "max_vocab_size": args.max_vocab_size,
        "min_token_freq": args.min_token_freq,
        "labels": list(keep_labels),
    }
    digest = hashlib.blake2b(json.dumps(payload, sort_keys=True).encode("utf-8"), digest_size=16).hexdigest()
    return cache_dir / f"token_map_{digest}"


def _build_token_vocab(
    args: argparse.Namespace,
    label_to_idx: Dict[str, int],
    text_transform: Optional[Callable[[str], str]],
) -> Dict[str, int]:
    token_counter: Counter[str] = Counter()
    split_cache: Dict[str, str] = {}
    iterator = _iter_prepared_chunks(
        csv_path=args.csv_path,
        sep=args.sep,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        text_col=args.text_col,
        label_col=args.label_col,
        group_col=args.group_col,
        min_text_chars=args.min_text_chars,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_hash=args.split_hash,
        split_cache=split_cache,
        text_transform=text_transform,
    )
    for splits, texts, labels in tqdm(iterator, desc="Cache pass 1/2 - building vocab", unit="rows"):
        if len(texts) == 0:
            continue
        y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in labels), dtype=np.int64, count=len(labels))
        train_mask = (splits == "train") & (y_idx >= 0)
        if not bool(np.any(train_mask)):
            continue
        for text in texts[train_mask]:
            token_counter.update(tokenize_text(text))

    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
    for token, count in token_counter.most_common():
        if count < int(args.min_token_freq):
            continue
        if len(vocab) >= int(args.max_vocab_size):
            break
        vocab[token] = len(vocab)
    if len(vocab) <= 2:
        LOGGER.warning("Tokenizer vocab is near-empty. Falling back to only special tokens.")
    return vocab


def _iter_labeled_records(
    args: argparse.Namespace,
    label_to_idx: Dict[str, int],
    text_transform: Optional[Callable[[str], str]],
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    split_cache: Dict[str, str] = {}
    iterator = _iter_prepared_chunks(
        csv_path=args.csv_path,
        sep=args.sep,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        text_col=args.text_col,
        label_col=args.label_col,
        group_col=args.group_col,
        min_text_chars=args.min_text_chars,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_hash=args.split_hash,
        split_cache=split_cache,
        text_transform=text_transform,
    )
    for splits, texts, labels in iterator:
        if len(texts) == 0:
            continue
        y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in labels), dtype=np.int64, count=len(labels))
        valid = y_idx >= 0
        if not bool(np.any(valid)):
            continue
        yield splits[valid], texts[valid], y_idx[valid]


def _load_sharded_token_cache(cache_root: Path) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for token cache loading.")
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_files = manifest.get("shards", [])
    if not isinstance(shard_files, list) or not shard_files:
        raise ValueError(f"Cache manifest has no shards: {manifest_path}")
    vocab_path = cache_root / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Cache vocab missing: {vocab_path}")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError("Cache vocab.json is invalid.")

    ids_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    lens_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    split_parts: List[torch.Tensor] = []
    for shard_name in shard_files:
        shard_path = cache_root / str(shard_name)
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch versions
            payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid shard payload: {shard_path}")
        ids_parts.append(payload["token_ids"].to(dtype=torch.int32))
        if "attention_mask" in payload:
            mask_parts.append(payload["attention_mask"].to(dtype=torch.uint8))
        else:
            mask_parts.append((payload["token_ids"] != 0).to(dtype=torch.uint8))
        lens_parts.append(payload["token_lens"].to(dtype=torch.int32))
        label_parts.append(payload["labels"].to(dtype=torch.int64))
        split_parts.append(payload["splits"].to(dtype=torch.int8))

    token_ids = torch.cat(ids_parts, dim=0)
    attention_mask = torch.cat(mask_parts, dim=0)
    token_lens = torch.cat(lens_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    splits = torch.cat(split_parts, dim=0)
    return {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "token_lens": token_lens,
        "labels": labels,
        "splits": splits,
        "vocab": vocab,
        "meta": manifest,
    }


def _load_or_build_token_cache(
    args: argparse.Namespace,
    keep_labels: Sequence[str],
    label_to_idx: Dict[str, int],
    text_transform: Optional[Callable[[str], str]],
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for token-map cache loading.")
    cache_root = _token_cache_path(args, keep_labels)
    manifest_path = cache_root / "manifest.json"
    force_rebuild = bool(args.cache_rebuild) or bool(getattr(args, "rebuild_cache", False))
    if manifest_path.exists() and not force_rebuild:
        try:
            payload = _load_sharded_token_cache(cache_root)
            LOGGER.info("Loaded token-map cache: %s", cache_root)
            return payload
        except Exception as exc:
            LOGGER.warning("Token cache is invalid or outdated (%s). Rebuilding: %s", exc, cache_root)

    vocab = _build_token_vocab(args=args, label_to_idx=label_to_idx, text_transform=text_transform)
    split_id_map = {"train": 0, "val": 1, "test": 2}
    shard_size = max(1000, int(args.cache_shard_size))
    max_tokens_cap = max(int(args.chunk_len), int(args.max_tokens_cap))

    cache_root.mkdir(parents=True, exist_ok=True)
    for old_shard in cache_root.glob("shard_*.pt"):
        old_shard.unlink(missing_ok=True)

    shard_names: List[str] = []
    shard_idx = 0
    total_rows = 0
    buf_ids: List[np.ndarray] = []
    buf_lens: List[np.ndarray] = []
    buf_labels: List[np.ndarray] = []
    buf_splits: List[np.ndarray] = []
    buf_rows = 0

    def _flush() -> None:
        nonlocal shard_idx, buf_rows, total_rows
        if buf_rows <= 0:
            return
        ids_np = np.concatenate(buf_ids, axis=0)
        lens_np = np.concatenate(buf_lens, axis=0)
        labels_np = np.concatenate(buf_labels, axis=0)
        splits_np = np.concatenate(buf_splits, axis=0)
        shard_name = f"shard_{shard_idx:05d}.pt"
        shard_path = cache_root / shard_name
        payload = {
            "token_ids": torch.from_numpy(ids_np.astype(np.int32, copy=False)),
            "attention_mask": torch.from_numpy((ids_np != 0).astype(np.uint8, copy=False)),
            "token_lens": torch.from_numpy(lens_np.astype(np.int32, copy=False)),
            "labels": torch.from_numpy(labels_np.astype(np.int64, copy=False)),
            "splits": torch.from_numpy(splits_np.astype(np.int8, copy=False)),
        }
        torch.save(payload, shard_path)
        shard_names.append(shard_name)
        total_rows += int(ids_np.shape[0])
        shard_idx += 1
        buf_ids.clear()
        buf_lens.clear()
        buf_labels.clear()
        buf_splits.clear()
        buf_rows = 0

    for splits, texts, y_idx in tqdm(
        _iter_labeled_records(args=args, label_to_idx=label_to_idx, text_transform=text_transform),
        desc="Cache pass 2/2 - encoding token maps",
        unit="rows",
    ):
        n = int(len(texts))
        if n == 0:
            continue
        ids = np.zeros((n, max_tokens_cap), dtype=np.int32)
        lens = np.zeros((n,), dtype=np.int32)
        for i, text in enumerate(texts):
            ids_i, len_i = encode_text_to_token_ids_with_length(text, vocab=vocab, max_len=max_tokens_cap)
            ids[i] = ids_i
            lens[i] = int(len_i)
        split_ids = np.fromiter((split_id_map.get(str(s), -1) for s in splits), dtype=np.int8, count=n)
        keep = split_ids >= 0
        if not bool(np.any(keep)):
            continue

        kept_ids = ids[keep]
        kept_lens = lens[keep]
        kept_labels = y_idx[keep].astype(np.int64, copy=False)
        kept_splits = split_ids[keep]
        buf_ids.append(kept_ids)
        buf_lens.append(kept_lens)
        buf_labels.append(kept_labels)
        buf_splits.append(kept_splits)
        buf_rows += int(kept_ids.shape[0])
        while buf_rows >= shard_size:
            _flush()
    _flush()

    if total_rows <= 0:
        raise ValueError("No token-map samples were cached. Check label/text settings.")
    vocab_path = cache_root / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    manifest = {
        "cache_root": str(cache_root),
        "n_rows": int(total_rows),
        "vocab_size": int(len(vocab)),
        "max_tokens_cap": int(max_tokens_cap),
        "chunk_len": int(args.chunk_len),
        "train_chunks_per_sample": int(args.train_chunks_per_sample),
        "eval_chunks_per_sample": int(args.eval_chunks_per_sample),
        "mask_locations": bool(args.mask_locations),
        "mask_prob": float(args.mask_prob),
        "shards": shard_names,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Saved token-map cache: %s | rows=%s shards=%s vocab=%s max_tokens_cap=%s",
        cache_root,
        total_rows,
        len(shard_names),
        len(vocab),
        max_tokens_cap,
    )
    return _load_sharded_token_cache(cache_root)


def _build_chunk_starts(
    *,
    length: int,
    chunk_len: int,
    chunks_per_sample: int,
    mode: str,
    sample_index: int,
    seed: int,
) -> np.ndarray:
    chunk_len = int(chunk_len)
    chunks_per_sample = max(1, int(chunks_per_sample))
    usable_len = max(0, int(length))
    max_start = max(0, usable_len - chunk_len)
    if max_start <= 0:
        return np.zeros((chunks_per_sample,), dtype=np.int32)
    if mode == "eval":
        if chunks_per_sample == 1:
            return np.asarray([max_start // 2], dtype=np.int32)
        starts = np.linspace(0, max_start, num=chunks_per_sample)
        return np.clip(np.rint(starts), 0, max_start).astype(np.int32)

    rng_seed = int((sample_index * 1103515245 + seed * 12345) % (2**32))
    rng = np.random.default_rng(rng_seed)
    if chunks_per_sample == 1:
        return np.asarray([int(rng.integers(0, max_start + 1))], dtype=np.int32)
    base = np.linspace(0, max_start, num=chunks_per_sample)
    jitter = max(1, int(max_start // max(4, chunks_per_sample * 2)))
    starts = []
    for b in base:
        offset = int(rng.integers(-jitter, jitter + 1))
        starts.append(int(np.clip(int(round(float(b))) + offset, 0, max_start)))
    return np.asarray(starts, dtype=np.int32)


class _TokenMapChunkDataset(Dataset):
    def __init__(
        self,
        *,
        token_ids: "torch.Tensor",
        token_lens: "torch.Tensor",
        labels: "torch.Tensor",
        indices: np.ndarray,
        chunk_len: int,
        chunks_per_sample: int,
        mode: str,
        seed: int,
        include_sample_id: bool,
    ) -> None:
        self.token_ids = token_ids
        self.token_lens = token_lens
        self.labels = labels
        self.mode = str(mode)
        self.chunk_len = int(chunk_len)
        self.include_sample_id = bool(include_sample_id)
        sample_indices = np.asarray(indices, dtype=np.int64)
        self.sample_indices = torch.as_tensor(sample_indices, dtype=torch.long)
        flat_src: List[int] = []
        flat_start: List[int] = []
        for src_idx in sample_indices:
            length = int(self.token_lens[int(src_idx)].item())
            starts = _build_chunk_starts(
                length=length,
                chunk_len=self.chunk_len,
                chunks_per_sample=int(chunks_per_sample),
                mode=self.mode,
                sample_index=int(src_idx),
                seed=int(seed),
            )
            for s in starts:
                flat_src.append(int(src_idx))
                flat_start.append(int(s))
        self.flat_src = torch.as_tensor(np.asarray(flat_src, dtype=np.int64), dtype=torch.long)
        self.flat_start = torch.as_tensor(np.asarray(flat_start, dtype=np.int32), dtype=torch.int32)
        self.item_labels_np = self.labels.index_select(0, self.flat_src).cpu().numpy().astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.flat_src.numel())

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        src_idx = int(self.flat_src[idx].item())
        start = int(self.flat_start[idx].item())
        end = start + self.chunk_len
        x = self.token_ids[src_idx, start:end].to(dtype=torch.long)
        out = {
            "token_ids": x,
            "label": self.labels[src_idx].to(dtype=torch.long),
        }
        if self.include_sample_id:
            out["sample_id"] = torch.tensor(src_idx, dtype=torch.long)
        return out


def _amp_is_enabled(args: argparse.Namespace, device: "torch.device") -> bool:
    mode = str(args.amp).lower()
    if mode == "off":
        return False
    if device.type != "cuda":
        return False
    return mode in {"auto", "on"}


def _maybe_apply_fast_mode(args: argparse.Namespace) -> None:
    if not bool(args.fast_mode):
        return
    if args.lr_schedule == "none":
        args.lr_schedule = "onecycle"
    if args.early_stopping_patience <= 0:
        args.early_stopping_patience = 3
    if args.num_workers <= 0:
        cpu_count = os.cpu_count() or 4
        args.num_workers = max(1, min(8, cpu_count // 2))
    if args.eval_every_epoch is False:
        args.eval_every_epoch = True
    LOGGER.info(
        "fast_mode enabled: lr_schedule=%s early_stopping_patience=%s num_workers=%s eval_every_epoch=%s",
        args.lr_schedule,
        args.early_stopping_patience,
        args.num_workers,
        args.eval_every_epoch,
    )


def _make_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
    device: "torch.device",
    sampler: Optional[object] = None,
) -> DataLoader:
    if DataLoader is None:
        raise RuntimeError("PyTorch DataLoader is unavailable.")
    num_workers = max(0, int(args.num_workers))
    pin_memory = bool(args.pin_memory) and device.type == "cuda"
    kwargs: Dict[str, object] = {
        "batch_size": int(batch_size),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    else:
        kwargs["shuffle"] = bool(shuffle)
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return DataLoader(dataset, **kwargs)


def _cuda_debug_state(device: "torch.device") -> str:
    if torch is None:
        return "cuda_state=unavailable(torch)"
    if device.type != "cuda":
        return "cuda_state=cpu"
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        alloc_mb = torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0)
        reserved_mb = torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0)
        peak_alloc_mb = torch.cuda.max_memory_allocated(idx) / (1024.0 * 1024.0)
        peak_reserved_mb = torch.cuda.max_memory_reserved(idx) / (1024.0 * 1024.0)
        return (
            f"cuda_mem_mb alloc={alloc_mb:.1f} reserved={reserved_mb:.1f} "
            f"peak_alloc={peak_alloc_mb:.1f} peak_reserved={peak_reserved_mb:.1f}"
        )
    except Exception as exc:  # pragma: no cover - defensive logging path
        return f"cuda_state_error={exc!r}"


def _startup_trace_enabled(args: argparse.Namespace, *, batch_idx: int) -> bool:
    return int(args.startup_trace_steps) > 0 and int(batch_idx) <= int(args.startup_trace_steps)


def _maybe_cuda_sync_for_trace(args: argparse.Namespace, device: "torch.device") -> None:
    if torch is None:
        return
    if not bool(args.startup_trace_sync):
        return
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)


def _build_mpsa_artifact(
    *,
    model_state_dict: Dict[str, "torch.Tensor"],
    vocab: Dict[str, int],
    label_to_idx: Dict[str, int],
    idx_to_label: List[str],
    config: Dict[str, object],
    metrics: Dict[str, Dict[str, float]],
    training_state: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model_type": MODEL_TYPE_MPSA_TOKEN,
        "vectorizer": None,
        "model": None,
        "model_state_dict": model_state_dict,
        "vocab": vocab,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "config": config,
        "metrics": metrics,
        "training_state": training_state,
    }


def _evaluate_mpsa_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: "torch.device",
    criterion: nn.Module,
    amp_enabled: bool,
    chunk_agg: str = "mean",
) -> Dict[str, float]:
    model.eval()
    sum_logits: Dict[int, torch.Tensor] = {}
    max_logits: Dict[int, torch.Tensor] = {}
    logsumexp_logits: Dict[int, torch.Tensor] = {}
    chunk_counts: Dict[int, int] = {}
    sample_labels: Dict[int, int] = {}
    with torch.no_grad():
        for batch in data_loader:
            x = batch["token_ids"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            sample_id = batch.get("sample_id", None)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                out = model(x)
                logits = out["accent"] if isinstance(out, dict) else out
            logits_cpu = logits.detach().float().cpu()
            y_cpu = y.detach().cpu()
            if sample_id is None:
                sample_id_cpu = torch.arange(logits_cpu.shape[0], dtype=torch.long)
            else:
                sample_id_cpu = sample_id.detach().cpu().to(dtype=torch.long)

            for i in range(int(logits_cpu.shape[0])):
                sid = int(sample_id_cpu[i].item())
                yi = int(y_cpu[i].item())
                sample_labels[sid] = yi
                l_i = logits_cpu[i]
                if sid not in chunk_counts:
                    sum_logits[sid] = l_i.clone()
                    max_logits[sid] = l_i.clone()
                    logsumexp_logits[sid] = l_i.clone()
                    chunk_counts[sid] = 1
                else:
                    sum_logits[sid] += l_i
                    max_logits[sid] = torch.maximum(max_logits[sid], l_i)
                    logsumexp_logits[sid] = torch.logaddexp(logsumexp_logits[sid], l_i)
                    chunk_counts[sid] += 1

    total_items = int(len(sample_labels))
    if total_items == 0:
        return {"n": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "log_loss": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

    sample_ids = sorted(sample_labels.keys())
    labels_eval = torch.as_tensor([sample_labels[sid] for sid in sample_ids], dtype=torch.long)
    if chunk_agg == "max":
        logits_eval = torch.stack([max_logits[sid] for sid in sample_ids], dim=0)
    elif chunk_agg == "logsumexp":
        logits_eval = torch.stack([logsumexp_logits[sid] for sid in sample_ids], dim=0)
    else:
        logits_eval = torch.stack([sum_logits[sid] / max(1, chunk_counts[sid]) for sid in sample_ids], dim=0)

    crit_weight = getattr(criterion, "weight", None)
    if isinstance(crit_weight, torch.Tensor):
        crit_weight = crit_weight.detach().cpu()
    loss = F.cross_entropy(logits_eval, labels_eval, weight=crit_weight)
    pred = torch.argmax(logits_eval, dim=1)
    total_correct = int((pred == labels_eval).sum().item())
    k = min(3, int(logits_eval.shape[1]))
    topk = torch.topk(logits_eval, k=k, dim=1).indices
    total_top3 = int((topk == labels_eval.unsqueeze(1)).any(dim=1).sum().item())

    y_np = labels_eval.numpy().astype(np.int64, copy=False)
    pred_np = pred.numpy().astype(np.int64, copy=False)
    n_classes = int(logits_eval.shape[1])
    confusion = np.bincount(y_np * n_classes + pred_np, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    nonzero = support > 0
    macro_f1 = float(np.mean(f1[nonzero])) if np.any(nonzero) else 0.0
    weighted_f1 = float(np.sum(f1 * support) / max(float(np.sum(support)), 1.0))

    return {
        "n": float(total_items),
        "accuracy": float(total_correct / total_items),
        "top3_accuracy": float(total_top3 / total_items),
        "log_loss": float(loss.detach().cpu().item()),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def _main_mpsa(args: argparse.Namespace) -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for --trainer mpsa.")
    if int(args.min_label_examples) < 500:
        LOGGER.warning(
            "min_label_examples=%s is below required floor. Overriding to 500 for this run.",
            int(args.min_label_examples),
        )
        args.min_label_examples = 500
    if args.input_repr != "token_map":
        LOGGER.warning("--trainer mpsa requires --input_repr token_map; overriding input_repr.")
        args.input_repr = "token_map"
    if int(args.chunk_len) <= 0:
        raise ValueError("--chunk_len must be >= 1")
    if int(args.max_tokens_cap) < int(args.chunk_len):
        raise ValueError("--max_tokens_cap must be >= --chunk_len")
    if int(args.train_chunks_per_sample) <= 0:
        raise ValueError("--train_chunks_per_sample must be >= 1")
    if int(args.eval_chunks_per_sample) <= 0:
        raise ValueError("--eval_chunks_per_sample must be >= 1")
    _maybe_apply_fast_mode(args)
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be >= 1")
    LOGGER.info(
        "[mpsa-setup] begin | startup_trace_steps=%s startup_trace_sync=%s",
        args.startup_trace_steps,
        bool(args.startup_trace_sync),
    )

    counts = _pass_label_counts(args)
    keep_labels = keep_labels_from_counts(
        counts=counts,
        min_label_examples=args.min_label_examples,
        top_k_labels=args.top_k_labels,
    )
    if len(keep_labels) < 2:
        raise ValueError("Need at least 2 labels after filtering.")
    kept_rows = int(sum(int(counts.get(lbl, 0)) for lbl in keep_labels))
    total_rows = int(sum(int(v) for v in counts.values()))
    dropped_rows = max(0, total_rows - kept_rows)
    dropped_classes = int(len(counts) - len(keep_labels))
    label_to_idx = {label: i for i, label in enumerate(keep_labels)}
    idx_to_label = list(keep_labels)
    LOGGER.info(
        "[mpsa-setup] labels ready | num_labels=%s kept_rows=%s dropped_rows=%s dropped_classes=%s min_examples_per_class=%s",
        len(keep_labels),
        kept_rows,
        dropped_rows,
        dropped_classes,
        int(args.min_label_examples),
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    LOGGER.info("[mpsa-setup] device resolved | device=%s", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        LOGGER.info(
            "[mpsa-setup] cuda backend | cudnn_benchmark=%s tf32=%s %s",
            bool(args.cudnn_benchmark),
            bool(torch.backends.cuda.matmul.allow_tf32),
            _cuda_debug_state(device),
        )
    if args.num_workers == 0:
        suggested_workers = max(1, min(8, (os.cpu_count() or 4) // 2))
        LOGGER.info(
            "DataLoader is using num_workers=0. For better throughput, try --num_workers %s --pin_memory --persistent_workers.",
            suggested_workers,
        )

    text_transform = _build_text_masker(args)
    if text_transform is None:
        LOGGER.info("[mpsa-setup] location masking disabled.")
    else:
        LOGGER.info(
            "[mpsa-setup] location masking enabled | mask_prob=%.3f location_list_path=%s",
            float(args.mask_prob),
            str(args.location_list_path) if args.location_list_path is not None else "",
        )

    token_cache = _load_or_build_token_cache(
        args=args,
        keep_labels=keep_labels,
        label_to_idx=label_to_idx,
        text_transform=text_transform,
    )
    token_ids_t = token_cache["token_ids"]
    token_lens_t = token_cache["token_lens"]
    labels_t = token_cache["labels"]
    splits_t = token_cache["splits"]
    vocab = token_cache["vocab"]
    cache_meta = token_cache.get("meta", {})
    if (
        not isinstance(token_ids_t, torch.Tensor)
        or not isinstance(token_lens_t, torch.Tensor)
        or not isinstance(labels_t, torch.Tensor)
        or not isinstance(splits_t, torch.Tensor)
    ):
        raise ValueError("Token cache tensors are invalid.")
    if not isinstance(vocab, dict):
        raise ValueError("Token cache vocab is invalid.")
    if int(token_ids_t.shape[1]) < int(args.chunk_len):
        raise ValueError(
            f"Cached token width {int(token_ids_t.shape[1])} is smaller than chunk_len={int(args.chunk_len)}. Rebuild cache."
        )
    LOGGER.info(
        "[mpsa-setup] cache ready | token_rows=%s vocab_size=%s max_tokens_cap=%s chunk_len=%s train_chunks_per_sample=%s eval_chunks_per_sample=%s",
        int(token_ids_t.shape[0]) if isinstance(token_ids_t, torch.Tensor) else -1,
        len(vocab) if isinstance(vocab, dict) else -1,
        int(token_ids_t.shape[1]) if isinstance(token_ids_t, torch.Tensor) else -1,
        int(args.chunk_len),
        int(args.train_chunks_per_sample),
        int(args.eval_chunks_per_sample),
    )

    splits_np = splits_t.cpu().numpy()
    train_idx = np.where(splits_np == 0)[0].astype(np.int64)
    val_idx = np.where(splits_np == 1)[0].astype(np.int64)
    test_idx = np.where(splits_np == 2)[0].astype(np.int64)
    if train_idx.size == 0:
        raise ValueError("Token cache has no train split rows.")
    if val_idx.size == 0:
        fallback = train_idx[: max(1, min(512, train_idx.size // 10))]
        LOGGER.warning("Validation split empty after filtering; using train fallback of size=%s.", fallback.size)
        val_idx = fallback
    if test_idx.size == 0:
        LOGGER.warning("Test split empty after filtering; using validation split.")
        test_idx = val_idx
    LOGGER.info(
        "[mpsa-setup] split sizes | train=%s val=%s test=%s",
        int(train_idx.size),
        int(val_idx.size),
        int(test_idx.size),
    )

    train_ds = _TokenMapChunkDataset(
        token_ids=token_ids_t,
        token_lens=token_lens_t,
        labels=labels_t,
        indices=train_idx,
        chunk_len=int(args.chunk_len),
        chunks_per_sample=int(args.train_chunks_per_sample),
        mode="train",
        seed=int(args.seed),
        include_sample_id=False,
    )
    val_ds = _TokenMapChunkDataset(
        token_ids=token_ids_t,
        token_lens=token_lens_t,
        labels=labels_t,
        indices=val_idx,
        chunk_len=int(args.chunk_len),
        chunks_per_sample=int(args.eval_chunks_per_sample),
        mode="eval",
        seed=int(args.seed),
        include_sample_id=True,
    )
    test_ds = _TokenMapChunkDataset(
        token_ids=token_ids_t,
        token_lens=token_lens_t,
        labels=labels_t,
        indices=test_idx,
        chunk_len=int(args.chunk_len),
        chunks_per_sample=int(args.eval_chunks_per_sample),
        mode="eval",
        seed=int(args.seed),
        include_sample_id=True,
    )
    if len(train_ds) <= 0:
        raise ValueError("Chunked train dataset is empty after filtering/cache build.")

    train_labels_np = train_ds.item_labels_np
    train_class_counts = np.bincount(train_labels_np, minlength=len(keep_labels)).astype(np.float64, copy=False)
    nonzero_train = train_class_counts > 0
    if np.any(nonzero_train):
        LOGGER.info(
            "[mpsa-setup] train label distribution (chunk-level) | classes=%s nonzero=%s min=%s median=%s max=%s imbalance_ratio=%.2f",
            len(keep_labels),
            int(np.sum(nonzero_train)),
            int(np.min(train_class_counts[nonzero_train])),
            float(np.median(train_class_counts[nonzero_train])),
            int(np.max(train_class_counts[nonzero_train])),
            float(np.max(train_class_counts[nonzero_train]) / max(np.min(train_class_counts[nonzero_train]), 1.0)),
        )

    class_loss_weights_np = np.ones(len(keep_labels), dtype=np.float32)
    if bool(args.class_weighted_loss):
        if float(args.class_weight_power) < 0.0:
            raise ValueError("--class_weight_power must be >= 0.")
        class_loss_weights = np.ones(len(keep_labels), dtype=np.float64)
        class_loss_weights[nonzero_train] = 1.0 / np.power(train_class_counts[nonzero_train], float(args.class_weight_power))
        mean_nonzero = float(np.mean(class_loss_weights[nonzero_train])) if np.any(nonzero_train) else 1.0
        if mean_nonzero > 0:
            class_loss_weights = class_loss_weights / mean_nonzero
        if float(args.class_weight_max) > 0.0:
            class_loss_weights = np.minimum(class_loss_weights, float(args.class_weight_max))
        class_loss_weights_np = class_loss_weights.astype(np.float32, copy=False)
        LOGGER.info(
            "[mpsa-setup] class-weighted loss enabled | power=%s cap=%s weight_range=[%.3f, %.3f]",
            float(args.class_weight_power),
            float(args.class_weight_max),
            float(np.min(class_loss_weights_np)),
            float(np.max(class_loss_weights_np)),
        )

    train_sampler: Optional[object] = None
    if bool(args.weighted_sampling):
        if WeightedRandomSampler is None:
            raise RuntimeError("WeightedRandomSampler is unavailable (PyTorch install issue).")
        if float(args.sampling_weight_power) < 0.0:
            raise ValueError("--sampling_weight_power must be >= 0.")
        sample_class_weights = np.ones(len(keep_labels), dtype=np.float64)
        sample_class_weights[nonzero_train] = 1.0 / np.power(train_class_counts[nonzero_train], float(args.sampling_weight_power))
        mean_sample_w = float(np.mean(sample_class_weights[nonzero_train])) if np.any(nonzero_train) else 1.0
        if mean_sample_w > 0:
            sample_class_weights = sample_class_weights / mean_sample_w
        per_sample_weights = sample_class_weights[train_labels_np].astype(np.float64, copy=False)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(per_sample_weights, dtype=torch.double),
            num_samples=int(len(train_ds)),
            replacement=bool(args.sampling_replacement),
        )
        LOGGER.info(
            "[mpsa-setup] weighted sampling enabled | power=%s replacement=%s sample_weight_range=[%.3f, %.3f]",
            float(args.sampling_weight_power),
            bool(args.sampling_replacement),
            float(np.min(per_sample_weights)),
            float(np.max(per_sample_weights)),
        )

    train_loader = _make_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        args=args,
        device=device,
        sampler=train_sampler,
    )
    val_loader = _make_loader(dataset=val_ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)
    test_loader = _make_loader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, args=args, device=device)
    LOGGER.info(
        "[mpsa-setup] dataloaders ready | batch_size=%s train_samples=%s val_samples=%s test_samples=%s train_steps=%s num_workers=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s weighted_sampling=%s",
        int(args.batch_size),
        int(len(train_ds)),
        int(len(val_ds)),
        int(len(test_ds)),
        int(len(train_loader)),
        int(args.num_workers),
        bool(args.pin_memory),
        bool(args.persistent_workers),
        int(args.prefetch_factor),
        bool(train_sampler is not None),
    )

    model = TextMPSADenseNetClassifier(
        num_classes=len(keep_labels),
        vocab_size=len(vocab),
        embed_dim=int(args.embed_dim),
        max_len=int(args.chunk_len),
        block_config=_parse_block_config(args.mpsa_block_config),
        growth_rate=int(args.mpsa_growth_rate),
        num_init_features=int(args.mpsa_num_init_features),
        bn_size=int(args.mpsa_bn_size),
        drop_rate=float(args.mpsa_drop_rate),
        embed_dropout=float(args.mpsa_embed_dropout),
        head_hidden_dim=int(args.head_hidden_dim),
        head_dropout=float(args.head_dropout),
    )
    model_ref: nn.Module = model
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    LOGGER.info(
        "[mpsa-setup] model ready | params=%s channels_last=%s %s",
        int(sum(p.numel() for p in model.parameters())),
        bool(args.channels_last),
        _cuda_debug_state(device),
    )
    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=str(args.compile_mode))
            model_ref = getattr(model, "_orig_mod", model)
            LOGGER.info("Enabled torch.compile (mode=%s).", args.compile_mode)
        except Exception as exc:  # pragma: no cover - compile support varies by platform
            LOGGER.warning("torch.compile failed; continuing without compile. Error: %s", exc)
            model_ref = model
    else:
        model_ref = model

    if not isinstance(model_ref, TextMPSADenseNetClassifier):
        raise RuntimeError("Unexpected model type after compile wrapping.")

    head_params = list(model_ref.head_parameters())
    embedding_params = list(model_ref.embedding.parameters())
    backbone_core_params = [p for name, p in model_ref.named_parameters() if name.startswith("backbone.")]
    backbone_params = embedding_params + backbone_core_params
    if not head_params:
        raise RuntimeError("Classifier head parameters are missing.")
    if not backbone_params:
        raise RuntimeError("Backbone parameters are missing.")

    head_lr = float(args.head_lr) if float(args.head_lr) > 0 else float(args.torch_lr)
    backbone_lr = float(args.backbone_lr) if float(args.backbone_lr) > 0 else float(max(1e-6, args.torch_lr * 0.3))
    param_groups = [
        {"params": head_params, "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]

    if args.torch_optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=float(args.torch_lr), weight_decay=args.torch_weight_decay)
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=float(args.torch_lr),
            momentum=args.torch_momentum,
            weight_decay=args.torch_weight_decay,
            nesterov=False,
        )
    criterion_weight = (
        torch.as_tensor(class_loss_weights_np, dtype=torch.float32, device=device)
        if bool(args.class_weighted_loss)
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    amp_enabled = _amp_is_enabled(args=args, device=device)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    else:  # pragma: no cover - fallback for older torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    LOGGER.info(
        "[mpsa-setup] optimizer ready | optimizer=%s amp_enabled=%s head_lr=%s backbone_lr=%s grad_accum_steps=%s",
        args.torch_optimizer,
        bool(amp_enabled),
        head_lr,
        backbone_lr,
        args.grad_accum_steps,
    )
    if int(args.log_every_batches) > 0:
        LOGGER.info("[mpsa-setup] batch progress logging enabled | every_n_batches=%s", int(args.log_every_batches))

    output_dir = args.output_dir
    history_jsonl_path = args.epoch_history_path if args.epoch_history_path is not None else (output_dir / "epoch_history.jsonl")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    epoch_history_rows: List[Dict[str, object]] = []
    best_val_macro_f1 = -1.0
    best_path = output_dir / "text_model_best.pt"
    autosave_path = output_dir / "text_model_autosave.joblib"
    last_autosave_time = time.monotonic()
    global_batches = 0
    global_train_updates = 0
    global_optimizer_steps = 0
    start_epoch = 1
    train_start = time.monotonic()
    max_train_seconds = max(0.0, float(args.max_train_minutes) * 60.0)
    stopped_early = False
    early_stopping_counter = 0
    best_early_stop_score = -float("inf")
    patience = int(args.early_stopping_patience)
    benchmark_times: List[float] = []
    benchmark_samples = 0

    if args.resume_checkpoint is not None:
        resume_artifact = joblib.load(args.resume_checkpoint)
        if not isinstance(resume_artifact, dict):
            raise ValueError(f"Resume checkpoint is not a valid artifact dict: {args.resume_checkpoint}")
        model_type = str(resume_artifact.get("model_type", ""))
        if model_type != MODEL_TYPE_MPSA_TOKEN:
            raise ValueError(
                f"Resume checkpoint model_type={model_type!r} is not compatible with --trainer mpsa ({MODEL_TYPE_MPSA_TOKEN!r})."
            )
        state_dict = resume_artifact.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Resume checkpoint missing model_state_dict: {args.resume_checkpoint}")
        incompatible = model_ref.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            LOGGER.warning(
                "Resume checkpoint had non-strict state mismatches | missing=%s unexpected=%s",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        training_state = resume_artifact.get("training_state", {})
        if isinstance(training_state, dict):
            saved_epoch = int(training_state.get("epoch", 0))
            epoch_complete = bool(training_state.get("epoch_complete", True))
            start_epoch = (saved_epoch + 1) if epoch_complete else max(1, saved_epoch)
            global_batches = int(training_state.get("global_batches", 0))
            global_optimizer_steps = int(training_state.get("global_optimizer_steps", 0))
            global_train_updates = int(training_state.get("global_train_updates", 0))
        LOGGER.info(
            "Resumed MPSA token-map training from %s | start_epoch=%s prev_batches=%s prev_optimizer_steps=%s",
            args.resume_checkpoint,
            start_epoch,
            global_batches,
            global_optimizer_steps,
        )
        if start_epoch > args.epochs:
            LOGGER.warning(
                "Resume checkpoint epoch=%s already >= requested epochs=%s, skipping additional training.",
                start_epoch - 1,
                args.epochs,
            )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, int(args.grad_accum_steps))))
    epochs_remaining = max(1, int(args.epochs) - int(start_epoch) + 1)
    scheduler: Optional[object]
    if args.lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[head_lr, backbone_lr],
            total_steps=max(1, steps_per_epoch * epochs_remaining),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=1000.0,
        )
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_remaining))
    else:
        scheduler = None

    freeze_backbone_epochs = max(0, int(args.freeze_backbone_epochs))
    freeze_embedding_epochs = max(0, int(args.freeze_embedding_epochs))

    def _apply_trainability_schedule(epoch_num: int) -> Tuple[float, bool]:
        for p in head_params:
            p.requires_grad = True
        if epoch_num <= freeze_backbone_epochs:
            backbone_frac = 0.0
        elif str(args.unfreeze_strategy) == "all":
            backbone_frac = 1.0
        else:
            stage = epoch_num - freeze_backbone_epochs
            if stage <= 1:
                backbone_frac = 0.33
            elif stage == 2:
                backbone_frac = 0.66
            else:
                backbone_frac = 1.0

        n_backbone = len(backbone_core_params)
        if backbone_frac <= 0.0:
            for p in backbone_core_params:
                p.requires_grad = False
        elif backbone_frac >= 1.0:
            for p in backbone_core_params:
                p.requires_grad = True
        else:
            n_trainable = max(1, int(round(n_backbone * backbone_frac)))
            cutoff = max(0, n_backbone - n_trainable)
            for idx, p in enumerate(backbone_core_params):
                p.requires_grad = idx >= cutoff

        embedding_trainable = backbone_frac > 0.0 and (epoch_num > freeze_embedding_epochs)
        for p in embedding_params:
            p.requires_grad = bool(embedding_trainable)
        return backbone_frac, bool(embedding_trainable)

    LOGGER.info("[mpsa-setup] entering training loop")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train(True)
        backbone_frac, embedding_trainable = _apply_trainability_schedule(epoch)
        LOGGER.info(
            "[mpsa-setup] epoch=%s trainability | backbone_fraction=%.2f embedding_trainable=%s strategy=%s",
            epoch,
            backbone_frac,
            embedding_trainable,
            str(args.unfreeze_strategy),
        )

        total_loss = 0.0
        total_items = 0
        total_correct = 0
        total_top3 = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_start = time.perf_counter()
            stage = "batch_loaded"
            should_step = False
            try:
                if _startup_trace_enabled(args, batch_idx=batch_idx):
                    LOGGER.info(
                        "[mpsa-trace] epoch=%s batch=%s stage=%s %s",
                        epoch,
                        batch_idx,
                        stage,
                        _cuda_debug_state(device),
                    )
                stage = "copy_to_device"
                x = batch["token_ids"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                _maybe_cuda_sync_for_trace(args, device)
                if _startup_trace_enabled(args, batch_idx=batch_idx):
                    LOGGER.info(
                        "[mpsa-trace] epoch=%s batch=%s stage=%s x_shape=%s y_shape=%s %s",
                        epoch,
                        batch_idx,
                        stage,
                        tuple(x.shape),
                        tuple(y.shape),
                        _cuda_debug_state(device),
                    )
                stage = "forward"
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    out = model(x)
                    logits = out["accent"] if isinstance(out, dict) else out
                    raw_loss = criterion(logits, y)
                    loss = raw_loss / max(1, int(args.grad_accum_steps))
                _maybe_cuda_sync_for_trace(args, device)
                if _startup_trace_enabled(args, batch_idx=batch_idx):
                    LOGGER.info(
                        "[mpsa-trace] epoch=%s batch=%s stage=%s loss=%.5f %s",
                        epoch,
                        batch_idx,
                        stage,
                        float(raw_loss.detach().cpu().item()),
                        _cuda_debug_state(device),
                    )

                stage = "backward"
                scaler.scale(loss).backward()
                _maybe_cuda_sync_for_trace(args, device)
                should_step = (batch_idx % max(1, int(args.grad_accum_steps)) == 0) or (batch_idx == len(train_loader))
                if should_step:
                    stage = "optimizer_step"
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_optimizer_steps += 1
                    if scheduler is not None and str(args.lr_schedule) == "onecycle":
                        scheduler.step()
                    _maybe_cuda_sync_for_trace(args, device)
                global_batches += 1
                global_train_updates += int(y.size(0))

                batch_size = int(y.size(0))
                total_loss += float(raw_loss.detach().cpu().item()) * batch_size
                total_items += batch_size
                pred = torch.argmax(logits, dim=1)
                total_correct += int((pred == y).sum().item())
                k = min(3, int(logits.shape[1]))
                topk = torch.topk(logits, k=k, dim=1).indices
                total_top3 += int((topk == y.unsqueeze(1)).any(dim=1).sum().item())

                log_every = int(args.log_every_batches)
                if log_every > 0 and (batch_idx % log_every == 0):
                    LOGGER.info(
                        "[mpsa-batch] epoch=%s batch=%s/%s loss=%.5f lr=%.6g global_batches=%s batch_ms=%.1f samples_per_sec=%.1f %s",
                        epoch,
                        batch_idx,
                        len(train_loader),
                        float(raw_loss.detach().cpu().item()),
                        float(optimizer.param_groups[0]["lr"]),
                        int(global_batches),
                        float((time.perf_counter() - batch_start) * 1000.0),
                        float(int(y.size(0)) / max(time.perf_counter() - batch_start, 1e-9)),
                        _cuda_debug_state(device),
                    )

                if args.benchmark_mode and should_step and global_batches > int(args.benchmark_warmup_steps):
                    if len(benchmark_times) < int(args.benchmark_steps):
                        elapsed = max(time.perf_counter() - batch_start, 1e-9)
                        benchmark_times.append(elapsed)
                        benchmark_samples += batch_size

                time_due = (time.monotonic() - last_autosave_time) >= (args.autosave_every_minutes * 60.0)
                batch_due = args.autosave_every_batches > 0 and (global_batches % args.autosave_every_batches == 0)
                if should_step and (time_due or batch_due):
                    stage = "autosave"
                    state_cpu = {k: v.detach().cpu() for k, v in model_ref.state_dict().items()}
                    artifact = _build_mpsa_artifact(
                        model_state_dict=state_cpu,
                        vocab=vocab,
                        label_to_idx=label_to_idx,
                        idx_to_label=idx_to_label,
                        config={},
                        metrics={"val": {}, "test": {}},
                        training_state={
                            "stage": "training",
                            "epoch": epoch,
                            "epoch_complete": False,
                            "global_batches": global_batches,
                            "global_optimizer_steps": global_optimizer_steps,
                            "global_train_updates": global_train_updates,
                            "backbone_fraction": float(backbone_frac),
                            "timestamp_unix": time.time(),
                        },
                    )
                    _atomic_joblib_dump(artifact, autosave_path)
                    last_autosave_time = time.monotonic()
            except Exception:
                LOGGER.exception(
                    "[mpsa-trace] crash | epoch=%s batch=%s stage=%s should_step=%s %s",
                    epoch,
                    batch_idx,
                    stage,
                    should_step,
                    _cuda_debug_state(device),
                )
                raise

            if args.smoke_steps > 0 and global_optimizer_steps >= int(args.smoke_steps):
                stopped_early = True
                break
            if max_train_seconds > 0 and (time.monotonic() - train_start) >= max_train_seconds:
                stopped_early = True
                break
        if scheduler is not None and str(args.lr_schedule) == "cosine":
            scheduler.step()

        train_loss = float(total_loss / max(1, total_items))
        train_acc = float(total_correct / max(1, total_items))
        train_top3 = float(total_top3 / max(1, total_items))
        epoch_seconds = float(max(time.monotonic() - epoch_start, 1e-9))
        row: Dict[str, object] = {
            "epoch": int(epoch),
            "run_id": run_id,
            "timestamp_unix": float(time.time()),
            "trainer": "mpsa",
            "seen_rows": int(total_items),
            "train_updates": int(total_items),
            "global_batches": int(global_batches),
            "global_train_updates": int(global_train_updates),
            "global_optimizer_steps": int(global_optimizer_steps),
            "epoch_seconds": epoch_seconds,
            "train_samples_per_sec": float(total_items / epoch_seconds),
            "train_loss_mean": train_loss,
            "train_loss_min": train_loss,
            "train_loss_max": train_loss,
            "train_accuracy": train_acc,
            "train_top3_accuracy": train_top3,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        should_run_val = bool(args.eval_every_epoch) or patience > 0
        if should_run_val:
            val_metrics = _evaluate_mpsa_loader(
                model=model,
                data_loader=val_loader,
                device=device,
                criterion=criterion,
                amp_enabled=amp_enabled,
                chunk_agg=str(args.eval_chunk_agg),
            )
            row["val_n"] = float(val_metrics["n"])
            row["val_accuracy"] = float(val_metrics["accuracy"])
            row["val_top3_accuracy"] = float(val_metrics["top3_accuracy"])
            row["val_log_loss"] = float(val_metrics["log_loss"])
            row["val_macro_f1"] = float(val_metrics["macro_f1"])
            row["val_weighted_f1"] = float(val_metrics["weighted_f1"])

            if float(val_metrics["macro_f1"]) > best_val_macro_f1:
                best_val_macro_f1 = float(val_metrics["macro_f1"])
                state_cpu = {k: v.detach().cpu() for k, v in model_ref.state_dict().items()}
                torch.save({"model_state_dict": state_cpu}, best_path)
                LOGGER.info("Saved new best token-map MPSA weights: %s", best_path)

            if patience > 0:
                early_score = float(val_metrics["macro_f1"])
                if early_score > (best_early_stop_score + float(args.early_stopping_min_delta)):
                    best_early_stop_score = early_score
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        LOGGER.info(
                            "Early stopping triggered at epoch=%s (patience=%s, metric=val_macro_f1).",
                            epoch,
                            patience,
                        )
                        stopped_early = True

        _append_history_jsonl(history_jsonl_path, row)
        epoch_history_rows.append(row)

        LOGGER.info(
            "Epoch %s/%s | train loss %.4f acc %.4f top3 %.4f%s",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            train_top3,
            (
                f" | val acc {float(row.get('val_accuracy', 0.0)):.4f} top3 {float(row.get('val_top3_accuracy', 0.0)):.4f} "
                f"macro_f1 {float(row.get('val_macro_f1', 0.0)):.4f} loss {float(row.get('val_log_loss', 0.0)):.4f}"
            )
            if should_run_val
            else "",
        )
        if stopped_early:
            break

    if best_path.exists():
        try:
            best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            best_payload = torch.load(best_path, map_location="cpu")
        state = best_payload.get("model_state_dict", {})
        if isinstance(state, dict):
            model_ref.load_state_dict(state, strict=False)
    final_state_cpu = {k: v.detach().cpu() for k, v in model_ref.state_dict().items()}

    criterion_eval = nn.CrossEntropyLoss()
    if args.skip_eval:
        val_metrics = {
            "n": 0.0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
            "log_loss": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
        test_metrics = {
            "n": 0.0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
            "log_loss": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }
    else:
        val_metrics = _evaluate_mpsa_loader(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion_eval,
            amp_enabled=amp_enabled,
            chunk_agg=str(args.eval_chunk_agg),
        )
        test_metrics = _evaluate_mpsa_loader(
            model=model,
            data_loader=test_loader,
            device=device,
            criterion=criterion_eval,
            amp_enabled=amp_enabled,
            chunk_agg=str(args.eval_chunk_agg),
        )

    config_for_artifact: Dict[str, object] = {
        "text_col": args.text_col,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "max_rows": args.max_rows,
        "chunksize": args.chunksize,
        "seed": args.seed,
        "split_hash": args.split_hash,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "min_text_chars": args.min_text_chars,
        "min_label_examples": args.min_label_examples,
        "top_k_labels": args.top_k_labels,
        "analyzer": args.analyzer,
        "ngram_range": [args.ngram_min, args.ngram_max],
        "n_features": args.n_features,
        "alpha": args.alpha,
        "trainer": args.trainer,
        "device": args.device,
        "torch_optimizer": args.torch_optimizer,
        "torch_momentum": args.torch_momentum,
        "torch_lr": args.torch_lr,
        "torch_weight_decay": args.torch_weight_decay,
        "geo_loss_mode": "none",
        "geo_loss_mix": 0.0,
        "geo_sigma_km": args.geo_sigma_km,
        "lat_col": args.lat_col,
        "lon_col": args.lon_col,
        "latlong_col": args.latlong_col,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "autosave_every_batches": args.autosave_every_batches,
        "autosave_every_minutes": args.autosave_every_minutes,
        "max_train_minutes": args.max_train_minutes,
        "skip_eval": bool(args.skip_eval),
        "eval_every_epoch": bool(args.eval_every_epoch),
        "epoch_history_path": str(args.epoch_history_path) if args.epoch_history_path is not None else "",
        "input_repr": args.input_repr,
        "cache_dir": str(_resolve_feature_cache_dir(args)),
        "cache_rows": int(cache_meta.get("n_rows", int(token_ids_t.shape[0])) if isinstance(cache_meta, dict) else int(token_ids_t.shape[0])),
        "cache_shard_size": int(args.cache_shard_size),
        "cache_rebuild": bool(args.cache_rebuild) or bool(getattr(args, "rebuild_cache", False)),
        "max_len": int(args.chunk_len),
        "max_tokens_cap": int(args.max_tokens_cap),
        "chunk_len": int(args.chunk_len),
        "train_chunks_per_sample": int(args.train_chunks_per_sample),
        "eval_chunks_per_sample": int(args.eval_chunks_per_sample),
        "eval_chunk_agg": str(args.eval_chunk_agg),
        "embed_dim": int(args.embed_dim),
        "max_vocab_size": int(args.max_vocab_size),
        "min_token_freq": int(args.min_token_freq),
        "mask_locations": bool(args.mask_locations),
        "location_list_path": str(args.location_list_path) if args.location_list_path is not None else "",
        "mask_prob": float(args.mask_prob),
        "mpsa_block_config": list(_parse_block_config(args.mpsa_block_config)),
        "mpsa_growth_rate": int(args.mpsa_growth_rate),
        "mpsa_num_init_features": int(args.mpsa_num_init_features),
        "mpsa_bn_size": int(args.mpsa_bn_size),
        "mpsa_drop_rate": float(args.mpsa_drop_rate),
        "mpsa_embed_dropout": float(args.mpsa_embed_dropout),
        "head_hidden_dim": int(args.head_hidden_dim),
        "head_dropout": float(args.head_dropout),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "amp": str(args.amp),
        "torch_compile": bool(args.torch_compile),
        "compile_mode": str(args.compile_mode),
        "channels_last": bool(args.channels_last),
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "grad_accum_steps": int(args.grad_accum_steps),
        "class_weighted_loss": bool(args.class_weighted_loss),
        "class_weight_power": float(args.class_weight_power),
        "class_weight_max": float(args.class_weight_max),
        "weighted_sampling": bool(args.weighted_sampling),
        "sampling_weight_power": float(args.sampling_weight_power),
        "sampling_replacement": bool(args.sampling_replacement),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "lr_schedule": str(args.lr_schedule),
        "freeze_embedding_epochs": int(args.freeze_embedding_epochs),
        "freeze_backbone_epochs": int(args.freeze_backbone_epochs),
        "unfreeze_strategy": str(args.unfreeze_strategy),
        "head_lr": float(head_lr),
        "backbone_lr": float(backbone_lr),
        "smoke_steps": int(args.smoke_steps),
        "benchmark_mode": bool(args.benchmark_mode),
        "benchmark_warmup_steps": int(args.benchmark_warmup_steps),
        "benchmark_steps": int(args.benchmark_steps),
        "log_every_batches": int(args.log_every_batches),
        "startup_trace_steps": int(args.startup_trace_steps),
        "startup_trace_sync": bool(args.startup_trace_sync),
        "fast_mode": bool(args.fast_mode),
    }
    if benchmark_times:
        mean_step = float(np.mean(benchmark_times))
        config_for_artifact["benchmark_step_ms"] = float(mean_step * 1000.0)
        config_for_artifact["benchmark_samples_per_sec"] = float(benchmark_samples / max(np.sum(benchmark_times), 1e-9))
        LOGGER.info(
            "Benchmark | mean_step=%.2f ms | approx_samples_per_sec=%.1f over %s measured steps",
            mean_step * 1000.0,
            float(config_for_artifact["benchmark_samples_per_sec"]),
            len(benchmark_times),
        )

    last_epoch_completed = int(epoch_history_rows[-1]["epoch"]) if epoch_history_rows else max(0, start_epoch - 1)
    artifact = _build_mpsa_artifact(
        model_state_dict=final_state_cpu,
        vocab=vocab,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        config=config_for_artifact,
        metrics={"val": val_metrics, "test": test_metrics},
        training_state={
            "stage": "complete",
            "epoch": int(last_epoch_completed),
            "global_batches": int(global_batches),
            "global_optimizer_steps": int(global_optimizer_steps),
            "global_train_updates": int(global_train_updates),
            "stopped_early": bool(stopped_early),
            "best_val_macro_f1": float(best_val_macro_f1),
            "timestamp_unix": time.time(),
        },
    )
    model_path = output_dir / "text_model.joblib"
    _atomic_joblib_dump(artifact, model_path)
    _save_history_table(output_dir, epoch_history_rows)

    with (output_dir / "metrics_text.json").open("w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    with (output_dir / "label_mapping_text.json").open("w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)
    LOGGER.info(
        "Saved token-map MPSA artifact: %s | val_acc=%.4f test_acc=%.4f",
        model_path,
        float(val_metrics.get("accuracy", 0.0)),
        float(test_metrics.get("accuracy", 0.0)),
    )


def _append_history_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _save_history_table(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "epoch_history.csv", index=False)
    with (output_dir / "epoch_history.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _atomic_joblib_dump(obj: object, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    joblib.dump(obj, tmp_path)
    tmp_path.replace(target_path)


def _load_resume_artifact(path: Path, expected_model_types: Optional[List[str]] = None) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Resume checkpoint is not a valid artifact dict: {path}")

    required_keys = ["vectorizer", "model", "label_to_idx", "idx_to_label"]
    missing = [k for k in required_keys if k not in artifact]
    if missing:
        raise ValueError(f"Resume checkpoint missing keys {missing}: {path}")

    model_type = str(artifact.get("model_type", ""))
    if expected_model_types is not None:
        if model_type not in expected_model_types:
            raise ValueError(f"Checkpoint model_type={model_type!r} not in expected={expected_model_types}.")
    elif model_type and model_type not in {MODEL_TYPE_SKLEARN, MODEL_TYPE_TORCH}:
        raise ValueError(f"Unsupported model_type={model_type} in {path}")
    return artifact


def _validate_resume_config(args: argparse.Namespace, checkpoint_config: Dict[str, object]) -> None:
    keys_to_match = [
        "text_col",
        "label_col",
        "group_col",
        "max_rows",
        "min_text_chars",
        "seed",
        "split_hash",
        "val_fraction",
        "test_fraction",
        "analyzer",
        "n_features",
        "trainer",
        "geo_loss_mode",
        "geo_loss_mix",
        "geo_sigma_km",
        "lat_col",
        "lon_col",
        "latlong_col",
    ]
    mismatches: List[str] = []
    for key in keys_to_match:
        if key not in checkpoint_config:
            continue
        checkpoint_value = checkpoint_config[key]
        args_value = getattr(args, key)
        if checkpoint_value != args_value:
            mismatches.append(f"{key}: checkpoint={checkpoint_value!r} current={args_value!r}")
    checkpoint_ngram = checkpoint_config.get("ngram_range")
    if checkpoint_ngram is not None:
        current_ngram = [args.ngram_min, args.ngram_max]
        if list(checkpoint_ngram) != current_ngram:
            mismatches.append(f"ngram_range: checkpoint={checkpoint_ngram!r} current={current_ngram!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Resume config mismatch detected. Re-run with the same data/split settings as the original run. "
            + mismatch_text
        )


def _resolve_torch_device(device_arg: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("PyTorch is required for --trainer torch.")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _csr_to_torch_sparse(X, device: "torch.device") -> "torch.Tensor":
    try:
        crow_indices = torch.as_tensor(X.indptr, dtype=torch.int64, device=device)
        col_indices = torch.as_tensor(X.indices, dtype=torch.int64, device=device)
        values = torch.as_tensor(X.data, dtype=torch.float32, device=device)
        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=X.shape,
            device=device,
        )
    except Exception:
        # Fallback for environments without robust CSR sparse ops.
        coo = X.tocoo()
        indices = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
        values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def _export_torch_linear_model(weight: "torch.Tensor", bias: "torch.Tensor", n_classes: int) -> TextLinearDecisionModel:
    weight_np = weight.detach().cpu().numpy().astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32)
    classes_np = np.arange(n_classes, dtype=np.int64)
    return TextLinearDecisionModel(weights=weight_np, bias=bias_np, classes_=classes_np)


def _build_artifact(
    *,
    vectorizer: HashingVectorizer,
    model: object,
    label_to_idx: Dict[str, int],
    idx_to_label: List[str],
    config: Dict[str, object],
    metrics: Dict[str, Dict[str, float]],
    training_state: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model_type": MODEL_TYPE_SKLEARN if isinstance(model, SGDClassifier) else MODEL_TYPE_TORCH,
        "vectorizer": vectorizer,
        "model": model,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "config": config,
        "metrics": metrics,
        "training_state": training_state,
    }


def _train_streaming(
    args: argparse.Namespace,
    keep_labels: List[str],
    label_to_idx: Dict[str, int],
    vectorizer: HashingVectorizer,
    model: SGDClassifier,
    output_dir: Path,
    config_for_artifact: Dict[str, object],
    on_epoch_end: Optional[Callable[[int, Dict[str, object], object], None]] = None,
) -> Dict[str, object]:
    classes = np.arange(len(keep_labels), dtype=np.int64)
    first_fit = not hasattr(model, "classes_")
    rng = np.random.default_rng(args.seed)
    autosave_path = output_dir / "text_model_autosave.joblib"
    last_autosave_time = time.monotonic()
    training_state_cfg = config_for_artifact.get("resume_training_state", {})
    if isinstance(training_state_cfg, dict):
        start_epoch = int(training_state_cfg.get("start_epoch", 1))
        global_batches = int(training_state_cfg.get("global_batches", 0))
        global_train_updates = int(training_state_cfg.get("global_train_updates", 0))
    else:
        start_epoch = 1
        global_batches = 0
        global_train_updates = 0
    start_epoch = max(1, start_epoch)
    last_completed_epoch = start_epoch - 1
    split_cache: Dict[str, str] = {}
    train_start_time = time.monotonic()
    max_train_seconds = max(0.0, float(args.max_train_minutes) * 60.0)
    stopped_early = False

    if start_epoch > args.epochs:
        LOGGER.warning(
            "Resume checkpoint is already at epoch %s and target epochs=%s; skipping additional training.",
            start_epoch - 1,
            args.epochs,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.monotonic()
        iterator = _iter_prepared_chunks(
            csv_path=args.csv_path,
            sep=args.sep,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            text_col=args.text_col,
            label_col=args.label_col,
            group_col=args.group_col,
            min_text_chars=args.min_text_chars,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            split_hash=args.split_hash,
            split_cache=split_cache,
        )
        seen = 0
        trained = 0

        def _fit_batch(batch_texts: np.ndarray, batch_labels: np.ndarray) -> None:
            nonlocal first_fit, trained, last_autosave_time, global_batches, global_train_updates
            nonlocal stopped_early
            if batch_texts.size == 0:
                return
            X = vectorizer.transform(batch_texts)
            y = batch_labels.astype(np.int64, copy=False)
            if first_fit:
                model.partial_fit(X, y, classes=classes)
                first_fit = False
            else:
                model.partial_fit(X, y)
            trained += int(batch_texts.size)
            global_train_updates += int(batch_texts.size)
            global_batches += 1

            time_due = (time.monotonic() - last_autosave_time) >= (args.autosave_every_minutes * 60.0)
            batch_due = args.autosave_every_batches > 0 and (global_batches % args.autosave_every_batches == 0)
            if not first_fit and (time_due or batch_due):
                artifact = _build_artifact(
                    vectorizer=vectorizer,
                    model=model,
                    label_to_idx=label_to_idx,
                    idx_to_label=keep_labels,
                    config=config_for_artifact,
                    metrics={"val": {}, "test": {}},
                    training_state={
                        "stage": "training",
                        "epoch": epoch,
                        "epoch_complete": False,
                        "global_batches": global_batches,
                        "global_train_updates": global_train_updates,
                        "timestamp_unix": time.time(),
                    },
                )
                _atomic_joblib_dump(artifact, autosave_path)
                last_autosave_time = time.monotonic()
                LOGGER.info(
                    "Autosaved checkpoint at epoch=%s batches=%s updates=%s -> %s",
                    epoch,
                    global_batches,
                    global_train_updates,
                    autosave_path,
                )
            if max_train_seconds > 0 and (time.monotonic() - train_start_time) >= max_train_seconds:
                stopped_early = True

        with tqdm(desc=f"Pass 2/3 - epoch {epoch}/{args.epochs}", unit="rows") as pbar:
            for splits, texts, labels in iterator:
                if stopped_early:
                    break
                chunk_size = int(len(texts))
                seen += chunk_size
                pbar.update(chunk_size)
                if chunk_size == 0:
                    continue
                train_mask = splits == "train"
                if not bool(np.any(train_mask)):
                    continue
                train_texts = texts[train_mask]
                train_labels = labels[train_mask]
                y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in train_labels), dtype=np.int64, count=len(train_labels))
                valid_mask = y_idx >= 0
                if not bool(np.any(valid_mask)):
                    continue
                train_texts = train_texts[valid_mask]
                y_idx = y_idx[valid_mask]
                for start in range(0, len(train_texts), args.batch_size):
                    if stopped_early:
                        break
                    end = min(start + args.batch_size, len(train_texts))
                    _fit_batch(train_texts[start:end], y_idx[start:end])
        LOGGER.info("Epoch %s done: seen=%s train_updates=%s", epoch, seen, trained)

        if not first_fit:
            artifact = _build_artifact(
                vectorizer=vectorizer,
                model=model,
                label_to_idx=label_to_idx,
                idx_to_label=keep_labels,
                config=config_for_artifact,
                metrics={"val": {}, "test": {}},
                training_state={
                    "stage": "training",
                    "epoch": epoch,
                    "epoch_complete": True,
                    "global_batches": global_batches,
                    "global_train_updates": global_train_updates,
                    "timestamp_unix": time.time(),
                },
            )
            _atomic_joblib_dump(artifact, autosave_path)
            last_autosave_time = time.monotonic()
            LOGGER.info("Autosaved end-of-epoch checkpoint -> %s", autosave_path)
        last_completed_epoch = epoch

        # light online reshuffle signal by perturbing random state (no data reorder in stream)
        _ = rng.integers(0, 1_000_000)
        epoch_seconds = float(max(time.monotonic() - epoch_start_time, 1e-9))
        epoch_stats: Dict[str, object] = {
            "epoch": int(epoch),
            "seen_rows": int(seen),
            "train_updates": int(trained),
            "global_batches": int(global_batches),
            "global_train_updates": int(global_train_updates),
            "epoch_seconds": epoch_seconds,
            "train_samples_per_sec": float(trained / epoch_seconds),
            "trainer": "sklearn",
        }
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_stats, model)
        if stopped_early:
            LOGGER.info("Stopping early because max_train_minutes was reached.")
            break

    if first_fit:
        raise ValueError("No training samples found after filtering. Check label/text columns and thresholds.")
    return {
        "last_completed_epoch": int(last_completed_epoch),
        "global_batches": int(global_batches),
        "global_train_updates": int(global_train_updates),
        "stopped_early": bool(stopped_early),
    }


def _train_streaming_torch(
    args: argparse.Namespace,
    keep_labels: List[str],
    label_to_idx: Dict[str, int],
    vectorizer: HashingVectorizer,
    output_dir: Path,
    config_for_artifact: Dict[str, object],
    geo_target_matrix: Optional[np.ndarray],
    initial_weight: Optional[np.ndarray],
    initial_bias: Optional[np.ndarray],
    on_epoch_end: Optional[Callable[[int, Dict[str, object], object], None]] = None,
) -> Tuple[TextLinearDecisionModel, Dict[str, object]]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for --trainer torch.")

    device = _resolve_torch_device(args.device)
    n_classes = len(keep_labels)
    n_features = int(args.n_features)
    autosave_path = output_dir / "text_model_autosave.joblib"
    last_autosave_time = time.monotonic()

    training_state_cfg = config_for_artifact.get("resume_training_state", {})
    if isinstance(training_state_cfg, dict):
        start_epoch = int(training_state_cfg.get("start_epoch", 1))
        global_batches = int(training_state_cfg.get("global_batches", 0))
        global_train_updates = int(training_state_cfg.get("global_train_updates", 0))
    else:
        start_epoch = 1
        global_batches = 0
        global_train_updates = 0
    start_epoch = max(1, start_epoch)
    last_completed_epoch = start_epoch - 1
    split_cache: Dict[str, str] = {}
    train_start_time = time.monotonic()
    max_train_seconds = max(0.0, float(args.max_train_minutes) * 60.0)
    stopped_early = False

    weight = torch.nn.Parameter(torch.zeros((n_features, n_classes), dtype=torch.float32, device=device))
    bias = torch.nn.Parameter(torch.zeros((n_classes,), dtype=torch.float32, device=device))

    if initial_weight is not None:
        iw = np.asarray(initial_weight, dtype=np.float32)
        if iw.shape != (n_features, n_classes):
            raise ValueError(f"Resume weight shape mismatch: expected {(n_features, n_classes)}, got {iw.shape}")
        weight.data.copy_(torch.as_tensor(iw, dtype=torch.float32, device=device))
    if initial_bias is not None:
        ib = np.asarray(initial_bias, dtype=np.float32)
        if ib.shape != (n_classes,):
            raise ValueError(f"Resume bias shape mismatch: expected {(n_classes,)}, got {ib.shape}")
        bias.data.copy_(torch.as_tensor(ib, dtype=torch.float32, device=device))

    if args.torch_optimizer == "adamw":
        optimizer = torch.optim.AdamW([weight, bias], lr=args.torch_lr, weight_decay=args.torch_weight_decay)
    else:
        optimizer = torch.optim.SGD(
            [weight, bias],
            lr=args.torch_lr,
            momentum=args.torch_momentum,
            weight_decay=args.torch_weight_decay,
            nesterov=False,
        )

    geo_targets_t: Optional["torch.Tensor"] = None
    if geo_target_matrix is not None and args.geo_loss_mix > 0.0:
        geo_targets_t = torch.as_tensor(geo_target_matrix, dtype=torch.float32, device=device)

    LOGGER.info("Torch trainer device: %s", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    LOGGER.info(
        "Torch optimizer: %s | lr=%s | weight_decay=%s | momentum=%s",
        args.torch_optimizer,
        args.torch_lr,
        args.torch_weight_decay,
        args.torch_momentum if args.torch_optimizer == "sgd" else 0.0,
    )
    if geo_targets_t is not None:
        LOGGER.info("Torch loss: mixed hard CE + geo-soft CE (mix=%.3f)", float(args.geo_loss_mix))
    else:
        LOGGER.info("Torch loss: hard CE only")
    if start_epoch > args.epochs:
        LOGGER.warning(
            "Resume checkpoint is already at epoch %s and target epochs=%s; skipping additional training.",
            start_epoch - 1,
            args.epochs,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.monotonic()
        iterator = _iter_prepared_chunks(
            csv_path=args.csv_path,
            sep=args.sep,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            text_col=args.text_col,
            label_col=args.label_col,
            group_col=args.group_col,
            min_text_chars=args.min_text_chars,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            split_hash=args.split_hash,
            split_cache=split_cache,
        )
        seen = 0
        trained = 0
        epoch_losses: List[float] = []

        def _fit_batch(batch_texts: np.ndarray, batch_labels: np.ndarray) -> None:
            nonlocal trained, last_autosave_time, global_batches, global_train_updates
            nonlocal stopped_early
            if batch_texts.size == 0:
                return

            X_csr = vectorizer.transform(batch_texts)
            X_t = _csr_to_torch_sparse(X_csr, device=device)
            y_t = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = torch.sparse.mm(X_t, weight) + bias
            hard_loss = F.cross_entropy(logits, y_t)
            if geo_targets_t is not None:
                target_soft = geo_targets_t.index_select(0, y_t)
                log_probs = F.log_softmax(logits, dim=1)
                soft_loss = -(target_soft * log_probs).sum(dim=1).mean()
                mix = float(args.geo_loss_mix)
                loss = (1.0 - mix) * hard_loss + mix * soft_loss
            else:
                loss = hard_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

            trained += int(batch_texts.size)
            global_train_updates += int(batch_texts.size)
            global_batches += 1

            time_due = (time.monotonic() - last_autosave_time) >= (args.autosave_every_minutes * 60.0)
            batch_due = args.autosave_every_batches > 0 and (global_batches % args.autosave_every_batches == 0)
            if time_due or batch_due:
                model_snapshot = _export_torch_linear_model(weight=weight, bias=bias, n_classes=n_classes)
                artifact = _build_artifact(
                    vectorizer=vectorizer,
                    model=model_snapshot,
                    label_to_idx=label_to_idx,
                    idx_to_label=keep_labels,
                    config=config_for_artifact,
                    metrics={"val": {}, "test": {}},
                    training_state={
                        "stage": "training",
                        "epoch": epoch,
                        "epoch_complete": False,
                        "global_batches": global_batches,
                        "global_train_updates": global_train_updates,
                        "timestamp_unix": time.time(),
                    },
                )
                _atomic_joblib_dump(artifact, autosave_path)
                last_autosave_time = time.monotonic()
                LOGGER.info(
                    "Autosaved checkpoint at epoch=%s batches=%s updates=%s loss=%.4f -> %s",
                    epoch,
                    global_batches,
                    global_train_updates,
                    float(loss.detach().cpu().item()),
                    autosave_path,
                )
            if max_train_seconds > 0 and (time.monotonic() - train_start_time) >= max_train_seconds:
                stopped_early = True

        with tqdm(desc=f"Pass 2/3 - epoch {epoch}/{args.epochs}", unit="rows") as pbar:
            for splits, texts, labels in iterator:
                if stopped_early:
                    break
                chunk_size = int(len(texts))
                seen += chunk_size
                pbar.update(chunk_size)
                if chunk_size == 0:
                    continue
                train_mask = splits == "train"
                if not bool(np.any(train_mask)):
                    continue
                train_texts = texts[train_mask]
                train_labels = labels[train_mask]
                y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in train_labels), dtype=np.int64, count=len(train_labels))
                valid_mask = y_idx >= 0
                if not bool(np.any(valid_mask)):
                    continue
                train_texts = train_texts[valid_mask]
                y_idx = y_idx[valid_mask]
                for start in range(0, len(train_texts), args.batch_size):
                    if stopped_early:
                        break
                    end = min(start + args.batch_size, len(train_texts))
                    _fit_batch(train_texts[start:end], y_idx[start:end])
        LOGGER.info("Epoch %s done: seen=%s train_updates=%s", epoch, seen, trained)

        model_snapshot = _export_torch_linear_model(weight=weight, bias=bias, n_classes=n_classes)
        artifact = _build_artifact(
            vectorizer=vectorizer,
            model=model_snapshot,
            label_to_idx=label_to_idx,
            idx_to_label=keep_labels,
            config=config_for_artifact,
            metrics={"val": {}, "test": {}},
            training_state={
                "stage": "training",
                "epoch": epoch,
                "epoch_complete": True,
                "global_batches": global_batches,
                "global_train_updates": global_train_updates,
                "timestamp_unix": time.time(),
            },
        )
        _atomic_joblib_dump(artifact, autosave_path)
        last_autosave_time = time.monotonic()
        LOGGER.info("Autosaved end-of-epoch checkpoint -> %s", autosave_path)
        last_completed_epoch = epoch
        epoch_seconds = float(max(time.monotonic() - epoch_start_time, 1e-9))
        loss_mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_min = float(np.min(epoch_losses)) if epoch_losses else 0.0
        loss_max = float(np.max(epoch_losses)) if epoch_losses else 0.0
        epoch_stats: Dict[str, object] = {
            "epoch": int(epoch),
            "seen_rows": int(seen),
            "train_updates": int(trained),
            "global_batches": int(global_batches),
            "global_train_updates": int(global_train_updates),
            "epoch_seconds": epoch_seconds,
            "train_samples_per_sec": float(trained / epoch_seconds),
            "train_loss_mean": loss_mean,
            "train_loss_min": loss_min,
            "train_loss_max": loss_max,
            "trainer": "torch",
        }
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_stats, model_snapshot)
        if stopped_early:
            LOGGER.info("Stopping early because max_train_minutes was reached.")
            break

    model_out = _export_torch_linear_model(weight=weight, bias=bias, n_classes=n_classes)
    progress = {
        "last_completed_epoch": int(last_completed_epoch),
        "global_batches": int(global_batches),
        "global_train_updates": int(global_train_updates),
        "stopped_early": bool(stopped_early),
    }
    return model_out, progress


def _evaluate(
    args: argparse.Namespace,
    keep_labels: List[str],
    label_to_idx: Dict[str, int],
    vectorizer: HashingVectorizer,
    model: object,
    split_name: str,
    split_cache: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    y_all: List[int] = []
    probs_all: List[np.ndarray] = []

    iterator = _iter_prepared_chunks(
        csv_path=args.csv_path,
        sep=args.sep,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        text_col=args.text_col,
        label_col=args.label_col,
        group_col=args.group_col,
        min_text_chars=args.min_text_chars,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_hash=args.split_hash,
        split_cache=split_cache,
    )
    with tqdm(desc=f"Pass 3/3 - eval {split_name}", unit="rows") as pbar:
        for splits, texts, labels in iterator:
            chunk_size = int(len(texts))
            pbar.update(chunk_size)
            if chunk_size == 0:
                continue
            mask = splits == split_name
            if not bool(np.any(mask)):
                continue
            eval_texts = texts[mask]
            eval_labels = labels[mask]
            y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in eval_labels), dtype=np.int64, count=len(eval_labels))
            valid_mask = y_idx >= 0
            if not bool(np.any(valid_mask)):
                continue
            eval_texts = eval_texts[valid_mask]
            y_idx = y_idx[valid_mask]
            for start in range(0, len(eval_texts), args.batch_size):
                end = min(start + args.batch_size, len(eval_texts))
                X = vectorizer.transform(eval_texts[start:end])
                scores = np.asarray(model.decision_function(X))
                if len(model.classes_) > 2 and scores.ndim == 1:
                    scores = scores.reshape(1, -1)
                probs = scores_to_probs(scores)
                y_all.extend(y_idx[start:end].tolist())
                probs_all.extend([p for p in probs])

    if not y_all:
        return {"n": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "log_loss": 0.0}

    y = np.asarray(y_all, dtype=np.int64)
    probs = np.asarray(probs_all, dtype=np.float64)
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y))
    top3 = topk_accuracy(probs, y, k=3)
    eps = 1e-12
    loss = float(-np.log(probs[np.arange(len(y)), y] + eps).mean())
    return {"n": float(len(y)), "accuracy": acc, "top3_accuracy": top3, "log_loss": loss}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_arg_parser().parse_args()

    if args.val_fraction + args.test_fraction >= 0.95:
        raise ValueError("val_fraction + test_fraction must be < 0.95")
    if int(args.min_label_examples) < 500:
        LOGGER.warning(
            "min_label_examples=%s is below required floor. Overriding to 500 for this run.",
            int(args.min_label_examples),
        )
        args.min_label_examples = 500
    if args.trainer in {"torch", "mpsa"} and torch is None:
        raise RuntimeError("PyTorch is not available but a torch-based trainer was requested.")
    if not (0.0 <= float(args.geo_loss_mix) <= 1.0):
        raise ValueError("--geo_loss_mix must be in [0, 1].")
    if args.geo_loss_mode != "none" and args.trainer != "torch":
        LOGGER.warning("geo_loss_mode is only supported for --trainer torch; disabling geo loss.")
        args.geo_loss_mode = "none"
    if args.eval_every_epoch and args.skip_eval:
        LOGGER.warning("--eval_every_epoch requested but --skip_eval is set; per-epoch validation will be skipped.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Scanning CSV: %s", args.csv_path)
    if args.trainer == "mpsa":
        _main_mpsa(args)
        return

    start_epoch = 1
    initial_global_batches = 0
    initial_global_train_updates = 0
    initial_weight: Optional[np.ndarray] = None
    initial_bias: Optional[np.ndarray] = None
    resumed = False

    config_for_artifact: Dict[str, object] = {
        "text_col": args.text_col,
        "label_col": args.label_col,
        "group_col": args.group_col,
        "max_rows": args.max_rows,
        "chunksize": args.chunksize,
        "seed": args.seed,
        "split_hash": args.split_hash,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "min_text_chars": args.min_text_chars,
        "min_label_examples": args.min_label_examples,
        "top_k_labels": args.top_k_labels,
        "analyzer": args.analyzer,
        "ngram_range": [args.ngram_min, args.ngram_max],
        "n_features": args.n_features,
        "alpha": args.alpha,
        "trainer": args.trainer,
        "device": args.device,
        "torch_optimizer": args.torch_optimizer,
        "torch_momentum": args.torch_momentum,
        "torch_lr": args.torch_lr,
        "torch_weight_decay": args.torch_weight_decay,
        "geo_loss_mode": args.geo_loss_mode,
        "geo_loss_mix": args.geo_loss_mix,
        "geo_sigma_km": args.geo_sigma_km,
        "lat_col": args.lat_col,
        "lon_col": args.lon_col,
        "latlong_col": args.latlong_col,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "autosave_every_batches": args.autosave_every_batches,
        "autosave_every_minutes": args.autosave_every_minutes,
        "max_train_minutes": args.max_train_minutes,
        "skip_eval": bool(args.skip_eval),
        "eval_every_epoch": bool(args.eval_every_epoch),
        "epoch_history_path": str(args.epoch_history_path) if args.epoch_history_path is not None else "",
    }

    if args.trainer == "sklearn":
        expected_model_types = [MODEL_TYPE_SKLEARN]
    else:
        expected_model_types = [MODEL_TYPE_TORCH]

    if args.resume_checkpoint is not None:
        resume_artifact = _load_resume_artifact(args.resume_checkpoint, expected_model_types=expected_model_types)
        checkpoint_config = resume_artifact.get("config", {})
        if isinstance(checkpoint_config, dict):
            _validate_resume_config(args, checkpoint_config)
            merged_config = dict(checkpoint_config)
            merged_config.update(config_for_artifact)
            config_for_artifact = merged_config

        raw_label_to_idx = resume_artifact["label_to_idx"]
        raw_idx_to_label = resume_artifact["idx_to_label"]
        if not isinstance(raw_label_to_idx, dict) or not isinstance(raw_idx_to_label, list):
            raise ValueError("Invalid label mappings in resume checkpoint.")
        label_to_idx = {str(k): int(v) for k, v in raw_label_to_idx.items()}
        idx_to_label = [str(v) for v in raw_idx_to_label]
        keep_labels = idx_to_label

        vectorizer = resume_artifact["vectorizer"]
        model = resume_artifact["model"]
        if not isinstance(vectorizer, HashingVectorizer):
            raise ValueError("Resume checkpoint vectorizer must be a HashingVectorizer.")
        if args.trainer == "sklearn":
            if not isinstance(model, SGDClassifier):
                raise ValueError("Resume checkpoint model must be an SGDClassifier for --trainer sklearn.")
        else:
            if not isinstance(model, TextLinearDecisionModel):
                raise ValueError("Resume checkpoint model must be TextLinearDecisionModel for --trainer torch.")
            initial_weight = np.asarray(model.weights, dtype=np.float32)
            initial_bias = np.asarray(model.bias, dtype=np.float32)

        training_state = resume_artifact.get("training_state", {})
        if isinstance(training_state, dict):
            saved_epoch = int(training_state.get("epoch", 1))
            epoch_complete = bool(training_state.get("epoch_complete", False))
            initial_global_batches = int(training_state.get("global_batches", 0))
            initial_global_train_updates = int(training_state.get("global_train_updates", 0))
            start_epoch = saved_epoch + 1 if epoch_complete else max(saved_epoch, 1)
            if saved_epoch > 0 and not epoch_complete:
                LOGGER.warning(
                    "Resuming from the start of epoch %s (mid-epoch checkpoints cannot resume exact row position).",
                    saved_epoch,
                )
        resumed = True
        LOGGER.info(
            "Resuming (%s) from checkpoint=%s | labels=%s | start_epoch=%s | prev_batches=%s",
            args.trainer,
            args.resume_checkpoint,
            len(keep_labels),
            start_epoch,
            initial_global_batches,
        )
    else:
        counts = _pass_label_counts(args)
        keep_labels = keep_labels_from_counts(
            counts=counts,
            min_label_examples=args.min_label_examples,
            top_k_labels=args.top_k_labels,
        )
        if len(keep_labels) < 2:
            raise ValueError("Need at least 2 labels after filtering.")

        label_to_idx = {label: i for i, label in enumerate(keep_labels)}
        idx_to_label = keep_labels
        LOGGER.info("Keeping %s labels for training.", len(keep_labels))
        vectorizer, sgd_model = _build_model_and_vectorizer(args)
        model = sgd_model

    geo_target_matrix: Optional[np.ndarray] = None
    if args.trainer == "torch" and args.geo_loss_mode == "centroid" and args.geo_loss_mix > 0.0:
        geo_target_matrix = _build_geo_target_matrix(args=args, keep_labels=keep_labels)
        if geo_target_matrix is None:
            LOGGER.warning("Geo loss requested but could not build centroid targets. Falling back to hard CE.")
            args.geo_loss_mode = "none"

    config_for_artifact["geo_loss_mode"] = args.geo_loss_mode
    config_for_artifact["resumed"] = resumed
    config_for_artifact["resume_checkpoint"] = str(args.resume_checkpoint) if args.resume_checkpoint is not None else ""
    config_for_artifact["resume_training_state"] = {
        "start_epoch": int(start_epoch),
        "global_batches": int(initial_global_batches),
        "global_train_updates": int(initial_global_train_updates),
    }
    history_jsonl_path = args.epoch_history_path if args.epoch_history_path is not None else (args.output_dir / "epoch_history.jsonl")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    epoch_history_rows: List[Dict[str, object]] = []
    eval_split_cache: Dict[str, str] = {}

    def _on_epoch_end(epoch: int, epoch_stats: Dict[str, object], epoch_model: object) -> None:
        row: Dict[str, object] = dict(epoch_stats)
        row["run_id"] = run_id
        row["timestamp_unix"] = float(time.time())
        if args.eval_every_epoch and not args.skip_eval:
            val_metrics_epoch = _evaluate(
                args=args,
                keep_labels=keep_labels,
                label_to_idx=label_to_idx,
                vectorizer=vectorizer,
                model=epoch_model,
                split_name="val",
                split_cache=eval_split_cache,
            )
            row["val_n"] = float(val_metrics_epoch["n"])
            row["val_accuracy"] = float(val_metrics_epoch["accuracy"])
            row["val_top3_accuracy"] = float(val_metrics_epoch["top3_accuracy"])
            row["val_log_loss"] = float(val_metrics_epoch["log_loss"])
            LOGGER.info(
                "Epoch %s val | n=%s acc=%.4f top3=%.4f loss=%.4f",
                epoch,
                int(val_metrics_epoch["n"]),
                val_metrics_epoch["accuracy"],
                val_metrics_epoch["top3_accuracy"],
                val_metrics_epoch["log_loss"],
            )
        epoch_history_rows.append(row)
        _append_history_jsonl(history_jsonl_path, row)

    if args.trainer == "sklearn":
        if not isinstance(model, SGDClassifier):
            raise ValueError("Expected SGDClassifier for sklearn trainer.")
        training_progress = _train_streaming(
            args=args,
            keep_labels=keep_labels,
            label_to_idx=label_to_idx,
            vectorizer=vectorizer,
            model=model,
            output_dir=args.output_dir,
            config_for_artifact=config_for_artifact,
            on_epoch_end=_on_epoch_end,
        )
    else:
        model, training_progress = _train_streaming_torch(
            args=args,
            keep_labels=keep_labels,
            label_to_idx=label_to_idx,
            vectorizer=vectorizer,
            output_dir=args.output_dir,
            config_for_artifact=config_for_artifact,
            geo_target_matrix=geo_target_matrix,
            initial_weight=initial_weight,
            initial_bias=initial_bias,
            on_epoch_end=_on_epoch_end,
        )

    _save_history_table(args.output_dir, epoch_history_rows)

    if args.skip_eval:
        LOGGER.info("Skipping pass-3 evaluation because --skip_eval is enabled.")
        val_metrics = {"n": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "log_loss": 0.0}
        test_metrics = {"n": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "log_loss": 0.0}
    else:
        eval_split_cache: Dict[str, str] = {}
        val_metrics = _evaluate(
            args=args,
            keep_labels=keep_labels,
            label_to_idx=label_to_idx,
            vectorizer=vectorizer,
            model=model,
            split_name="val",
            split_cache=eval_split_cache,
        )
        test_metrics = _evaluate(
            args=args,
            keep_labels=keep_labels,
            label_to_idx=label_to_idx,
            vectorizer=vectorizer,
            model=model,
            split_name="test",
            split_cache=eval_split_cache,
        )

    LOGGER.info(
        "Validation | n=%s acc=%.4f top3=%.4f loss=%.4f",
        int(val_metrics["n"]),
        val_metrics["accuracy"],
        val_metrics["top3_accuracy"],
        val_metrics["log_loss"],
    )
    LOGGER.info(
        "Test       | n=%s acc=%.4f top3=%.4f loss=%.4f",
        int(test_metrics["n"]),
        test_metrics["accuracy"],
        test_metrics["top3_accuracy"],
        test_metrics["log_loss"],
    )

    artifact = _build_artifact(
        vectorizer=vectorizer,
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        config=config_for_artifact,
        metrics={"val": val_metrics, "test": test_metrics},
        training_state={
            "stage": "complete",
            "epoch": int(training_progress["last_completed_epoch"]),
            "global_batches": int(training_progress["global_batches"]),
            "global_train_updates": int(training_progress["global_train_updates"]),
            "stopped_early": bool(training_progress.get("stopped_early", False)),
            "timestamp_unix": time.time(),
        },
    )
    model_path = args.output_dir / "text_model.joblib"
    _atomic_joblib_dump(artifact, model_path)

    with (args.output_dir / "metrics_text.json").open("w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    with (args.output_dir / "label_mapping_text.json").open("w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)

    LOGGER.info("Saved text model artifact: %s", model_path)


if __name__ == "__main__":
    main()
