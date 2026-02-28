from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from conase_geo.models.text_mpsa import TextMPSADenseNetClassifier
from conase_geo.text_utils import encode_text_to_token_ids, normalize_text, scores_to_probs, split_from_group

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


LOGGER = logging.getLogger(__name__)
MODEL_TYPE_MPSA_TOKEN = "text_mpsa_token_map"

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
_MID_SENTENCE_CAP_RE = re.compile(r"(?<!^)(?<![\.\!\?]\s)\b[A-Z][a-z]{2,}\b")
_ASR_BRACKET_RE = re.compile(r"\[(?:music|applause|laughter|noise|silence)[^\]]*\]", flags=re.IGNORECASE)
_ASR_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_ASR_FILLER_RE = re.compile(r"\b(?:um+|uh+|like|you know|kinda|sorta)\b", flags=re.IGNORECASE)


@dataclass
class SplitPlan:
    selected_split: str
    split_source: str
    split_column: str
    heldout_mapped_to: str


class RegexEntityMasker:
    def __init__(self, location_terms: Sequence[str]) -> None:
        self.location_terms = [t.strip() for t in location_terms if t and t.strip()]
        self.location_pattern: Optional[re.Pattern[str]] = None
        if self.location_terms:
            escaped = sorted((re.escape(t) for t in self.location_terms), key=len, reverse=True)
            self.location_pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

    def __call__(self, text: str) -> str:
        out = _URL_RE.sub("[ENT]", text)
        out = _HANDLE_RE.sub("[ENT]", out)
        out = _MID_SENTENCE_CAP_RE.sub("[ENT]", out)
        if self.location_pattern is not None:
            out = self.location_pattern.sub("[ENT]", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out


class SpacyEntityMasker:
    def __init__(self, location_terms: Sequence[str]) -> None:
        try:
            import spacy  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError("spaCy is not installed. Use --entity_mask_mode regex or install spaCy.") from exc
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "spaCy model en_core_web_sm is unavailable. Install it or use --entity_mask_mode regex."
            ) from exc
        self.location_terms = [t.strip() for t in location_terms if t and t.strip()]
        self.location_pattern: Optional[re.Pattern[str]] = None
        if self.location_terms:
            escaped = sorted((re.escape(t) for t in self.location_terms), key=len, reverse=True)
            self.location_pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)

    def __call__(self, text: str) -> str:
        out = _URL_RE.sub("[ENT]", text)
        out = _HANDLE_RE.sub("[ENT]", out)
        doc = self.nlp(out)
        spans: List[Tuple[int, int]] = []
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "FAC", "PERSON", "ORG", "NORP"}:
                spans.append((ent.start_char, ent.end_char))
        if spans:
            pieces: List[str] = []
            cursor = 0
            for start, end in sorted(spans):
                if start < cursor:
                    continue
                pieces.append(out[cursor:start])
                pieces.append("[ENT]")
                cursor = end
            pieces.append(out[cursor:])
            out = "".join(pieces)
        out = _MID_SENTENCE_CAP_RE.sub("[ENT]", out)
        if self.location_pattern is not None:
            out = self.location_pattern.sub("[ENT]", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out


class TextBatchPredictor:
    def __init__(
        self,
        *,
        artifact: Dict[str, Any],
        device: str,
        amp_enabled: bool,
        compile_model: bool,
        compile_mode: str,
    ) -> None:
        self.artifact = artifact
        self.model_type = str(artifact.get("model_type", ""))
        self.idx_to_label = [str(x) for x in artifact.get("idx_to_label", [])]
        self.n_classes = len(self.idx_to_label)
        if self.n_classes < 2:
            raise ValueError("Checkpoint must have at least 2 classes.")

        self.device_str = device
        self.amp_enabled = bool(amp_enabled)
        self.compile_model = bool(compile_model)
        self.compile_mode = str(compile_mode)

        self.vectorizer = artifact.get("vectorizer")
        self.linear_model = artifact.get("model")

        self.torch_device = None
        self.torch_model = None
        self.vocab: Optional[Dict[str, int]] = None
        self.max_len: int = 384

        if self.model_type == MODEL_TYPE_MPSA_TOKEN:
            if torch is None:
                raise RuntimeError("PyTorch is required for token-map MPSA evaluation.")
            self._init_mpsa_model()
        else:
            if self.vectorizer is None or self.linear_model is None:
                raise ValueError("Checkpoint does not contain a usable linear text model.")

    def _init_mpsa_model(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for token-map MPSA evaluation.")
        config = self.artifact.get("config", {})
        if not isinstance(config, dict):
            config = {}

        state_dict = self.artifact.get("model_state_dict")
        vocab = self.artifact.get("vocab")
        if not isinstance(state_dict, dict):
            raise ValueError("Token-map checkpoint is missing model_state_dict.")
        if not isinstance(vocab, dict):
            raise ValueError("Token-map checkpoint is missing vocab.")
        self.vocab = {str(k): int(v) for k, v in vocab.items()}

        block_cfg_raw = config.get("mpsa_block_config", [6, 12, 24, 16])
        if not isinstance(block_cfg_raw, Sequence) or len(block_cfg_raw) != 4:
            block_cfg_raw = [6, 12, 24, 16]
        block_cfg = tuple(int(v) for v in block_cfg_raw)

        self.max_len = int(config.get("chunk_len", config.get("max_len", 384)))
        embed_dim = int(config.get("embed_dim", 128))

        model = TextMPSADenseNetClassifier(
            num_classes=self.n_classes,
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            max_len=self.max_len,
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

        use_cuda = self.device_str == "cuda" and torch.cuda.is_available()
        self.torch_device = torch.device("cuda" if use_cuda else "cpu")
        if self.device_str == "cuda" and not use_cuda:
            LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")

        model = model.to(self.torch_device)
        model.eval()

        if self.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=self.compile_mode)
                LOGGER.info("Enabled torch.compile for evaluation (mode=%s)", self.compile_mode)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("torch.compile failed; using eager mode. error=%s", exc)
        self.torch_model = model

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        if self.model_type == MODEL_TYPE_MPSA_TOKEN:
            return self._predict_proba_mpsa(texts)
        return self._predict_proba_linear(texts)

    def _predict_proba_linear(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        scores = np.asarray(self.linear_model.decision_function(X))
        if self.n_classes > 2 and scores.ndim == 1:
            scores = scores.reshape(1, -1)
        probs = scores_to_probs(scores)
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"Checkpoint/model class mismatch: probs has {probs.shape[1]} classes but idx_to_label has {self.n_classes}."
            )
        return probs.astype(np.float32, copy=False)

    def _predict_proba_mpsa(self, texts: Sequence[str]) -> np.ndarray:
        if torch is None or self.torch_model is None or self.torch_device is None or self.vocab is None:
            raise RuntimeError("MPSA predictor is not initialized.")
        ids = np.stack(
            [encode_text_to_token_ids(text, vocab=self.vocab, max_len=self.max_len) for text in texts],
            axis=0,
        )
        x_cpu = torch.as_tensor(ids, dtype=torch.long)
        x = x_cpu.to(self.torch_device, non_blocking=(self.torch_device.type == "cuda"))
        with torch.no_grad():
            with torch.autocast(
                device_type=self.torch_device.type,
                dtype=torch.float16,
                enabled=(self.amp_enabled and self.torch_device.type == "cuda"),
            ):
                out = self.torch_model(x)
                logits = out["accent"] if isinstance(out, dict) else out
                probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy().astype(np.float32, copy=False)


class ReservoirSampler:
    def __init__(self, k: int, seed: int) -> None:
        self.k = max(0, int(k))
        self.rng = random.Random(int(seed))
        self.n_seen = 0
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        if self.k <= 0:
            return
        self.n_seen += 1
        if len(self.rows) < self.k:
            self.rows.append(row)
            return
        j = self.rng.randint(1, self.n_seen)
        if j <= self.k:
            self.rows[j - 1] = row


class RunningStats:
    def __init__(self, n_classes: int, thresholds: Sequence[float], ece_bins: int) -> None:
        self.n_classes = int(n_classes)
        self.thresholds = list(thresholds)
        self.ece_bins = int(ece_bins)

        self.n = 0
        self.topk_hits: Dict[int, int] = {1: 0, 3: 0, 5: 0}

        self.support = np.zeros(self.n_classes, dtype=np.int64)
        self.tp = np.zeros(self.n_classes, dtype=np.int64)
        self.fp = np.zeros(self.n_classes, dtype=np.int64)
        self.fn = np.zeros(self.n_classes, dtype=np.int64)
        self.top3_hits_by_class = np.zeros(self.n_classes, dtype=np.int64)
        self.top5_hits_by_class = np.zeros(self.n_classes, dtype=np.int64)

        self.nll_sum = 0.0
        self.brier_sum = 0.0

        self.conf_correct_sum = 0.0
        self.conf_incorrect_sum = 0.0
        self.n_correct = 0
        self.n_incorrect = 0
        self.overconfident_wrong = 0

        self.ece_count = np.zeros(self.ece_bins, dtype=np.int64)
        self.ece_conf_sum = np.zeros(self.ece_bins, dtype=np.float64)
        self.ece_correct_sum = np.zeros(self.ece_bins, dtype=np.float64)

        self.entropy_sum = 0.0
        self.entropy_hist_bins = 256
        self.entropy_hist = np.zeros(self.entropy_hist_bins, dtype=np.int64)

        self.coverage_count = np.zeros(len(self.thresholds), dtype=np.int64)
        self.coverage_correct = np.zeros(len(self.thresholds), dtype=np.int64)
        self.coverage_tp = np.zeros((len(self.thresholds), self.n_classes), dtype=np.int64)
        self.coverage_fp = np.zeros((len(self.thresholds), self.n_classes), dtype=np.int64)
        self.coverage_fn = np.zeros((len(self.thresholds), self.n_classes), dtype=np.int64)

        self.confusion: Optional[np.ndarray] = None

    def enable_confusion(self) -> None:
        if self.confusion is None:
            self.confusion = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)


class BucketAccumulator:
    def __init__(self, n_classes: int) -> None:
        self.n = 0
        self.top1 = 0
        self.top5 = 0
        self.tp = np.zeros(n_classes, dtype=np.int64)
        self.fp = np.zeros(n_classes, dtype=np.int64)
        self.fn = np.zeros(n_classes, dtype=np.int64)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming evaluation for text region classifier checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "heldout"], default="heldout")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile_mode", type=str, default="max-autotune")
    parser.add_argument("--topk", type=str, default="1,3,5")
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--save_pred_sample", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--entity_mask_eval", action="store_true")
    parser.add_argument("--entity_mask_mode", type=str, choices=["regex", "spacy"], default="regex")

    parser.add_argument("--slices", dest="slices", action="store_true")
    parser.add_argument("--no-slices", dest="slices", action="store_false")
    parser.set_defaults(slices=None)

    parser.add_argument("--calibration", dest="calibration", action="store_true")
    parser.add_argument("--no-calibration", dest="calibration", action="store_false")
    parser.set_defaults(calibration=None)

    parser.add_argument("--confusions", dest="confusions", action="store_true")
    parser.add_argument("--no-confusions", dest="confusions", action="store_false")
    parser.set_defaults(confusions=None)

    parser.add_argument("--coverage_curve", dest="coverage_curve", action="store_true")
    parser.add_argument("--no-coverage_curve", dest="coverage_curve", action="store_false")
    parser.set_defaults(coverage_curve=None)

    parser.add_argument("--per_class", dest="per_class", action="store_true")
    parser.add_argument("--no-per_class", dest="per_class", action="store_false")
    parser.set_defaults(per_class=None)

    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sep", type=str, default="")
    parser.add_argument("--text_col", type=str, default="")
    parser.add_argument("--label_col", type=str, default="")
    parser.add_argument("--group_col", type=str, default="")
    parser.add_argument("--split_col", type=str, default="")
    parser.add_argument("--id_col", type=str, default="video_id")
    parser.add_argument("--channel_col", type=str, default="channel_id")
    parser.add_argument("--chunksize", type=int, default=8000)
    parser.add_argument("--min_text_chars", type=int, default=-1)
    parser.add_argument("--val_fraction", type=float, default=-1.0)
    parser.add_argument("--test_fraction", type=float, default=-1.0)
    parser.add_argument("--split_hash", type=str, choices=["blake2", "crc32"], default="")
    parser.add_argument("--log_every_batches", type=int, default=100)
    return parser


def _parse_csv_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _parse_topk(raw: str, n_classes: int) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        k = int(t)
        if k <= 0:
            continue
        out.append(k)
    out.extend([1, 3, 5])
    uniq = sorted(set(min(int(n_classes), k) for k in out if k > 0))
    return uniq


def _resolve_args_with_checkpoint(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    if args.sep == "":
        args.sep = str(cfg.get("sep", "|")) if "sep" in cfg else "|"
    if args.text_col == "":
        args.text_col = str(cfg.get("text_col", "text"))
    if args.label_col == "":
        args.label_col = str(cfg.get("label_col", "state"))
    if args.group_col == "":
        args.group_col = str(cfg.get("group_col", "channel_id"))
    if args.split_hash == "":
        args.split_hash = str(cfg.get("split_hash", "blake2"))
    if args.min_text_chars < 0:
        args.min_text_chars = int(cfg.get("min_text_chars", 20))
    if args.val_fraction < 0:
        args.val_fraction = float(cfg.get("val_fraction", 0.1))
    if args.test_fraction < 0:
        args.test_fraction = float(cfg.get("test_fraction", 0.1))
    if args.csv_path is None:
        cfg_csv = str(cfg.get("csv_path", "")).strip()
        if cfg_csv:
            args.csv_path = Path(cfg_csv)
    return args


def _apply_fast_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(args.fast):
        if args.slices is None:
            args.slices = True
        if args.calibration is None:
            args.calibration = True
        if args.confusions is None:
            args.confusions = True
        if args.coverage_curve is None:
            args.coverage_curve = True
        if args.per_class is None:
            args.per_class = True
        return args

    if args.slices is None:
        args.slices = False
    if args.calibration is None:
        args.calibration = False
    if args.confusions is None:
        args.confusions = False
    if args.coverage_curve is None:
        args.coverage_curve = False
    if args.per_class is None:
        args.per_class = False
    args.entity_mask_eval = False
    args.save_pred_sample = min(int(args.save_pred_sample), 1000)
    args.log_every_batches = max(0, min(int(args.log_every_batches), 200))
    return args


def _read_header_columns(csv_path: Path, sep: str) -> List[str]:
    header = pd.read_csv(
        csv_path,
        sep=sep,
        dtype=str,
        nrows=0,
        keep_default_na=False,
        na_filter=False,
    )
    return [str(c).strip() for c in header.columns]


def _pick_split_column(args: argparse.Namespace, columns: Sequence[str]) -> str:
    cols = {c.lower(): c for c in columns}
    if args.split_col:
        key = args.split_col.strip().lower()
        if key in cols:
            return cols[key]
    for candidate in ["split", "set", "fold", "subset"]:
        if candidate in cols:
            return cols[candidate]
    return ""


def _normalize_split_token(raw: str) -> str:
    token = str(raw or "").strip().lower().replace("-", "_")
    mapping = {
        "dev": "val",
        "valid": "val",
        "validation": "val",
        "holdout": "heldout",
        "held_out": "heldout",
    }
    return mapping.get(token, token)


def _build_split_plan(
    *,
    requested_split: str,
    split_col: str,
    csv_path: Path,
    sep: str,
    text_col: str,
    label_col: str,
    group_col: str,
) -> SplitPlan:
    if split_col:
        seen_values: set[str] = set()
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            dtype=str,
            chunksize=4000,
            keep_default_na=False,
            na_filter=False,
            on_bad_lines="skip",
            usecols=[split_col],
        )
        for chunk in reader:
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if split_col not in chunk.columns:
                continue
            vals = chunk[split_col].astype(str)
            for v in vals.head(500).tolist():
                seen_values.add(_normalize_split_token(v))
            if len(seen_values) >= 8:
                break
        if requested_split in seen_values:
            return SplitPlan(
                selected_split=requested_split,
                split_source="column",
                split_column=split_col,
                heldout_mapped_to="",
            )
        if requested_split == "heldout" and "test" in seen_values:
            LOGGER.info("Split column has no heldout; mapping --split heldout to split=test.")
            return SplitPlan(
                selected_split="test",
                split_source="column",
                split_column=split_col,
                heldout_mapped_to="test",
            )
        if requested_split in {"test", "val", "train"} and requested_split not in seen_values:
            fallback = "test" if "test" in seen_values else ("val" if "val" in seen_values else "train")
            LOGGER.warning("Requested split=%s not found in split column; falling back to %s.", requested_split, fallback)
            return SplitPlan(
                selected_split=fallback,
                split_source="column",
                split_column=split_col,
                heldout_mapped_to="",
            )

    mapped = "test" if requested_split == "heldout" else requested_split
    if requested_split == "heldout":
        LOGGER.info("No explicit heldout split column found; using deterministic channel-held-out mapping (heldout->test).")
    return SplitPlan(
        selected_split=mapped,
        split_source="hash",
        split_column="",
        heldout_mapped_to=("test" if requested_split == "heldout" else ""),
    )


def _get_location_terms_from_repo(root: Path) -> List[str]:
    paths = list(root.rglob("*state*.txt")) + list(root.rglob("*province*.txt")) + list(root.rglob("*cities*.txt"))
    terms: List[str] = []
    for path in paths:
        try:
            if path.stat().st_size > 2_000_000:
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                terms.append(item)
        except Exception:
            continue
    return terms


def _build_entity_masker(
    mode: str,
    *,
    location_terms: Sequence[str],
) -> Callable[[str], str]:
    if mode == "spacy":
        return SpacyEntityMasker(location_terms)
    return RegexEntityMasker(location_terms)


def _compute_word_count_array(texts: Sequence[str]) -> np.ndarray:
    return np.asarray([len(_WORD_RE.findall(t)) for t in texts], dtype=np.int32)


def _compute_asr_score_array(texts: Sequence[str]) -> np.ndarray:
    s = pd.Series(list(texts), dtype=str)
    bracket = s.str.count(_ASR_BRACKET_RE)
    timestamp = s.str.count(_ASR_TIMESTAMP_RE)
    filler = s.str.count(_ASR_FILLER_RE)
    score = bracket.to_numpy(dtype=np.int32) + timestamp.to_numpy(dtype=np.int32) + filler.to_numpy(dtype=np.int32)
    return score


def _length_bucket_ids(word_counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(word_counts, dtype=np.int8)
    out[(word_counts >= 150) & (word_counts < 300)] = 1
    out[(word_counts >= 300) & (word_counts <= 500)] = 2
    out[word_counts > 500] = 3
    return out


def _confidence_bucket_ids(conf: np.ndarray) -> np.ndarray:
    out = np.floor(conf * 5.0).astype(np.int8)
    out = np.clip(out, 0, 4)
    return out


def _update_basic_class_stats(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    top3_hit: np.ndarray,
    top5_hit: np.ndarray,
    stats: RunningStats,
) -> None:
    stats.n += int(len(y))
    stats.topk_hits[1] += int(np.sum(pred == y))
    stats.topk_hits[3] += int(np.sum(top3_hit))
    stats.topk_hits[5] += int(np.sum(top5_hit))

    stats.support += np.bincount(y, minlength=stats.n_classes)
    correct = pred == y
    if np.any(correct):
        stats.tp += np.bincount(y[correct], minlength=stats.n_classes)
    mismatch = ~correct
    if np.any(mismatch):
        stats.fn += np.bincount(y[mismatch], minlength=stats.n_classes)
        stats.fp += np.bincount(pred[mismatch], minlength=stats.n_classes)

    if np.any(top3_hit):
        stats.top3_hits_by_class += np.bincount(y[top3_hit], minlength=stats.n_classes)
    if np.any(top5_hit):
        stats.top5_hits_by_class += np.bincount(y[top5_hit], minlength=stats.n_classes)


def _update_bucket_stats(
    *,
    bucket: BucketAccumulator,
    y: np.ndarray,
    pred: np.ndarray,
    top5_hit: np.ndarray,
) -> None:
    if len(y) == 0:
        return
    bucket.n += int(len(y))
    correct = pred == y
    bucket.top1 += int(np.sum(correct))
    bucket.top5 += int(np.sum(top5_hit))
    if np.any(correct):
        bucket.tp += np.bincount(y[correct], minlength=bucket.tp.shape[0])
    mismatch = ~correct
    if np.any(mismatch):
        bucket.fn += np.bincount(y[mismatch], minlength=bucket.fn.shape[0])
        bucket.fp += np.bincount(pred[mismatch], minlength=bucket.fp.shape[0])


def _precision_recall_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp_f = tp.astype(np.float64)
    fp_f = fp.astype(np.float64)
    fn_f = fn.astype(np.float64)
    precision = np.divide(tp_f, tp_f + fp_f, out=np.zeros_like(tp_f), where=(tp_f + fp_f) > 0)
    recall = np.divide(tp_f, tp_f + fn_f, out=np.zeros_like(tp_f), where=(tp_f + fn_f) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp_f), where=(precision + recall) > 0)
    return precision, recall, f1


def _macro_weighted_metrics(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, support: np.ndarray) -> Dict[str, float]:
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    support_f = support.astype(np.float64)
    support_sum = float(np.sum(support_f))
    weighted_precision = float(np.sum(precision * support_f) / support_sum) if support_sum > 0 else 0.0
    weighted_recall = float(np.sum(recall * support_f) / support_sum) if support_sum > 0 else 0.0
    weighted_f1 = float(np.sum(f1 * support_f) / support_sum) if support_sum > 0 else 0.0

    macro_precision = float(np.mean(precision)) if precision.size else 0.0
    macro_recall_all = float(np.mean(recall)) if recall.size else 0.0
    valid = support > 0
    macro_recall_present = float(np.mean(recall[valid])) if np.any(valid) else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall_all,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": macro_recall_present,
    }


def _median_from_hist(hist: np.ndarray, max_value: float) -> float:
    total = int(np.sum(hist))
    if total <= 0:
        return 0.0
    midpoint = (total - 1) // 2
    cumsum = 0
    for i, c in enumerate(hist.tolist()):
        cumsum += int(c)
        if cumsum > midpoint:
            return float((i + 0.5) / max(1, len(hist)) * max_value)
    return float(max_value)


def _bucket_to_metrics(bucket: BucketAccumulator) -> Dict[str, float]:
    if bucket.n <= 0:
        return {"support": 0.0, "top1": 0.0, "top5": 0.0, "macroF1": 0.0}
    _, _, f1 = _precision_recall_f1(bucket.tp, bucket.fp, bucket.fn)
    return {
        "support": float(bucket.n),
        "top1": float(bucket.top1 / bucket.n),
        "top5": float(bucket.top5 / bucket.n),
        "macroF1": float(np.mean(f1)) if f1.size else 0.0,
    }


def _quantile_edges_from_counts(score_counts: Dict[int, int]) -> Tuple[int, int, int]:
    if not score_counts:
        return (0, 0, 0)
    items = sorted((int(k), int(v)) for k, v in score_counts.items())
    total = sum(v for _, v in items)
    if total <= 0:
        return (0, 0, 0)

    def _locate(p: float) -> int:
        target = p * total
        running = 0
        for score, cnt in items:
            running += cnt
            if running >= target:
                return int(score)
        return int(items[-1][0])

    return (_locate(0.25), _locate(0.50), _locate(0.75))


def _compute_top_confusions(confusion: np.ndarray, idx_to_label: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_classes = confusion.shape[0]
    per_class_rows: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []

    off_diag = confusion.copy()
    np.fill_diagonal(off_diag, 0)

    for true_idx in range(n_classes):
        row = off_diag[true_idx]
        support = int(np.sum(confusion[true_idx]))
        if support <= 0:
            continue
        mistaken_total = int(np.sum(row))
        if mistaken_total <= 0:
            continue
        top_idx = np.argsort(row)[::-1][:3]
        rank = 1
        for pred_idx in top_idx:
            cnt = int(row[pred_idx])
            if cnt <= 0:
                continue
            per_class_rows.append(
                {
                    "true_label": idx_to_label[true_idx],
                    "pred_label": idx_to_label[pred_idx],
                    "count": cnt,
                    "rate_within_true": float(cnt / support),
                    "rank": rank,
                }
            )
            rank += 1

    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            if true_idx == pred_idx:
                continue
            cnt = int(off_diag[true_idx, pred_idx])
            if cnt <= 0:
                continue
            global_rows.append(
                {
                    "true_label": idx_to_label[true_idx],
                    "pred_label": idx_to_label[pred_idx],
                    "count": cnt,
                }
            )

    global_rows_sorted = sorted(global_rows, key=lambda x: x["count"], reverse=True)[:20]
    return pd.DataFrame(per_class_rows), pd.DataFrame(global_rows_sorted)


def _default_output_dir(checkpoint: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "eval" / f"{checkpoint.stem}_{ts}"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _run_streaming_eval(
    *,
    args: argparse.Namespace,
    predictor: TextBatchPredictor,
    split_plan: SplitPlan,
    label_to_idx: Dict[str, int],
    idx_to_label: List[str],
    output_dir: Path,
    text_transform: Optional[Callable[[str], str]],
    eval_name: str,
) -> Dict[str, Any]:
    n_classes = len(idx_to_label)
    topk_list = _parse_topk(args.topk, n_classes)
    topk_hit_counts = {int(k): 0 for k in topk_list}

    thresholds = _parse_csv_float_list(args.thresholds)
    if not thresholds:
        thresholds = [0.5]
    thresholds = sorted([max(0.0, min(1.0, t)) for t in thresholds])

    stats = RunningStats(n_classes=n_classes, thresholds=thresholds, ece_bins=max(1, int(args.ece_bins)))
    if bool(args.confusions) or bool(args.per_class) or bool(args.slices):
        stats.enable_confusion()

    sample = ReservoirSampler(k=int(args.save_pred_sample), seed=int(args.seed) + (17 if eval_name == "masked" else 0))

    do_calibration = bool(args.calibration)
    do_slices = bool(args.slices)
    do_coverage = bool(args.coverage_curve)

    length_keys = ["lt150", "150_299", "300_500", "gt500"]
    conf_keys = ["0.0_0.2", "0.2_0.4", "0.4_0.6", "0.6_0.8", "0.8_1.0"]
    length_buckets = {k: BucketAccumulator(n_classes=n_classes) for k in length_keys}
    conf_buckets = {k: BucketAccumulator(n_classes=n_classes) for k in conf_keys}
    asr_score_acc: Dict[int, BucketAccumulator] = {}
    asr_score_counts: Dict[int, int] = {}

    usecols = {args.text_col, args.label_col}
    if args.group_col:
        usecols.add(args.group_col)
    if args.id_col:
        usecols.add(args.id_col)
    if args.channel_col:
        usecols.add(args.channel_col)
    if split_plan.split_source == "column" and split_plan.split_column:
        usecols.add(split_plan.split_column)

    reader = pd.read_csv(
        args.csv_path,
        sep=args.sep,
        dtype=str,
        chunksize=int(args.chunksize),
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
        usecols=lambda c: str(c).strip() in usecols,
    )

    batch_counter = 0
    rows_scanned = 0
    rows_after_filter = 0
    group_split_cache: Dict[str, str] = {}
    start = time.perf_counter()

    for chunk in reader:
        chunk.columns = [str(c).strip() for c in chunk.columns]
        rows_scanned += len(chunk)

        if args.text_col not in chunk.columns or args.label_col not in chunk.columns:
            raise ValueError(f"CSV is missing required columns: {args.text_col}, {args.label_col}")

        texts = chunk[args.text_col].astype(str).map(normalize_text)
        labels = chunk[args.label_col].astype(str).str.strip()

        base_mask = (texts.str.len() >= int(args.min_text_chars)) & (labels != "")
        if not bool(base_mask.any()):
            continue

        texts = texts[base_mask]
        labels = labels[base_mask]
        chunk_sel = chunk.loc[base_mask]

        y_idx = np.fromiter((label_to_idx.get(lbl, -1) for lbl in labels.to_numpy(dtype=str)), dtype=np.int64, count=len(labels))
        known_mask = y_idx >= 0
        if not bool(np.any(known_mask)):
            continue

        texts_np = texts.to_numpy(dtype=str)[known_mask]
        labels_np = y_idx[known_mask]
        chunk_sel = chunk_sel.iloc[np.nonzero(known_mask)[0]].copy()

        if split_plan.split_source == "column" and split_plan.split_column in chunk_sel.columns:
            split_arr = chunk_sel[split_plan.split_column].astype(str).map(_normalize_split_token).to_numpy(dtype=str)
        else:
            if args.group_col in chunk_sel.columns:
                groups = chunk_sel[args.group_col].astype(str).to_numpy(dtype=str)
            elif args.id_col in chunk_sel.columns:
                groups = chunk_sel[args.id_col].astype(str).to_numpy(dtype=str)
            else:
                groups = np.full(len(chunk_sel), "", dtype=object).astype(str)
            split_arr = np.empty(len(groups), dtype=object)
            for i, g in enumerate(groups):
                key = str(g or "")
                split_val = group_split_cache.get(key)
                if split_val is None:
                    split_val = split_from_group(
                        group_value=key,
                        seed=int(args.seed),
                        val_fraction=float(args.val_fraction),
                        test_fraction=float(args.test_fraction),
                        hash_method=str(args.split_hash),
                    )
                    group_split_cache[key] = split_val
                split_arr[i] = split_val
            split_arr = split_arr.astype(str)

        split_mask = split_arr == split_plan.selected_split
        if not bool(np.any(split_mask)):
            continue

        texts_np = texts_np[split_mask]
        labels_np = labels_np[split_mask]
        chunk_sel = chunk_sel.iloc[np.nonzero(split_mask)[0]].copy()

        if args.max_eval_samples > 0:
            remaining = int(args.max_eval_samples) - int(stats.n)
            if remaining <= 0:
                break
            if len(texts_np) > remaining:
                texts_np = texts_np[:remaining]
                labels_np = labels_np[:remaining]
                chunk_sel = chunk_sel.iloc[:remaining].copy()

        if len(texts_np) == 0:
            continue

        rows_after_filter += len(texts_np)

        if do_slices:
            word_counts_all = _compute_word_count_array(texts_np)
            asr_scores_all = _compute_asr_score_array(texts_np)
            length_ids_all = _length_bucket_ids(word_counts_all)
        else:
            asr_scores_all = np.zeros(len(texts_np), dtype=np.int32)
            length_ids_all = np.zeros(len(texts_np), dtype=np.int8)

        id_values = chunk_sel[args.id_col].astype(str).to_numpy(dtype=str) if args.id_col in chunk_sel.columns else np.full(len(texts_np), "", dtype=object)
        channel_values = (
            chunk_sel[args.channel_col].astype(str).to_numpy(dtype=str)
            if args.channel_col in chunk_sel.columns
            else np.full(len(texts_np), "", dtype=object)
        )

        for start_idx in range(0, len(texts_np), int(args.batch_size)):
            end_idx = min(start_idx + int(args.batch_size), len(texts_np))
            bt = texts_np[start_idx:end_idx].tolist()
            by = labels_np[start_idx:end_idx]
            basr = asr_scores_all[start_idx:end_idx]
            blen = length_ids_all[start_idx:end_idx]
            b_id = id_values[start_idx:end_idx]
            b_channel = channel_values[start_idx:end_idx]

            if text_transform is not None:
                bt = [text_transform(t) for t in bt]

            probs = predictor.predict_proba(bt)
            if probs.shape[0] != len(by):
                raise RuntimeError("Prediction batch size mismatch.")

            ranked = np.argsort(-probs, axis=1)
            pred = ranked[:, 0].astype(np.int64)
            top3_hit = (ranked[:, : min(3, ranked.shape[1])] == by[:, None]).any(axis=1)
            top5_hit = (ranked[:, : min(5, ranked.shape[1])] == by[:, None]).any(axis=1)
            for k in topk_list:
                hit_k = (ranked[:, : min(k, ranked.shape[1])] == by[:, None]).any(axis=1)
                topk_hit_counts[int(k)] += int(np.sum(hit_k))

            _update_basic_class_stats(y=by, pred=pred, top3_hit=top3_hit, top5_hit=top5_hit, stats=stats)

            p_true = probs[np.arange(len(by)), by]
            p_true = np.clip(p_true, 1e-12, 1.0)
            stats.nll_sum += float(np.sum(-np.log(p_true)))
            probs_sq = np.sum(probs * probs, axis=1)
            stats.brier_sum += float(np.sum(probs_sq - 2.0 * p_true + 1.0))

            conf = np.max(probs, axis=1)
            correct = pred == by
            stats.n_correct += int(np.sum(correct))
            stats.n_incorrect += int(np.sum(~correct))
            stats.conf_correct_sum += float(np.sum(conf[correct])) if np.any(correct) else 0.0
            stats.conf_incorrect_sum += float(np.sum(conf[~correct])) if np.any(~correct) else 0.0
            stats.overconfident_wrong += int(np.sum((~correct) & (conf >= 0.9)))

            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
            stats.entropy_sum += float(np.sum(entropy))
            max_entropy = float(math.log(max(2, n_classes)))
            if max_entropy > 0:
                ent_idx = np.floor((entropy / max_entropy) * stats.entropy_hist_bins).astype(np.int64)
                ent_idx = np.clip(ent_idx, 0, stats.entropy_hist_bins - 1)
                stats.entropy_hist += np.bincount(ent_idx, minlength=stats.entropy_hist_bins)

            if do_calibration:
                eidx = np.floor(conf * stats.ece_bins).astype(np.int64)
                eidx = np.clip(eidx, 0, stats.ece_bins - 1)
                stats.ece_count += np.bincount(eidx, minlength=stats.ece_bins)
                stats.ece_conf_sum += np.bincount(eidx, weights=conf, minlength=stats.ece_bins)
                stats.ece_correct_sum += np.bincount(eidx, weights=correct.astype(np.float64), minlength=stats.ece_bins)

            if do_coverage:
                for i, t in enumerate(stats.thresholds):
                    m = conf >= t
                    if not np.any(m):
                        continue
                    y_t = by[m]
                    p_t = pred[m]
                    c_t = p_t == y_t
                    stats.coverage_count[i] += int(len(y_t))
                    stats.coverage_correct[i] += int(np.sum(c_t))
                    if np.any(c_t):
                        stats.coverage_tp[i] += np.bincount(y_t[c_t], minlength=n_classes)
                    mm = ~c_t
                    if np.any(mm):
                        stats.coverage_fn[i] += np.bincount(y_t[mm], minlength=n_classes)
                        stats.coverage_fp[i] += np.bincount(p_t[mm], minlength=n_classes)

            if stats.confusion is not None:
                flat = by * n_classes + pred
                stats.confusion += np.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)

            if do_slices:
                conf_ids = _confidence_bucket_ids(conf)
                for bid in range(4):
                    mask_b = blen == bid
                    if np.any(mask_b):
                        _update_bucket_stats(
                            bucket=length_buckets[length_keys[bid]],
                            y=by[mask_b],
                            pred=pred[mask_b],
                            top5_hit=top5_hit[mask_b],
                        )
                for bid in range(5):
                    mask_b = conf_ids == bid
                    if np.any(mask_b):
                        _update_bucket_stats(
                            bucket=conf_buckets[conf_keys[bid]],
                            y=by[mask_b],
                            pred=pred[mask_b],
                            top5_hit=top5_hit[mask_b],
                        )
                for score in np.unique(basr):
                    mask_s = basr == score
                    if not np.any(mask_s):
                        continue
                    score_i = int(score)
                    if score_i not in asr_score_acc:
                        asr_score_acc[score_i] = BucketAccumulator(n_classes=n_classes)
                        asr_score_counts[score_i] = 0
                    asr_score_counts[score_i] += int(np.sum(mask_s))
                    _update_bucket_stats(
                        bucket=asr_score_acc[score_i],
                        y=by[mask_s],
                        pred=pred[mask_s],
                        top5_hit=top5_hit[mask_s],
                    )

            top3_labels = ranked[:, : min(3, ranked.shape[1])]
            for i in range(len(by)):
                sample_row = {
                    "row_index_stream": int(stats.n - len(by) + i + 1),
                    "id": str(b_id[i]),
                    "channel_id": str(b_channel[i]),
                    "true_label": idx_to_label[int(by[i])],
                    "pred_label": idx_to_label[int(pred[i])],
                    "confidence": float(conf[i]),
                    "correct": bool(correct[i]),
                    "top3": "|".join(idx_to_label[int(x)] for x in top3_labels[i]),
                    "text": str(bt[i])[:800],
                }
                sample.add(sample_row)

            batch_counter += 1
            if args.log_every_batches > 0 and (batch_counter % int(args.log_every_batches) == 0):
                elapsed = max(1e-9, time.perf_counter() - start)
                sps = stats.n / elapsed
                LOGGER.info(
                    "[%s] batch=%s samples=%s sps=%.1f top1=%.4f",
                    eval_name,
                    batch_counter,
                    stats.n,
                    sps,
                    stats.topk_hits[1] / max(1, stats.n),
                )

        if args.max_eval_samples > 0 and stats.n >= int(args.max_eval_samples):
            break

    elapsed = max(1e-9, time.perf_counter() - start)

    core = _macro_weighted_metrics(tp=stats.tp, fp=stats.fp, fn=stats.fn, support=stats.support)
    mean_conf_correct = stats.conf_correct_sum / max(1, stats.n_correct)
    mean_conf_incorrect = stats.conf_incorrect_sum / max(1, stats.n_incorrect)
    max_entropy = float(math.log(max(2, n_classes)))
    entropy_median = _median_from_hist(stats.entropy_hist, max_entropy) if max_entropy > 0 else 0.0

    topk_metrics: Dict[str, float] = {}
    for k in topk_list:
        hits = int(topk_hit_counts.get(int(k), 0))
        topk_metrics[f"top{k}_accuracy"] = float(hits / max(1, stats.n))

    metrics: Dict[str, Any] = {
        "split_requested": args.split,
        "split_used": split_plan.selected_split,
        "split_source": split_plan.split_source,
        "heldout_mapped_to": split_plan.heldout_mapped_to,
        "n_samples": int(stats.n),
        "rows_scanned": int(rows_scanned),
        "rows_after_basic_filter": int(rows_after_filter),
        "wall_time_sec": float(elapsed),
        "samples_per_sec": float(stats.n / elapsed),
        "nll": float(stats.nll_sum / max(1, stats.n)),
        "brier": float(stats.brier_sum / max(1, stats.n)),
        "mean_confidence_correct": float(mean_conf_correct),
        "mean_confidence_incorrect": float(mean_conf_incorrect),
        "overconfident_error_rate": float(stats.overconfident_wrong / max(1, stats.n)),
        "entropy_mean": float(stats.entropy_sum / max(1, stats.n)),
        "entropy_median": float(entropy_median),
        **topk_metrics,
        **core,
    }

    if bool(args.per_class):
        precision, recall, f1 = _precision_recall_f1(stats.tp, stats.fp, stats.fn)
        support = stats.support.astype(np.int64)
        top3_recall = np.divide(
            stats.top3_hits_by_class.astype(np.float64),
            np.maximum(support.astype(np.float64), 1.0),
        )
        top5_recall = np.divide(
            stats.top5_hits_by_class.astype(np.float64),
            np.maximum(support.astype(np.float64), 1.0),
        )
        per_class_df = pd.DataFrame(
            {
                "class_index": np.arange(n_classes, dtype=np.int64),
                "label": idx_to_label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "top3_recall": top3_recall,
                "top5_recall": top5_recall,
            }
        )
        per_class_df.to_csv(output_dir / "per_class.csv", index=False)

    calibration_payload: Dict[str, Any] = {}
    if do_calibration:
        bin_rows: List[Dict[str, float]] = []
        ece = 0.0
        for i in range(stats.ece_bins):
            count = int(stats.ece_count[i])
            if count <= 0:
                avg_conf = 0.0
                acc = 0.0
            else:
                avg_conf = float(stats.ece_conf_sum[i] / count)
                acc = float(stats.ece_correct_sum[i] / count)
            ece += abs(acc - avg_conf) * (count / max(1, stats.n))
            lo = i / stats.ece_bins
            hi = (i + 1) / stats.ece_bins
            bin_rows.append(
                {
                    "bin": i,
                    "low": lo,
                    "high": hi,
                    "count": count,
                    "avg_conf": avg_conf,
                    "accuracy": acc,
                }
            )
        calibration_payload = {
            "ece": float(ece),
            "bins": bin_rows,
            "mean_confidence_correct": float(mean_conf_correct),
            "mean_confidence_incorrect": float(mean_conf_incorrect),
            "overconfident_error_rate": float(stats.overconfident_wrong / max(1, stats.n)),
        }
        (output_dir / "calibration.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
        metrics["ece"] = float(ece)

    coverage_rows: List[Dict[str, float]] = []
    if do_coverage:
        for i, t in enumerate(stats.thresholds):
            cov_n = int(stats.coverage_count[i])
            coverage = float(cov_n / max(1, stats.n))
            acc_at_cov = float(stats.coverage_correct[i] / max(1, cov_n)) if cov_n > 0 else 0.0
            _, _, f1_t = _precision_recall_f1(stats.coverage_tp[i], stats.coverage_fp[i], stats.coverage_fn[i])
            macro_f1_cov = float(np.mean(f1_t)) if cov_n > 0 else 0.0
            coverage_rows.append(
                {
                    "threshold": float(t),
                    "coverage": coverage,
                    "n": cov_n,
                    "accuracy": acc_at_cov,
                    "macroF1": macro_f1_cov,
                }
            )
        pd.DataFrame(coverage_rows).to_csv(output_dir / "coverage_curve.csv", index=False)
        (output_dir / "coverage_curve.json").write_text(json.dumps(coverage_rows, indent=2), encoding="utf-8")

    slices_payload: Dict[str, Any] = {}
    if do_slices:
        q1, q2, q3 = _quantile_edges_from_counts(asr_score_counts)
        asr_quant_buckets = {
            "q1": BucketAccumulator(n_classes=n_classes),
            "q2": BucketAccumulator(n_classes=n_classes),
            "q3": BucketAccumulator(n_classes=n_classes),
            "q4": BucketAccumulator(n_classes=n_classes),
        }
        for score, acc in asr_score_acc.items():
            if score <= q1:
                key = "q1"
            elif score <= q2:
                key = "q2"
            elif score <= q3:
                key = "q3"
            else:
                key = "q4"
            tgt = asr_quant_buckets[key]
            tgt.n += acc.n
            tgt.top1 += acc.top1
            tgt.top5 += acc.top5
            tgt.tp += acc.tp
            tgt.fp += acc.fp
            tgt.fn += acc.fn

        slices_payload = {
            "length_buckets": {k: _bucket_to_metrics(v) for k, v in length_buckets.items()},
            "asr_score_quantiles": {
                "edges": {"q25": int(q1), "q50": int(q2), "q75": int(q3)},
                "buckets": {k: _bucket_to_metrics(v) for k, v in asr_quant_buckets.items()},
            },
            "confidence_buckets": {k: _bucket_to_metrics(v) for k, v in conf_buckets.items()},
        }
        (output_dir / "slices.json").write_text(json.dumps(slices_payload, indent=2), encoding="utf-8")

    if stats.confusion is not None and bool(args.confusions):
        np.save(output_dir / "confusion_matrix.npy", stats.confusion)
        per_cls_conf, global_conf = _compute_top_confusions(stats.confusion, idx_to_label)
        rows: List[pd.DataFrame] = []
        if not per_cls_conf.empty:
            per_cls_conf.insert(0, "scope", "per_class")
            rows.append(per_cls_conf)
        if not global_conf.empty:
            global_conf.insert(0, "scope", "global")
            rows.append(global_conf)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(output_dir / "top_confusions.csv", index=False)

    if sample.rows:
        pd.DataFrame(sample.rows).to_csv(output_dir / "preds_sample.csv", index=False)

    args_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metrics_payload: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_name": args.checkpoint.name,
        "evaluation": eval_name,
        "metrics": metrics,
        "args": args_payload,
    }
    if calibration_payload:
        metrics_payload["calibration"] = calibration_payload
    if coverage_rows:
        metrics_payload["coverage_curve"] = coverage_rows
    if slices_payload:
        metrics_payload["slices"] = slices_payload

    metrics_path = output_dir / ("metrics_masked.json" if eval_name == "masked" else "metrics.json")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary_lines = [
        f"# Evaluation Summary ({eval_name})",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Split requested: `{args.split}`",
        f"- Split used: `{split_plan.selected_split}` ({split_plan.split_source})",
        f"- Samples: {stats.n}",
        f"- Throughput: {metrics['samples_per_sec']:.1f} samples/sec",
        f"- Time: {metrics['wall_time_sec']:.2f} sec",
        f"- Top-1: {metrics.get('top1_accuracy', 0.0):.4f}",
        f"- Top-3: {metrics.get('top3_accuracy', 0.0):.4f}",
        f"- Top-5: {metrics.get('top5_accuracy', 0.0):.4f}",
        f"- Macro F1: {metrics.get('macro_f1', 0.0):.4f}",
        f"- Weighted F1: {metrics.get('weighted_f1', 0.0):.4f}",
        f"- NLL: {metrics.get('nll', 0.0):.4f}",
        f"- Brier: {metrics.get('brier', 0.0):.4f}",
    ]
    if "ece" in metrics:
        summary_lines.append(f"- ECE: {metrics['ece']:.4f}")

    summary_path = output_dir / ("summary_masked.md" if eval_name == "masked" else "summary.md")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return metrics_payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_arg_parser().parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch is not None:
        torch.manual_seed(int(args.seed))

    artifact = joblib.load(args.checkpoint)
    if not isinstance(artifact, dict):
        raise ValueError("Checkpoint artifact is not a dictionary.")

    cfg = artifact.get("config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    args = _resolve_args_with_checkpoint(args, cfg)
    if args.csv_path is None:
        raise ValueError("--csv_path is required (not found in checkpoint config).")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args = _apply_fast_defaults(args)

    use_cuda = args.device == "cuda" and torch is not None and torch.cuda.is_available()
    amp_enabled = bool(args.amp) if args.amp is not None else bool(use_cuda)

    output_dir = args.output_dir or _default_output_dir(args.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_label_to_idx = artifact.get("label_to_idx")
    raw_idx_to_label = artifact.get("idx_to_label")
    if not isinstance(raw_label_to_idx, dict) or not isinstance(raw_idx_to_label, list):
        raise ValueError("Checkpoint is missing label mapping.")
    label_to_idx = {str(k): int(v) for k, v in raw_label_to_idx.items()}
    idx_to_label = [str(x) for x in raw_idx_to_label]

    columns = _read_header_columns(args.csv_path, args.sep)
    split_col = _pick_split_column(args, columns)
    split_plan = _build_split_plan(
        requested_split=args.split,
        split_col=split_col,
        csv_path=args.csv_path,
        sep=args.sep,
        text_col=args.text_col,
        label_col=args.label_col,
        group_col=args.group_col,
    )

    predictor = TextBatchPredictor(
        artifact=artifact,
        device=args.device,
        amp_enabled=amp_enabled,
        compile_model=bool(args.compile),
        compile_mode=args.compile_mode,
    )

    location_terms = _get_location_terms_from_repo(Path.cwd())
    if not location_terms:
        location_terms = idx_to_label

    LOGGER.info(
        "Starting eval | model_type=%s split=%s source=%s batch_size=%s amp=%s compile=%s fast=%s",
        predictor.model_type,
        split_plan.selected_split,
        split_plan.split_source,
        args.batch_size,
        amp_enabled,
        bool(args.compile),
        bool(args.fast),
    )

    base_payload = _run_streaming_eval(
        args=args,
        predictor=predictor,
        split_plan=split_plan,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        output_dir=output_dir,
        text_transform=None,
        eval_name="base",
    )

    if args.entity_mask_eval:
        masker = _build_entity_masker(
            mode=args.entity_mask_mode,
            location_terms=location_terms,
        )
        masked_payload = _run_streaming_eval(
            args=args,
            predictor=predictor,
            split_plan=split_plan,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            output_dir=output_dir,
            text_transform=masker,
            eval_name="masked",
        )

        m0 = base_payload.get("metrics", {})
        m1 = masked_payload.get("metrics", {})
        delta = {
            "delta_top1": _safe_float(m1.get("top1_accuracy")) - _safe_float(m0.get("top1_accuracy")),
            "delta_top5": _safe_float(m1.get("top5_accuracy")) - _safe_float(m0.get("top5_accuracy")),
            "delta_macroF1": _safe_float(m1.get("macro_f1")) - _safe_float(m0.get("macro_f1")),
            "delta_ECE": _safe_float(m1.get("ece")) - _safe_float(m0.get("ece")),
            "delta_NLL": _safe_float(m1.get("nll")) - _safe_float(m0.get("nll")),
        }
        (output_dir / "metrics_delta.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")

    LOGGER.info("Evaluation complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
