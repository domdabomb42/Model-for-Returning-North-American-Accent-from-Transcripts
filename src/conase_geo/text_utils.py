from __future__ import annotations

import hashlib
from pathlib import Path
import re
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
_PROPER_NOUN_SEQ_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

_US_STATE_NAMES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
]
_US_STATE_ABBR = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]
_CA_PROV_TERR_NAMES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia",
    "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan", "Northwest Territories", "Nunavut", "Yukon",
]
_CA_PROV_TERR_ABBR = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "PEI", "QC", "SK", "NT", "NU", "YT"]

_CURATED_CITY_TERMS = [
    "NYC", "New York City", "Los Angeles", "LA", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio",
    "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "San Francisco", "Indianapolis", "Seattle", "Denver", "Boston", "El Paso", "Detroit", "Nashville",
    "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson",
    "Fresno", "Sacramento", "Mesa", "Atlanta", "Kansas City", "Colorado Springs", "Miami", "Raleigh",
    "Omaha", "Minneapolis", "New Orleans", "Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa", "Edmonton",
]


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("@", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_location_terms(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        return []
    out: List[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            out.append(item)
    except Exception:
        return []
    return out


def _compile_location_name_pattern(location_terms: Sequence[str]) -> Optional[re.Pattern[str]]:
    clean_terms = sorted({t.strip() for t in location_terms if t and t.strip()}, key=len, reverse=True)
    if not clean_terms:
        return None
    escaped = [re.escape(t) for t in clean_terms]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def _compile_location_abbr_pattern(abbr_terms: Sequence[str]) -> Optional[re.Pattern[str]]:
    clean_terms = sorted({t.strip().upper() for t in abbr_terms if t and t.strip()}, key=len, reverse=True)
    if not clean_terms:
        return None
    escaped = [re.escape(t) for t in clean_terms]
    # Match uppercase abbreviations only to avoid masking common lowercase words like "in" (Indiana).
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(escaped) + r")(?![A-Za-z])")


class LocationMasker:
    def __init__(
        self,
        *,
        extra_terms: Optional[Sequence[str]] = None,
        mask_token: str = "[LOC]",
        mask_prob: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.mask_token = str(mask_token)
        self.mask_prob = float(mask_prob)
        self.rng = np.random.default_rng(int(seed))

        name_terms: List[str] = []
        name_terms.extend(_US_STATE_NAMES)
        name_terms.extend(_CA_PROV_TERR_NAMES)
        name_terms.extend(_CURATED_CITY_TERMS)
        if extra_terms is not None:
            name_terms.extend([str(t) for t in extra_terms if str(t).strip()])

        self.name_pattern = _compile_location_name_pattern(name_terms)
        self.abbr_pattern = _compile_location_abbr_pattern(_US_STATE_ABBR + _CA_PROV_TERR_ABBR)

        self.city_state_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*,\s*([A-Z]{2}|[A-Z][a-z]+)\b"
        )
        self.loc_phrase_pattern = re.compile(
            r"\b(i am from|i'm from|im from|i live in|we live in|here in|out of|based in)\s+"
            r"([A-Za-z][A-Za-z\.\-]*(?:\s+[A-Za-z][A-Za-z\.\-]*){0,3})",
            flags=re.IGNORECASE,
        )
        self.in_loc_pattern = re.compile(
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
        )

    def __call__(self, text: object) -> str:
        s = normalize_text(text)
        if not s:
            return s
        if self.mask_prob < 1.0:
            if self.mask_prob <= 0.0:
                return s
            if float(self.rng.random()) > self.mask_prob:
                return s

        out = _URL_RE.sub(self.mask_token, s)
        out = _HANDLE_RE.sub(self.mask_token, out)
        out = self.city_state_pattern.sub(self.mask_token, out)
        out = self.loc_phrase_pattern.sub(lambda m: f"{m.group(1)} {self.mask_token}", out)
        out = self.in_loc_pattern.sub(lambda m: f"in {self.mask_token}", out)
        if self.name_pattern is not None:
            out = self.name_pattern.sub(self.mask_token, out)
        if self.abbr_pattern is not None:
            out = self.abbr_pattern.sub(self.mask_token, out)
        out = _PROPER_NOUN_SEQ_RE.sub(self.mask_token, out)
        out = re.sub(r"(?:\s*" + re.escape(self.mask_token) + r"\s*){2,}", f" {self.mask_token} ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out


def tokenize_text(text: object) -> List[str]:
    clean = normalize_text(text).lower()
    if not clean:
        return []
    return _TOKEN_RE.findall(clean)


def encode_text_to_token_ids(
    text: object,
    *,
    vocab: Dict[str, int],
    max_len: int,
    unk_id: int = 1,
) -> np.ndarray:
    ids = np.zeros(int(max_len), dtype=np.int32)
    tokens = tokenize_text(text)
    if not tokens:
        return ids
    for i, token in enumerate(tokens[: int(max_len)]):
        ids[i] = int(vocab.get(token, unk_id))
    return ids


def encode_text_to_token_ids_with_length(
    text: object,
    *,
    vocab: Dict[str, int],
    max_len: int,
    unk_id: int = 1,
) -> Tuple[np.ndarray, int]:
    ids = np.zeros(int(max_len), dtype=np.int32)
    tokens = tokenize_text(text)
    if not tokens:
        return ids, 0
    use_len = min(int(max_len), len(tokens))
    for i, token in enumerate(tokens[:use_len]):
        ids[i] = int(vocab.get(token, unk_id))
    return ids, int(use_len)


def split_from_group(
    group_value: object,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    *,
    hash_method: str = "blake2",
) -> str:
    group = str(group_value or "")
    key = f"{seed}|{group}".encode("utf-8", errors="ignore")
    if hash_method == "crc32":
        u = (zlib.crc32(key) & 0xFFFFFFFF) / float(2**32)
    else:
        h = hashlib.blake2b(key, digest_size=8).digest()
        u = int.from_bytes(h, byteorder="big") / float(2**64)
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def scores_to_probs(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        pos = 1.0 / (1.0 + np.exp(-scores))
        return np.stack([1.0 - pos, pos], axis=1)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    expv = np.exp(shifted)
    denom = np.sum(expv, axis=1, keepdims=True) + 1e-12
    return expv / denom


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if probs.size == 0:
        return 0.0
    top_k = min(k, probs.shape[1])
    idx = np.argpartition(probs, -top_k, axis=1)[:, -top_k:]
    hit = (idx == y_true[:, None]).any(axis=1)
    return float(np.mean(hit))


def keep_labels_from_counts(
    counts: Dict[str, int],
    min_label_examples: int,
    top_k_labels: int,
) -> List[str]:
    items = [(label, c) for label, c in counts.items() if c >= min_label_examples]
    items.sort(key=lambda x: x[1], reverse=True)
    if top_k_labels > 0:
        items = items[:top_k_labels]
    return [label for label, _ in items]


def read_csv_in_chunks(
    csv_path: str,
    sep: str,
    chunksize: int,
    max_rows: int,
    usecols: Any = None,
) -> Iterable[pd.DataFrame]:
    seen = 0
    reader = pd.read_csv(
        csv_path,
        sep=sep,
        dtype=str,
        chunksize=chunksize,
        usecols=usecols,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="skip",
    )
    for chunk in reader:
        if max_rows > 0:
            remaining = max_rows - seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        seen += len(chunk)
        yield canonicalize_columns(chunk)
