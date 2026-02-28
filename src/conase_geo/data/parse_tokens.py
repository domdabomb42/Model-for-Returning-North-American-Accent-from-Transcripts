from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class TokenStamp:
    token: str
    pos: str
    time: float


def parse_token_stamp(raw_token: str) -> Optional[TokenStamp]:
    """Parse token_POS_time from the right to support tokens with underscores."""
    if not isinstance(raw_token, str):
        return None
    raw_token = raw_token.strip()
    if not raw_token:
        return None
    parts = raw_token.rsplit("_", 2)
    if len(parts) != 3:
        return None
    token, pos, time_str = parts
    if not token or not pos:
        return None
    try:
        timestamp = float(time_str)
    except (TypeError, ValueError):
        return None
    return TokenStamp(token=token, pos=pos, time=timestamp)


def parse_text_pos(text_pos: str) -> List[TokenStamp]:
    if not isinstance(text_pos, str):
        return []
    results: List[TokenStamp] = []
    for piece in text_pos.split():
        stamp = parse_token_stamp(piece)
        if stamp is not None:
            results.append(stamp)
    return results


def tokens_in_window(tokens: Sequence[TokenStamp], clip_start: float, clip_end: float) -> List[TokenStamp]:
    return [t for t in tokens if clip_start <= t.time <= clip_end]


def window_text(tokens: Sequence[TokenStamp]) -> str:
    return " ".join(t.token for t in tokens)


def serialize_tokens_for_window(tokens: Sequence[TokenStamp], clip_start: float) -> str:
    payload = [{"token": t.token, "pos": t.pos, "time": max(0.0, t.time - clip_start)} for t in tokens]
    return json.dumps(payload, ensure_ascii=True)


def parse_token_times_json(raw_value: object) -> List[float]:
    """Parse token_times_json that may be list[dict] or list[number]."""
    if raw_value is None:
        return []
    if isinstance(raw_value, float):
        if raw_value != raw_value:
            return []
    if isinstance(raw_value, list):
        data = raw_value
    else:
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []

    out: List[float] = []
    for item in data:
        if isinstance(item, (int, float)):
            out.append(float(item))
            continue
        if isinstance(item, dict):
            value = item.get("time")
            if value is None:
                continue
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def to_token_times(tokens: Iterable[TokenStamp]) -> List[float]:
    return [t.time for t in tokens]
