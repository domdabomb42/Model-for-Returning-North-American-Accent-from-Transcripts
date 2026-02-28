from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from conase_geo.config import TIMING_FEATURE_NAMES


def compute_timing_features(
    token_times: Sequence[float],
    clip_start: float,
    clip_end: float,
    pause_threshold: float = 0.35,
) -> Dict[str, float]:
    duration = max(float(clip_end) - float(clip_start), 1e-6)
    clean_times = sorted(float(t) for t in token_times if t is not None)

    n_tokens = len(clean_times)
    words_per_second = float(n_tokens / duration)

    if n_tokens >= 2:
        intervals = np.diff(np.asarray(clean_times, dtype=np.float64))
        intervals = intervals[intervals >= 0.0]
    else:
        intervals = np.asarray([], dtype=np.float64)

    pauses = intervals[intervals > pause_threshold] if intervals.size else np.asarray([], dtype=np.float64)

    pause_mean = float(pauses.mean()) if pauses.size else 0.0
    pause_median = float(np.median(pauses)) if pauses.size else 0.0
    pause_std = float(pauses.std()) if pauses.size else 0.0
    pause_max = float(pauses.max()) if pauses.size else 0.0
    pause_p90 = float(np.percentile(pauses, 90)) if pauses.size else 0.0
    pause_rate = float(pauses.size / duration)

    if intervals.size >= 2 and float(intervals.mean()) > 1e-9:
        rhythm_cv = float(intervals.std() / intervals.mean())
    else:
        rhythm_cv = 0.0

    return {
        "words_per_second": words_per_second,
        "pause_mean": pause_mean,
        "pause_median": pause_median,
        "pause_std": pause_std,
        "pause_max": pause_max,
        "pause_p90": pause_p90,
        "pause_rate": pause_rate,
        "rhythm_cv": rhythm_cv,
        "n_tokens": float(n_tokens),
        "clip_duration": float(duration),
    }


def timing_feature_vector(
    feature_dict: Dict[str, float],
    feature_names: Sequence[str] = TIMING_FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray([float(feature_dict.get(name, 0.0)) for name in feature_names], dtype=np.float32)
