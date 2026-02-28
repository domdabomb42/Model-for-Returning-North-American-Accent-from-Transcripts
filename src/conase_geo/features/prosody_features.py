from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import librosa
import numpy as np

from conase_geo.config import DEFAULT_SAMPLE_RATE, PROSODY_FEATURE_NAMES

LOGGER = logging.getLogger(__name__)


def _zero_features() -> Dict[str, float]:
    return {name: 0.0 for name in PROSODY_FEATURE_NAMES}


def compute_prosody_features(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if waveform.size == 0:
        return _zero_features()

    y = waveform.astype(np.float32, copy=False)
    duration_seconds = float(len(y) / max(sample_rate, 1))

    rms = librosa.feature.rms(y=y).squeeze()
    energy_mean = float(np.mean(rms)) if rms.size else 0.0
    energy_std = float(np.std(rms)) if rms.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(y).squeeze()
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0

    f0 = np.asarray([], dtype=np.float32)
    voiced_mask = np.asarray([], dtype=bool)
    try:
        raw_f0 = librosa.yin(
            y,
            fmin=75.0,
            fmax=350.0,
            sr=sample_rate,
            frame_length=2048,
            hop_length=256,
        )
        raw_f0 = np.asarray(raw_f0, dtype=np.float32)
        voiced_mask = np.isfinite(raw_f0) & (raw_f0 > 0.0)
        f0 = raw_f0[voiced_mask]
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.debug("librosa.yin failed: %s", exc)

    if f0.size:
        pitch_mean = float(np.mean(f0))
        pitch_std = float(np.std(f0))
        pitch_median = float(np.median(f0))
        voiced_fraction = float(np.mean(voiced_mask))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_median = 0.0
        if rms.size:
            threshold = float(np.percentile(rms, 30))
            voiced_fraction = float(np.mean(rms > threshold))
        else:
            voiced_fraction = 0.0

    return {
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_median": pitch_median,
        "voiced_fraction": voiced_fraction,
        "duration_seconds": duration_seconds,
        "zcr_mean": zcr_mean,
    }


def compute_prosody_from_file(
    audio_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_seconds: float | None = None,
) -> Dict[str, float]:
    path = Path(audio_path)
    if not path.exists():
        return _zero_features()
    y, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=True, duration=max_seconds)
    return compute_prosody_features(y, sr)


def prosody_feature_vector(
    feature_dict: Dict[str, float],
    feature_names: Sequence[str] = PROSODY_FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray([float(feature_dict.get(name, 0.0)) for name in feature_names], dtype=np.float32)
