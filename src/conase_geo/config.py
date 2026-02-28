from dataclasses import dataclass
from typing import Literal

LabelType = Literal["state", "location_topk", "geo_grid"]

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CLIP_SECONDS = 10.0
DEFAULT_PAUSE_THRESHOLD_SECONDS = 0.35

TIMING_FEATURE_NAMES = [
    "words_per_second",
    "pause_mean",
    "pause_median",
    "pause_std",
    "pause_max",
    "pause_p90",
    "pause_rate",
    "rhythm_cv",
    "n_tokens",
    "clip_duration",
]

PROSODY_FEATURE_NAMES = [
    "energy_mean",
    "energy_std",
    "pitch_mean",
    "pitch_std",
    "pitch_median",
    "voiced_fraction",
    "duration_seconds",
    "zcr_mean",
]

AUX_FEATURE_NAMES = TIMING_FEATURE_NAMES + PROSODY_FEATURE_NAMES


@dataclass(frozen=True)
class ManifestConfig:
    label_type: LabelType = "state"
    clip_seconds: float = DEFAULT_CLIP_SECONDS
    clips_per_video: int = 2
    max_videos: int = 2_000
    min_label_examples: int = 200
    top_k_locations: int = 200
    geo_cell_size_deg: float = 1.5
    pause_threshold_seconds: float = DEFAULT_PAUSE_THRESHOLD_SECONDS


@dataclass(frozen=True)
class TrainingConfig:
    model: str = "audio_plus_timing"
    encoder: str = "wav2vec2_base"
    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    split_by: str = "channel_id"
    pause_threshold_seconds: float = DEFAULT_PAUSE_THRESHOLD_SECONDS
    max_audio_seconds: float = DEFAULT_CLIP_SECONDS
