from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from conase_geo.config import PROSODY_FEATURE_NAMES, TIMING_FEATURE_NAMES
from conase_geo.data.parse_tokens import parse_token_times_json
from conase_geo.features.prosody_features import compute_prosody_from_file, prosody_feature_vector
from conase_geo.features.timing_features import compute_timing_features, timing_feature_vector
from conase_geo.train import build_model

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

LOGGER = logging.getLogger(__name__)


def _load_audio_tensor(audio_path: Path, sample_rate: int, max_audio_seconds: float) -> torch.Tensor:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for audio-based prediction.")
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)
    wav = wav.squeeze(0).float()

    max_samples = int(sample_rate * max_audio_seconds)
    if wav.numel() > max_samples:
        wav = wav[:max_samples]
    elif wav.numel() < max_samples:
        wav = F.pad(wav, (0, max_samples - wav.numel()))
    return wav


def _load_mpsa_tensor(
    audio_path: Path,
    sample_rate: int,
    max_audio_seconds: float,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for MPSA-DenseNet prediction.")
    wav = _load_audio_tensor(audio_path, sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mfcc,
            "center": True,
        },
    )
    mfcc = mfcc_transform(wav.unsqueeze(0)).squeeze(0)  # [n_mfcc, frames]
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.unsqueeze(0).repeat(2, 1, 1).float()


def _parse_idx_to_label(payload: object, label_to_idx: dict) -> List[str]:
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        pairs = sorted([(int(k), str(v)) for k, v in payload.items()], key=lambda x: x[0])
        return [v for _, v in pairs]
    return [label for label, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]


class GeoPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        chosen_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.device = torch.device(chosen_device if device != "auto" else "cpu" if not torch.cuda.is_available() else "cuda")
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.model_type = str(self.checkpoint["model_type"])
        self.encoder_name = str(self.checkpoint.get("encoder_name", "mfcc"))
        self.label_to_idx = dict(self.checkpoint["label_to_idx"])
        self.idx_to_label = _parse_idx_to_label(self.checkpoint.get("idx_to_label"), self.label_to_idx)
        self.sample_rate = int(self.checkpoint.get("sample_rate", 16_000))
        self.max_audio_seconds = float(self.checkpoint.get("max_audio_seconds", 10.0))
        self.pause_threshold = float(self.checkpoint.get("pause_threshold", 0.35))
        self.mpsa_config: Dict[str, Any] = dict(self.checkpoint.get("mpsa_config", {}))
        self.mpsa_n_mfcc = int(self.checkpoint.get("mpsa_n_mfcc", 64))
        self.mpsa_n_fft = int(self.checkpoint.get("mpsa_n_fft", 1024))
        self.mpsa_hop_length = int(self.checkpoint.get("mpsa_hop_length", 256))

        if self.model_type == "timing_only":
            feature_dim = len(TIMING_FEATURE_NAMES)
        elif self.model_type == "audio_plus_timing":
            feature_dim = len(TIMING_FEATURE_NAMES) + len(PROSODY_FEATURE_NAMES)
        else:
            feature_dim = 0

        self.model = build_model(
            model_type=self.model_type,
            encoder_name=self.encoder_name,
            num_classes=len(self.label_to_idx),
            feature_dim=feature_dim,
            freeze_encoder=True,
            sample_rate=self.sample_rate,
            mpsa_config=self.mpsa_config,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def _timing_vector(self, token_times: Sequence[float]) -> np.ndarray:
        clip_duration = max(self.max_audio_seconds, 1e-3)
        if not token_times:
            return np.zeros(len(TIMING_FEATURE_NAMES), dtype=np.float32)
        stats = compute_timing_features(
            token_times=token_times,
            clip_start=0.0,
            clip_end=clip_duration,
            pause_threshold=self.pause_threshold,
        )
        return timing_feature_vector(stats, TIMING_FEATURE_NAMES)

    def predict(
        self,
        audio_path: str | Path,
        top_k: int = 5,
        token_times_json: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        token_times = parse_token_times_json(token_times_json) if token_times_json else []
        if self.model_type == "timing_only" and not token_times:
            LOGGER.warning("timing_only model received no token_times_json. Using zero timing features.")

        with torch.no_grad():
            if self.model_type == "timing_only":
                timing = torch.as_tensor(self._timing_vector(token_times), dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(timing)
            elif self.model_type == "audio_only":
                wav = _load_audio_tensor(path, self.sample_rate, self.max_audio_seconds).unsqueeze(0).to(self.device)
                logits = self.model(wav)
            elif self.model_type == "mpsa_densenet":
                mpsa_tensor = _load_mpsa_tensor(
                    audio_path=path,
                    sample_rate=self.sample_rate,
                    max_audio_seconds=self.max_audio_seconds,
                    n_mfcc=self.mpsa_n_mfcc,
                    n_fft=self.mpsa_n_fft,
                    hop_length=self.mpsa_hop_length,
                ).unsqueeze(0).to(self.device)
                out = self.model(mpsa_tensor)
                logits = out["accent"] if isinstance(out, dict) else out
            else:
                wav = _load_audio_tensor(path, self.sample_rate, self.max_audio_seconds).unsqueeze(0).to(self.device)
                timing_vec = self._timing_vector(token_times)
                prosody_stats = compute_prosody_from_file(path, sample_rate=self.sample_rate, max_seconds=self.max_audio_seconds)
                prosody_vec = prosody_feature_vector(prosody_stats, PROSODY_FEATURE_NAMES)
                feature_vec = np.concatenate([timing_vec, prosody_vec], axis=0).astype(np.float32)
                features = torch.as_tensor(feature_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(wav, features)

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            k = min(max(1, top_k), probs.shape[0])
            values, indices = torch.topk(probs, k=k)

        output: List[Tuple[str, float]] = []
        for prob, idx in zip(values.tolist(), indices.tolist()):
            label = self.idx_to_label[idx] if idx < len(self.idx_to_label) else str(idx)
            output.append((label, float(prob)))
        return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict location label from a short audio sample.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--token_times_json",
        type=str,
        default=None,
        help="Optional JSON string or path to JSON file for timing_only/audio_plus_timing.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _resolve_token_times_json(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    maybe_path = Path(raw)
    if maybe_path.exists():
        return maybe_path.read_text(encoding="utf-8")
    return raw


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()
    predictor = GeoPredictor(checkpoint_path=args.checkpoint, device=args.device)
    preds = predictor.predict(
        audio_path=args.audio,
        top_k=args.top_k,
        token_times_json=_resolve_token_times_json(args.token_times_json),
    )
    for rank, (label, prob) in enumerate(preds, start=1):
        print(f"{rank}. {label}\t{prob:.4f}")


if __name__ == "__main__":
    main()
