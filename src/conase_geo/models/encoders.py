from __future__ import annotations

import logging

import torch
from torch import nn

try:
    import torchaudio
except Exception:  # pragma: no cover - optional runtime dependency failure
    torchaudio = None

LOGGER = logging.getLogger(__name__)


class MFCCEncoder(nn.Module):
    def __init__(self, sample_rate: int = 16_000, n_mfcc: int = 40) -> None:
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for MFCCEncoder.")
        self.output_dim = n_mfcc
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.float()
        mfcc = self.mfcc(wav)  # [B, n_mfcc, T]
        return mfcc.mean(dim=-1)


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for Wav2Vec2Encoder.")
        self.freeze = freeze
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model()
        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, self.sample_rate, dtype=torch.float32)
            features, _ = self.model.extract_features(dummy)
            self.output_dim = int(features[-1].shape[-1])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.float()
        if self.freeze:
            with torch.no_grad():
                features, _ = self.model.extract_features(wav)
        else:
            features, _ = self.model.extract_features(wav)
        hidden = features[-1]  # [B, T, C]
        return hidden.mean(dim=1)


def build_audio_encoder(name: str, freeze: bool = True, sample_rate: int = 16_000) -> nn.Module:
    key = name.lower().strip()
    if key == "wav2vec2_base":
        try:
            return Wav2Vec2Encoder(freeze=freeze)
        except Exception as exc:
            LOGGER.warning("Failed to load wav2vec2_base (%s). Falling back to MFCC.", exc)
            return MFCCEncoder(sample_rate=sample_rate)
    if key == "mfcc":
        return MFCCEncoder(sample_rate=sample_rate)
    raise ValueError(f"Unknown encoder: {name}")
