from __future__ import annotations

import torch
from torch import nn


class TimingOnlyClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, timing_features: torch.Tensor) -> torch.Tensor:
        return self.net(timing_features)


class AudioOnlyClassifier(nn.Module):
    def __init__(self, audio_encoder: nn.Module, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.head = nn.Sequential(
            nn.Linear(self.audio_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        audio_emb = self.audio_encoder(audio_waveform)
        return self.head(audio_emb)


class AudioPlusTimingClassifier(nn.Module):
    def __init__(
        self,
        audio_encoder: nn.Module,
        timing_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        fusion_dim = self.audio_encoder.output_dim + timing_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, audio_waveform: torch.Tensor, timing_features: torch.Tensor) -> torch.Tensor:
        audio_emb = self.audio_encoder(audio_waveform)
        fused = torch.cat([audio_emb, timing_features], dim=-1)
        return self.head(fused)
