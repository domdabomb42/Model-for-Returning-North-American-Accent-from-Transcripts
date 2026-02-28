from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from conase_geo.models.mpsa_densenet import MPSAConfig, MPSADenseNet


class TextMPSADenseNetClassifier(nn.Module):
    """
    Text classifier that maps token ids to a 2D map consumed by MPSA-DenseNet.

    Input:
      token_ids: [B, max_len]
    Internal 2D map:
      [B, 1, embed_dim, max_len]
    """

    def __init__(
        self,
        *,
        num_classes: int,
        vocab_size: int,
        embed_dim: int,
        max_len: int,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        growth_rate: int = 32,
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        embed_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)
        self.embedding = nn.Embedding(int(vocab_size), int(embed_dim), padding_idx=0)
        self.embed_dropout = nn.Dropout(float(embed_dropout)) if float(embed_dropout) > 0 else nn.Identity()

        config = MPSAConfig(
            block_config=tuple(block_config),
            growth_rate=int(growth_rate),
            num_init_features=int(num_init_features),
            bn_size=int(bn_size),
            drop_rate=float(drop_rate),
            in_channels=1,
            num_age_classes=0,
            num_gender_classes=0,
        )
        self.backbone = MPSADenseNet(num_classes=int(num_classes), config=config)
        feat_dim = int(self.backbone.accent_head.in_features)
        hidden = max(16, int(head_hidden_dim))
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(float(head_dropout)),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(hidden, int(num_classes)),
        )

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.embedding(token_ids)  # [B, max_len, embed_dim]
        x = self.embed_dropout(x)
        x = x.transpose(1, 2).unsqueeze(1).contiguous()  # [B, 1, embed_dim, max_len]
        feat = self.backbone.forward_features(x)
        return {"accent": self.classifier_head(feat)}

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.classifier_head.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.embedding.parameters():
            yield p
        for p in self.backbone.parameters():
            yield p
