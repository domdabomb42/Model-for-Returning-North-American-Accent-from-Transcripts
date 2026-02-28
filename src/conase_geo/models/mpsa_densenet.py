from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _safe_group(channels: int, preferred: int) -> int:
    group = max(1, min(channels, preferred))
    while group > 1 and channels % group != 0:
        group -= 1
    return max(1, group)


class SEWeightModule(nn.Module):
    """Squeeze-and-Excitation weighting used by PSA branches."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return self.sigmoid(y)


class PSAModule(nn.Module):
    """
    Parallel split attention block.

    Four kernel branches process the same input, then branch attention is
    normalized with softmax across branches before re-concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int] = (3, 5, 7, 9),
        groups: Sequence[int] = (1, 2, 4, 8),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("PSAModule expects out_channels divisible by 4.")
        if len(kernels) != 4 or len(groups) != 4:
            raise ValueError("PSAModule expects exactly 4 kernels and 4 group values.")

        split_channels = out_channels // 4
        self.split_channels = split_channels
        self.branch_count = 4

        convs = []
        se_layers = []
        for k, g in zip(kernels, groups):
            group = _safe_group(split_channels, g)
            convs.append(
                nn.Conv2d(
                    in_channels,
                    split_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=group,
                    bias=False,
                )
            )
            se_layers.append(SEWeightModule(split_channels, reduction=reduction))
        self.convs = nn.ModuleList(convs)
        self.se_blocks = nn.ModuleList(se_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.convs]  # 4 x [B, C/4, H, W]
        branch_feats = torch.stack(feats, dim=1)  # [B, 4, C/4, H, W]

        attn = [se(feat) for se, feat in zip(self.se_blocks, feats)]  # 4 x [B, C/4, 1, 1]
        branch_attn = torch.stack(attn, dim=1)  # [B, 4, C/4, 1, 1]
        branch_attn = torch.softmax(branch_attn, dim=1)

        weighted = branch_feats * branch_attn
        b, _, c, h, w = weighted.shape
        return weighted.view(b, 4 * c, h, w)


class _MPSADenseLayer(nn.Module):
    """DenseNet layer where the 3x3 conv is replaced by a PSA module."""

    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, inter_channels, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.psa = PSAModule(inter_channels, growth_rate)
        self.drop_rate = float(drop_rate)

    def forward(self, prev_features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(prev_features, dim=1)
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.psa(self.relu2(self.norm2(x)))
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class _MPSADenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _MPSADenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, dim=1)


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


@dataclass(frozen=True)
class MPSAConfig:
    block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
    growth_rate: int = 32
    num_init_features: int = 64
    bn_size: int = 4
    drop_rate: float = 0.0
    in_channels: int = 2
    num_age_classes: int = 0
    num_gender_classes: int = 0


class MPSADenseNet(nn.Module):
    """
    MPSA-DenseNet style network for accent classification.

    Main output key is `accent`. Optional auxiliary heads `age` and `gender`
    are created when class counts are > 0.
    """

    def __init__(self, num_classes: int, config: MPSAConfig = MPSAConfig()) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.output_dim = self.num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(config.num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = config.num_init_features
        blocks = []
        for i, num_layers in enumerate(config.block_config):
            block = _MPSADenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=config.bn_size,
                growth_rate=config.growth_rate,
                drop_rate=config.drop_rate,
            )
            blocks.append(block)
            num_features = num_features + num_layers * config.growth_rate
            if i != len(config.block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2
        self.blocks = nn.Sequential(*blocks)
        self.norm_final = nn.BatchNorm2d(num_features)

        self.accent_head = nn.Linear(num_features, self.num_classes)
        self.age_head = nn.Linear(num_features, config.num_age_classes) if config.num_age_classes > 0 else None
        self.gender_head = nn.Linear(num_features, config.num_gender_classes) if config.num_gender_classes > 0 else None

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.forward_features(x)
        out: Dict[str, torch.Tensor] = {"accent": self.accent_head(feat)}
        if self.age_head is not None:
            out["age"] = self.age_head(feat)
        if self.gender_head is not None:
            out["gender"] = self.gender_head(feat)
        return out
