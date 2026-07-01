"""1D ResNet encoder."""

from __future__ import annotations
import torch.nn as nn
from torch import Tensor
from encoders.base import Encoder


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class ResNet1DEncoder(Encoder):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 3,
        d_model: int = 128,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock(base_channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(base_channels, d_model)
        self._d = d_model

    @property
    def output_dim(self) -> int:
        return self._d

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.blocks(self.stem(x))).squeeze(-1)
        return self.proj(x)
