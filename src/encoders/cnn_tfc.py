"""3-block Conv1D encoder — port of TS-TCC base_Model backbone."""

from __future__ import annotations
import torch.nn as nn
from torch import Tensor
from encoders.base import Encoder


class TFCConvEncoder(Encoder):
    """3-block Conv1D with BN+ReLU+MaxPool, AdaptiveAvgPool, linear proj.

    Ported from TFC-pretraining/code/baselines/TS-TCC/models/model.py.
    in_channels set from channel_policy (1 for first_only, C for all).
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_size: int = 8,
        stride: int = 1,
        cnn_dropout: float = 0.35,
        final_out_channels: int = 128,
        d_model: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=kernel_size // 2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.Dropout(cnn_dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels), nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(final_out_channels, d_model)
        self._d = d_model

    @property
    def output_dim(self) -> int:
        return self._d

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv3(self.conv2(self.conv1(x)))   # (B, final_out_channels, T')
        x = self.pool(x).squeeze(-1)                 # (B, final_out_channels)
        return self.proj(x)                          # (B, d_model)
