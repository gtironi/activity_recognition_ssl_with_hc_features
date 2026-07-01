from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module, ABC):
    """All encoders: (B, C_in, T) → (B, D). C_in set by channel_policy at runtime."""

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, T) → z: (B, D)"""
