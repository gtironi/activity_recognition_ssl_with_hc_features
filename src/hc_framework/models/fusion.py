"""Concrete FusionModule implementations.

Each module merges signal and hand-crafted feature embeddings into a single vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from hc_framework.models.base import FusionModule


class ConcatFusion(FusionModule):
    """Concatenation along the feature axis.

    Migrated from ``layers.fusion.ConcatFusion``.
    """

    def __init__(self, signal_dim: int, hc_dim: int):
        super().__init__()
        self._output_dim = signal_dim + hc_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z_signal: Tensor, z_hc: Tensor) -> Tensor:
        return torch.cat((z_signal, z_hc), dim=1)
