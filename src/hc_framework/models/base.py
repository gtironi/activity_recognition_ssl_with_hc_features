"""Abstract base classes for the modular hybrid architecture.

Four pluggable components define the hybrid pipeline:

    x_signal  --> [SignalEncoder]      --+
                                          +--> [FusionModule] --> [ClassificationHead] --> logits
    x_features --> [HandcraftedBranch] --+

Each ABC requires an ``output_dim`` property so that downstream modules can
be constructed with the correct input size without performing a forward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from encoders.base import Encoder

# All concrete encoders (shared + hybrid-only) live in ``encoders`` and implement
# this same (B, C, T) -> (B, output_dim) interface.
SignalEncoder = Encoder


class HandcraftedBranch(nn.Module, ABC):
    """Projects pre-computed hand-crafted features into a fixed-size embedding.

    Input:  (B, n_features).
    Output: (B, output_dim).
    """

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(self, x_features: Tensor) -> Tensor: ...


class FusionModule(nn.Module, ABC):
    """Merges signal and hand-crafted feature embeddings into a single representation.

    Input:  z_signal (B, enc_dim), z_hc (B, hc_dim).
    Output: (B, output_dim).
    """

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(self, z_signal: Tensor, z_hc: Tensor) -> Tensor: ...


class ClassificationHead(nn.Module, ABC):
    """Maps a fused (or encoder-only) representation to class logits.

    Input:  (B, in_dim).
    Output: (B, num_classes).
    """

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor: ...
