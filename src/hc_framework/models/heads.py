"""Concrete ClassificationHead implementations.

Each head maps a fused representation (B, in_dim) to logits (B, num_classes).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from hc_framework.models.base import ClassificationHead


class MLPHead(ClassificationHead):
    """Two-layer MLP: Linear -> ReLU -> Dropout -> Linear, with Kaiming init."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class LinearHead(ClassificationHead):
    """Single linear layer — minimal parameter count."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: Tensor) -> Tensor:
        return self.fc(z)


class PatchTSTHFClassificationHead(ClassificationHead):
    """HuggingFace PatchTST built-in classification head.

    Expects PatchTST hidden states of shape (B, C, num_patches, d_model) and
    returns logits (B, num_classes).
    """

    needs_patchtst_hidden: bool = True

    def __init__(self, config, num_classes: int):
        super().__init__()
        # Import here to avoid transformers dependency for non-PatchTST runs
        from transformers.models.patchtst.modeling_patchtst import PatchTSTClassificationHead

        config.num_targets = num_classes
        self._head = PatchTSTClassificationHead(config=config)

    def forward(self, z: Tensor) -> Tensor:
        return self._head(z)
