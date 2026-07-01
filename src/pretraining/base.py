from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn


class PretextMethod(ABC):

    @property
    def skip_pretrain(self) -> bool:
        return False

    @abstractmethod
    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        """Returns dict with 'loss' (scalar Tensor) + optional diagnostics."""

    def parameters(self):
        """Return extra learnable parameters (projectors etc.) beyond encoder."""
        return iter([])
