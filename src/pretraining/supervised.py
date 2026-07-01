"""SupervisedBaseline: no pretraining, skip_pretrain=True."""

from __future__ import annotations
import torch
import torch.nn as nn
from pretraining.base import PretextMethod


class SupervisedBaseline(PretextMethod):
    @property
    def skip_pretrain(self) -> bool:
        return True

    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        return {"loss": torch.tensor(0.0)}
