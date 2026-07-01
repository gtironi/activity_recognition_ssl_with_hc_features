"""SimCLR: NT-Xent on 2 augmented views of same window."""

from __future__ import annotations
import torch
import torch.nn as nn
from pretraining.augmentations import augment_time
from pretraining.base import PretextMethod
from pretraining.losses import NTXentLoss


class SimCLRMethod(PretextMethod):
    def __init__(self, encoder_dim: int, projection_dim: int,
                 temperature: float, cfg):
        self._projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim),
        )
        self._cfg = cfg
        self._temperature = temperature
        self._proj_dim = projection_dim
        self._device = None
        self._criterion = None  # built lazily (need device + batch_size)

    def parameters(self):
        return self._projector.parameters()

    def _get_criterion(self, device, batch_size):
        if self._criterion is None or self._device != device:
            self._device = device
            self._criterion = NTXentLoss(device, batch_size, self._temperature)
        elif self._criterion.batch_size != batch_size:
            self._criterion = NTXentLoss(device, batch_size, self._temperature)
        return self._criterion

    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        x, _, _ = batch            # (B, C, T)
        device = x.device
        B = x.size(0)

        self._projector.to(device)
        v1 = augment_time(x, self._cfg)
        v2 = augment_time(x, self._cfg)

        z1 = self._projector(encoder(v1))
        z2 = self._projector(encoder(v2))

        criterion = self._get_criterion(device, B)
        loss = criterion(z1, z2)
        return {"loss": loss}
