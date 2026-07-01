"""TS-TCC: contextual contrastive on weak/strong augmented views."""

from __future__ import annotations
import torch
import torch.nn as nn
from pretraining.augmentations import augment_time
from pretraining.base import PretextMethod
from pretraining.losses import NTXentLoss


class TSTCCMethod(PretextMethod):
    """Contextual Contrastive only (CC) — no temporal contrasting (full TC
    requires sequence-level encoder output, out of scope for v1)."""

    def __init__(self, encoder_dim: int, cfg):
        self._cfg = cfg
        self._proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self._criterion_cc = None
        self._device = None

    def parameters(self):
        return self._proj.parameters()

    def _get_criterion(self, device, batch_size):
        if self._criterion_cc is None or self._device != device:
            self._device = device
            # temperature 0.2 for CC (from TS-TCC paper)
            self._criterion_cc = NTXentLoss(device, batch_size, temperature=0.2)
        elif self._criterion_cc.batch_size != batch_size:
            self._criterion_cc = NTXentLoss(device, batch_size, temperature=0.2)
        return self._criterion_cc

    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        x, _, _ = batch
        device = x.device
        B = x.size(0)
        self._proj.to(device)

        x_weak   = augment_time(x, self._cfg)
        x_strong = augment_time(augment_time(x, self._cfg), self._cfg)

        z_w = self._proj(encoder(x_weak))
        z_s = self._proj(encoder(x_strong))

        crit = self._get_criterion(device, B)
        loss_cc = crit(z_w, z_s)
        return {"loss": loss_cc, "loss_cc": loss_cc}
