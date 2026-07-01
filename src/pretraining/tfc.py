"""TFC (Time-Frequency Consistency) pretraining method.

Owns 2 encoders (time + frequency). Ported from TFC-pretraining/code/TFC/.
Loss: lam*(loss_t + loss_f) + l_TF   [from trainer.py lines 119-137]
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pretraining.augmentations import augment_freq, augment_time
from pretraining.base import PretextMethod
from pretraining.losses import NTXentLoss


class TFCMethod(PretextMethod):
    def __init__(self, encoder_factory, projection_dim: int,
                 temperature: float, lam: float, cfg):
        self.encoder_t = encoder_factory()   # time branch
        self.encoder_f = encoder_factory()   # frequency branch
        D = self.encoder_t.output_dim
        self.projector_t = nn.Sequential(
            nn.Linear(D, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, projection_dim),
        )
        self.projector_f = nn.Sequential(
            nn.Linear(D, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, projection_dim),
        )
        self._lam = lam
        self._temperature = temperature
        self._proj_dim = projection_dim
        self._cfg = cfg
        self._criterion = None
        self._device = None

    def get_time_encoder(self):
        return self.encoder_t

    def parameters(self):
        params = (list(self.encoder_f.parameters()) +
                  list(self.projector_t.parameters()) +
                  list(self.projector_f.parameters()))
        return iter(params)

    def _get_criterion(self, device, batch_size):
        if self._criterion is None or self._device != device:
            self._device = device
            self._criterion = NTXentLoss(device, batch_size, self._temperature)
        elif self._criterion.batch_size != batch_size:
            self._criterion = NTXentLoss(device, batch_size, self._temperature)
        return self._criterion

    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        # encoder arg = encoder_t (already set, same object)
        x_t, _, _ = batch                               # (B, C, T)
        device = x_t.device
        B = x_t.size(0)

        self.encoder_f.to(device)
        self.projector_t.to(device)
        self.projector_f.to(device)

        x_f = torch.fft.fft(x_t).abs()                 # (B, C, T) frequency magnitude

        aug_t = augment_time(x_t, self._cfg)
        aug_f = augment_freq(x_f, self._cfg)

        h_t  = encoder(x_t);            z_t  = self.projector_t(h_t)
        h_f  = self.encoder_f(x_f);     z_f  = self.projector_f(h_f)
        h_ta = encoder(aug_t);           z_ta = self.projector_t(h_ta)
        h_fa = self.encoder_f(aug_f);    z_fa = self.projector_f(h_fa)

        crit = self._get_criterion(device, B)

        loss_t  = crit(h_t, h_ta)
        loss_f  = crit(h_f, h_fa)
        l_TF    = crit(z_t, z_f)
        # triplet-style consistency losses
        l_1 = crit(z_t, z_fa); l_2 = crit(z_ta, z_f); l_3 = crit(z_ta, z_fa)
        loss_c  = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        loss = self._lam * (loss_t + loss_f) + l_TF

        return {"loss": loss, "loss_t": loss_t, "loss_f": loss_f,
                "l_TF": l_TF, "loss_c": loss_c}


def build_tfc_method(pretext_cfg, encoder_cfg, in_channels, window_len, device):
    from encoders import build_encoder

    def factory():
        return build_encoder(encoder_cfg, in_channels=in_channels,
                              context_length=window_len)

    return TFCMethod(
        encoder_factory=factory,
        projection_dim=pretext_cfg.projection_dim,
        temperature=pretext_cfg.temperature,
        lam=pretext_cfg.tfc_lambda,
        cfg=pretext_cfg,
    )
