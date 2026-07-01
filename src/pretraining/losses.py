"""NT-Xent loss — ported from TFC-pretraining/code/TFC/loss.py:NTXentLoss_poly.

CUDA-specific calls replaced with device-agnostic equivalents.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalised temperature-scaled cross-entropy (NT-Xent / InfoNCE).

    Ported from TFC NTXentLoss_poly; device-agnostic.
    """

    def __init__(self, device: torch.device, batch_size: int,
                 temperature: float = 0.2, use_cosine_similarity: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        if use_cosine_similarity:
            self._cos = nn.CosineSimilarity(dim=-1)
            self._sim_fn = self._cosine_sim
        else:
            self._sim_fn = self._dot_sim
        # build mask once (reuse across calls)
        self._mask = self._build_mask(batch_size, device)

    @staticmethod
    def _build_mask(bs: int, device: torch.device) -> torch.Tensor:
        diag = np.eye(2 * bs)
        l1 = np.eye(2 * bs, 2 * bs, k=-bs)
        l2 = np.eye(2 * bs, 2 * bs, k=bs)
        mask = torch.from_numpy((diag + l1 + l2)).bool()
        return (~mask).to(device)

    def _cosine_sim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._cos(x.unsqueeze(1), y.unsqueeze(0))

    @staticmethod
    def _dot_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        B = zis.size(0)
        if B != self.batch_size:
            # rebuild mask for this batch size (last batch may be smaller)
            mask = self._build_mask(B, self.device)
        else:
            mask = self._mask

        reps = torch.cat([zjs, zis], dim=0)          # (2B, D)
        sim = self._sim_fn(reps, reps)                # (2B, 2B)

        l_pos = torch.diag(sim, B)
        r_pos = torch.diag(sim, -B)
        positives = torch.cat([l_pos, r_pos]).view(2 * B, 1)
        negatives = sim[mask].view(2 * B, -1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * B, device=self.device).long()
        CE = self.criterion(logits, labels)

        # poly loss correction (from TFC)
        onehot = torch.cat([torch.ones(2 * B, 1),
                             torch.zeros(2 * B, negatives.shape[-1])], dim=-1).to(self.device).long()
        pt = torch.mean(onehot * F.softmax(logits, dim=-1))
        loss = CE / (2 * B) + B * (1.0 / B - pt)
        return loss
