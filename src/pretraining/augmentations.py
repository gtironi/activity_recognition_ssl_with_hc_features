"""IMU augmentations — ported from TFC augmentations.py, CPU-safe.

All functions accept/return numpy arrays or Tensors with shape (B, C, T).
"""

from __future__ import annotations
import numpy as np
import torch


def jitter(x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
    return x * factor


def permutation(x: torch.Tensor, max_segments: int = 5) -> torch.Tensor:
    B, C, T = x.shape
    out = x.clone()
    for i in range(B):
        num_segs = np.random.randint(1, max(2, max_segments))
        if num_segs > 1:
            split_points = sorted(np.random.choice(T - 2, num_segs - 1, replace=False) + 1)
            splits = np.split(np.arange(T), split_points)
            # use explicit list-of-arrays concatenation to avoid inhomogeneous array error
            perm_order = np.random.permutation(len(splits))
            perm = np.concatenate([splits[j] for j in perm_order])
            out[i] = x[i, :, perm]
    return out


def remove_frequency(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    mask = torch.empty_like(x).uniform_() > ratio
    return x * mask


def add_frequency(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    mask = torch.empty_like(x).uniform_() > (1 - ratio)
    max_amp = x.abs().max()
    noise = torch.rand_like(x) * (max_amp * 0.1)
    return x + mask * noise


def augment_time(x: torch.Tensor, cfg) -> torch.Tensor:
    """Single strong time-domain augmentation for SimCLR/TS-TCC views."""
    x = jitter(x, sigma=cfg.jitter_sigma)
    x = scaling(x, sigma=cfg.scale_sigma)
    if np.random.random() < 0.5:
        x = permutation(x, max_segments=cfg.permutation_max_segs)
    return x


def augment_freq(x_f: torch.Tensor, cfg) -> torch.Tensor:
    """Frequency-domain augmentation."""
    return remove_frequency(x_f, cfg.freq_remove_ratio) + add_frequency(x_f, cfg.freq_add_ratio)
