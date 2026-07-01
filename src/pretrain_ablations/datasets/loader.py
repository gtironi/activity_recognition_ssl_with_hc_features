"""Canonical .pt loader with channel_policy + per-channel z-score normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.registry import load_registry, resolve_processed_root


def apply_channel_policy(samples: torch.Tensor, policy: str) -> torch.Tensor:
    """samples: (N, C, T). Returns sliced tensor based on policy."""
    if policy == "first_only":
        return samples[:, :1, :]
    if policy == "all":
        return samples
    if policy.startswith("first_n:"):
        k = int(policy.split(":")[1])
        return samples[:, :k, :]
    raise ValueError(f"Unknown channel_policy={policy!r}")


def apply_resample(samples: torch.Tensor, resample_t: int) -> torch.Tensor:
    """samples: (N, C, T). Linearly resample the time axis to `resample_t` (no-op if 0 or already T)."""
    if resample_t <= 0 or samples.shape[-1] == resample_t:
        return samples
    return F.interpolate(samples, size=resample_t, mode="linear", align_corners=False)


class CanonicalDataset(Dataset):
    """Holds (samples, labels, groups) tensors with optional normalization."""

    def __init__(
        self,
        samples: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        self.samples = samples
        self.labels = labels
        self.groups = groups
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = self.samples[idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)
        return x, self.labels[idx], self.groups[idx]


def _load_split(processed_root: Path, split: str, channel_policy: str, resample_t: int = 0
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    pt_path = processed_root / f"{split}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing canonical .pt: {pt_path}")
    d = torch.load(pt_path, weights_only=False)
    samples = d["samples"].float()
    samples = apply_channel_policy(samples, channel_policy)
    samples = apply_resample(samples, resample_t)
    labels = d["labels"].long()
    groups = d["groups"].long()
    meta = d.get("meta", {})
    return samples, labels, groups, meta


def build_dataloaders(data_cfg) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train/val/test loaders and a meta dict.

    meta = {
        "num_classes": int,
        "num_channels_after_policy": int (= C_in for encoder),
        "window_len": int (= T),
        "class_names": list[str],
        "id2label": dict,
        "channel_names": list[str],
        "dataset_id": str,
    }
    """
    registry = load_registry(data_cfg.registry_path)
    info = registry[data_cfg.dataset_id]

    if info.get("task") == "forecasting_sanity":
        raise ValueError(
            f"dataset_id={data_cfg.dataset_id} has task=forecasting_sanity. "
            "Cannot use in classification pipeline."
        )

    processed_root = resolve_processed_root(data_cfg.dataset_id, registry)

    resample_t = getattr(data_cfg, "resample_t", 0)
    train_x, train_y, train_g, train_meta = _load_split(processed_root, "train", data_cfg.channel_policy, resample_t)
    val_x, val_y, val_g, _ = _load_split(processed_root, "val", data_cfg.channel_policy, resample_t)
    test_x, test_y, test_g, _ = _load_split(processed_root, "test", data_cfg.channel_policy, resample_t)

    # Normalize per channel using train stats
    if data_cfg.normalize:
        # samples shape: (N, C, T) → mean/std over (N, T) per channel
        mean = train_x.mean(dim=(0, 2), keepdim=True).squeeze(0)  # (C, 1)
        std = train_x.std(dim=(0, 2), keepdim=True).squeeze(0)    # (C, 1)
    else:
        mean = std = None

    train_ds = CanonicalDataset(train_x, train_y, train_g, mean, std)
    val_ds = CanonicalDataset(val_x, val_y, val_g, mean, std)
    test_ds = CanonicalDataset(test_x, test_y, test_g, mean, std)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True,
                          num_workers=data_cfg.num_workers, pin_memory=pin,
                          drop_last=data_cfg.batch_size > 1)
    val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False,
                        num_workers=data_cfg.num_workers, pin_memory=pin)
    test_dl = DataLoader(test_ds, batch_size=data_cfg.batch_size, shuffle=False,
                         num_workers=data_cfg.num_workers, pin_memory=pin)

    # Build meta
    id2label = train_meta.get("id2label", {})
    num_classes = len(id2label) if id2label else int(train_y.max().item()) + 1
    class_names = [id2label.get(str(i), str(i)) for i in range(num_classes)]
    channel_names = train_meta.get("channel_names", [])
    if data_cfg.channel_policy == "first_only":
        channel_names = channel_names[:1]
    elif data_cfg.channel_policy.startswith("first_n:"):
        k = int(data_cfg.channel_policy.split(":")[1])
        channel_names = channel_names[:k]

    _, C_in, T = train_x.shape
    meta = {
        "num_classes": num_classes,
        "num_channels_after_policy": C_in,
        "window_len": T,
        "class_names": class_names,
        "id2label": id2label,
        "channel_names": channel_names,
        "dataset_id": data_cfg.dataset_id,
    }
    return train_dl, val_dl, test_dl, meta
