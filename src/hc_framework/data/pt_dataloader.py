"""Hybrid dataloader backed by the canonical `dataset/processed/<id>/{train,val,test}.pt`
tensors (same files consumed by `pretrain_ablations`), instead of windowed parquets.

This is what lets the hybrid model reuse SSL encoder checkpoints (per-dataset or pooled)
that were pretrained on the canonical tensors: `channel_policy` and `resample_t` reshape
the signals to match the checkpoint's expected (C, T), while hand-crafted features
come from the `.pt`'s `features` tensor (added by `scripts/core/pretrain_export.py`).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from hc_framework.data.dataloader import CalfHybridDataset
from pretrain_ablations.datasets.loader import apply_channel_policy, apply_resample
from common.registry import load_registry, resolve_processed_root


def _load_pt_split(processed_root, split: str, channel_policy: str, resample_t: int):
    d = torch.load(processed_root / f"{split}.pt", weights_only=False)
    samples = apply_channel_policy(d["samples"].float(), channel_policy)
    samples = apply_resample(samples, resample_t)
    features = d.get("features")
    if features is None:
        features = torch.zeros((samples.shape[0], 0), dtype=torch.float32)
    return samples, features.float(), d["labels"].long(), d.get("meta", {})


def prepare_train_val_test_loaders_from_pt(
    dataset_id: str,
    channel_policy: str = "all",
    resample_t: int = 0,
    batch_size: int = 64,
    num_workers: int = 2,
    registry_path: str = "dataset_registry.yaml",
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, int, int, int, None]:
    """Build train/val/test loaders from `dataset/processed/<dataset_id>/{train,val,test}.pt`.

    Returns:
    (train_dl, val_dl, test_dl, class_names, num_classes, n_feats, in_channels, None).
    """
    registry = load_registry(registry_path)
    processed_root = resolve_processed_root(dataset_id, registry)

    sig_tr, feat_tr, y_tr, meta = _load_pt_split(processed_root, "train", channel_policy, resample_t)
    sig_val, feat_val, y_val, _ = _load_pt_split(processed_root, "val", channel_policy, resample_t)
    sig_te, feat_te, y_te, _ = _load_pt_split(processed_root, "test", channel_policy, resample_t)

    # Per-channel z-score using train stats (matches pretrain_ablations.datasets.loader).
    mean = sig_tr.mean(dim=(0, 2), keepdim=True)
    std = sig_tr.std(dim=(0, 2), keepdim=True)

    def _norm(x: torch.Tensor) -> np.ndarray:
        return ((x - mean) / (std + 1e-6)).numpy()

    sig_tr_n, sig_val_n, sig_te_n = _norm(sig_tr), _norm(sig_val), _norm(sig_te)

    n_feats = feat_tr.shape[1]
    if n_feats > 0:
        scaler = StandardScaler().fit(feat_tr.numpy())
        feat_tr_n = scaler.transform(feat_tr.numpy())
        feat_val_n = scaler.transform(feat_val.numpy())
        feat_te_n = scaler.transform(feat_te.numpy())
    else:
        feat_tr_n, feat_val_n, feat_te_n = feat_tr.numpy(), feat_val.numpy(), feat_te.numpy()

    id2label = meta.get("id2label", {})
    num_classes = len(id2label) if id2label else int(y_tr.max().item()) + 1
    class_names = np.array([id2label.get(str(i), str(i)) for i in range(num_classes)])
    in_channels = sig_tr_n.shape[1]

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        CalfHybridDataset(sig_tr_n, feat_tr_n, y_tr.numpy()),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin,
    )
    val_dl = DataLoader(
        CalfHybridDataset(sig_val_n, feat_val_n, y_val.numpy()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin,
    )
    test_dl = DataLoader(
        CalfHybridDataset(sig_te_n, feat_te_n, y_te.numpy()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin,
    )

    return train_dl, val_dl, test_dl, class_names, num_classes, n_feats, in_channels, None
