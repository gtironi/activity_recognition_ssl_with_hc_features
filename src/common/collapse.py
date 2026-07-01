"""kNN collapse heuristic for representation quality during pretraining."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


MAX_SAMPLES = 5000


def knn_collapse_heuristic(
    encoder: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    k: int = 5,
    max_samples: int = MAX_SAMPLES,
    seed: int = 42,
) -> float:
    """kNN fit-and-score on fixed subsample of val embeddings.

    Returns value in [0,1]. Near 1/num_classes → likely collapsed.
    This is a HEURISTIC, not a generalisation metric.
    """
    encoder.eval()
    embs, labs = [], []
    n = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            if n >= max_samples:
                break
            z = encoder(x.to(device)).cpu().numpy()
            embs.append(z); labs.append(y.numpy())
            n += len(z)

    Z = np.concatenate(embs)[:max_samples]
    y = np.concatenate(labs)[:max_samples]

    if len(Z) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Z), max_samples, replace=False)
        Z, y = Z[idx], y[idx]

    from sklearn.neighbors import KNeighborsClassifier
    k_actual = min(k, len(Z) - 1)
    neigh = KNeighborsClassifier(n_neighbors=k_actual)
    neigh.fit(Z, y)
    return float(neigh.score(Z, y))
