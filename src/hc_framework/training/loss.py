from __future__ import annotations

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Weights aligned to indices 0..num_classes-1 (absent classes → weight 1.0)."""
    labels = labels.astype(np.int64)
    present = np.unique(labels)
    raw = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(num_classes, dtype=np.float32)
    for i, c in enumerate(present):
        full[c] = raw[i]
    return torch.as_tensor(full, dtype=torch.float32)
