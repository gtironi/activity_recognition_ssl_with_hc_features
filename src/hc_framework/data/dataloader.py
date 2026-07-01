from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CalfHybridDataset(Dataset):
    def __init__(self, signals: np.ndarray, features: np.ndarray, labels: np.ndarray):
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]
