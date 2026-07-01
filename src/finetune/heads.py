"""Classification heads: LinearProbe + MLPHead."""

from __future__ import annotations
import torch.nn as nn
from torch import Tensor


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: Tensor) -> Tensor:
        return self.fc(z)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        curr = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(curr, hidden_dim), nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            curr = hidden_dim
        layers.append(nn.Linear(curr, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def build_head(cfg, in_dim: int, num_classes: int) -> nn.Module:
    if cfg.name == "linear":
        return LinearProbe(in_dim, num_classes)
    if cfg.name == "mlp":
        return MLPHead(in_dim, cfg.hidden_dim, num_classes, cfg.num_layers, cfg.dropout)
    raise ValueError(f"Unknown head name={cfg.name!r}")
