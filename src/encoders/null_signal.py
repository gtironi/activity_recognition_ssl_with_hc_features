"""No-op encoder for handcrafted-only baselines."""

from __future__ import annotations

from torch import Tensor

from encoders.base import Encoder


class NullSignalEncoder(Encoder):
    """Returns a zero embedding; intended to be ignored by the model forward."""

    def __init__(self, output_dim: int = 1):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        return x.new_zeros((batch, self._output_dim))
