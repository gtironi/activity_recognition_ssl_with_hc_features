"""HuggingFace PatchTSMixer encoder wrapper."""

from __future__ import annotations

import torch
from torch import Tensor

from encoders.base import Encoder


def _best_patch_length(T: int, requested: int) -> int:
    pl = min(requested, T)
    while pl > 1 and T % pl != 0:
        pl -= 1
    return pl


class PatchTSMixerEncoder(Encoder):
    """HuggingFace PatchTSMixer wrapper. Permutes (B, C, T) -> (B, T, C) internally."""

    def __init__(
        self,
        context_length: int = 75,
        in_channels: int = 3,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from transformers import PatchTSMixerConfig, PatchTSMixerModel

        pl = _best_patch_length(context_length, patch_length)
        ps = min(patch_stride, pl)
        config = PatchTSMixerConfig(
            context_length=context_length,
            patch_length=pl,
            patch_stride=ps,
            num_input_channels=in_channels,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
        self._backbone = PatchTSMixerModel(config)
        self._output_dim = d_model

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            sd = state.get("model_state_dict", state)
            self._backbone.load_state_dict(sd, strict=False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        out = self._backbone(past_values=x.permute(0, 2, 1))
        return out.last_hidden_state.mean(dim=(1, 2))
