"""HuggingFace PatchTST encoder wrapper."""

from __future__ import annotations

import torch
from torch import Tensor

from encoders.base import Encoder


def _best_patch_length(T: int, requested: int) -> int:
    pl = min(requested, T)
    while pl > 1 and T % pl != 0:
        pl -= 1
    return pl


class PatchTSTEncoder(Encoder):
    """Wraps HF ``PatchTSTModel``. Permutes ``(B, C, T)`` -> ``(B, T, C)`` internally
    (HF expects channels-last). Mean-pools over channels/patches -> ``(B, d_model)``.

    If ``pretrained_path`` is given, loads encoder weights saved from a
    ``PatchTSTForPretraining`` checkpoint (strips the ``"model."`` prefix).
    """

    def __init__(
        self,
        context_length: int = 75,
        in_channels: int = 3,
        patch_length: int = 12,
        patch_stride: int = 12,
        d_model: int = 128,
        num_heads: int = 16,
        num_layers: int = 3,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
        ffn_dim: int = 512,
        revin: bool = True,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        from transformers import PatchTSTConfig, PatchTSTModel

        pl = _best_patch_length(context_length, patch_length)
        ps = min(patch_stride, pl)

        config = PatchTSTConfig(
            num_input_channels=in_channels,
            context_length=context_length,
            patch_length=pl,
            patch_stride=ps,
            d_model=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=head_dropout,
            revin=revin,
            channel_attention=False,
        )
        self._backbone = PatchTSTModel(config)
        self._d_model = d_model

        if pretrained_path is not None:
            self.load_pretrained_encoder(pretrained_path)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, T) -> (B, T, C) — HuggingFace expects channels-last
        out = self._backbone(past_values=x.permute(0, 2, 1))
        # last_hidden_state: (B, num_channels, num_patches, d_model)
        return out.last_hidden_state.mean(dim=(1, 2))

    def forward_hidden(self, x: Tensor) -> Tensor:
        """Return PatchTST last_hidden_state for HF-style classification heads.

        Shape: (B, C, num_patches, d_model)
        """
        out = self._backbone(past_values=x.permute(0, 2, 1))
        return out.last_hidden_state

    def apply_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> None:
        """Wrap the backbone with LoRA adapters (base weights frozen, only adapters train)."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=list(target_modules), bias="none",
        )
        self._backbone = get_peft_model(self._backbone, config)

    def load_pretrained_encoder(self, path: str) -> None:
        """Load backbone weights from a PatchTSTForPretraining checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        encoder_state = {
            k[len("model."):]: v
            for k, v in sd.items()
            if k.startswith("model.")
        }
        self._backbone.load_state_dict(encoder_state or sd, strict=False)
