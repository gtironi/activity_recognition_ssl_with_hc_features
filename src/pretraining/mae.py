"""MAE pretraining via HF PatchTSTForPretraining / PatchTSMixerForPretraining."""

from __future__ import annotations
import torch
import torch.nn as nn
from pretraining.base import PretextMethod


def _best_patch_length(T: int, requested: int) -> int:
    pl = min(requested, T)
    while pl > 1 and T % pl != 0:
        pl -= 1
    return pl


class MAEMethod(PretextMethod):
    """Wraps HF ForPretraining models. Encoder passed in to share config.

    After pretraining, call extract_encoder() to get PatchTSTEncoder /
    PatchTSMixerEncoder with backbone weights loaded.
    """

    def __init__(self, encoder, cfg):
        self._encoder_ref = encoder   # reference only; not trained via this method
        self._cfg = cfg
        self._backbone_name = type(encoder).__name__

        C_in = None
        ctx = None
        # probe encoder config
        if hasattr(encoder, "_backbone"):
            bc = encoder._backbone.config
            C_in = bc.num_input_channels
            ctx  = bc.context_length

        assert C_in is not None and ctx is not None, \
            "MAEMethod only works with PatchTSTEncoder or PatchTSMixerEncoder"

        pl = _best_patch_length(ctx, cfg.patch_length if hasattr(cfg, "patch_length") else 8)

        if "PatchTST" in self._backbone_name:
            from transformers import PatchTSTConfig, PatchTSTForPretraining
            bc = encoder._backbone.config
            pretrain_config = PatchTSTConfig(
                num_input_channels=C_in, context_length=ctx,
                patch_length=pl, patch_stride=pl,
                d_model=bc.d_model, num_attention_heads=bc.num_attention_heads,
                num_hidden_layers=bc.num_hidden_layers,
                dropout=bc.dropout, mask_type="random",
                random_mask_ratio=cfg.mask_ratio,
            )
            self._model = PatchTSTForPretraining(pretrain_config)
            # share backbone weights: ForPretraining.model.encoder ↔ PatchTSTModel
            self._model.model.load_state_dict(encoder._backbone.state_dict(), strict=False)
        else:
            from transformers import PatchTSMixerConfig, PatchTSMixerForPretraining
            bc = encoder._backbone.config
            pretrain_config = PatchTSMixerConfig(
                context_length=ctx, patch_length=pl, patch_stride=pl,
                num_input_channels=C_in, d_model=bc.d_model,
                num_layers=bc.num_layers, dropout=bc.dropout,
                mask_ratio=cfg.mask_ratio,
            )
            self._model = PatchTSMixerForPretraining(pretrain_config)

        self._ctx = ctx
        self._C_in = C_in

    def parameters(self):
        return self._model.parameters()

    def pretrain_step(self, encoder: nn.Module, batch: tuple) -> dict:
        x, _, _ = batch                    # (B, C_in, T)
        self._model = self._model.to(x.device)
        out = self._model(past_values=x.permute(0, 2, 1))   # HF expects (B, T, C)
        return {"loss": out.loss}

    def extract_encoder(self):
        """Return encoder with backbone weights from pretrained ForPretraining model."""
        enc = self._encoder_ref
        if "PatchTST" in self._backbone_name:
            enc._backbone.load_state_dict(self._model.model.state_dict(), strict=False)
        else:
            # PatchTSMixer: try to copy backbone
            try:
                enc._backbone.load_state_dict(
                    {k: v for k, v in self._model.state_dict().items()
                     if not k.startswith("decoder")}, strict=False)
            except Exception:
                pass
        return enc
