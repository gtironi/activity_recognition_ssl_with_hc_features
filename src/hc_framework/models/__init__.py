"""Modular hybrid architecture — factory and public API.

Usage::

    from hc_framework.models import build_hybrid_model

    model = build_hybrid_model(
        encoder_name="robust",
        input_mode="hybrid",
        num_classes=19,
        n_hc_feats=120,
    )
"""

from __future__ import annotations

from encoders import (
    CNNLSTMEncoder,
    NullSignalEncoder,
    ResNet1DEncoder,
    RobustCNNLSTMEncoder,
    TFCConvEncoder,
)
from hc_framework.models.fusion import ConcatFusion
from hc_framework.models.hc_branches import MLPHandcraftedBranch
from hc_framework.models.heads import LinearHead, MLPHead, PatchTSTHFClassificationHead
from hc_framework.models.model import HybridModel

_ENCODER_REGISTRY: dict[str, type] = {
    "cnn_lstm": CNNLSTMEncoder,
    "robust": RobustCNNLSTMEncoder,
    "cnn_tfc": TFCConvEncoder,
    "resnet1d": ResNet1DEncoder,
}

_HC_BRANCH_REGISTRY: dict[str, type] = {
    "mlp": MLPHandcraftedBranch,
}


def build_hybrid_model(
    encoder_name: str,
    input_mode: str = "hybrid",
    num_classes: int = 19,
    n_hc_feats: int = 120,
    in_channels: int = 3,
    head_name: str = "mlp",
    head_hidden_dim: int = 256,
    head_dropout: float = 0.4,
    hc_branch_name: str = "mlp",
    hc_hidden_dim: int | None = None,
    hc_dropout: float = 0.3,
    **encoder_kwargs,
) -> HybridModel:
    """Build a HybridModel from component names.

    Parameters
    ----------
    encoder_name : str
        ``"cnn_lstm"`` | ``"robust"`` | ``"cnn_tfc"`` | ``"resnet1d"`` |
        ``"patchtst"`` | ``"patchtsmixer"`` | ``"hc_mlp"``.
    input_mode : str
        ``"deep_only"`` | ``"hybrid"`` | ``"handcrafted_only"``.
    num_classes : int
        Number of output classes.
    n_hc_feats : int
        Number of pre-computed hand-crafted features (ignored in deep_only mode).
    head_hidden_dim : int
        Hidden dimension of the MLPHead.
    head_dropout : float
        Dropout rate in the MLPHead.
    hc_branch_name : str
        ``"mlp"`` (identity pass-through). Default ``"mlp"``.
    hc_hidden_dim : int | None
        Hidden dimension of the hand-crafted feature branch.  Defaults to ``encoder.output_dim``.
    hc_dropout : float
        Dropout rate in the hand-crafted feature branch.
    **encoder_kwargs
        Extra keyword arguments forwarded to the encoder constructor.
    """
    # Build encoder
    if encoder_name == "hc_mlp":
        if input_mode != "handcrafted_only":
            raise ValueError("hc_mlp requires input_mode='handcrafted_only'.")
        encoder = NullSignalEncoder()
    elif encoder_name == "patchtst":
        from encoders import PatchTSTEncoder

        encoder_kwargs.setdefault("in_channels", in_channels)
        encoder = PatchTSTEncoder(**encoder_kwargs)
    elif encoder_name == "patchtsmixer":
        from encoders import PatchTSMixerEncoder

        encoder_kwargs.setdefault("in_channels", in_channels)
        encoder = PatchTSMixerEncoder(**encoder_kwargs)
    elif encoder_name in _ENCODER_REGISTRY:
        encoder_kwargs.setdefault("in_channels", in_channels)
        encoder = _ENCODER_REGISTRY[encoder_name](**encoder_kwargs)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name!r}. "
            f"Available: {sorted(list(_ENCODER_REGISTRY) + ['patchtst', 'patchtsmixer', 'hc_mlp'])}"
        )

    if hc_branch_name not in _HC_BRANCH_REGISTRY:
        raise ValueError(
            f"Unknown hc_branch_name: {hc_branch_name!r}. "
            f"Available: {sorted(_HC_BRANCH_REGISTRY)}"
        )
    HCBranchCls = _HC_BRANCH_REGISTRY[hc_branch_name]

    # Build optional hand-crafted feature branch + fusion
    hc_branch = None
    fusion = None
    if input_mode == "hybrid":
        hc_hidden = hc_hidden_dim if hc_hidden_dim is not None else encoder.output_dim
        hc_branch = HCBranchCls(n_hc_feats, hc_hidden, dropout=hc_dropout)
        fusion = ConcatFusion(encoder.output_dim, hc_branch.output_dim)
        head_in_dim = fusion.output_dim
    elif input_mode == "handcrafted_only":
        hc_hidden = hc_hidden_dim if hc_hidden_dim is not None else n_hc_feats
        hc_branch = HCBranchCls(n_hc_feats, hc_hidden, dropout=hc_dropout)
        head_in_dim = hc_branch.output_dim
    elif input_mode == "deep_only":
        head_in_dim = encoder.output_dim
    else:
        raise ValueError(
            f"Unknown input_mode: {input_mode!r}. Use 'deep_only', 'hybrid', or 'handcrafted_only'."
        )

    if head_name == "mlp":
        head = MLPHead(head_in_dim, head_hidden_dim, num_classes, dropout=head_dropout)
    elif head_name == "linear":
        head = LinearHead(head_in_dim, num_classes)
    elif head_name == "patchtst_hf":
        if encoder_name != "patchtst" or input_mode != "deep_only":
            raise ValueError("head_name='patchtst_hf' requires encoder_name='patchtst' and input_mode='deep_only'.")
        # Reuse the same config already inside the encoder backbone
        cfg = encoder._backbone.config  # noqa: SLF001 (intentional: minimal wiring)
        head = PatchTSTHFClassificationHead(cfg, num_classes)
    else:
        raise ValueError("Unknown head_name. Use 'mlp', 'linear', or 'patchtst_hf'.")

    return HybridModel(encoder, hc_branch, fusion, head, input_mode)
