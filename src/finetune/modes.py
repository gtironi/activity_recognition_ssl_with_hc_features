"""apply_finetune_mode → optimizer param_groups."""

from __future__ import annotations
import torch.nn as nn


def apply_finetune_mode(
    encoder: nn.Module,
    head: nn.Module,
    mode: str,
    lr: float,
    encoder_lr_factor: float = 0.1,
    partial_k: int = 1,
) -> list[dict]:
    """Returns param_groups for optimizer."""
    if mode in ("freeze", "linear_probe"):
        for p in encoder.parameters():
            p.requires_grad = False
        return [{"params": list(head.parameters()), "lr": lr}]

    if mode == "full":
        for p in encoder.parameters():
            p.requires_grad = True
        return [{"params": list(encoder.parameters()) + list(head.parameters()), "lr": lr}]

    if mode == "partial_k":
        for p in encoder.parameters():
            p.requires_grad = False
        blocks = list(encoder.named_children())
        for _, module in blocks[-partial_k:]:
            for p in module.parameters():
                p.requires_grad = True
        unfrozen = [p for p in encoder.parameters() if p.requires_grad]
        return [{"params": unfrozen, "lr": lr},
                {"params": list(head.parameters()), "lr": lr}]

    if mode == "discriminative_lr":
        for p in encoder.parameters():
            p.requires_grad = True
        return [{"params": list(encoder.parameters()), "lr": lr * encoder_lr_factor},
                {"params": list(head.parameters()), "lr": lr}]

    raise ValueError(f"Unknown finetune mode={mode!r}")
