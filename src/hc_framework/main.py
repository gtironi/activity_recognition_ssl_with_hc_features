"""
Entry point for supervised, fine-tune, and test experiments.

Usage from repository root:
  PYTHONPATH=src python -m hc_framework.main --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from hc_framework.models import build_hybrid_model
from hc_framework.training.evaluation_report import save_test_evaluation_artifacts
from hc_framework.training.metrics import classification_metrics_numpy
from hc_framework.training.trainer import Trainer
from hc_framework.utils.logging import setup_logging
from hc_framework.utils.repro import set_seed

logger = logging.getLogger(__name__)


def _ensure_src_on_path():
    here = Path(__file__).resolve()
    src_root = here.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Activity Recognition — experiments CLI")

    # --- Mode & model ---
    p.add_argument("--mode", choices=("supervised", "finetune", "test"), required=True)
    p.add_argument(
        "--model",
        type=str,
        default="robust",
        help="Encoder name: cnn_lstm | robust | patchtst | hc_mlp",
    )
    p.add_argument(
        "--input_mode",
        choices=("deep_only", "hybrid", "handcrafted_only"),
        default="hybrid",
        help=(
            "deep_only = encoder → head; hybrid = encoder + handcrafted → fusion → head; "
            "handcrafted_only = handcrafted → head"
        ),
    )

    # --- Data ---
    p.add_argument(
        "--from_pt", type=str, required=True,
        help=(
            "dataset_id to load from dataset/processed/<id>/{train,val,test}.pt. "
            "Use with --channel_policy/--resample_t to match SSL encoder checkpoints."
        ),
    )
    p.add_argument("--channel_policy", type=str, default="all",
                   help="--from_pt: 'all' | 'first_only' | 'first_n:K'.")
    p.add_argument("--resample_t", type=int, default=0,
                   help="--from_pt: resample window length to T (0 = keep native).")
    p.add_argument("--registry_path", type=str, default="dataset_registry.yaml",
                   help="--from_pt: path to dataset_registry.yaml.")

    # --- Checkpoints ---
    p.add_argument("--checkpoint", type=str, default="", help="Resume supervised training from this checkpoint.")
    p.add_argument("--patchtst_checkpoint", type=str, default="", help="Pretrained PatchTST checkpoint to load.")
    p.add_argument("--output_dir", type=str, default="experiments/runs")

    # --- Training hyperparameters ---
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (if omitted, uses mode default)")
    p.add_argument("--hidden_lstm", type=int, default=None)
    p.add_argument("--no_class_weights", action="store_true", help="Supervised: disable class balancing")
    p.add_argument("--init_encoder_from", type=str, default="",
                   help="Load only encoder weights from this checkpoint (e.g. TS2Vec pretrain). Skipped if empty.")
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Supervised/finetune: freeze signal encoder (train head / handcrafted / fusion only).",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="Wrap the PatchTST encoder with LoRA adapters (--model patchtst only).",
    )
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank (--lora).")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (--lora).")

    # --- PatchTST-specific ---
    p.add_argument("--context_length", type=int, default=75, help="Window length T used by PatchTST.")
    p.add_argument("--patch_len", type=int, default=12, help="Patch length (PatchTST).")
    p.add_argument("--stride", type=int, default=12, help="Stride between patches (PatchTST).")
    p.add_argument("--revin", type=int, default=1, choices=(0, 1), help="Reversible instance normalization (1=on).")
    p.add_argument("--n_layers", type=int, default=3, help="Number of Transformer layers (PatchTST).")
    p.add_argument("--n_heads", type=int, default=16, help="Number of attention heads (PatchTST).")
    p.add_argument("--d_model", type=int, default=128, help="Transformer d_model (PatchTST).")
    p.add_argument("--d_ff", type=int, default=512, help="Transformer FFN dimension (PatchTST).")
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="PatchTST attention + feed-forward dropout (ignored for non-PatchTST models).",
    )
    p.add_argument(
        "--head_dropout",
        type=float,
        default=0.2,
        help="PatchTST classification head dropout (HF config).",
    )
    p.add_argument(
        "--head",
        type=str,
        default="mlp",
        choices=("mlp", "linear", "patchtst_hf"),
        help="Classification head: mlp | linear | patchtst_hf (requires --model patchtst --input_mode deep_only).",
    )

    return p.parse_args()


def _build_encoder_kwargs(args) -> dict:
    """Collect encoder-specific kwargs from CLI args."""
    kwargs = {}
    if args.hidden_lstm is not None:
        kwargs["hidden_lstm"] = args.hidden_lstm
    # PatchTST kwargs (only used if encoder_name == "patchtst")
    if args.model in ("patchtst",):
        kwargs.update(
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            patch_length=args.patch_len,
            patch_stride=args.stride,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            ffn_dim=args.d_ff,
            revin=bool(args.revin),
        )
        if args.patchtst_checkpoint:
            kwargs["pretrained_path"] = args.patchtst_checkpoint
    return kwargs


def _prepare_labeled_loaders(args):
    from hc_framework.data.pt_dataloader import prepare_train_val_test_loaders_from_pt

    return prepare_train_val_test_loaders_from_pt(
        args.from_pt,
        channel_policy=args.channel_policy,
        resample_t=args.resample_t,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        registry_path=args.registry_path,
    )


def main():
    _ensure_src_on_path()
    args = parse_args()
    set_seed(args.seed)

    if args.lora and args.model != "patchtst":
        raise SystemExit("--lora is only supported with --model patchtst")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("device=%s", device)

    # ---- Test mode ----
    if args.mode == "test":
        train_dl, val_dl, test_dl, class_names, num_classes, n_feats, in_channels, _ = _prepare_labeled_loaders(args)
        encoder_kwargs = _build_encoder_kwargs(args)
        model = build_hybrid_model(
            encoder_name=args.model,
            input_mode=args.input_mode,
            num_classes=num_classes,
            n_hc_feats=n_feats,
            in_channels=in_channels,
            head_name=args.head,
            **encoder_kwargs,
        ).to(device)
        logger.info("model (%s/%s):\n%s", args.model, args.input_mode, model)
        ckpt = args.checkpoint or str(out / "best.pt")
        trainer = Trainer(model, device, out)
        res = trainer.evaluate(test_dl, ckpt)
        metrics = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test"
        )
        logger.info("checkpoint=%s", ckpt)
        logger.info("test accuracy=%.4f macro_f1=%.4f", metrics["accuracy"], metrics["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return

    # ---- Supervised / Finetune ----
    train_dl, val_dl, test_dl, class_names, num_classes, n_feats, in_channels, _ = _prepare_labeled_loaders(args)
    logger.info("classes=%d n_hc_feats=%d in_channels=%d", len(class_names), n_feats, in_channels)

    encoder_kwargs = _build_encoder_kwargs(args)
    model = build_hybrid_model(
        encoder_name=args.model,
        input_mode=args.input_mode,
        num_classes=num_classes,
        n_hc_feats=n_feats,
        in_channels=in_channels,
        head_name=args.head,
        **encoder_kwargs,
    ).to(device)
    logger.info("model (%s/%s):\n%s", args.model, args.input_mode, model)

    if args.init_encoder_from:
        src = Path(args.init_encoder_from)
        if not src.is_file():
            raise SystemExit(f"--init_encoder_from: file not found: {src}")
        state = torch.load(src, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Two on-disk formats are accepted:
        #   (a) sensors-style: flat keys prefixed `encoder.` (e.g. saved by Trainer)
        #   (b) pretrain_ablations-style: `{"encoder": <state_dict>}` with raw encoder keys
        if isinstance(state, dict) and "encoder" in state and isinstance(state["encoder"], dict):
            enc_state = state["encoder"]
        else:
            enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        if not enc_state:
            raise SystemExit(f"--init_encoder_from: no encoder keys found in {src}")
        miss, unex = model.encoder.load_state_dict(enc_state, strict=False)
        logger.info("Loaded encoder from %s (loaded=%d, missing=%d, unexpected=%d)",
                    src, len(enc_state), len(miss), len(unex))

    if args.lora:
        model.encoder.apply_lora(r=args.lora_r, alpha=args.lora_alpha)
        n_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.encoder.parameters())
        logger.info("LoRA applied to encoder: trainable=%d/%d (%.2f%%)",
                    n_trainable, n_total, 100.0 * n_trainable / n_total)

    trainer = Trainer(model, device, out)

    if args.mode == "supervised":
        lr = args.lr if args.lr is not None else 1e-3
        resume = args.checkpoint if args.checkpoint else None
        trainer.train_supervised(
            train_dl,
            val_dl,
            num_classes,
            epochs=args.epochs,
            lr=lr,
            use_class_weights=not args.no_class_weights,
            resume_from=resume,
            freeze_encoder=args.freeze_encoder,
        )
        res = trainer.evaluate(test_dl, out / "best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test_stage1"
        )
        logger.info("test stage1: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return

    if args.mode == "finetune":
        load_from = args.checkpoint or (out / "best.pt")
        lr = args.lr if args.lr is not None else 1e-5
        # Stage 2: fold val into train, then run a short plain-CE finetune.
        # No early stopping, no selection — last-epoch checkpoint is saved.
        from hc_framework.data.dataloader import CalfHybridDataset
        from torch.utils.data import DataLoader as _DL
        tr_ds, va_ds = train_dl.dataset, val_dl.dataset
        ft_train_ds = CalfHybridDataset(
            torch.cat([tr_ds.signals, va_ds.signals], dim=0).numpy(),
            torch.cat([tr_ds.features, va_ds.features], dim=0).numpy(),
            torch.cat([tr_ds.labels, va_ds.labels], dim=0).numpy(),
        )
        ft_train_dl = _DL(
            ft_train_ds,
            batch_size=train_dl.batch_size,
            shuffle=True,
            num_workers=train_dl.num_workers,
            pin_memory=train_dl.pin_memory,
        )
        logger.info(
            "finetune: train+val = %d samples (train=%d, val=%d); test_dl is log-only",
            len(ft_train_ds), len(tr_ds), len(va_ds),
        )
        if trainer.finetune(
            ft_train_dl,
            load_path=load_from,
            epochs=args.epochs,
            lr=lr,
            freeze_encoder=args.freeze_encoder,
        ) is None:
            raise SystemExit(f"Fine-tune cancelled: checkpoint not found at {load_from}")
        res = trainer.evaluate(test_dl, out / "finetuned_best.pt")
        m = classification_metrics_numpy(res["y_true"], res["y_pred"])
        paths = save_test_evaluation_artifacts(
            res["y_true"], res["y_pred"], class_names, out, stem="test_stage2"
        )
        logger.info("test stage2: acc=%.4f macro_f1=%.4f", m["accuracy"], m["f1_macro"])
        logger.info("saved confusion matrix: %s", paths["png_path"])
        logger.info("saved per-class metrics: %s", paths["json_path"])
        return


if __name__ == "__main__":
    main()
