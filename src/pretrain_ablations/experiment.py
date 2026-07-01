"""Main experiment orchestrator.

Usage:
    python -m pretrain_ablations.experiment --config configs/foo.yaml
    python -m pretrain_ablations.experiment --config configs/foo.yaml --override seed=42 encoder.name=patchtst
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(cfg, output_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{cfg.data.dataset_id}_{cfg.encoder.name}_{cfg.pretext.method}_{cfg.finetune.mode}_s{cfg.seed}"
    if cfg.run_name and cfg.run_name != "unnamed":
        name = cfg.run_name
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _evaluate_test(encoder, head, test_dl, device, class_names):
    encoder.eval(); head.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y, _ in test_dl:
            logits = head(encoder(x.to(device)))
            preds.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg) -> Path:
    from pretrain_ablations.config import save_resolved_yaml
    from pretrain_ablations.datasets.loader import build_dataloaders
    from common.registry import load_registry
    from finetune.heads import build_head
    from encoders import build_encoder
    from pretrain_ablations.eval.confusion import save_confusion_matrix
    from pretrain_ablations.eval.embedding_viz import extract_embeddings, save_tsne
    from common.metrics import compute_metrics
    from finetune.modes import apply_finetune_mode
    from finetune.trainer import FinetuneTrainer
    from common.repro import save_environment, save_manifest, set_seed

    # -----------------------------------------------------------------
    # 0. Guard: no forecasting_sanity datasets
    # -----------------------------------------------------------------
    registry = load_registry(cfg.data.registry_path)
    info = registry.get(cfg.data.dataset_id, {})
    if info.get("task") == "forecasting_sanity":
        raise ValueError(
            f"dataset_id={cfg.data.dataset_id!r} has task=forecasting_sanity — "
            "not usable in classification pipeline."
        )

    # -----------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------
    set_seed(cfg.seed)
    output_root = Path(cfg.output_root)
    run_dir = _make_run_dir(cfg, output_root)
    artifacts = run_dir / "artifacts"
    eval_dir = run_dir / "eval"
    for d in [artifacts / "checkpoints", artifacts / "logs",
               eval_dir / "embeddings"]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run dir: {run_dir}")
    save_resolved_yaml(cfg, artifacts / "config.yaml")
    save_manifest(artifacts / "manifest.json", cfg)
    save_environment(artifacts / "environment.txt")

    # -----------------------------------------------------------------
    # 2. Data
    # -----------------------------------------------------------------
    train_dl, val_dl, test_dl, meta = build_dataloaders(cfg.data)
    num_classes = meta["num_classes"]
    C_in = meta["num_channels_after_policy"]
    T = meta["window_len"]
    class_names = meta["class_names"]
    logger.info(f"Data: {cfg.data.dataset_id} | num_classes={num_classes} C_in={C_in} T={T}")

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    logger.info(f"Device: {device}")

    # -----------------------------------------------------------------
    # 3. Validate compatibility
    # -----------------------------------------------------------------
    method_name = cfg.pretext.method
    encoder_name = cfg.encoder.name

    if method_name == "mae" and encoder_name not in ("patchtst", "patchtsmixer"):
        raise ValueError(f"MAE pretraining requires patchtst or patchtsmixer, got {encoder_name!r}")

    if (method_name == "supervised" and cfg.finetune.mode in ("freeze", "linear_probe")
            and not cfg.encoder.init_from):
        raise ValueError(
            f"finetune.mode={cfg.finetune.mode!r} makes no sense without pretrained encoder. "
            "Use mode=full or partial_k with supervised, or set encoder.init_from=<ckpt>."
        )

    # -----------------------------------------------------------------
    # 4. Build encoder + pretext method
    # -----------------------------------------------------------------
    from pretraining.supervised import SupervisedBaseline

    if method_name == "tfc":
        from pretraining.tfc import TFCMethod, build_tfc_method
        method = build_tfc_method(cfg.pretext, cfg.encoder, C_in, T, device)
        encoder = method.get_time_encoder().to(device)
    else:
        encoder = build_encoder(cfg.encoder, in_channels=C_in, context_length=T).to(device)
        if method_name == "supervised":
            method = SupervisedBaseline()
        elif method_name == "simclr":
            from pretraining.simclr import SimCLRMethod
            method = SimCLRMethod(encoder.output_dim, cfg.pretext.projection_dim,
                                   cfg.pretext.temperature, cfg.pretext)
        elif method_name == "mae":
            from pretraining.mae import MAEMethod
            method = MAEMethod(encoder, cfg.pretext)
        elif method_name == "tstcc":
            from pretraining.tstcc import TSTCCMethod
            method = TSTCCMethod(encoder.output_dim, cfg.pretext)
        else:
            raise ValueError(f"Unknown pretext method={method_name!r}")

    # -----------------------------------------------------------------
    # 4b. Transfer: load pretrained encoder weights, skip pretrain
    # -----------------------------------------------------------------
    if cfg.encoder.init_from:
        ckpt = torch.load(cfg.encoder.init_from, map_location=device, weights_only=False)
        missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
        logger.info(f"Loaded encoder weights from {cfg.encoder.init_from} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")
        method = SupervisedBaseline()  # encoder already pretrained; skip pretrain below

    # save model summary (encoder)
    summary_lines = [f"=== Encoder: {encoder_name} ===", str(encoder), ""]
    (artifacts / "model_summary.txt").write_text("\n".join(summary_lines))
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder params: {n_params:,}")

    # -----------------------------------------------------------------
    # 5. Pretrain
    # -----------------------------------------------------------------
    do_pretrain = not method.skip_pretrain

    if do_pretrain:
        from common.collapse import knn_collapse_heuristic
        chance = 1.0 / num_classes

        method_params = list(method.parameters())
        pretrain_params = list(encoder.parameters()) + method_params
        if method_name == "tfc":
            pretrain_params += list(method.encoder_f.parameters())

        optimizer = torch.optim.AdamW(pretrain_params, lr=cfg.pretext.lr,
                                       weight_decay=cfg.pretext.weight_decay)

        pretrain_history = []
        best_knn = -1.0
        best_enc_state = None
        log_lines = []
        collapse_count = 0

        for epoch in range(1, cfg.pretext.epochs + 1):
            encoder.train()
            if method_name == "tfc":
                method.encoder_f.train()

            batch_losses = []
            for batch in train_dl:
                batch = tuple(t.to(device) for t in batch)
                optimizer.zero_grad(set_to_none=True)
                out = method.pretrain_step(encoder, batch)
                loss = out["loss"]
                loss.backward()
                if cfg.pretext.grad_clip:
                    nn.utils.clip_grad_norm_(pretrain_params, cfg.pretext.grad_clip)
                optimizer.step()
                batch_losses.append(loss.item())

            avg_loss = float(np.mean(batch_losses))
            row = {"epoch": epoch, "loss": round(avg_loss, 4)}

            if epoch % cfg.pretext.collapse_check_every == 0 or epoch == 1:
                encoder.eval()
                knn = knn_collapse_heuristic(encoder, val_dl, device,
                                              k=cfg.pretext.collapse_knn_k,
                                              max_samples=cfg.pretext.collapse_max_samples)
                row["knn_collapse_heuristic"] = round(knn, 4)
                row["chance_level"] = round(chance, 4)
                if knn <= chance * 1.1:
                    collapse_count += 1
                else:
                    collapse_count = 0
                if collapse_count >= 3 and epoch > 10:
                    row["collapse_flag"] = True
                    logger.warning(f"Collapse detected at epoch {epoch} (knn={knn:.3f})")

                if knn > best_knn:
                    best_knn = knn
                    best_enc_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                    torch.save({"encoder": best_enc_state},
                               artifacts / "checkpoints" / "pretrain_best.pt")

                line = (f"Pretrain Ep {epoch:03d}/{cfg.pretext.epochs} "
                        f"loss={avg_loss:.4f} knn={knn:.3f} (chance={chance:.3f})")
            else:
                line = f"Pretrain Ep {epoch:03d}/{cfg.pretext.epochs} loss={avg_loss:.4f}"

            pretrain_history.append(row)
            log_lines.append(line)
            logger.info(line)

        # save last
        torch.save({"encoder": encoder.state_dict()},
                   artifacts / "checkpoints" / "pretrain_last.pt")
        # load best
        if best_enc_state is not None:
            encoder.load_state_dict(best_enc_state)
        (artifacts / "logs" / "pretrain.log").write_text("\n".join(log_lines) + "\n")
        with open(eval_dir / "metrics_pretrain.json", "w") as f:
            json.dump(pretrain_history, f, indent=2)

        # post-pretrain embeddings + t-SNE
        encoder.eval()
        emb_pre, lbl_pre = extract_embeddings(encoder, val_dl, device)
        if cfg.eval.save_embeddings:
            np.save(eval_dir / "embeddings" / "post_pretrain_val.npy", emb_pre)
            np.save(eval_dir / "embeddings" / "post_pretrain_val_labels.npy", lbl_pre)
        save_tsne(emb_pre, lbl_pre, class_names,
                  eval_dir / "embeddings" / "tsne_post_pretrain.png", cfg.eval)
        logger.info(f"Post-pretrain t-SNE saved. best_knn={best_knn:.3f}")

        # for MAE: swap to encoder-only model
        if method_name == "mae":
            encoder = method.extract_encoder()
            encoder.to(device)
            # the in-loop pretrain_best.pt held pre-extraction weights for MAE; overwrite
            # with the extracted encoder so init_from gets the right state_dict.
            torch.save({"encoder": encoder.state_dict()},
                       artifacts / "checkpoints" / "pretrain_best.pt")

    # -----------------------------------------------------------------
    # 5b. pretrain_only: stop here, encoder checkpoint is the deliverable
    # -----------------------------------------------------------------
    if cfg.pretext.pretrain_only:
        torch.save({"encoder": encoder.state_dict()},
                   artifacts / "checkpoints" / "pretrain_last.pt")
        summary = (
            f"{run_dir.name} | pretrain_only | dataset={cfg.data.dataset_id} "
            f"| encoder={cfg.encoder.name} | method={cfg.pretext.method}"
        )
        (eval_dir / "summary.txt").write_text(summary + "\n")
        logger.info(f"\nDone (pretrain_only). {summary}")
        return run_dir

    # -----------------------------------------------------------------
    # 6. Build head + append to model_summary
    # -----------------------------------------------------------------
    head = build_head(cfg.head, encoder.output_dim, num_classes).to(device)
    with open(artifacts / "model_summary.txt", "a") as f:
        f.write(f"\n=== Head: {cfg.head.name} ===\n{head}\n")

    # -----------------------------------------------------------------
    # 7. Finetune
    # -----------------------------------------------------------------
    param_groups = apply_finetune_mode(
        encoder, head, cfg.finetune.mode, cfg.finetune.lr,
        cfg.finetune.encoder_lr_factor, cfg.finetune.partial_k,
    )
    ft = FinetuneTrainer(
        encoder, head, device,
        ckpt_dir=artifacts / "checkpoints",
        log_path=artifacts / "logs" / "finetune.log",
        metrics_path=eval_dir / "metrics_finetune.json",
    )
    ft_result = ft.train(train_dl, val_dl, cfg.finetune, param_groups, num_classes, class_names)
    logger.info(f"Finetune best val_acc={ft_result['best_val_acc']:.3f}")

    # -----------------------------------------------------------------
    # 8. Test evaluation
    # -----------------------------------------------------------------
    y_true, y_pred = _evaluate_test(encoder, head, test_dl, device, class_names)
    metrics = compute_metrics(y_true, y_pred, class_names)
    with open(eval_dir / "metrics_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(y_true, y_pred, class_names, eval_dir)

    # post-finetune embeddings + t-SNE
    encoder.eval()
    emb_post, lbl_post = extract_embeddings(encoder, test_dl, device)
    if cfg.eval.save_embeddings:
        np.save(eval_dir / "embeddings" / "post_finetune_test.npy", emb_post)
        np.save(eval_dir / "embeddings" / "post_finetune_test_labels.npy", lbl_post)
    save_tsne(emb_post, lbl_post, class_names,
              eval_dir / "embeddings" / "tsne_post_finetune.png", cfg.eval)

    # -----------------------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------------------
    pretrain_tag = cfg.pretext.method if do_pretrain else "supervised"
    summary = (
        f"{run_dir.name} | dataset={cfg.data.dataset_id} | encoder={cfg.encoder.name} "
        f"| method={pretrain_tag} | finetune={cfg.finetune.mode} | seed={cfg.seed} "
        f"| acc={metrics['accuracy']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} "
        f"| macro_f1={metrics['macro_f1']:.4f}"
    )
    (eval_dir / "summary.txt").write_text(summary + "\n")
    logger.info(f"\nDone. {summary}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from pretrain_ablations.config import load_config

    p = argparse.ArgumentParser(description="Run a single pretrain/finetune experiment")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--override", nargs="*", default=[],
                   help="Dot-list overrides, e.g. seed=42 encoder.name=patchtst")
    args = p.parse_args()

    cfg = load_config(args.config, args.override or None)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
