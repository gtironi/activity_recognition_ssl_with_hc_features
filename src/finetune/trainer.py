"""FinetuneTrainer: supervised train loop with early stopping + checkpointing."""

from __future__ import annotations
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class FinetuneTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        device: torch.device,
        ckpt_dir: Path,
        log_path: Path,
        metrics_path: Path,
    ):
        self.encoder = encoder
        self.head = head
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.log_path = Path(log_path)
        self.metrics_path = Path(metrics_path)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        cfg,
        param_groups: list[dict],
        num_classes: int,
        class_names: list[str],
    ) -> dict:
        class_weights = None
        if cfg.class_weights:
            labels_np = np.array([y.item() for _, y, _ in train_dl.dataset])
            from sklearn.utils.class_weight import compute_class_weight
            present = np.unique(labels_np)
            raw = compute_class_weight("balanced", classes=present, y=labels_np)
            full = np.ones(num_classes, dtype=np.float32)
            for i, c in enumerate(present):
                full[c] = raw[i]
            class_weights = torch.tensor(full, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor
        )

        best_val_acc = 0.0
        best_wts_enc = copy.deepcopy(self.encoder.state_dict())
        best_wts_head = copy.deepcopy(self.head.state_dict())
        stall = 0
        history = []

        log_lines = []
        for epoch in range(1, cfg.epochs + 1):
            self.encoder.train(); self.head.train()
            tr_loss = tr_correct = tr_total = 0
            for x, y, _ in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.head(self.encoder(x))
                loss = criterion(logits, y)
                loss.backward()
                if cfg.grad_clip:
                    nn.utils.clip_grad_norm_(
                        [p for pg in param_groups for p in pg["params"]], cfg.grad_clip)
                optimizer.step()
                tr_loss += loss.item() * x.size(0)
                tr_correct += (logits.argmax(1) == y).sum().item()
                tr_total += x.size(0)

            self.encoder.eval(); self.head.eval()
            vl_loss = vl_correct = vl_total = 0
            with torch.no_grad():
                for x, y, _ in val_dl:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.head(self.encoder(x))
                    vl_loss += criterion(logits, y).item() * x.size(0)
                    vl_correct += (logits.argmax(1) == y).sum().item()
                    vl_total += y.size(0)

            avg_tr = tr_loss / tr_total; tr_acc = tr_correct / tr_total
            avg_vl = vl_loss / vl_total; vl_acc = vl_correct / vl_total
            scheduler.step(avg_vl)

            row = {"epoch": epoch, "train_loss": round(avg_tr, 4),
                   "train_acc": round(tr_acc, 4), "val_loss": round(avg_vl, 4),
                   "val_acc": round(vl_acc, 4)}
            history.append(row)
            line = f"Ep {epoch:03d} | tr_loss={avg_tr:.4f} tr_acc={tr_acc:.3f} | vl_loss={avg_vl:.4f} vl_acc={vl_acc:.3f}"
            log_lines.append(line)
            logger.info(line)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_wts_enc = copy.deepcopy(self.encoder.state_dict())
                best_wts_head = copy.deepcopy(self.head.state_dict())
                torch.save({"encoder": best_wts_enc, "head": best_wts_head},
                           self.ckpt_dir / "finetune_best.pt")
                stall = 0
            else:
                stall += 1
                if stall >= cfg.early_stopping_patience:
                    logger.info(f"Early stop at epoch {epoch}")
                    break

        # save last
        torch.save({"encoder": self.encoder.state_dict(), "head": self.head.state_dict()},
                   self.ckpt_dir / "finetune_last.pt")

        # load best
        self.encoder.load_state_dict(best_wts_enc)
        self.head.load_state_dict(best_wts_head)

        self.log_path.write_text("\n".join(log_lines) + "\n")
        with open(self.metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        return {"best_val_acc": best_val_acc, "history": history}
