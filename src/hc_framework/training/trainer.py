from __future__ import annotations

import copy
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

from hc_framework.training.loss import balanced_class_weights

logger = logging.getLogger(__name__)


def _iter_trainable_params(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def _balanced_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy restricted to classes present in y_true (no warning)."""
    labels_present = np.unique(y_true)
    return float(
        recall_score(
            y_true, y_pred, labels=labels_present, average="macro", zero_division=0
        )
    )

class Trainer:
    """Supervised training, fine-tuning, and val/test evaluation loops."""

    def __init__(self, model: nn.Module, device: torch.device, output_dir: str | Path):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_signal_encoder_freeze(self, freeze: bool) -> None:
        """Freeze (or unfreeze) the signal encoder only (PatchTST / CNN+LSTM / robust)."""
        enc = getattr(self.model, "encoder", None)
        if enc is None or type(enc).__name__ == "NullSignalEncoder":
            return
        for p in enc.parameters():
            p.requires_grad = not freeze
        logger.info("Signal encoder requires_grad=%s", not freeze)

    def train_supervised(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        num_classes: int,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        use_class_weights: bool = True,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.3,
        early_stopping_patience: int = 25,
        grad_clip: float = 1.0,
        checkpoint_name: str = "best.pt",
        resume_from: str | Path | None = None,
        freeze_encoder: bool = False,
    ) -> nn.Module:
        best_wts = copy.deepcopy(self.model.state_dict())
        best_bal_acc = -math.inf
        stall = 0
        start_epoch = 0
        ckpt_path = self.output_dir / checkpoint_name
        resume_ckpt = None
        if resume_from is not None and Path(resume_from).is_file():
            resume_ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            self.model.load_state_dict(resume_ckpt["model_state_dict"])
            best_bal_acc = resume_ckpt["best_bal_acc"]
            best_wts = resume_ckpt["best_wts"]
            stall = resume_ckpt["stall"]
            start_epoch = resume_ckpt["epoch"] + 1
            logger.info(
                "Resuming training from epoch %d (best_bal_acc=%.2f%%)",
                start_epoch, best_bal_acc,
            )

        self._apply_signal_encoder_freeze(freeze_encoder)

        labels = train_dl.dataset.labels.cpu().numpy()
        cw = None
        if use_class_weights:
            cw = balanced_class_weights(labels, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=cw) if cw is not None else nn.CrossEntropyLoss()
        logger.info("class_weights=%s", cw is not None)
        optimizer = torch.optim.AdamW(_iter_trainable_params(self.model), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

        if resume_ckpt is not None and not freeze_encoder and "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load optimizer/scheduler state (%s); starting fresh.", e)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loss = 0.0
            t_ys, t_preds = [], []
            for x_sig, x_feat, y in train_dl:
                x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x_sig, x_feat)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(_iter_trainable_params(self.model)), grad_clip
                    )
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                t_ys.append(y.detach().cpu().numpy())
                t_preds.append(logits.argmax(1).detach().cpu().numpy())

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_bal_acc = 100.0 * _balanced_acc(
                np.concatenate(t_ys), np.concatenate(t_preds)
            )

            self.model.eval()
            val_loss = 0.0
            v_ys, v_preds = [], []
            with torch.no_grad():
                for x_sig, x_feat, y in val_dl:
                    x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                    logits = self.model(x_sig, x_feat)
                    val_loss += criterion(logits, y).item() * x_sig.size(0)
                    v_ys.append(y.cpu().numpy())
                    v_preds.append(logits.argmax(1).cpu().numpy())
            avg_val_loss = val_loss / len(val_dl.dataset)
            val_bal_acc = 100.0 * _balanced_acc(
                np.concatenate(v_ys), np.concatenate(v_preds)
            )

            logger.info(
                "Ep %03d/%d | train_loss=%.4f acc=%.2f%% | val_loss=%.4f val_acc=%.2f%%",
                epoch + 1, epochs, avg_train_loss, train_bal_acc, avg_val_loss, val_bal_acc,
            )
            scheduler.step(avg_val_loss)

            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                best_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), ckpt_path)
                stall = 0
            else:
                stall += 1
                if stall >= early_stopping_patience:
                    logger.info("Early stopping (no improvement for %d epochs).", early_stopping_patience)
                    break

            # Periodic checkpoint for resume
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_bal_acc": best_bal_acc,
                    "best_wts": best_wts,
                    "stall": stall,
                },
                self.output_dir / "checkpoint.pt",
            )

        self.model.load_state_dict(best_wts)
        return self.model

    def finetune(
        self,
        train_dl: DataLoader,
        load_path: str | Path,
        epochs: int = 10,
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        checkpoint_name: str = "finetuned_best.pt",
        freeze_encoder: bool = False,
    ) -> nn.Module | None:
        """Stage 2: short plain-CE finetune on train (val already folded in by caller).
        No early stopping, no model selection. Saves only the last-epoch checkpoint."""
        load_path = Path(load_path)
        if not load_path.is_file():
            logger.warning("Checkpoint not found: %s", load_path)
            return None
        self.model.load_state_dict(torch.load(load_path, map_location=self.device, weights_only=True))
        self._apply_signal_encoder_freeze(freeze_encoder)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_iter_trainable_params(self.model), lr=lr, weight_decay=weight_decay)
        ckpt_path = self.output_dir / checkpoint_name

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            t_ys, t_preds = [], []
            for x_sig, x_feat, y in train_dl:
                x_sig, x_feat, y = x_sig.to(self.device), x_feat.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x_sig, x_feat)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(_iter_trainable_params(self.model)), grad_clip
                    )
                optimizer.step()
                train_loss += loss.item() * x_sig.size(0)
                t_ys.append(y.detach().cpu().numpy())
                t_preds.append(logits.argmax(1).detach().cpu().numpy())

            avg_train_loss = train_loss / len(train_dl.dataset)
            train_bal_acc = 100.0 * _balanced_acc(
                np.concatenate(t_ys), np.concatenate(t_preds)
            )
            logger.info(
                "Finetune Ep %03d/%d | train_loss=%.4f acc=%.2f%%",
                epoch + 1, epochs, avg_train_loss, train_bal_acc,
            )

        # Save the last-epoch checkpoint (no selection).
        torch.save(self.model.state_dict(), ckpt_path)
        return self.model

    def evaluate(self, data_loader: DataLoader, checkpoint: str | Path | None = None) -> dict:
        if checkpoint and Path(checkpoint).is_file():
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for x_sig, x_feat, y in data_loader:
                x_sig, x_feat = x_sig.to(self.device), x_feat.to(self.device)
                logits = self.model(x_sig, x_feat)
                pred = logits.argmax(1).cpu().numpy()
                ys.append(y.numpy())
                preds.append(pred)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        acc = float((y_true == y_pred).mean())
        return {"accuracy": acc, "y_true": y_true, "y_pred": y_pred}
