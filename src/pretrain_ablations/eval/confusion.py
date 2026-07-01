"""Confusion matrix: PNG + raw npy."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list[str], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    np.save(out_dir / "confusion_matrix_raw.npy", cm)

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if val > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
