"""Save confusion matrix PNG and per-class / aggregate metrics under ``output_dir``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from hc_framework.training.metrics import classification_metrics_numpy


def _build_per_behavior_rows(
    class_names: np.ndarray,
    report: dict,
) -> tuple[list[str], list[float], list[float], list[float], list[int]]:
    """Return (names, recall, precision, f1, support) aligned to ``class_names``."""
    names_out: list[str] = []
    recalls: list[float] = []
    precs: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for c in class_names:
        key = str(c)
        if key not in report:
            names_out.append(key)
            recalls.append(0.0)
            precs.append(0.0)
            f1s.append(0.0)
            supports.append(0)
            continue
        block = report[key]
        names_out.append(key)
        recalls.append(float(block["recall"]))
        precs.append(float(block["precision"]))
        f1s.append(float(block["f1-score"]))
        supports.append(int(block["support"]))
    return names_out, recalls, precs, f1s, supports


def save_test_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    output_dir: str | Path,
    *,
    stem: str = "test",
) -> dict:
    """
    Write ``{stem}_confusion_matrix.png`` and ``{stem}_classification_metrics.json``.

    Per-behavior **recall** is the fraction of true instances of that behavior predicted
    correctly (same as one-vs-rest "true-class accuracy"). JSON also includes precision and F1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(class_names), dtype=int)
    display_labels = [str(c) for c in class_names]

    overall = classification_metrics_numpy(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )
    names, recalls, precs, f1s, supports = _build_per_behavior_rows(class_names, report)

    payload = {
        "overall": {
            "accuracy": overall["accuracy"],
            "balanced_accuracy": overall["balanced_accuracy"],
            "f1_macro": overall["f1_macro"],
            "f1_weighted": overall["f1_weighted"],
        },
        "per_behavior": {
            n: {
                "recall_true_class_accuracy": r,
                "precision": p,
                "f1": f,
                "support": s,
            }
            for n, r, p, f, s in zip(names, recalls, precs, f1s, supports)
        },
    }
    json_path = output_dir / f"{stem}_classification_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_cls = len(class_names)
    fig_h = min(22.0, max(10.0, 0.45 * n_cls + 6.0))
    fig_w = min(24.0, max(12.0, 0.55 * n_cls + 8.0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.28, top=0.92, bottom=0.06)
    ax_cm = fig.add_subplot(gs[0])
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=display_labels,
        normalize="true",
        xticks_rotation=55,
        ax=ax_cm,
        cmap="Blues",
        colorbar=True,
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")

    ax_tab = fig.add_subplot(gs[1])
    ax_tab.axis("off")
    cell_text = [[f"{r:.3f}", f"{f:.3f}", str(s)] for r, f, s in zip(recalls, f1s, supports)]
    tbl = ax_tab.table(
        cellText=cell_text,
        rowLabels=names,
        colLabels=["Recall\n(true-class acc.)", "F1", "Support"],
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.15)

    fig.suptitle(
        f"acc={overall['accuracy']:.4f}  |  "
        f"bal_acc={overall['balanced_accuracy']:.4f}  |  "
        f"F1 macro={overall['f1_macro']:.4f}  |  "
        f"F1 weighted={overall['f1_weighted']:.4f}",
        fontsize=11,
        y=0.98,
    )
    png_path = output_dir / f"{stem}_confusion_matrix.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return {
        "json_path": str(json_path),
        "png_path": str(png_path),
        "overall": overall,
        "confusion_matrix": cm.tolist(),
    }
