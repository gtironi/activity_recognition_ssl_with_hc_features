"""Classification metrics."""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    class_names: list[str]) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":     float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1":    {class_names[i]: float(v) for i, v in enumerate(per_class)
                            if i < len(class_names)},
    }
