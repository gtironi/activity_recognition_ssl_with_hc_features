from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def classification_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels_present = np.unique(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            recall_score(
                y_true, y_pred, labels=labels_present, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
