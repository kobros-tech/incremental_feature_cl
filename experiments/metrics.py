# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Evaluation helpers for target-versus-rest continual experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def target_metrics(y_binary: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Evaluate a target-positive binary task using decision scores (threshold zero)."""
    y_binary = np.asarray(y_binary, dtype=bool)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_binary) != len(scores):
        raise ValueError("y_binary and scores must have the same length.")
    predicted = scores >= 0
    positives = y_binary
    negatives = ~y_binary
    recall = float(predicted[positives].mean()) if positives.any() else float("nan")
    specificity = float((~predicted[negatives]).mean()) if negatives.any() else float("nan")
    balanced = (
        (recall + specificity) / 2
        if np.isfinite(recall) and np.isfinite(specificity)
        else float("nan")
    )
    try:
        auc = float(roc_auc_score(y_binary, scores))
    except ValueError:
        auc = float("nan")
    return {
        "target_recall": recall,
        "negative_specificity": specificity,
        "balanced_accuracy": float(balanced),
        "roc_auc": auc,
    }
