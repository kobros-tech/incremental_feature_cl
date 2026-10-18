# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Evaluation helpers for target-versus-rest continual experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def target_metrics(
    y_binary: np.ndarray, scores: np.ndarray, threshold: float = 0.0
) -> dict[str, float]:
    """Evaluate a target-positive binary task at the supplied decision threshold."""
    y_binary = np.asarray(y_binary, dtype=bool)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_binary) != len(scores):
        raise ValueError("y_binary and scores must have the same length.")
    predicted = scores >= threshold
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


def select_balanced_accuracy_threshold(
    y_binary: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Select a threshold using validation balanced accuracy only.

    Predictions use scores >= threshold. Ties are resolved by choosing the threshold
    closest to zero, then the numerically smaller threshold.
    """
    y_binary = np.asarray(y_binary, dtype=bool)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_binary) != len(scores):
        raise ValueError("y_binary and scores must have the same length.")
    if not y_binary.any() or y_binary.all():
        raise ValueError("Threshold calibration requires both target and negative examples.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Threshold calibration scores must all be finite.")

    # Compute candidate balanced accuracies in one pass after sorting. Avoid
    # recalculating ROC AUC for every possible threshold on larger validation sets.
    candidates = np.concatenate([np.unique(scores), [np.nextafter(float(np.max(scores)), np.inf)]])
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_targets = y_binary[order]
    suffix_targets = np.concatenate(
        [np.cumsum(sorted_targets[::-1])[::-1], np.array([0], dtype=int)]
    )
    positions = np.searchsorted(sorted_scores, candidates, side="left")
    true_positives = suffix_targets[positions]
    predicted_positives = len(scores) - positions
    false_positives = predicted_positives - true_positives
    true_negatives = int((~y_binary).sum()) - false_positives
    balanced = 0.5 * (
        true_positives / int(y_binary.sum()) + true_negatives / int((~y_binary).sum())
    )
    best_score = float(np.max(balanced))
    tied = candidates[np.isclose(balanced, best_score, rtol=0.0, atol=1e-12)]
    threshold = float(min(tied, key=lambda value: (abs(value), value)))
    return threshold, target_metrics(y_binary, scores, threshold)
