# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Target-vs-rest metrics.  Negatives are restricted to negative classes seen so far."""

from __future__ import annotations

import numpy as np


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def binary_target_metrics(
    pred_bin: np.ndarray, labels: np.ndarray, target_class: int, seen_negatives
) -> dict[str, float]:
    pos = labels == target_class
    neg = np.isin(labels, list(seen_negatives))
    p = pred_bin.astype(bool)
    tp, fn = int((p & pos).sum()), int((~p & pos).sum())
    fp, tn = int((p & neg).sum()), int((~p & neg).sum())
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        "target_accuracy": recall,
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": _div(2 * precision * recall, precision + recall),
        "negative_accuracy": _div(tn, tn + fp),
        "false_positive_rate": _div(fp, fp + tn),
        "false_negative_rate": _div(fn, fn + tp),
        "overall_binary_accuracy": _div(tp + tn, tp + tn + fp + fn),
        "n_target": int(pos.sum()),
        "n_seen_negative": int(neg.sum()),
    }
