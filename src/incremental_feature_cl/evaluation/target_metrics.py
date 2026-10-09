# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Target-vs-rest metrics.  Negatives are restricted to negative classes seen so far.

Why more than "accuracy": with ``K`` seen classes only ~``1/K`` of the evaluated samples are
positives, so plain binary accuracy is dominated by the negatives (it reaches ``1 - 1/K`` for a
model that never fires).  ``balanced_accuracy`` = (recall + negative accuracy) / 2 is the number
to compare models with, and ``auc`` is threshold-free: if recall / negative accuracy swing from
one experience to the next while ``auc`` stays put, the learned ranking is stable and only the
decision threshold moved.
"""

from __future__ import annotations

import numpy as np


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float | None:
    """P(score_pos > score_neg) + 0.5 P(tie), via average ranks (Mann-Whitney U)."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    scores = np.concatenate([pos_scores, neg_scores]).astype(np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    # average rank over ties
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(scores)]])
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0  # ranks are 1-based
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def binary_target_metrics(
    pred_bin: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    seen_negatives,
    scores: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Metrics of one target-vs-rest decision.

    ``pred_bin`` is the thresholded decision (``score > 0``); ``scores`` (optional) are the raw
    logits and enable ``auc``.  Samples of classes that are neither the target nor in
    ``seen_negatives`` are ignored.
    """
    pos = labels == target_class
    neg = np.isin(labels, list(seen_negatives))
    p = pred_bin.astype(bool)
    tp, fn = int((p & pos).sum()), int((~p & pos).sum())
    fp, tn = int((p & neg).sum()), int((~p & neg).sum())
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    negative_accuracy = _div(tn, tn + fp)
    return {
        "target_accuracy": recall,
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": _div(2 * precision * recall, precision + recall),
        "negative_accuracy": negative_accuracy,
        "false_positive_rate": _div(fp, fp + tn),
        "false_negative_rate": _div(fn, fn + tp),
        "balanced_accuracy": (recall + negative_accuracy) / 2.0,
        "auc": roc_auc(scores[pos], scores[neg]) if scores is not None else None,
        "overall_binary_accuracy": _div(tp + tn, tp + tn + fp + fn),
        "n_target": int(pos.sum()),
        "n_seen_negative": int(neg.sum()),
    }
