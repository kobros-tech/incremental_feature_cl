# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Scoring a set of independent target-vs-rest skills.

Every skill ``c`` is judged exactly like a target-mode run with target ``c``: its own binary
decision (``score > 0``) on the test samples of class ``c`` (positives) and of the other seen
classes (negatives), through :func:`binary_target_metrics`.  Skill scores are independent logits
that were never trained to be compared with each other, so the ``argmax`` accuracy is reported
only as a separate, secondary diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .target_metrics import binary_target_metrics

_AVERAGED = (
    "target_recall",
    "target_precision",
    "target_f1",
    "negative_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_accuracy",
    "auc",
    "overall_binary_accuracy",
)


def per_skill_metrics(
    scores: np.ndarray, labels: np.ndarray, class_ids: Sequence[int], seen: Sequence[int]
) -> dict[int, dict]:
    """``{class id: binary_target_metrics}`` for every skill whose class has been seen.

    ``scores[:, j]`` is the logit of the skill of ``class_ids[j]``.
    """
    seen_set = {int(c) for c in seen}
    out: dict[int, dict] = {}
    for j, c in enumerate(class_ids):
        c = int(c)
        if c not in seen_set:
            continue
        negatives = seen_set - {c}
        out[c] = binary_target_metrics(scores[:, j] > 0, labels, c, negatives, scores[:, j])
    return out


def macro_metrics(per_skill: dict[int, dict]) -> dict[str, float | int | None]:
    """Mean of every per-skill metric (same keys as one ``binary_target_metrics`` result)."""
    out: dict[str, float | int | None] = {}
    for key in _AVERAGED:
        values = [m[key] for m in per_skill.values() if m.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    out["target_accuracy"] = out["target_recall"]
    out["n_skills"] = len(per_skill)
    out["n_target"] = int(sum(m["n_target"] for m in per_skill.values()))
    out["n_seen_negative"] = int(np.mean([m["n_seen_negative"] for m in per_skill.values()]))
    return out


def argmax_accuracy(
    scores: np.ndarray, labels: np.ndarray, class_ids: Sequence[int], seen: Sequence[int]
) -> float:
    """Multiclass accuracy of ``argmax`` over raw skill logits on samples of seen classes (diagnostic)."""
    ids = np.asarray(class_ids)
    predicted = ids[scores.argmax(axis=1)]
    mask = np.isin(labels, list(seen))
    return float((predicted[mask] == labels[mask]).mean()) if mask.any() else float("nan")
