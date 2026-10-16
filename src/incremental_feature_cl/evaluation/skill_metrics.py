# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Scoring a set of independent target-vs-rest skills.

Every skill ``c`` is judged exactly like a target-mode run with target ``c``: its own binary
decision (``score > 0``) on the test samples of class ``c`` (positives) and of the other seen
classes (negatives), through :func:`binary_target_metrics`.  Skill scores are independent logits that were never trained to be compared with each other.
Evaluation therefore uses each skill's binary target-vs-rest decision and does not report
multiclass accuracy derived from the argmax of skill scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
    seen: Sequence[int],
    thresholds: Mapping[int, float] | None = None,
) -> dict[int, dict]:
    """``{class id: binary_target_metrics}`` for every skill whose class has been seen.

    ``scores[:, j]`` is the logit of the skill of ``class_ids[j]``.  The decision is
    ``score > thresholds[class id]`` (``0`` when ``thresholds`` is ``None`` or lacks the class),
    so the default reproduces the plain ``score > 0`` evaluation.  Threshold-free ``auc`` does not
    depend on it.  Each result also records the ``threshold`` that was used.
    """
    thresholds = thresholds or {}
    seen_set = {int(c) for c in seen}
    out: dict[int, dict] = {}
    for j, c in enumerate(class_ids):
        c = int(c)
        if c not in seen_set:
            continue
        negatives = seen_set - {c}
        t = float(thresholds.get(c, 0.0))
        m = binary_target_metrics(scores[:, j] > t, labels, c, negatives, scores[:, j])
        m["threshold"] = t
        out[c] = m
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
