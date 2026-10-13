# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Decision-level evaluation of independent target-vs-rest skills, without ``argmax``.

Two problems are addressed here.

1. **The decision threshold.**  A skill's logit is compared with ``0`` by default.  On a stream
   with many classes only ``~1/K`` of the evaluated samples are positives, so a threshold of
   ``0`` can give an acceptable recall and a poor precision / F1 even when the ranking (AUC) is
   good.  :func:`select_thresholds` chooses one threshold per skill that maximises F1 on a
   *held-out validation set* (never the test set), and every report states the thresholds used.

2. **Skills are not comparable with each other.**  ``argmax`` over independently trained logits
   silently assumes they are, and it always answers, even when no skill recognises the sample.
   :func:`decision_report` therefore never takes an ``argmax``.  It asks which skills *fire*
   (``score_j > threshold_j``) and reports what that set looks like:

   * ``none_fired_rate`` (no skill recognises the sample) and ``multi_fired_rate`` (several do);
   * ``exact_match_rate``: the true skill is the only one that fires.  This is the strict
     decision accuracy: abstentions and ambiguous samples both count as errors;
   * ``precision_when_single``: how often the only skill that fires is the right one;
   * per-class recall / precision / false-positive rate and the full *firing matrix*
     ``[true class, skill]`` = fraction of the class's samples on which the skill fires.

The thresholds never feed back into training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def best_f1_threshold(scores: np.ndarray, is_positive: np.ndarray) -> tuple[float, float | None]:
    """Threshold ``t`` (decision ``score > t``) maximising F1; returns ``(t, f1)``.

    Candidates are the mid-points between consecutive distinct scores, so the result does not
    sit on a sample's score.  Ties are broken towards the threshold closest to ``0``.  Without
    positives, without negatives, or when no threshold gives a non-zero F1, the default ``0`` is
    returned (``f1`` is then ``None``).
    """
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(is_positive, dtype=bool)
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == len(scores):
        return 0.0, None
    order = np.argsort(-scores, kind="mergesort")
    s, p = scores[order], pos[order]
    tp, fp = np.cumsum(p), np.cumsum(~p)
    last = np.r_[s[1:] != s[:-1], True]  # last row of every run of equal scores
    tp, fp, cut = tp[last], fp[last], s[last]
    f1 = 2.0 * tp / (tp + fp + n_pos)  # == 2PR / (P + R)
    lower = np.r_[cut[1:], cut[-1] - 1.0]
    threshold = (cut + lower) / 2.0
    best = f1.max()
    if best <= 0:
        return 0.0, None
    candidates = np.flatnonzero(f1 >= best - 1e-12)
    pick = candidates[np.argmin(np.abs(threshold[candidates]))]
    return float(threshold[pick]), float(f1[pick])


def select_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
    seen: Sequence[int],
) -> dict[int, float]:
    """One F1-maximising threshold per skill from *validation* scores.

    ``scores[:, j]`` belongs to the skill of ``class_ids[j]``.  The positives of a skill are the
    validation samples of its class, the negatives those of the other seen classes (the natural
    class mix, which is what the test-time F1 sees too).
    """
    seen_set = {int(c) for c in seen}
    keep = np.isin(labels, list(seen_set))
    out: dict[int, float] = {}
    for j, c in enumerate(class_ids):
        c = int(c)
        if c not in seen_set:
            continue
        out[c], _ = best_f1_threshold(scores[keep, j], labels[keep] == c)
    return out


def decision_report(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
    seen: Sequence[int],
    thresholds: Mapping[int, float] | None = None,
    *,
    include_matrix: bool = True,
) -> dict[str, Any]:
    """Which skills fire on which samples (see the module docstring); no ``argmax`` anywhere.

    Only test samples of seen classes are used.  ``thresholds`` maps class id -> threshold
    (missing / ``None`` means ``0``).
    """
    ids = [int(c) for c in class_ids]
    seen_set = {int(c) for c in seen}
    thresholds = thresholds or {}
    thr = np.array([float(thresholds.get(c, 0.0)) for c in ids])
    keep = np.isin(labels, list(seen_set))
    s, y = np.asarray(scores)[keep], np.asarray(labels)[keep]
    n = len(y)
    if n == 0:
        raise ValueError("decision_report needs test samples of at least one seen class")
    col = {c: j for j, c in enumerate(ids)}
    true_col = np.array([col[int(v)] for v in y])
    fired = s > thr[None, :]
    n_fired = fired.sum(1)
    true_fired = fired[np.arange(n), true_col]
    exact = true_fired & (n_fired == 1)
    single = n_fired == 1

    per_class: dict[int, dict[str, float | int | None]] = {}
    for c in ids:
        if c not in seen_set:
            continue
        j, is_c = col[c], y == c
        n_c = int(is_c.sum())
        tp = int(fired[is_c, j].sum())
        fp = int(fired[~is_c, j].sum())
        per_class[c] = {
            "n": n_c,
            "recall": tp / n_c if n_c else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "false_positive_rate": fp / int((~is_c).sum()) if (~is_c).any() else None,
            "exact_match_rate": float(exact[is_c].mean()) if n_c else None,
            "threshold": float(thr[j]),
        }

    def mean_of(key: str) -> float | None:
        v = [m[key] for m in per_class.values() if m[key] is not None]
        return float(np.mean(v)) if v else None

    report: dict[str, Any] = {
        "n_samples": int(n),
        "n_skills": len(per_class),
        "none_fired_rate": float((n_fired == 0).mean()),
        "single_fired_rate": float(single.mean()),
        "multi_fired_rate": float((n_fired > 1).mean()),
        "mean_skills_fired": float(n_fired.mean()),
        "mean_wrong_skills_fired": float((n_fired - true_fired).mean()),
        "true_skill_fires_rate": float(true_fired.mean()),
        "exact_match_rate": float(exact.mean()),
        "precision_when_single": float(exact.sum() / single.sum()) if single.any() else None,
        "macro_recall": mean_of("recall"),
        "macro_precision": mean_of("precision"),
        "macro_false_positive_rate": mean_of("false_positive_rate"),
        "macro_exact_match_rate": mean_of("exact_match_rate"),
        "per_class": per_class,
        "thresholds": {c: float(thr[col[c]]) for c in per_class},
    }
    if include_matrix:
        rows = [c for c in ids if c in seen_set]
        report["firing_matrix_classes"] = rows
        report["firing_matrix"] = [
            fired[y == c].mean(0).tolist() if (y == c).any() else [None] * len(ids) for c in rows
        ]
        report["firing_matrix_skills"] = ids
    return report
