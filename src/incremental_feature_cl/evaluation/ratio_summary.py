# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Summaries for the target:negative ratio sweep (one row per run, aggregated across targets).

All inputs are :class:`RunResult` objects (or rows derived from them), so a sweep can be
re-summarised from saved ``results.json`` files without retraining.

Metric notes
------------
* ``negative_accuracy`` is measured on the negative classes *seen so far* (same as the target metrics).
* ``balanced_accuracy = (target_recall + negative_accuracy) / 2``. Overall binary accuracy depends on
  the test-set class mix, so balanced accuracy is the fairer headline when comparing ratios.
* ``target_recall_peak_to_final_drop`` = max earlier target recall - final target recall. This is
  exactly the standard per-class forgetting (``forgetting_per_class``) of the target class, whose
  per-class accuracy *is* its recall (tested). It is measured against the *peak*, so a model that
  dipped and then partly recovered still reports the drop from its best earlier recall. It says
  nothing about the negative classes; ``average_forgetting`` is the mean over *all* classes.
* Requested vs realized ratio. ``ratio`` / ``ratio_label`` in a row are the **requested** ratio, an
  upper bound on negatives per target: negatives are never duplicated, so while the cumulative
  negative pool is smaller than requested an experience trains on fewer negatives than asked
  (requested 1:5 -> realized 1:3). The ``realized`` columns are derived from the recorded
  ``n_train_target`` / ``n_train_negative`` (so results saved before the ratio report existed still
  work): ``fraction_experiences_at_requested_ratio`` is the share of experiences that got all
  requested negatives, ``first_experience_at_requested_ratio`` the first one that did,
  ``final_negatives_per_target`` / ``final_realized_ratio_label`` describe the last experience and
  ``mean_negatives_per_target`` averages over experiences. Averages over experiences
  (``average_*``) mix saturated and unsaturated experiences; the ``final_*`` metrics at a late
  experience usually have the full requested ratio.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..data.streams import format_ratio, ratio_report
from .result_store import RunResult

METRIC_KEYS = (
    "final_target_recall",
    "final_negative_accuracy",
    "final_balanced_accuracy",
    "final_target_f1",
    "final_false_positive_rate",
    "final_overall_binary_accuracy",
    "average_target_recall",
    "average_negative_accuracy",
    "average_balanced_accuracy",
    "target_recall_peak_to_final_drop",
    "average_forgetting",
    "mean_negatives_per_target",
    "final_negatives_per_target",
    "requested_negatives_per_target",
    "fraction_experiences_at_requested_ratio",
    "total_train_samples",
    "total_train_time_s",
)


def ratio_sort_key(ratio: float | None) -> tuple[int, float]:
    """5:1 ... 1:5 (descending targets-per-negative), the cumulative reference last."""
    return (1, 0.0) if ratio is None else (0, -float(ratio))


def run_curves(result: RunResult) -> dict[str, list[float]]:
    """Per-experience target recall / negative accuracy / balanced accuracy of one run."""
    tm = [e["target_metrics"] for e in result.experiences]
    recall = [m["target_recall"] for m in tm]
    neg = [m["negative_accuracy"] for m in tm]
    return {
        "target_recall": recall,
        "negative_accuracy": neg,
        "balanced_accuracy": [(r + n) / 2 for r, n in zip(recall, neg)],
    }


def realized_profile(result: RunResult) -> list[dict[str, Any]]:
    """Requested-vs-realized ratio report of every experience (see ``data.streams.ratio_report``).

    Derived from ``n_train_target`` / ``n_train_negative`` and the configured ratio rather than the
    stored ``ratio_satisfied`` fields, so it also works for results saved without them.
    """
    ratio = result.config.get("target_to_negatives")
    return [
        ratio_report(e["n_train_target"], e["n_train_negative"], ratio) for e in result.experiences
    ]


def run_metrics(result: RunResult) -> dict[str, Any]:
    """Final / average metrics of one target-vs-rest run."""
    c = run_curves(result)
    rec, neg, bal = c["target_recall"], c["negative_accuracy"], c["balanced_accuracy"]
    last = result.experiences[-1]["target_metrics"]
    n_t = [e.get("n_train_target") for e in result.experiences]
    n_n = [e.get("n_train_negative") for e in result.experiences]
    per_target = [n / t for t, n in zip(n_t, n_n) if t and n is not None]
    summary = result.summary or result.compute_summary()
    profile = realized_profile(result)
    checked = [p["ratio_satisfied"] for p in profile if p["ratio_satisfied"] is not None]
    first_ok = next((i for i, p in enumerate(profile) if p["ratio_satisfied"]), None)
    requested = result.config.get("target_to_negatives")
    return {
        "final_target_recall": rec[-1],
        "final_negative_accuracy": neg[-1],
        "final_balanced_accuracy": bal[-1],
        "final_target_f1": last["target_f1"],
        "final_false_positive_rate": last["false_positive_rate"],
        "final_overall_binary_accuracy": last["overall_binary_accuracy"],
        "average_target_recall": float(np.mean(rec)),
        "average_negative_accuracy": float(np.mean(neg)),
        "average_balanced_accuracy": float(np.mean(bal)),
        "target_recall_peak_to_final_drop": (max(rec[:-1]) - rec[-1]) if len(rec) > 1 else 0.0,
        "average_forgetting": summary.get("average_forgetting"),
        "mean_negatives_per_target": float(np.mean(per_target)) if per_target else None,
        "final_negatives_per_target": (n_n[-1] / n_t[-1]) if n_t[-1] else None,
        "final_realized_ratio_label": profile[-1]["realized_ratio_label"],
        "requested_negatives_per_target": None if requested is None else 1.0 / requested,
        "fraction_experiences_at_requested_ratio": (float(np.mean(checked)) if checked else None),
        "first_experience_at_requested_ratio": first_ok if checked else None,
        "total_train_samples": sum(e.get("n_train_samples") or 0 for e in result.experiences),
        "total_train_time_s": summary.get("total_train_time_s"),
    }


def make_row(
    result: RunResult, target_class: int, ratio: float | None, pos_weight_label: str
) -> dict[str, Any]:
    return {
        "target_class": target_class,
        "ratio": ratio,  # the *requested* ratio (targets per negative); see module docstring
        "ratio_label": format_ratio(ratio),
        "pos_weight": pos_weight_label,
        **run_metrics(result),
    }


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std across targets for each (pos_weight, ratio); ordered 5:1 ... 1:5, cumulative last."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["pos_weight"], r["ratio_label"]), []).append(r)
    out = []
    for (pw, label), rs in groups.items():
        agg: dict[str, Any] = {
            "pos_weight": pw,
            "ratio": rs[0]["ratio"],
            "ratio_label": label,
            "n_targets": len(rs),
        }
        for k in METRIC_KEYS:
            vals = _finite(r[k] for r in rs)
            agg[f"mean_{k}"] = float(np.mean(vals)) if vals else None
            agg[f"std_{k}"] = float(np.std(vals)) if vals else None
        out.append(agg)
    pws = list(dict.fromkeys(r["pos_weight"] for r in rows))
    out.sort(key=lambda a: (pws.index(a["pos_weight"]), *ratio_sort_key(a["ratio"])))
    return out


def mean_curves(curves: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Mean over targets of per-experience curves (all runs share the number of experiences)."""
    return {k: np.mean([c[k] for c in curves], axis=0).tolist() for k in curves[0]}
