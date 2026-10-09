# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plot A (target/overall accuracy vs experience) and Plot D (per-class accuracy vs experience)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..evaluation.result_store import RunResult
from ._common import finish


def plot_accuracy_vs_experience(result: RunResult, path):
    """Plot A. Target mode: target accuracy (+ negative acc / F1). Multiclass: seen/all accuracy."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [e["index"] for e in result.experiences]
    if result.mode == "target":
        tm = [e["target_metrics"] for e in result.experiences]
        who = (
            f"target class {result.target_class}"
            if result.target_class is not None
            else "mean over skills"
        )
        ax.plot(
            x,
            [m["target_accuracy"] for m in tm],
            "o-",
            lw=2.5,
            label=f"{who} accuracy (recall)",
        )
        ax.plot(x, [m["negative_accuracy"] for m in tm], "s--", label="seen-negative accuracy")
        ax.plot(x, [m["target_f1"] for m in tm], "^:", label="target F1")
        if all(m.get("balanced_accuracy") is not None for m in tm):
            ax.plot(x, [m["balanced_accuracy"] for m in tm], "d-.", label="balanced accuracy")
        ax.set_title("Target-vs-rest: target accuracy vs experience")
    else:
        ax.plot(x, result.series("accuracy_seen"), "o-", lw=2.5, label="accuracy (seen classes)")
        ax.plot(x, result.series("accuracy_all"), "s--", label="accuracy (all classes)")
        ax.set_title("Accuracy vs experience")
    ax.set_xlabel("experience index")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return finish(fig, path)


def plot_class_accuracy_curves(result: RunResult, path):
    """Plot D: one line per class, starting at the experience where it was introduced."""
    acc = result.accuracy_matrix()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    first = result.class_first_experience
    for c in range(acc.shape[1]):
        if c not in first:
            continue
        t0 = first[c]
        is_t = c == result.target_class
        ax.plot(
            range(t0, acc.shape[0]),
            acc[t0:, c],
            lw=2.8 if is_t else 0.8,
            alpha=1.0 if is_t else 0.45,
            color="crimson" if is_t else None,
            label=f"target {c}" if is_t else None,
            zorder=5 if is_t else 1,
        )
    ax.set_xlabel("experience index")
    ax.set_ylabel("class accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-class accuracy over experiences (drops = forgetting)")
    ax.grid(alpha=0.3)
    if result.target_class is not None:
        ax.legend()
    return finish(fig, path)
