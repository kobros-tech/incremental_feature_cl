# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plots for the target:negative ratio sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from ..evaluation.ratio_summary import ratio_sort_key
from ._common import finish

_PANELS = [
    (
        "Target recall vs negative accuracy (final)",
        [
            ("final_target_recall", "target recall"),
            ("final_negative_accuracy", "negative accuracy"),
        ],
    ),
    (
        "Balanced accuracy",
        [
            ("final_balanced_accuracy", "final"),
            ("average_balanced_accuracy", "mean over experiences"),
        ],
    ),
    (
        "Target F1 and false-positive rate (final)",
        [("final_target_f1", "target F1"), ("final_false_positive_rate", "false-positive rate")],
    ),
    (
        "Forgetting",
        [
            ("target_recall_peak_to_final_drop", "target recall: peak -> final drop"),
            ("average_forgetting", "avg forgetting (all classes)"),
        ],
    ),
]


def plot_ratio_main(aggregate, path):
    """The headline figure: balanced accuracy vs the *requested* ratio (mean +/- std over targets,
    one line per loss weighting), next to how many negatives per target were actually *realized*.

    Left: final and mean-over-experiences balanced accuracy. Right: realized vs requested negatives
    per target (final experience; mean over experiences) with the ``realized = requested`` diagonal;
    points below it mean the cumulative negative pool was too small to reach the requested ratio,
    which also affects the ``average`` (over experiences) metrics.
    """
    numeric = [a for a in aggregate if a["ratio"] is not None]
    labels = []
    for a in sorted(aggregate, key=lambda a: ratio_sort_key(a["ratio"])):
        if a["ratio_label"] not in labels:
            labels.append(a["ratio_label"])
    xs = {lab: i for i, lab in enumerate(labels)}
    pws = list(dict.fromkeys(a["pos_weight"] for a in aggregate))
    styles = ["-", "--", ":", "-."]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3))
    for pi, pw in enumerate(pws):
        aggs = sorted(
            (a for a in aggregate if a["pos_weight"] == pw), key=lambda a: xs[a["ratio_label"]]
        )
        x = np.array([xs[a["ratio_label"]] for a in aggs])
        tag = "" if len(pws) == 1 else f" [{pw}]"
        for key, name, color in (
            ("final_balanced_accuracy", "final", "C0"),
            ("average_balanced_accuracy", "mean over experiences", "C1"),
        ):
            m = np.array([a[f"mean_{key}"] for a in aggs], dtype=float)
            sd = np.array([a[f"std_{key}"] for a in aggs], dtype=float)
            ax.errorbar(
                x, m, yerr=sd, fmt=f"o{styles[pi % 4]}", color=color, capsize=3, label=name + tag
            )
        rx = [
            (
                1 / a["ratio"],
                a["mean_final_negatives_per_target"],
                a["mean_mean_negatives_per_target"],
            )
            for a in aggs
            if a["ratio"] is not None and a["mean_final_negatives_per_target"] is not None
        ]
        if rx:
            req, fin, avg = zip(*rx)
            bx.plot(req, fin, f"o{styles[pi % 4]}", color="C0", label="final experience" + tag)
            bx.plot(req, avg, f"s{styles[pi % 4]}", color="C1", label="mean over experiences" + tag)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("requested target : negative ratio")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(-0.02, 1.02)
    if "cumulative" in labels and len(labels) > 1:
        ax.axvline(len(labels) - 1.5, color="gray", ls=":", lw=1)  # reference, not a ratio
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Balanced accuracy vs requested ratio", fontsize=10)
    if numeric:
        reqs = [1 / a["ratio"] for a in numeric]
        bx.plot(
            [min(reqs), max(reqs)],
            [min(reqs), max(reqs)],
            color="gray",
            ls=":",
            lw=1,
            label="realized = requested",
        )
    bx.set_xscale("log")
    bx.set_yscale("log")
    if numeric:
        ticks = sorted({1 / a["ratio"] for a in numeric})
        for axis in (bx.xaxis, bx.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_locator(NullLocator())
    bx.set_xlabel("requested negatives per target")
    bx.set_ylabel("realized negatives per target")
    bx.grid(alpha=0.3, which="both")
    bx.legend(fontsize=8)
    bx.set_title("Realized vs requested ratio (below the line = pool too small)", fontsize=10)
    fig.suptitle("Target:negative ratio sweep (mean over targets)")
    return finish(fig, path)


def plot_ratio_final_metrics(rows, aggregate, path):
    """Final metrics vs ratio (5:1 -> 1:5, cumulative last). Mean +/- std over targets;
    thin lines are individual targets. One line style per loss-weighting setting."""
    labels = []
    for a in sorted(aggregate, key=lambda a: ratio_sort_key(a["ratio"])):
        if a["ratio_label"] not in labels:
            labels.append(a["ratio_label"])
    xs = {lab: i for i, lab in enumerate(labels)}
    pws = list(dict.fromkeys(a["pos_weight"] for a in aggregate))
    styles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (title, series) in zip(axes.flat, _PANELS):
        for j, (key, name) in enumerate(series):
            color = f"C{j}"
            for pi, pw in enumerate(pws):
                aggs = [
                    a for a in aggregate if a["pos_weight"] == pw and a[f"mean_{key}"] is not None
                ]
                aggs.sort(key=lambda a: xs[a["ratio_label"]])
                if not aggs:
                    continue
                x = np.array([xs[a["ratio_label"]] for a in aggs])
                m = np.array([a[f"mean_{key}"] for a in aggs])
                sd = np.array([a[f"std_{key}"] for a in aggs])
                ls = styles[pi % len(styles)]
                lab = name if len(pws) == 1 else f"{name} [{pw}]"
                ax.errorbar(x, m, yerr=sd, fmt=f"o{ls}", color=color, capsize=3, lw=2, label=lab)
                if max(a["n_targets"] for a in aggs) > 1:
                    for tc in sorted({r["target_class"] for r in rows}):
                        pts = [
                            (xs[r["ratio_label"]], r[key])
                            for r in rows
                            if r["target_class"] == tc
                            and r["pos_weight"] == pw
                            and r[key] is not None
                        ]
                        if pts:
                            pts.sort()
                            ax.plot(*zip(*pts), color=color, alpha=0.18, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.3 if title == "Forgetting" else -0.02, 1.02)  # drops can be negative
        if "cumulative" in labels and len(labels) > 1:
            ax.axvline(len(labels) - 1.5, color="gray", ls=":", lw=1)  # reference, not a ratio
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("target : negative training ratio")
    fig.suptitle("Effect of the target:negative ratio (final experience)")
    return finish(fig, path)


def plot_ratio_curves(curves_by_ratio, path, title_suffix=""):
    """Per-experience mean (over targets) curves, one line per ratio. ``curves_by_ratio``:
    ``{(ratio, label): {"target_recall": [...], "negative_accuracy": [...], "balanced_accuracy": [...]}}``."""
    items = sorted(curves_by_ratio.items(), key=lambda kv: ratio_sort_key(kv[0][0]))
    cmap = plt.get_cmap("viridis")
    n_num = max(1, sum(1 for (r, _), _c in items if r is not None) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    k = 0
    for (ratio, label), c in items:
        style = (
            {"color": "black", "ls": "--", "lw": 2}
            if ratio is None
            else {"color": cmap(k / n_num), "lw": 1.8}
        )
        k += ratio is not None
        for ax, key in zip(axes, ("target_recall", "negative_accuracy", "balanced_accuracy")):
            ax.plot(range(len(c[key])), c[key], marker="o", ms=3, label=label, **style)
    for ax, t in zip(
        axes, ("target recall", "negative accuracy (seen negatives)", "balanced accuracy")
    ):
        ax.set_title(t + title_suffix, fontsize=10)
        ax.set_xlabel("experience index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title="target:negative")
    return finish(fig, path)
