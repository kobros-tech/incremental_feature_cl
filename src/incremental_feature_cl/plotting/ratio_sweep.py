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
    """Headline ratio-sweep figure.

    Left: final balanced accuracy versus requested ratio.
    Right: requested versus realized negatives per target.

    Results are aggregated over targets. Different loss-weighting settings
    are shown with different line styles.
    """
    numeric = [a for a in aggregate if a["ratio"] is not None]

    labels = []
    for a in sorted(aggregate, key=lambda a: ratio_sort_key(a["ratio"])):
        if a["ratio_label"] not in labels:
            labels.append(a["ratio_label"])

    xs = {label: i for i, label in enumerate(labels)}
    pws = list(dict.fromkeys(a["pos_weight"] for a in aggregate))
    styles = ["-", "--", ":", "-."]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3))

    # Performance panel.
    for pi, pw in enumerate(pws):
        aggs = sorted(
            (
                a
                for a in aggregate
                if a["pos_weight"] == pw and a["mean_final_balanced_accuracy"] is not None
            ),
            key=lambda a: xs[a["ratio_label"]],
        )
        if not aggs:
            continue

        x = np.array([xs[a["ratio_label"]] for a in aggs])
        mean = np.array(
            [a["mean_final_balanced_accuracy"] for a in aggs],
            dtype=float,
        )
        std = np.array(
            [a["std_final_balanced_accuracy"] for a in aggs],
            dtype=float,
        )

        label = "final balanced accuracy" if len(pws) == 1 else f"final balanced accuracy [{pw}]"

        ax.errorbar(
            x,
            mean,
            yerr=std,
            fmt=f"o{styles[pi % len(styles)]}",
            capsize=3,
            label=label,
        )

    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("requested target : negative ratio")
    ax.set_ylabel("final balanced accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    if "cumulative" in labels and len(labels) > 1:
        ax.axvline(
            len(labels) - 1.5,
            ls=":",
            lw=1,
        )

    ax.set_title("Final balanced accuracy", fontsize=10)
    ax.legend(fontsize=8)

    # Sampling-fidelity panel.
    for pi, pw in enumerate(pws):
        aggs = sorted(
            (
                a
                for a in aggregate
                if a["pos_weight"] == pw
                and a["ratio"] is not None
                and a["mean_final_negatives_per_target"] is not None
            ),
            key=lambda a: ratio_sort_key(a["ratio"]),
        )
        if not aggs:
            continue

        requested = np.array(
            [1 / a["ratio"] for a in aggs],
            dtype=float,
        )
        realized = np.array(
            [a["mean_final_negatives_per_target"] for a in aggs],
            dtype=float,
        )

        label = "realized" if len(pws) == 1 else f"realized [{pw}]"

        bx.plot(
            requested,
            realized,
            marker="o",
            linestyle=styles[pi % len(styles)],
            label=label,
        )

    if numeric:
        requested = np.array(
            [1 / a["ratio"] for a in numeric],
            dtype=float,
        )
        lo = requested.min()
        hi = requested.max()

        bx.plot(
            [lo, hi],
            [lo, hi],
            linestyle=":",
            lw=1,
            label="requested = realized",
        )

    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("requested negatives per target")
    bx.set_ylabel("realized negatives per target")
    bx.set_title("Sampling fidelity", fontsize=10)
    bx.grid(alpha=0.3, which="both")
    bx.legend(fontsize=8)

    if numeric:
        ticks = sorted({1 / a["ratio"] for a in numeric})
        for axis in (bx.xaxis, bx.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.set_minor_locator(NullLocator())

    fig.suptitle("Target:negative ratio sweep")
    return finish(fig, path)


def plot_ratio_final_metrics(rows, aggregate, path):
    """Final target-specific metrics versus the requested ratio.

    Aggregate mean +/- std is shown as the main line. Individual target
    trajectories are shown only when multiple target classes are present.
    """
    labels = []
    for a in sorted(aggregate, key=lambda a: ratio_sort_key(a["ratio"])):
        if a["ratio_label"] not in labels:
            labels.append(a["ratio_label"])

    xs = {label: i for i, label in enumerate(labels)}
    pws = list(dict.fromkeys(a["pos_weight"] for a in aggregate))
    styles = ["-", "--", ":", "-."]

    panels = [
        ("Final target recall", "final_target_recall"),
        ("Final negative accuracy", "final_negative_accuracy"),
        ("Final balanced accuracy", "final_balanced_accuracy"),
        ("Target recall forgetting", "target_recall_peak_to_final_drop"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        sharex=True,
    )

    multiple_targets = (
        max(
            (a["n_targets"] for a in aggregate),
            default=1,
        )
        > 1
    )

    for ax, (title, key) in zip(axes.flat, panels):
        for pi, pw in enumerate(pws):
            aggs = [a for a in aggregate if a["pos_weight"] == pw and a[f"mean_{key}"] is not None]
            aggs.sort(key=lambda a: xs[a["ratio_label"]])

            if not aggs:
                continue

            x = np.array([xs[a["ratio_label"]] for a in aggs])
            mean = np.array(
                [a[f"mean_{key}"] for a in aggs],
                dtype=float,
            )
            std = np.array(
                [a[f"std_{key}"] for a in aggs],
                dtype=float,
            )

            label = "mean +/- std" if len(pws) == 1 else f"mean +/- std [{pw}]"

            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt=f"o{styles[pi % len(styles)]}",
                capsize=3,
                label=label,
            )

            if multiple_targets:
                for target_class in sorted({r["target_class"] for r in rows}):
                    points = [
                        (xs[r["ratio_label"]], r[key])
                        for r in rows
                        if r["target_class"] == target_class
                        and r["pos_weight"] == pw
                        and r[key] is not None
                    ]

                    if points:
                        points.sort()
                        px, py = zip(*points)
                        ax.plot(
                            px,
                            py,
                            alpha=0.2,
                            lw=0.8,
                        )

        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

        if key == "target_recall_peak_to_final_drop":
            ax.set_ylim(-0.3, 1.02)
        else:
            ax.set_ylim(-0.02, 1.02)

        if "cumulative" in labels and len(labels) > 1:
            ax.axvline(
                len(labels) - 1.5,
                ls=":",
                lw=1,
            )

        ax.legend(fontsize=7)

    for ax in axes[1]:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("requested target : negative training ratio")

    fig.suptitle("Final metrics vs target:negative ratio")
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
