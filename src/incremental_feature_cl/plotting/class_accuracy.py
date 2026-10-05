# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plot B (accuracy vs class), C (accuracy heatmap), E (forgetting vs class)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..evaluation.forgetting import forgetting_per_class
from ..evaluation.result_store import RunResult
from ._common import finish


def plot_accuracy_vs_class(result: RunResult, path, experience: int = -1):
    """Plot B: accuracy of every class after a chosen experience (default: the last)."""
    acc = result.accuracy_matrix()
    t = experience % acc.shape[0]
    seen = set(result.experiences[t]["seen_classes"])
    cols = [
        "crimson" if c == result.target_class else ("tab:blue" if c in seen else "lightgray")
        for c in range(acc.shape[1])
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(acc.shape[1]), np.nan_to_num(acc[t]), color=cols)
    ax.set_xlabel("class index")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Accuracy per class after experience {t} (grey = not yet seen)")
    return finish(fig, path)


def plot_accuracy_heatmap(result: RunResult, path):
    """Plot C: rows = experience, columns = class, cell = class accuracy."""
    acc = result.accuracy_matrix()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        np.ma.masked_invalid(acc),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
        origin="upper",
        interpolation="nearest",
    )
    ax.set_xlabel("class index")
    ax.set_ylabel("experience index")
    ax.set_title("Class accuracy after each experience")
    fig.colorbar(im, ax=ax, label="accuracy")
    if result.target_class is not None:
        ax.axvline(result.target_class, color="red", lw=0.6, alpha=0.7)
    return finish(fig, path)


def plot_forgetting_vs_class(result: RunResult, path):
    """Plot E: max previous accuracy - final accuracy, per class."""
    f = forgetting_per_class(result.accuracy_matrix(), result.class_first_experience)
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["crimson" if c == result.target_class else "tab:orange" for c in range(len(f))]
    ax.bar(range(len(f)), np.nan_to_num(f), color=cols)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("class index")
    ax.set_ylabel("forgetting")
    ax.set_title("Forgetting per class (max previous - final accuracy)")
    return finish(fig, path)
