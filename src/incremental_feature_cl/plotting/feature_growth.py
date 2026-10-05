# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plot F (feature-space growth), G (new-feature utilisation), H (expansion invariance probe)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..evaluation.result_store import RunResult
from ._common import finish


def plot_feature_growth(result: RunResult, path):
    x = [e["index"] for e in result.experiences]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(x, result.series("feature_dim"), where="post", color="tab:blue", lw=2)
    ax.set_xlabel("experience index")
    ax.set_ylabel("feature dimensionality", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(x, result.series("parameter_count"), "o--", color="tab:green")
    ax2.set_ylabel("parameter count", color="tab:green")
    ax.set_title("Feature-space and parameter growth")
    ax.grid(alpha=0.3)
    return finish(fig, path)


def plot_new_feature_utilization(result: RunResult, path):
    """||W_k|| per classifier block, and mean |W_k phi_k| of the newest block, over experiences."""
    x = [e["index"] for e in result.experiences]
    n_blocks = max(len(e["classifier_block_norms"]) for e in result.experiences)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k in range(n_blocks):
        ys = [
            e["classifier_block_norms"][k] if k < len(e["classifier_block_norms"]) else None
            for e in result.experiences
        ]
        pts = [(xi, yi) for xi, yi in zip(x, ys) if yi is not None]
        axes[0].plot(*zip(*pts), "o-", label="backbone block" if k == 0 else f"new block {k}")
    axes[0].set_xlabel("experience index")
    axes[0].set_ylabel("||W_k|| (Frobenius)")
    axes[0].set_title("Classifier weight norm per feature block")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    newest = [
        e["block_contribution"][-1] if len(e["block_contribution"]) > 1 else None
        for e in result.experiences
    ]
    pts = [(xi, yi) for xi, yi in zip(x, newest) if yi is not None]
    if pts:
        axes[1].plot(*zip(*pts), "o-", color="tab:red")
    axes[1].set_xlabel("experience index")
    axes[1].set_ylabel("mean |W_new phi_new|")
    axes[1].set_title("Logit contribution of the newest block")
    axes[1].grid(alpha=0.3)
    return finish(fig, path)


def plot_expansion_probe(result: RunResult, path):
    """Did expansion / optimisation change predictions of already-seen classes?"""
    ex = [e for e in result.experiences if e.get("probe")]
    if not ex:
        return None
    x = [e["index"] for e in ex]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(x, [max(e["probe"]["logit_max_abs_diff_expansion"], 1e-12) for e in ex], "o-")
    axes[0].set_title("max |logit change| caused by expansion\n(0 => invariant; floor 1e-12)")
    axes[0].set_xlabel("experience index")
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        x, [e["probe"]["pred_agreement_expansion"] for e in ex], "o-", label="after expansion"
    )
    axes[1].plot(
        x, [e["probe"]["pred_agreement_after_training"] for e in ex], "s--", label="after training"
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Old-prediction agreement with previous state")
    axes[1].set_xlabel("experience index")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    return finish(fig, path)
