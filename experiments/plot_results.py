# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plot saved JSON results without retraining the model."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STREAM_METHODS = ("naive", "replay")


def _curve(runs: list[dict], key: str) -> np.ndarray:
    """Mean ``key`` over the classes seen so far; shape (n_seeds, n_experiences)."""
    return np.array(
        [[np.mean([m[key] for m in step.values()]) for step in r["metrics"]] for r in runs]
    )


def _arrival_matrix(runs: list[dict], key: str) -> np.ndarray:
    """``[experience t, arrival group a]``: mean ``key`` after t of the classes that arrived at a.

    Averaged over seeds.  Class order is shuffled per seed, so classes are grouped by the
    experience they arrived in; cells with ``t < a`` (class not yet seen) are NaN.
    """
    n = len(runs[0]["metrics"])
    out = np.full((len(runs), n, n), np.nan)
    for i, r in enumerate(runs):
        for t, step in enumerate(r["metrics"]):
            for a in range(t + 1):
                vals = [step[c][key] for c, arr in r["arrival"].items() if arr == a and c in step]
                out[i, t, a] = np.mean(vals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells with t < a are all-NaN by design
        return np.nanmean(out, axis=0)


def plot_continual(data: dict, output_dir: Path) -> list[Path]:
    """Plots for ``experiments.continual_eval`` results; returns the written paths."""
    runs = data["runs"]
    methods = list(runs)
    saved: list[Path] = []

    def save(fig, name):
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
        print(f"Saved: {path}")

    for key in ("recall", "accuracy"):
        fig, ax = plt.subplots()
        for m in methods:
            c = _curve(runs[m], key)
            x = np.arange(c.shape[1])
            ax.plot(x, c.mean(0), marker="o", label=m)
            if len(c) > 1:
                ax.fill_between(
                    x, c.mean(0) - c.std(0, ddof=1), c.mean(0) + c.std(0, ddof=1), alpha=0.15
                )
        ax.set_xlabel("Experience")
        ax.set_ylabel(f"Mean {key} over classes seen so far")
        ax.set_xticks(range(len(x)))
        ax.set_ylim(0, 1.02)
        ax.legend()
        save(fig, f"{key}_over_experiences.png")

    for m in (m for m in STREAM_METHODS if m in runs):
        mat = _arrival_matrix(runs[m], "recall")
        fig, ax = plt.subplots()
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", origin="upper")
        ax.set_xlabel("Experience in which the classes arrived")
        ax.set_ylabel("Evaluated after experience")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        for t in range(mat.shape[0]):
            for a in range(t + 1):
                ax.text(
                    a,
                    t,
                    f"{mat[t, a]:.2f}",
                    ha="center",
                    va="center",
                    color="k" if mat[t, a] > 0.6 else "w",
                    fontsize=8,
                )
        ax.set_title(f"Recall of each arrival group - {m}")
        fig.colorbar(im, ax=ax, label="recall")
        save(fig, f"recall_matrix_{m}.png")

    fig, ax = plt.subplots()
    keys = ("final_recall", "final_specificity", "final_accuracy", "recall_forgetting")
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [data["aggregate"][m][k]["mean"] for k in keys]
        errs = [np.nan_to_num(data["aggregate"][m][k]["std"]) for k in keys]
        ax.bar(np.arange(len(keys)) + i * width, vals, width, yerr=errs, capsize=3, label=m)
    ax.set_xticks(np.arange(len(keys)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([k.replace("_", " ") for k in keys])
    ax.set_ylim(0, 1.02)
    ax.legend()
    save(fig, "summary_bars.png")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "runs" in data and "aggregate" in data:
        plot_continual(data, args.output_dir)
        return

    if "summary" in data:
        summary = data["summary"]
        if summary and "ratio" in summary[0]:
            labels = [
                "all negatives" if row["ratio"] is None else str(row["ratio"]) for row in summary
            ]
            xlabel = "target_to_negatives (target samples per negative)"
        else:
            labels = [
                "replay off" if row["replay_mem_size"] == 0 else f"memory={row['replay_mem_size']}"
                for row in summary
            ]
            xlabel = "Replay memory capacity"
        means = [row["mean_final_balanced_accuracy"] for row in summary]
        errors = [row["std_final_balanced_accuracy"] or 0.0 for row in summary]
        valid = [
            (label, mean, error)
            for label, mean, error in zip(labels, means, errors)
            if mean is not None
        ]
        if not valid:
            raise ValueError("No finite summary metrics found in the results file.")
        labels, means, errors = zip(*valid)
        fig, ax = plt.subplots()
        ax.bar(range(len(labels)), means, yerr=errors, capsize=4)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final balanced accuracy")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = args.output_dir / "ratio_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved: {path}")
        return

    history = data.get("history")
    if not history:
        raise ValueError("Expected a single-run file containing a non-empty 'history' list.")
    x = [row["experience"] for row in history]
    for metric in ("target_recall", "negative_specificity", "balanced_accuracy", "roc_auc"):
        fig, ax = plt.subplots()
        ax.plot(x, [row[metric] for row in history], marker="o")
        ax.set_xlabel("Experience")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = args.output_dir / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
