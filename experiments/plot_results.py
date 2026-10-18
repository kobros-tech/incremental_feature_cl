# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plot saved JSON results without retraining the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
