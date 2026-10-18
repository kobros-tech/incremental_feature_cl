# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Compare replay disabled with one or more bounded-memory settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.run_target import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["digits", "synthetic", "cifar100"], default="digits")
    parser.add_argument("--target-class", type=int, default=5)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--replay-sizes", type=int, nargs="+", default=[0, 200])
    parser.add_argument("--new-feature-dim", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--target-to-negatives", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/replay_comparison.json"))
    args = parser.parse_args()

    if any(size != 0 and size < 2 for size in args.replay_sizes):
        parser.error("each replay size must be 0 (off) or at least 2")

    runs = []
    for replay_size in args.replay_sizes:
        for seed in args.seeds:
            result = run_experiment(
                dataset=args.dataset,
                target_class=args.target_class,
                n_experiences=args.n_experiences,
                seed=seed,
                new_feature_dim=args.new_feature_dim,
                n_epochs=args.n_epochs,
                replay_mem_size=replay_size,
                target_to_negatives=args.target_to_negatives,
            )
            runs.append(
                {
                    "replay_mem_size": replay_size,
                    "seed": seed,
                    "final": result["history"][-1],
                }
            )

    summary = []
    for replay_size in args.replay_sizes:
        final_rows = [run["final"] for run in runs if run["replay_mem_size"] == replay_size]
        scores = [
            row["balanced_accuracy"] for row in final_rows if np.isfinite(row["balanced_accuracy"])
        ]
        forgetting = [
            row["target_recall_forgetting"]
            for row in final_rows
            if row["target_recall_forgetting"] is not None
        ]
        summary.append(
            {
                "replay_mem_size": replay_size,
                "n_runs": len(scores),
                "mean_final_balanced_accuracy": float(np.mean(scores)) if scores else None,
                "std_final_balanced_accuracy": float(np.std(scores)) if scores else None,
                "mean_final_target_recall_forgetting": (
                    float(np.mean(forgetting)) if forgetting else None
                ),
            }
        )

    output = {
        "dataset": args.dataset,
        "target_class": args.target_class,
        "n_experiences": args.n_experiences,
        "seeds": args.seeds,
        "runs": runs,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
