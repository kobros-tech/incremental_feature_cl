# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Compare target:negative sampling ratios over multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.data import resolve_dataset
from experiments.run_target import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="digits",
        help="digits, synthetic, cifar100, or the path of an extracted CIFAR-100 directory.",
    )
    parser.add_argument("--target-class", type=int, default=5)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 1 / 3, 0.5, 1.0, 2.0])
    parser.add_argument("--include-all-negatives", action="store_true")
    parser.add_argument("--new-feature-dim", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--replay-mem-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("results/ratio_comparison.json"))
    args = parser.parse_args()
    try:
        dataset_name, data_dir = resolve_dataset(args.dataset)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    ratios: list[float | None] = list(args.ratios)
    if args.include_all_negatives:
        ratios.insert(0, None)
    runs = []
    for ratio in ratios:
        for seed in args.seeds:
            result = run_experiment(
                dataset=args.dataset,
                target_class=args.target_class,
                n_experiences=args.n_experiences,
                seed=seed,
                new_feature_dim=args.new_feature_dim,
                n_epochs=args.n_epochs,
                replay_mem_size=args.replay_mem_size,
                target_to_negatives=ratio,
            )
            runs.append(
                {
                    "ratio": ratio,
                    "seed": seed,
                    "final": result["history"][-1],
                }
            )

    summary = []
    for ratio in ratios:
        values = [
            run["final"]["balanced_accuracy"]
            for run in runs
            if run["ratio"] == ratio and np.isfinite(run["final"]["balanced_accuracy"])
        ]
        summary.append(
            {
                "ratio": ratio,
                "n_runs": len(values),
                "mean_final_balanced_accuracy": float(np.mean(values)) if values else None,
                "std_final_balanced_accuracy": float(np.std(values)) if values else None,
            }
        )

    output = {
        "dataset": dataset_name,
        "data_dir": None if data_dir is None else str(data_dir),
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
