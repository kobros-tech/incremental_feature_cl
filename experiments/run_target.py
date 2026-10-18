# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Run a reproducible class-incremental target-versus-rest experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.data import class_incremental_indices, load_dataset
from experiments.metrics import target_metrics
from incremental_feature_cl import IncrementalFeatureMapModel


def run_experiment(
    dataset: str = "digits",
    target_class: int = 5,
    n_experiences: int = 5,
    seed: int = 0,
    new_feature_dim: int = 8,
    n_epochs: int = 5,
    replay_mem_size: int = 200,
    target_to_negatives: float | None = None,
    train_blocks: bool = True,
    data_dir: str | Path | None = None,
) -> dict:
    X_train, X_test, y_train, y_test = load_dataset(dataset, seed, data_dir=data_dir)
    if target_class not in np.unique(y_train):
        raise ValueError(f"target_class={target_class} is absent from the training split.")

    model = IncrementalFeatureMapModel(
        new_feature_dim=new_feature_dim,
        n_epochs=n_epochs,
        replay_mem_size=replay_mem_size,
        target_to_negatives=target_to_negatives,
        train_blocks=train_blocks,
        random_state=seed,
    )
    y_test_binary = y_test == target_class
    history = []
    target_seen = False
    best_target_recall = None
    streams = class_incremental_indices(y_train, n_experiences)

    for experience, indices in enumerate(streams):
        y_current = (y_train[indices] == target_class).astype(int)
        model.partial_fit(X_train[indices], y_current, classes=np.array([0, 1]))
        metrics = target_metrics(y_test_binary, model.decision_function(X_test))
        arriving_classes = np.unique(y_train[indices]).astype(int).tolist()
        target_seen = target_seen or target_class in arriving_classes
        target_recall = metrics["target_recall"]
        if target_seen and np.isfinite(target_recall):
            if best_target_recall is None or target_recall > best_target_recall:
                best_target_recall = target_recall
            forgetting = max(0.0, best_target_recall - target_recall)
        else:
            forgetting = None
        history.append(
            {
                "experience": experience,
                "classes_arriving": arriving_classes,
                "n_train": len(indices),
                "n_target_train": int(y_current.sum()),
                "n_negative_train": int((y_current == 0).sum()),
                "feature_dim": int(model.feature_dim_),
                "replay_mem_size": int(replay_mem_size),
                "target_to_negatives": target_to_negatives,
                "best_target_recall": best_target_recall,
                "target_recall_forgetting": forgetting,
                **metrics,
            }
        )

    return {
        "dataset": dataset,
        "target_class": int(target_class),
        "n_experiences": int(n_experiences),
        "seed": int(seed),
        "new_feature_dim": int(new_feature_dim),
        "n_epochs": int(n_epochs),
        "replay_mem_size": int(replay_mem_size),
        "target_to_negatives": target_to_negatives,
        "train_blocks": bool(train_blocks),
        "history": history,
        "final_target_recall_forgetting": history[-1]["target_recall_forgetting"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["digits", "synthetic", "cifar100"], default="digits")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Extracted CIFAR-100 directory (contains train/test or a cifar-100-python subdirectory).",
    )
    parser.add_argument("--target-class", type=int, default=5)
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new-feature-dim", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--replay-mem-size", type=int, default=200)
    parser.add_argument(
        "--target-to-negatives",
        type=float,
        default=None,
        help="Target:negative ratio as a float (0.2 means 1:5); omit to keep all negatives.",
    )
    parser.add_argument("--fixed-blocks", action="store_true", help="Do not train feature blocks.")
    parser.add_argument("--output", type=Path, default=Path("results/target_run.json"))
    args = parser.parse_args()

    result = run_experiment(
        dataset=args.dataset,
        target_class=args.target_class,
        n_experiences=args.n_experiences,
        seed=args.seed,
        new_feature_dim=args.new_feature_dim,
        n_epochs=args.n_epochs,
        replay_mem_size=args.replay_mem_size,
        target_to_negatives=args.target_to_negatives,
        train_blocks=not args.fixed_blocks,
        data_dir=args.data_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["history"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
