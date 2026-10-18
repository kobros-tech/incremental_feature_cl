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


def _score_diagnostics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Summarise target/negative decision scores at the model's zero threshold."""
    target_scores = scores[y_true]
    negative_scores = scores[~y_true]
    return {
        "target_score_mean": float(np.mean(target_scores)),
        "target_score_median": float(np.median(target_scores)),
        "target_score_p10": float(np.percentile(target_scores, 10)),
        "target_fraction_score_ge_zero": float(np.mean(target_scores >= 0.0)),
        "negative_score_mean": float(np.mean(negative_scores)),
        "negative_score_median": float(np.median(negative_scores)),
        "negative_score_p90": float(np.percentile(negative_scores, 90)),
        "negative_fraction_score_ge_zero": float(np.mean(negative_scores >= 0.0)),
        "target_minus_negative_mean_score": float(
            np.mean(target_scores) - np.mean(negative_scores)
        ),
    }


def _coefficient_diagnostics(model: IncrementalFeatureMapModel) -> dict:
    """Report coefficient norms by raw input and appended feature block."""
    offset = int(model.n_features_in_)
    block_norms = []
    for block in model.blocks_:
        end = offset + block.out_dim
        block_norms.append(float(np.linalg.norm(model.coef_[0, offset:end])))
        offset = end
    return {
        "input_coefficient_l2": float(np.linalg.norm(model.coef_[0, : model.n_features_in_])),
        "feature_block_coefficient_l2": block_norms,
        "intercept": float(model.intercept_[0]),
    }


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
    replay_pos_weight: str = "current",
) -> dict:
    if n_experiences < 1:
        raise ValueError("n_experiences must be at least 1.")

    X_train, X_test, y_train, y_test = load_dataset(dataset, seed, data_dir=data_dir)
    if target_class not in np.unique(y_train):
        raise ValueError(f"target_class={target_class} is absent from the training split.")

    y_test_binary = y_test == target_class
    if not np.any(y_test_binary) or not np.any(~y_test_binary):
        raise ValueError("The test split must contain both target and negative examples.")

    model = IncrementalFeatureMapModel(
        new_feature_dim=new_feature_dim,
        n_epochs=n_epochs,
        replay_mem_size=replay_mem_size,
        target_to_negatives=target_to_negatives,
        train_blocks=train_blocks,
        replay_pos_weight=replay_pos_weight,
        random_state=seed,
    )
    history = []
    target_seen = False
    best_target_recall = None
    streams = class_incremental_indices(y_train, n_experiences)

    for experience, indices in enumerate(streams):
        y_current = (y_train[indices] == target_class).astype(int)
        if hasattr(model, "classes_"):
            before_score_diagnostics = _score_diagnostics(
                y_test_binary, model.decision_function(X_test)
            )
            coef_before = model.coef_.copy()
            intercept_before = float(model.intercept_[0])
            memory_before = {
                "negative": len(model._mem[0]),
                "target": len(model._mem[1]),
            }
        else:
            before_score_diagnostics = None
            coef_before = None
            intercept_before = None
            memory_before = None

        model.partial_fit(X_train[indices], y_current, classes=np.array([0, 1]))
        scores_after = model.decision_function(X_test)
        metrics = target_metrics(y_test_binary, scores_after)
        after_score_diagnostics = _score_diagnostics(y_test_binary, scores_after)
        if coef_before is None:
            coefficient_change = None
        else:
            old_width = coef_before.shape[1]
            coefficient_change = {
                "existing_coefficient_delta_l2": float(
                    np.linalg.norm(model.coef_[0, :old_width] - coef_before[0])
                ),
                "newly_added_coefficient_l2": float(np.linalg.norm(model.coef_[0, old_width:])),
                "intercept_before": intercept_before,
                "intercept_after": float(model.intercept_[0]),
                "intercept_delta": float(model.intercept_[0] - intercept_before),
            }
        memory_after = {
            "negative": len(model._mem[0]),
            "target": len(model._mem[1]),
        }
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
                "scores_before_update": before_score_diagnostics,
                "scores_after_update": after_score_diagnostics,
                "coefficient_change": coefficient_change,
                "coefficients_after_update": _coefficient_diagnostics(model),
                "memory_before_update": memory_before,
                "memory_after_update": memory_after,
                "fit_diagnostics": model.last_fit_diagnostics_,
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
        "replay_pos_weight": replay_pos_weight,
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
    parser.add_argument(
        "--replay-pos-weight",
        choices=["current", "uniform"],
        default="current",
        help=(
            "Positive-class weighting for replay samples: 'current' reuses the "
            "current experience's weight; 'uniform' uses weight 1.0 for replay."
        ),
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
        replay_pos_weight=args.replay_pos_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["history"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
