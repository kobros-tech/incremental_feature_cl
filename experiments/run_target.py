# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Run a reproducible class-incremental target-versus-rest experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.data import DATASETS, class_incremental_indices, load_dataset, resolve_dataset
from experiments.metrics import select_balanced_accuracy_threshold, target_metrics
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
    dataset: str | Path = "digits",
    target_class: int = 5,
    n_experiences: int = 5,
    seed: int = 0,
    new_feature_dim: int = 8,
    n_epochs: int = 5,
    replay_mem_size: int = 200,
    target_to_negatives: float | None = None,
    train_blocks: bool = True,
    replay_pos_weight: str = "current",
    target_every_experience: bool = True,
) -> dict:
    if n_experiences < 1:
        raise ValueError("n_experiences must be at least 1.")

    name, data_dir = resolve_dataset(dataset)
    X_train, X_test, y_train, y_test = load_dataset(dataset, seed)
    classes = np.unique(y_train)
    # Reserve validation data from the training split only. Keep at least one fitting
    # example per class, including for the tiny fixture datasets used by the tests.
    n_validation = max(len(classes), int(np.ceil(0.2 * len(y_train))))
    if n_validation > len(y_train) - len(classes):
        raise ValueError(
            "The training split is too small to reserve validation examples while "
            "keeping at least one fitting example per class."
        )
    fit_indices, validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=n_validation,
        random_state=seed,
        stratify=y_train,
    )
    X_fit, y_fit = X_train[fit_indices], y_train[fit_indices]
    X_validation, y_validation = X_train[validation_indices], y_train[validation_indices]
    if target_class not in np.unique(y_fit):
        raise ValueError(f"target_class={target_class} is absent from the training split.")

    y_validation_binary = y_validation == target_class
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
    recall_at_arrival = None
    recall_at_natural_arrival = None
    streams = class_incremental_indices(y_fit, n_experiences)
    target_indices = np.flatnonzero(y_fit == target_class)
    target_memory_initialized = False
    fixed_threshold = None
    fixed_threshold_validation_metrics = None
    fixed_threshold_calibration_experience = None
    fixed_threshold_validation_classes = None
    fixed_threshold_target_recall_at_calibration = None
    natural_target_arrival_experience = next(
        (exp for exp, stream in enumerate(streams) if np.any(y_fit[stream] == target_class)),
        None,
    )
    first_training_exposure_experience = None
    seen_training_classes = set()

    for experience, stream_indices in enumerate(streams):
        arriving_classes = np.unique(y_fit[stream_indices]).astype(int).tolist()
        indices = stream_indices
        target_added_to_experience = False
        if (
            target_every_experience
            and experience > natural_target_arrival_experience
            and target_class not in arriving_classes
        ):
            # Reintroduce the target only after its natural class arrival. Earlier
            # experiences remain a genuine pre-arrival stream; test data stays held out.
            indices = np.concatenate([stream_indices, target_indices])
            target_added_to_experience = True
        y_current = (y_fit[indices] == target_class).astype(int)
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

        positive_memory_before = None
        positive_seen_before = None
        if target_every_experience and target_memory_initialized:
            positive_memory_before = model._mem[1].copy()
            positive_seen_before = model._mem_seen[1]

        model.partial_fit(X_fit[indices], y_current, classes=np.array([0, 1]))

        if positive_memory_before is not None:
            # Repeated training exposure must not make duplicate copies of the same target
            # images dominate the reservoir; keep the positive reservoir from first exposure.
            model._mem[1] = positive_memory_before
            model._mem_seen[1] = positive_seen_before
            model.last_fit_diagnostics_["memory_target"] = len(model._mem[1])
            model.last_fit_diagnostics_["memory_total"] = sum(map(len, model._mem))

        if int(y_current.sum()):
            target_memory_initialized = True

        scores_after = model.decision_function(X_test)
        metrics = target_metrics(y_test_binary, scores_after)
        training_classes = np.unique(y_fit[indices]).astype(int).tolist()
        seen_training_classes.update(training_classes)
        if target_class in training_classes and first_training_exposure_experience is None:
            first_training_exposure_experience = experience

        # Calibration may use validation examples only from classes observed in training
        # up to this experience. This excludes future-class labels while allowing strict
        # class-once runs to calibrate on the target plus previously seen negative classes.
        calibration_allowed_classes = sorted(seen_training_classes)
        calibration_mask = np.isin(y_validation, calibration_allowed_classes)
        y_calibration = y_validation_binary[calibration_mask]
        calibration_classes = (
            sorted(np.unique(y_validation[calibration_mask]).astype(int).tolist())
            if np.any(calibration_mask)
            else []
        )
        validation_scores = (
            model.decision_function(X_validation[calibration_mask])
            if np.any(calibration_mask)
            else np.array([], dtype=float)
        )
        validation_threshold = None
        validation_threshold_metrics = None
        if (
            target_class in calibration_allowed_classes
            and y_calibration.any()
            and (~y_calibration).any()
        ):
            validation_threshold, validation_threshold_metrics = select_balanced_accuracy_threshold(
                y_calibration, validation_scores
            )

        if fixed_threshold is None and validation_threshold is not None:
            # Freeze the first valid, class-arrival-aware validation threshold. If the
            # target arrives alone and no negative class has arrived, wait for a valid
            # calibration set rather than using future-class validation examples.
            fixed_threshold = float(validation_threshold)
            fixed_threshold_validation_metrics = validation_threshold_metrics
            fixed_threshold_calibration_experience = experience
            fixed_threshold_validation_classes = calibration_classes

        threshold_calibration = {
            "method": "validation_balanced_accuracy_seen_classes",
            "calibration_experience": (experience if validation_threshold is not None else None),
            "validation_classes": calibration_classes,
            "calibration_allowed_classes": calibration_allowed_classes,
            "n_validation": int(calibration_mask.sum()),
            "threshold": (None if validation_threshold is None else float(validation_threshold)),
            "validation": validation_threshold_metrics,
            "test": (
                None
                if validation_threshold is None
                else target_metrics(y_test_binary, scores_after, threshold=validation_threshold)
            ),
        }
        fixed_threshold_calibration = {
            "method": "first_valid_seen-class_validation_balanced_accuracy",
            "calibration_experience": fixed_threshold_calibration_experience,
            "validation_classes": fixed_threshold_validation_classes,
            "threshold": fixed_threshold,
            "validation_at_calibration": fixed_threshold_validation_metrics,
            "test": (
                None
                if fixed_threshold is None
                else target_metrics(y_test_binary, scores_after, threshold=fixed_threshold)
            ),
        }
        fixed_threshold_test_recall = (
            None
            if fixed_threshold_calibration["test"] is None
            else fixed_threshold_calibration["test"]["target_recall"]
        )
        if experience == fixed_threshold_calibration_experience:
            fixed_threshold_target_recall_at_calibration = fixed_threshold_test_recall
        fixed_threshold_recall_change_since_calibration = (
            None
            if fixed_threshold_target_recall_at_calibration is None
            or fixed_threshold_test_recall is None
            else float(fixed_threshold_test_recall - fixed_threshold_target_recall_at_calibration)
        )
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
        target_seen = target_seen or target_class in training_classes
        target_recall = metrics["target_recall"]
        if target_seen and recall_at_arrival is None and np.isfinite(target_recall):
            recall_at_arrival = target_recall
        if experience == natural_target_arrival_experience and np.isfinite(target_recall):
            recall_at_natural_arrival = target_recall
        change_since_arrival = (
            None if recall_at_arrival is None else float(target_recall - recall_at_arrival)
        )
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
                "classes_in_training_batch": training_classes,
                "natural_target_arrival_experience": natural_target_arrival_experience,
                "first_training_exposure_experience": first_training_exposure_experience,
                "fixed_threshold_calibration_experience": fixed_threshold_calibration_experience,
                "target_every_experience": bool(target_every_experience),
                "target_added_to_experience": target_added_to_experience,
                "n_train": len(indices),
                "n_validation": len(y_validation),
                "n_target_train": int(y_current.sum()),
                "n_negative_train": int((y_current == 0).sum()),
                "feature_dim": int(model.feature_dim_),
                "replay_mem_size": int(replay_mem_size),
                "target_to_negatives": target_to_negatives,
                "best_target_recall": best_target_recall,
                "target_recall_forgetting": forgetting,
                "target_recall_forgetting_at_zero_threshold": forgetting,
                # Explicitly distinguish the default zero threshold from calibrated thresholds.
                "target_recall_at_zero_threshold": metrics["target_recall"],
                "target_recall_at_validation_selected_threshold": (
                    None
                    if threshold_calibration["test"] is None
                    else threshold_calibration["test"]["target_recall"]
                ),
                "target_recall_at_fixed_threshold": fixed_threshold_test_recall,
                "fixed_threshold_target_recall_at_calibration": (
                    fixed_threshold_target_recall_at_calibration
                ),
                "fixed_threshold_target_recall_change_since_calibration": (
                    fixed_threshold_recall_change_since_calibration
                ),
                "target_recall_at_arrival": recall_at_arrival,
                "target_recall_at_first_training_exposure": recall_at_arrival,
                "target_recall_at_natural_class_arrival": recall_at_natural_arrival,
                "target_recall_change_since_arrival": change_since_arrival,
                "scores_before_update": before_score_diagnostics,
                "scores_after_update": after_score_diagnostics,
                "coefficient_change": coefficient_change,
                "coefficients_after_update": _coefficient_diagnostics(model),
                "memory_before_update": memory_before,
                "memory_after_update": memory_after,
                "fit_diagnostics": model.last_fit_diagnostics_,
                "threshold_calibration": threshold_calibration,
                "fixed_threshold_calibration": fixed_threshold_calibration,
                **metrics,
            }
        )

    return {
        "dataset": name,
        "data_dir": (str(data_dir) if data_dir is not None and data_dir.is_dir() else None),
        "data_source": None if data_dir is None else str(data_dir),
        "target_class": int(target_class),
        "validation_size": len(y_validation),
        "natural_target_arrival_experience": natural_target_arrival_experience,
        "first_training_exposure_experience": first_training_exposure_experience,
        "fixed_threshold_calibration_experience": fixed_threshold_calibration_experience,
        "threshold_calibration_method": "validation_balanced_accuracy_seen_classes_each_experience",
        "fixed_threshold_calibration_method": "first_valid_seen_class_validation_balanced_accuracy",
        "n_experiences": int(n_experiences),
        "seed": int(seed),
        "new_feature_dim": int(new_feature_dim),
        "n_epochs": int(n_epochs),
        "replay_mem_size": int(replay_mem_size),
        "target_to_negatives": target_to_negatives,
        "replay_pos_weight": replay_pos_weight,
        "target_every_experience": bool(target_every_experience),
        "train_blocks": bool(train_blocks),
        "history": history,
        "final_target_recall_forgetting": history[-1]["target_recall_forgetting"],
        "final_target_recall_forgetting_at_zero_threshold": history[-1][
            "target_recall_forgetting_at_zero_threshold"
        ],
        "final_target_recall_change_since_arrival": history[-1][
            "target_recall_change_since_arrival"
        ],
        "final_target_recall_at_zero_threshold": history[-1]["target_recall_at_zero_threshold"],
        "final_target_recall_at_validation_selected_threshold": history[-1][
            "target_recall_at_validation_selected_threshold"
        ],
        "final_target_recall_at_fixed_threshold": history[-1]["target_recall_at_fixed_threshold"],
        "final_fixed_threshold_target_recall_change_since_calibration": history[-1][
            "fixed_threshold_target_recall_change_since_calibration"
        ],
        "target_recall_at_first_training_exposure": recall_at_arrival,
        "target_recall_at_natural_class_arrival": recall_at_natural_arrival,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="digits",
        help=(
            f"One of {', '.join(DATASETS)}, an extracted CIFAR-100 directory, or a local "
            "CIFAR-100 .tar/.tar.gz archive. 'cifar100' downloads and caches the official archive."
        ),
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
    parser.add_argument(
        "--target-every-experience",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include the same training target examples at natural arrival and in every "
            "later experience (default). Earlier experiences remain target-free. "
            "Use --no-target-every-experience for a strict class-once stream."
        ),
    )
    parser.add_argument("--fixed-blocks", action="store_true", help="Do not train feature blocks.")
    parser.add_argument("--output", type=Path, default=Path("results/target_run.json"))
    args = parser.parse_args()
    try:
        resolve_dataset(args.dataset)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

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
        replay_pos_weight=args.replay_pos_weight,
        target_every_experience=args.target_every_experience,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["history"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
