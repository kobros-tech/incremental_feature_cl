# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Class-incremental evaluation that reports recall, accuracy and forgetting together.

Every class arrives in exactly one experience (classes in a seed-shuffled order).  After each
experience the model is scored on the held-out test rows of every class seen so far, giving
matrices indexed ``[experience, class]``:

* recall ``R``      - fraction of class ``c`` test rows scored >= 0 by class ``c``'s binary model
* accuracy ``P``    - fraction of class ``c`` test rows whose arg-max over the *seen* classes is ``c``
* specificity ``S`` - fraction of other *seen*-class rows scored < 0 (stops recall being bought
  by predicting "target" everywhere); classes that have not arrived are never used as negatives
* AUC ``U``         - threshold-free ranking quality against the same seen-class negatives, so threshold drift can be told apart from
  genuine loss of information

Four methods are compared on the same data and seeds: ``naive`` (no replay), ``replay`` (the
package's bounded memory), ``joint`` (one offline fit on all data, no feature expansion) and
``joint_matched`` (offline training on all data with the same feature-expansion schedule and
number of passes as the stream, so capacity is matched).  Replay memory is accounted for in
total rows across all one-vs-rest models, because each class model keeps its own memory.  The
verdict compares ``replay`` with ``naive`` using paired differences over seeds and a Student-t
confidence interval, and reports each criterion separately instead of one opaque pass/fail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

from experiments.data import load_dataset, resolve_dataset
from incremental_feature_cl import IncrementalFeatureMapModel

METHODS = ("naive", "replay", "joint", "joint_matched")


def _class_groups(classes: np.ndarray, n_experiences: int, rng: np.random.RandomState):
    if not 1 <= n_experiences <= len(classes):
        raise ValueError(f"n_experiences must be between 1 and {len(classes)}.")
    return np.array_split(rng.permutation(classes), n_experiences)


def _make_model(method: str, new_feature_dim: int, n_epochs: int, replay_mem_size: int, seed: int):
    return OneVsRestClassifier(
        IncrementalFeatureMapModel(
            new_feature_dim=0 if method == "joint" else new_feature_dim,
            n_epochs=n_epochs,
            replay_mem_size=replay_mem_size if method == "replay" else 0,
            random_state=seed,
        )
    )


def _evaluate(model, X_test, y_test, seen: list[int]) -> dict[int, dict[str, float]]:
    """Per-seen-class recall, accuracy, specificity and AUC at the zero threshold."""
    scores = model.decision_function(X_test)  # (n, n_classes), columns follow model.classes_
    col = {int(c): i for i, c in enumerate(model.classes_)}
    seen_cols = [col[c] for c in seen]
    pred = np.asarray(seen)[np.argmax(scores[:, seen_cols], axis=1)]
    in_seen = np.isin(y_test, seen)
    out = {}
    for c in seen:
        s = scores[:, col[c]]
        is_c = y_test == c
        # Negatives are rows of *other seen* classes only: counting classes that have not
        # arrived yet would score the model on data it could not have learned from.
        others = in_seen & ~is_c
        mask = is_c | others
        out[c] = {
            "recall": float(np.mean(s[is_c] >= 0.0)),
            "accuracy": float(np.mean(pred[is_c & in_seen] == c)),
            "specificity": float(np.mean(s[others] < 0.0)) if others.any() else float("nan"),
            "auc": float(roc_auc_score(is_c[mask], s[mask])) if others.any() else float("nan"),
        }
    return out


def run_stream(
    dataset="digits",
    method="replay",
    seed=0,
    n_experiences=5,
    new_feature_dim=8,
    n_epochs=5,
    replay_mem_size=None,
    memory_fraction=0.25,
    max_classes=None,
) -> dict:
    """Run one method on one seed and return the ``[experience, class]`` metric matrices.

    ``replay_mem_size`` is the memory of *each* one-vs-rest class model.  When ``None`` it is
    derived from a total budget of ``memory_fraction`` of the training rows spread over the
    classes, so the default replay never stores more than that share of the training set.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    X_train, X_test, y_train, y_test = load_dataset(dataset, seed)
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    if max_classes is not None:
        classes = np.sort(rng.permutation(classes)[:max_classes])
        keep_tr, keep_te = np.isin(y_train, classes), np.isin(y_test, classes)
        X_train, y_train, X_test, y_test = (
            X_train[keep_tr],
            y_train[keep_tr],
            X_test[keep_te],
            y_test[keep_te],
        )
    groups = _class_groups(classes, n_experiences, rng)
    arrival = {int(c): e for e, g in enumerate(groups) for c in g}
    if replay_mem_size is None:
        replay_mem_size = max(2, int(memory_fraction * len(y_train) / len(classes)))
    model = _make_model(method, new_feature_dim, n_epochs, replay_mem_size, seed)

    metrics: list[dict[int, dict[str, float]]] = []
    if method in ("joint", "joint_matched"):
        if method == "joint":
            model.fit(X_train, y_train)
        else:  # same expansion schedule and number of passes as the stream, all data each time
            for _ in range(n_experiences):
                model.partial_fit(X_train, y_train, classes=classes)
        full = _evaluate(model, X_test, y_test, [int(c) for c in classes])
        # The joint model has no stream; report the same numbers after every experience for the
        # classes seen so far so that it can be read against the incremental methods.
        for e in range(n_experiences):
            seen = sorted(c for c, a in arrival.items() if a <= e)
            metrics.append({c: full[c] for c in seen})
    else:
        for e, group in enumerate(groups):
            rows = np.flatnonzero(np.isin(y_train, group))
            model.partial_fit(X_train[rows], y_train[rows], classes=classes)
            seen = sorted(c for c, a in arrival.items() if a <= e)
            metrics.append(_evaluate(model, X_test, y_test, seen))

    if method == "replay":
        stored = sum(sum(len(k) for k in est._mem) for est in model.estimators_)
    elif method == "naive":
        stored = 0
    else:  # offline methods need every training row
        stored = len(y_train)
    summary = summarise(metrics, arrival, n_experiences)
    summary["replay_memory_rows"] = float(stored)
    summary["replay_memory_fraction_of_train"] = float(stored / len(y_train))
    return {
        "method": method,
        "seed": int(seed),
        "replay_mem_size_per_class_model": int(replay_mem_size),
        "n_train_rows": len(y_train),
        "arrival": {str(c): a for c, a in sorted(arrival.items())},
        "metrics": [{str(c): m for c, m in step.items()} for step in metrics],
        "summary": summary,
    }


def summarise(metrics, arrival: dict[int, int], n_experiences: int) -> dict[str, float]:
    """Continual-learning summary of one run from its per-experience, per-class metrics."""
    last = n_experiences - 1
    final = metrics[last]
    old = [c for c, a in arrival.items() if a < last]  # classes that had time to be forgotten
    out: dict[str, float] = {}
    for key in ("recall", "accuracy", "specificity", "auc"):
        out[f"final_{key}"] = float(np.nanmean([final[c][key] for c in final]))
        out[f"average_incremental_{key}"] = float(
            np.nanmean([np.nanmean([m[key] for m in step.values()]) for step in metrics])
        )
        out[f"{key}_at_arrival"] = float(np.nanmean([metrics[arrival[c]][c][key] for c in arrival]))
    out["final_balanced_accuracy"] = (out["final_recall"] + out["final_specificity"]) / 2
    out["balanced_accuracy_at_arrival"] = (
        out["recall_at_arrival"] + out["specificity_at_arrival"]
    ) / 2
    for key in ("recall", "accuracy"):
        if old:
            # Forgetting: best value after arrival (before the final experience) minus final.
            drops = [
                max(metrics[t][c][key] for t in range(arrival[c], last)) - final[c][key]
                for c in old
            ]
            changes = [final[c][key] - metrics[arrival[c]][c][key] for c in old]
            out[f"{key}_forgetting"] = float(np.nanmean(drops))
            out[f"{key}_backward_transfer"] = float(np.nanmean(changes))
        else:
            out[f"{key}_forgetting"] = out[f"{key}_backward_transfer"] = float("nan")
    return out


def _paired(a: list[float], b: list[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean and t-interval of per-seed differences ``a - b``."""
    d = np.asarray(a) - np.asarray(b)
    mean = float(d.mean())
    if len(d) < 2:
        return {"mean": mean, "low": float("nan"), "high": float("nan"), "n": len(d)}
    half = float(stats.t.ppf((1 + confidence) / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean": mean, "low": mean - half, "high": mean + half, "n": len(d)}


def verdict(runs: dict[str, list[dict]], plasticity_margin: float, joint_margin: float) -> dict:
    """Criteria for a continual-learning claim, each judged on paired per-seed differences."""

    def col(method, key):
        return [r["summary"][key] for r in runs[method]]

    def diff(key, a="replay", b="naive"):
        return _paired(col(a, key), col(b, key))

    checks = {
        "retention_recall_forgetting_lower_than_naive": (
            diff("recall_forgetting"),
            lambda d: d["high"] < 0,
        ),
        "final_recall_higher_than_naive": (diff("final_recall"), lambda d: d["low"] > 0),
        "final_accuracy_higher_than_naive": (diff("final_accuracy"), lambda d: d["low"] > 0),
        "recall_not_bought_with_specificity": (
            diff("final_balanced_accuracy"),
            lambda d: d["low"] > 0,
        ),
        # Plasticity is judged on balanced accuracy: recall at arrival alone can be inflated by a
        # positive-biased threshold (a model that says "target" too often has high recall).
        "still_learns_new_classes": (
            diff("balanced_accuracy_at_arrival"),
            lambda d: d["low"] > -plasticity_margin,
        ),
    }
    informational = {
        "info_recall_at_arrival": diff("recall_at_arrival"),
        "info_auc_at_arrival": diff("auc_at_arrival"),
    }
    out = {name: {**d, "passed": bool(ok(d)), "gating": True} for name, (d, ok) in checks.items()}
    for name, d in informational.items():  # reported, never gating
        out[name] = {**d, "passed": bool(d["low"] > -plasticity_margin), "gating": False}
    fractions = col("replay", "replay_memory_fraction_of_train")
    out["replay_stores_less_than_the_training_set"] = {
        "mean": float(np.mean(fractions)),
        "low": float(np.min(fractions)),
        "high": float(np.max(fractions)),
        "n": len(fractions),
        "passed": bool(np.max(fractions) < 1.0),
        "gating": True,
    }
    if runs.get("joint_matched"):
        gap = diff("final_balanced_accuracy", a="joint_matched", b="replay")
        out["info_gap_to_capacity_matched_joint"] = {
            **gap,
            "passed": bool(gap["high"] <= joint_margin),
            "gating": False,
        }
    if runs.get("joint"):
        gap = diff("final_balanced_accuracy", a="joint", b="replay")
        out["within_margin_of_joint_training"] = {
            **gap,
            "passed": bool(gap["high"] <= joint_margin),
            "gating": True,
        }
    out["all_passed"] = all(v["passed"] for v in out.values() if v["gating"])
    return out


def run_comparison(
    dataset="digits",
    seeds=(0, 1, 2, 3, 4),
    methods=METHODS,
    plasticity_margin=0.05,
    joint_margin=0.15,
    **kwargs,
) -> dict:
    runs = {m: [run_stream(dataset, m, s, **kwargs) for s in seeds] for m in methods}
    aggregate = {}
    for m, rs in runs.items():
        keys = rs[0]["summary"].keys()
        aggregate[m] = {
            k: {
                "mean": float(np.nanmean([r["summary"][k] for r in rs])),
                "std": float(np.nanstd([r["summary"][k] for r in rs], ddof=1))
                if len(rs) > 1
                else float("nan"),
            }
            for k in keys
        }
    result = {"dataset": str(dataset), "seeds": list(seeds), "settings": kwargs}
    result.update({"runs": runs, "aggregate": aggregate})
    if {"naive", "replay"} <= set(methods):
        result["verdict"] = verdict(runs, plasticity_margin, joint_margin)
    return result


def format_report(result: dict) -> str:
    rows = [
        ("final_recall", "final recall"),
        ("final_specificity", "final specificity"),
        ("final_accuracy", "final accuracy"),
        ("final_auc", "final AUC"),
        ("recall_at_arrival", "recall at arrival"),
        ("recall_forgetting", "recall forgetting"),
        ("recall_backward_transfer", "recall backward transfer"),
        ("accuracy_forgetting", "accuracy forgetting"),
        ("average_incremental_recall", "avg incremental recall"),
        ("average_incremental_accuracy", "avg incremental accuracy"),
        ("replay_memory_rows", "rows retained (all class models)"),
        ("replay_memory_fraction_of_train", "fraction of training set retained"),
    ]
    methods = list(result["aggregate"])
    lines = ["| metric | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for key, label in rows:
        cells = [
            f"{result['aggregate'][m][key]['mean']:.3f} ± {result['aggregate'][m][key]['std']:.3f}"
            for m in methods
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if "verdict" in result:
        lines += ["", "| criterion (replay vs naive, paired) | mean diff | 95% CI | passed |"]
        lines.append("|---|---|---|---|")
        for name, v in result["verdict"].items():
            if name != "all_passed":
                lines.append(
                    f"| {name}{'' if v['gating'] else ' (not gating)'} | {v['mean']:+.3f} "
                    f"| [{v['low']:+.3f}, {v['high']:+.3f}] | {'yes' if v['passed'] else 'NO'} |"
                )
        lines.append(f"\nAll gating criteria passed: {result['verdict']['all_passed']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="digits")
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--new-feature-dim", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument(
        "--replay-mem-size",
        type=int,
        default=None,
        help="Replay memory of EACH one-vs-rest class model (total = this x n_classes). "
        "Default: derived from --memory-fraction.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.25,
        help="Total replay budget as a fraction of the training rows (default 0.25).",
    )
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--plasticity-margin", type=float, default=0.05)
    parser.add_argument("--joint-margin", type=float, default=0.15)
    parser.add_argument(
        "--fail-on-verdict",
        action="store_true",
        help="Exit with status 1 if any gating criterion fails (for CI).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/continual_eval.json"))
    args = parser.parse_args()
    try:
        resolve_dataset(args.dataset)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    result = run_comparison(
        args.dataset,
        seeds=args.seeds,
        plasticity_margin=args.plasticity_margin,
        joint_margin=args.joint_margin,
        n_experiences=args.n_experiences,
        new_feature_dim=args.new_feature_dim,
        n_epochs=args.n_epochs,
        replay_mem_size=args.replay_mem_size,
        memory_fraction=args.memory_fraction,
        max_classes=args.max_classes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_report(result))
    print(f"Saved: {args.output}")
    if args.fail_on_verdict and not result.get("verdict", {}).get("all_passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
