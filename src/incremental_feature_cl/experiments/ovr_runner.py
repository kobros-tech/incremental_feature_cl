# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Shared class-incremental loop for every one-vs-rest learner (Skills, sklearn controls).

The learner is any :class:`OneVsRestSkillModel` (or subclass); this module owns everything that
must be identical for a fair comparison: data, stream, frozen test features, the per-skill
target-vs-rest scoring (:mod:`evaluation.skill_metrics`), the refresh check and the saved result.

How a skill is scored.  Each skill is judged like a ``train_target`` run with that class as
target: its own decision (``logit > 0``) on the test samples of its class (positives) and of the
other seen classes (negatives).  The headline numbers are means over skills of ``recall``,
``neg_acc`` (specificity), ``bal_acc`` = (recall + neg_acc) / 2, ``f1`` and the threshold-free
``auc``.  ``argmax_acc`` (multiclass accuracy of the arg-max over independent skill logits) is a
separate diagnostic and is NOT comparable with ``train_target``'s ``acc_seen`` (a target-vs-rest
accuracy dominated by the negatives).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import build_class_incremental_stream
from ..evaluation import argmax_accuracy, macro_metrics, per_skill_metrics
from ..evaluation.result_store import RunResult
from ..models import OneVsRestSkillModel
from ..plotting import (
    plot_accuracy_heatmap,
    plot_accuracy_vs_class,
    plot_accuracy_vs_experience,
    plot_class_accuracy_curves,
    plot_feature_growth,
    plot_forgetting_vs_class,
)
from ..utils.reproducibility import collect_environment, resolve_device, seed_everything
from .common import build_model, get_data

ModelBuilder = Callable[[Any, torch.nn.Module, torch.device], OneVsRestSkillModel]


def materialize(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read through ``Dataset.__getitem__`` so uint8 images are converted and normalized."""
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def refresh_report(info: dict[str, Any], seen: list[int], new_classes: list[int]) -> dict[str, Any]:
    """Did this experience actually refresh the old skills? (derived from ``model.update``'s record)

    A refreshed old skill was trained on its class against old negatives (replay) and new
    negatives (``new_classes_in_old_skill_training`` = share of this experience's classes that
    were in its training negatives).
    """
    old = list(info["updated_old_skills"])
    trained = [c for c in old if info["skills"][c].get("trained")]
    coverage, old_coverage = [], []
    for c in trained:
        used = set(info["skills"][c].get("negatives_used_per_class", {}))
        wanted_new = set(new_classes) - {c}
        wanted_old = (set(seen) - set(new_classes)) - {c}
        coverage.append(len(used & wanted_new) / len(wanted_new) if wanted_new else 1.0)
        old_coverage.append(len(used & wanted_old) / len(wanted_old) if wanted_old else 1.0)
    complete = all(set(known) == set(seen) - {c} for c, known in info["known_negatives"].items())
    return {
        "old_skills": len(old),
        "old_skills_refreshed": len(trained),
        "new_classes_in_old_skill_training": float(np.mean(coverage)) if coverage else None,
        "min_new_class_coverage": float(min(coverage)) if coverage else None,
        "old_classes_in_old_skill_training": float(np.mean(old_coverage)) if old_coverage else None,
        "min_old_class_coverage": float(min(old_coverage)) if old_coverage else None,
        "every_skill_knows_all_other_seen_classes": complete,
        "mean_positives_per_old_update": (
            float(np.mean([info["skills"][c]["n_positive"] for c in trained])) if trained else None
        ),
        "mean_negatives_per_old_update": (
            float(np.mean([info["skills"][c]["n_negative"] for c in trained])) if trained else None
        ),
    }


def format_row(rec: dict[str, Any]) -> str:
    m, r = rec["target_metrics"], rec["refresh"]
    auc = f"{m['auc']:.3f}" if m["auc"] is not None else "  n/a"
    row = (
        f"  exp {rec['index']:>2} skills={rec['number_of_skills']:<3} "
        f"bal_acc={m['balanced_accuracy']:.3f} recall={m['target_recall']:.3f} "
        f"neg_acc={m['negative_accuracy']:.3f} f1={m['target_f1']:.3f} auc={auc} "
        f"| argmax_acc={rec['accuracy_seen']:.3f}"
    )
    if r["old_skills"]:
        row += f" | refreshed {r['old_skills_refreshed']}/{r['old_skills']} old skills"
        if r["old_skills_refreshed"]:
            row += (
                f" (negatives: new classes {r['new_classes_in_old_skill_training']:.0%}, "
                f"old classes {r['old_classes_in_old_skill_training']:.0%}; "
                f"pos/neg per update {r['mean_positives_per_old_update']:.0f}/"
                f"{r['mean_negatives_per_old_update']:.0f})"
            )
    if rec["train_loss"] is not None and not math.isnan(rec["train_loss"]):
        row += f" | loss={rec['train_loss']:.4f}"
    return row


def run_ovr_experiment(
    cfg,
    out_dir: Path,
    build: ModelBuilder,
    *,
    describe: str = "",
    config_extra: dict[str, Any] | None = None,
    make_plots: bool = True,
) -> RunResult:
    """Run one class-incremental one-vs-rest learner and save a :class:`RunResult`.

    ``build(cfg, backbone, device)`` returns the learner.  ``cfg.model.freeze_backbone`` is
    forced on: skills replay features.
    """
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.model.freeze_backbone = True
    device = torch.device(cfg.train.device)
    seed_everything(cfg.seed)

    train, test, num_classes, shape = get_data(cfg)
    if cfg.data.dataset == "synthetic" and "n_classes" not in cfg.data.synthetic:
        n = cfg.data.n_experiences
        cfg.data.synthetic = {**cfg.data.synthetic, "n_classes": max(10, math.ceil(10 / n) * n)}
        train, test, num_classes, shape = get_data(cfg)

    stream = build_class_incremental_stream(
        train, test, cfg.data.n_experiences, cfg.seed, cfg.data.class_order
    )
    backbone = build_model(cfg, shape, num_outputs=1).backbone.to(device)
    model = build(cfg, backbone, device).to(device)

    # The representation is frozen, so the test features are computed once.
    x_test, y_test = materialize(test, cfg.train.eval_mb_size)
    h_test = model.extract(x_test.to(device)).cpu()
    labels = y_test.numpy()

    result_cfg = {**cfg.to_dict(), **(config_extra or {})}
    records: list[dict[str, Any]] = []
    print(
        f"[{cfg.name}] dataset={cfg.data.dataset} experiences={cfg.data.n_experiences} "
        f"device={device} {describe}".rstrip(),
        flush=True,
    )
    for exp in stream.train:
        started = time.time()
        x, y = materialize(exp.dataset, cfg.train.train_mb_size)
        info = model.update(x, y, device=device)
        train_time = time.time() - started

        seen = stream.seen_classes(exp.index)
        with torch.no_grad():
            scores = model.decision_function_features(h_test.to(device)).cpu().numpy()
        per_skill = per_skill_metrics(scores, labels, model.class_ids, seen)
        macro = macro_metrics(per_skill)
        argmax_seen = argmax_accuracy(scores, labels, model.class_ids, seen)
        predicted = np.asarray(model.class_ids)[scores.argmax(axis=1)]
        refresh = refresh_report(info, seen, exp.new_classes)

        losses = [
            v["train_loss"]
            for v in info["skills"].values()
            if v.get("trained") and v.get("train_loss") is not None
        ]
        finite = [v for v in losses if not math.isnan(v)]
        skill_dims = {c: s.total_feature_dim for c, s in model.skills.items()}
        record = {
            "index": exp.index,
            "classes": exp.classes,
            "new_classes": exp.new_classes,
            "seen_classes": seen,
            "train_loss": float(np.mean(finite)) if finite else None,
            "train_time_s": train_time,
            # per_class_accuracy[c] = balanced accuracy of the skill of class c (None if unseen)
            "per_class_accuracy": [
                per_skill[c]["balanced_accuracy"] if c in per_skill else None
                for c in range(num_classes)
            ],
            # mean over skills of the per-skill target-vs-rest metrics (the headline numbers)
            "target_metrics": macro,
            "per_skill_metrics": per_skill,
            # arg-max over skill logits: diagnostic only
            "accuracy_seen": argmax_seen,
            "accuracy_all": float((predicted == labels).mean()),
            "refresh": refresh,
            "feature_dim": max(skill_dims.values(), default=model.feature_dim),
            "shared_feature_dim": model.feature_dim,
            "skill_feature_dims": skill_dims,
            "number_of_classifiers": model.num_skills,
            "number_of_skills": info["number_of_skills"],
            "new_skills": info["new_skills"],
            "updated_old_skills": info["updated_old_skills"],
            "known_negative_classes_per_skill": info["known_negatives"],
            "replay_count_per_skill": info["replay_count"],
            "skill_training": info["skills"],
            "comparison": info.get("comparison"),
            "parameter_count": model.parameter_count(),
            "trainable_parameter_count": info["trainable_parameter_count"],
        }
        records.append(record)
        print(format_row(record), flush=True)
        if refresh["old_skills_refreshed"] < refresh["old_skills"]:
            print(
                "  WARNING: some old skills were not updated (a skill without any stored positive "
                "exemplar cannot be refreshed: use --memory-per-class > 0)",
                flush=True,
            )
        if not refresh["every_skill_knows_all_other_seen_classes"]:
            print("  WARNING: a skill does not know every other seen class", flush=True)

    result = RunResult(
        result_cfg,
        collect_environment(),
        "target",  # every skill is a target-vs-rest learner; plots/summaries read target_metrics
        num_classes,
        stream.class_order,
        stream.class_first_experience(),
        records,
    )
    result.compute_summary()
    result.save(out_dir)
    if make_plots:
        plots = Path(out_dir) / "plots"
        plot_accuracy_vs_experience(result, plots / "A_accuracy_vs_experience.png")
        plot_accuracy_vs_class(result, plots / "B_accuracy_vs_class.png")
        plot_accuracy_heatmap(result, plots / "C_class_accuracy_heatmap.png")
        plot_class_accuracy_curves(result, plots / "D_class_accuracy_curves.png")
        plot_forgetting_vs_class(result, plots / "E_forgetting_vs_class.png")
        plot_feature_growth(result, plots / "F_feature_growth.png")

    s = result.summary
    auc = f" final_auc={s['final_auc']:.3f}" if s.get("final_auc") is not None else ""
    print(
        f"[{cfg.name}] final_bal_acc={s['final_balanced_accuracy']:.3f}{auc} "
        f"final_f1={s['final_target_f1']:.3f} final_argmax_acc={s['final_accuracy_seen']:.3f} "
        f"avg_forgetting={s['average_forgetting']:.3f} (balanced accuracy per skill) -> {out_dir}",
        flush=True,
    )
    return result
