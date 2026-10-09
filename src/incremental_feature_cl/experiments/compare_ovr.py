# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Continual target-vs-rest Skills on the class-incremental stream.

    python -m incremental_feature_cl.experiments.compare_ovr \
        --dataset cifar100 --n-experiences 5 --new-feature-dim 16 --train-epochs 2 \
        --ratio 1:5 --memory-per-class 100 --seed 1 --vs-target 0 17 50

Every class owns one persistent :class:`Skill` (class vs. the other seen classes).  After each
experience ALL skills are refreshed: the new classes become negatives of every old skill (and the
old classes negatives of every new skill), on top of each skill's replay memory.

How a skill is scored.  Each skill is judged like a ``train_target`` run with that class as
target: its own decision (``logit > 0``) on the test samples of its class (positives) and of the
other seen classes (negatives).  The headline numbers are therefore means over skills of
``recall``, ``neg_acc`` (specificity), ``bal_acc`` = (recall + neg_acc) / 2, ``f1`` and the
threshold-free ``auc``.  ``argmax_acc`` (multiclass accuracy of the arg-max over independent skill
logits) is printed only as a diagnostic: those logits were never trained to be compared with each
other, so it is far lower than the per-skill numbers and is NOT comparable with ``train_target``'s
``acc_seen`` (which is a target-vs-rest accuracy, dominated by the negatives).

``--vs-target K ...`` additionally runs ``train_target`` (IncrementalFeatureMapModel) for those
classes with the same data / seed / ratio and prints both side by side.  Protocol differences:
the target-mode run re-trains every experience on the full cumulative negative pool (sub-sampled
by ``--ratio``), while a skill only sees new data plus ``--memory-per-class`` exemplars per class,
so the target-mode run is the more generous of the two.

Skills replay *features*, so the backbone is always frozen here.  What limits a skill is the data
of its refreshes (``--memory-per-class`` positives and ``--ratio`` times as many negatives), not
the model: with enough exemplars it reaches the target-mode numbers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import build_class_incremental_stream
from ..evaluation import argmax_accuracy, macro_metrics, per_skill_metrics
from ..evaluation.result_store import RunResult
from ..models import OneVsRestSkillModel, SkillConfig
from ..plotting import (
    plot_accuracy_heatmap,
    plot_accuracy_vs_class,
    plot_accuracy_vs_experience,
    plot_class_accuracy_curves,
    plot_feature_growth,
    plot_forgetting_vs_class,
)
from ..utils.reproducibility import collect_environment, resolve_device, seed_everything
from .common import add_common_args, build_model, config_from_args, get_data
from .common import run_experiment as run_target_experiment


def _skill_config(cfg, args: argparse.Namespace) -> SkillConfig:
    if args.skill_pos_weight == "none":
        pos_weight: str | float | None = None
    elif args.skill_pos_weight == "balanced":
        pos_weight = "balanced"
    else:
        pos_weight = float(args.skill_pos_weight)
    return SkillConfig(
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
        epochs=cfg.train.train_epochs,
        batch_size=cfg.train.train_mb_size,
        feature_expansion_dim=cfg.model.new_feature_dim,
        target_to_negatives=cfg.target_to_negatives,
        pos_weight=pos_weight,
        memory_per_class=args.memory_per_class,
    )


def _materialize(dataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Read through ``Dataset.__getitem__`` so uint8 images are converted and normalized."""
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def refresh_report(info: dict[str, Any], seen: list[int], new_classes: list[int]) -> dict[str, Any]:
    """Did this experience actually refresh the old skills? (derived from ``model.update``'s record)"""
    old = list(info["updated_old_skills"])
    trained = [c for c in old if info["skills"][c].get("trained")]
    coverage = []
    for c in trained:
        used = set(info["skills"][c].get("negatives_used_per_class", {}))
        wanted = set(new_classes) - {c}
        coverage.append(len(used & wanted) / len(wanted) if wanted else 1.0)
    complete = all(set(known) == set(seen) - {c} for c, known in info["known_negatives"].items())
    return {
        "old_skills": len(old),
        "old_skills_refreshed": len(trained),
        "new_classes_in_old_skill_training": float(np.mean(coverage)) if coverage else None,
        "min_new_class_coverage": float(min(coverage)) if coverage else None,
        "every_skill_knows_all_other_seen_classes": complete,
        "mean_positives_per_old_update": (
            float(np.mean([info["skills"][c]["n_positive"] for c in trained])) if trained else None
        ),
        "mean_negatives_per_old_update": (
            float(np.mean([info["skills"][c]["n_negative"] for c in trained])) if trained else None
        ),
    }


def _format_row(rec: dict[str, Any]) -> str:
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
                f" (new classes in their negatives: {r['new_classes_in_old_skill_training']:.0%}, "
                f"pos/neg per update: {r['mean_positives_per_old_update']:.0f}/"
                f"{r['mean_negatives_per_old_update']:.0f})"
            )
    if rec["train_loss"] is not None:
        row += f" | loss={rec['train_loss']:.4f}"
    return row


def _vs_target(cfg, per_skill: dict[int, dict], targets: list[int], out_dir: Path) -> list[dict]:
    """Run target mode (IncrementalFeatureMapModel) for ``targets``; compare with the skills."""
    rows = []
    for k in targets:
        if k not in per_skill:
            raise ValueError(f"--vs-target {k}: class {k} has no skill (not a valid class id)")
        c = copy.deepcopy(cfg)
        c.mode, c.target_class, c.name = "target", k, ""
        c.name = c.auto_name()
        run = run_target_experiment(
            c, out_dir=out_dir / "vs_target" / f"target_{k}", make_plots=False, verbose=False
        )
        tm = run.experiences[-1]["target_metrics"]
        rows.append({"class": k, "skill": per_skill[k], "target_run": tm})

    keys = ("balanced_accuracy", "target_recall", "negative_accuracy", "target_f1", "auc")
    cols = ("bal", "recall", "neg", "f1", "auc")
    head = " ".join(f"{h:>7}" for h in cols)
    print("\nfinal experience, target-vs-rest over all seen classes:")
    print(f"{'':>6} | {'Skill':^39} | IncrementalFeatureMapModel (target mode)")
    print(f"{'class':>6} | {head} | {head}")

    def cell(m: dict, k: str) -> str:
        return f"{m[k]:7.3f}" if m.get(k) is not None else "    n/a"

    for r in rows:
        left = " ".join(cell(r["skill"], k) for k in keys)
        right = " ".join(cell(r["target_run"], k) for k in keys)
        print(f"{r['class']:>6} | {left}   | {right}")
    mean = {
        side: {
            k: float(np.mean([x[side][k] for x in rows if x[side].get(k) is not None] or [np.nan]))
            for k in keys
        }
        for side in ("skill", "target_run")
    }
    print(
        f"{'mean':>6} | {' '.join(cell(mean['skill'], k) for k in keys)}   "
        f"| {' '.join(cell(mean['target_run'], k) for k in keys)}"
    )
    return rows


def run_experiment(cfg, args: argparse.Namespace, out_dir: Path) -> RunResult:
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.model.freeze_backbone = True  # skills replay features
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
    skill_cfg = _skill_config(cfg, args)
    model = OneVsRestSkillModel(backbone, config=skill_cfg, seed=cfg.seed).to(device)

    # The representation is frozen, so the test features are computed once.
    x_test, y_test = _materialize(test, cfg.train.eval_mb_size)
    h_test = model.extract(x_test.to(device)).cpu()
    labels = y_test.numpy()

    cfg.name = cfg.name or _default_name(cfg, args)
    result_cfg = cfg.to_dict()
    result_cfg["skill_config"] = asdict(skill_cfg)
    result_cfg["ratio"] = args.ratio
    records: list[dict[str, Any]] = []

    print(
        f"[{cfg.name}] dataset={cfg.data.dataset} experiences={cfg.data.n_experiences} "
        f"device={device} ratio={args.ratio} memory/class={args.memory_per_class}",
        flush=True,
    )
    for exp in stream.train:
        started = time.time()
        x, y = _materialize(exp.dataset, cfg.train.train_mb_size)
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
        skill_dims = {c: s.model.feature_dim for c, s in model.skills.items()}
        record = {
            "index": exp.index,
            "classes": exp.classes,
            "new_classes": exp.new_classes,
            "seen_classes": seen,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_time_s": train_time,
            # per_class_accuracy[c] = balanced accuracy of the skill of class c (None if unseen)
            "per_class_accuracy": [
                per_skill[c]["balanced_accuracy"] if c in per_skill else None
                for c in range(num_classes)
            ],
            # mean over skills of the per-skill target-vs-rest metrics (the headline numbers)
            "target_metrics": macro,
            "per_skill_metrics": per_skill,
            # arg-max over independent skill logits: diagnostic only
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
            "parameter_count": model.parameter_count(),
            "trainable_parameter_count": info["trainable_parameter_count"],
        }
        records.append(record)
        print(_format_row(record), flush=True)
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
    if not args.no_plots:
        plots = out_dir / "plots"
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
        f"avg_forgetting(bal_acc per skill)={s['average_forgetting']:.3f} -> {out_dir}",
        flush=True,
    )

    if args.vs_target:
        rows = _vs_target(cfg, records[-1]["per_skill_metrics"], args.vs_target, out_dir)
        (out_dir / "vs_target.json").write_text(json.dumps(rows, indent=2, default=float))
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_common_args(parser, "cifar100", ratio_default="1:5")
    parser.add_argument(
        "--memory-per-class",
        type=int,
        default=100,
        help="replay exemplars kept per class by every skill (default 100; 0 disables replay). "
        "This is the data a refresh of an old skill can use: with --ratio 1:5 it trains on "
        "memory positives and 5x as many negatives, so very small values starve the skills "
        "(on digits, 20 exemplars gave AUC 0.95 vs 0.99 for target mode, 120 matched it). "
        "Not the same as --replay-mem-size, which only configures the --vs-target baseline.",
    )
    parser.add_argument(
        "--skill-pos-weight",
        default="balanced",
        help="balanced, none, or a numeric BCE positive-class weight",
    )
    parser.add_argument(
        "--vs-target",
        type=int,
        nargs="+",
        metavar="CLASS",
        help="also run train_target (IncrementalFeatureMapModel) for these classes and compare",
    )
    args = parser.parse_args(argv)
    if args.memory_per_class < 0:
        parser.error("--memory-per-class must be >= 0")
    cfg = config_from_args(args, "target", "cifar100")
    cfg.model.freeze_backbone = True
    cfg.name = cfg.name or _default_name(cfg, args)
    run_experiment(cfg, args, Path(cfg.output_dir) / cfg.name)


def _default_name(cfg, args) -> str:
    return (
        f"ovr_{cfg.data.dataset}_e{cfg.data.n_experiences}_d{cfg.model.new_feature_dim}"
        f"_r{args.ratio.replace(':', '-')}_m{args.memory_per_class}_s{cfg.seed}"
    )


if __name__ == "__main__":
    main()
