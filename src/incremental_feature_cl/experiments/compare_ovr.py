# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Continual target-vs-rest Skills on the class-incremental stream.

    python -m incremental_feature_cl.experiments.compare_ovr \
        --dataset cifar100 --n-experiences 5 --new-feature-dim 16 --train-epochs 2 \
        --ratio 1:5 --memory-per-class 100 --seed 1 --vs-target 0 17 50 --vs-sklearn replay

Every class owns one persistent :class:`Skill` (class vs. the other seen classes).  After each
experience ALL skills are refreshed: the new classes become negatives of every old skill (and the
old classes negatives of every new skill), on top of each skill's replay memory.  The log line of
every experience proves it (``refreshed 80/80 old skills (negatives: new classes 100%, old classes
100% ...)``).

The skills are configured with the same flags as ``train_target`` (``--new-feature-dim``,
``--activation``, ``--initialization``, ``--output-init``, ``--freeze-old-blocks``,
``--optimizer``, ``--lr`` ...), i.e. the same options as ``IncrementalFeatureMapModel``.

How a skill is scored: see ``ovr_runner``.  The headline numbers are means over skills of the
per-skill target-vs-rest metrics (``recall``, ``neg_acc``, ``bal_acc``, ``f1``, ``auc``);
``argmax_acc`` is only a diagnostic.  Because independent skills are not trained to be compared,
``--comparison-weight W`` (> 0) adds a joint *comparison phase* after each experience that trains
all skills on the replay exemplars with a softmax cross-entropy over skills (plus a per-skill
binary loss), which is what makes ``argmax_acc`` meaningful.

``--vs-target K ...`` additionally runs ``train_target`` (IncrementalFeatureMapModel) for those
classes and ``--vs-sklearn replay cumulative`` runs the scikit-learn controls
(``sklearn_control``) on the same data/seed/ratio, and prints everything side by side.  Protocol
differences: target mode and ``cumulative`` re-train every experience on the full cumulative
negative pool, while a skill (and ``replay``) only sees new data plus ``--memory-per-class``
exemplars per class.

Skills replay *features*, so the backbone is always frozen here.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..models import ComparisonConfig, OneVsRestSkillModel, SkillConfig
from ..models.skill import EXPANSIONS
from .common import add_common_args, config_from_args
from .common import run_experiment as run_target_experiment
from .ovr_runner import run_ovr_experiment

_COLUMNS = ("balanced_accuracy", "target_recall", "negative_accuracy", "target_f1", "auc")
_HEADERS = ("bal", "recall", "neg", "f1", "auc")


def skill_config_from(cfg, args: argparse.Namespace) -> SkillConfig:
    """Map the shared experiment config (ModelConfig / TrainConfig) onto a SkillConfig."""
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
        optimizer=cfg.train.optimizer,
        max_iter=args.skill_max_iter,
        feature_expansion_dim=cfg.model.new_feature_dim,
        activation=cfg.model.activation,
        initialization=cfg.model.initialization,
        output_init=cfg.model.output_init,
        freeze_old_blocks=cfg.model.freeze_old_blocks,
        expansion=args.skill_expansion,
        target_to_negatives=cfg.target_to_negatives,
        pos_weight=pos_weight,
        memory_per_class=args.memory_per_class,
    )


def comparison_config_from(args: argparse.Namespace) -> ComparisonConfig:
    return ComparisonConfig(
        weight=args.comparison_weight,
        bce_weight=args.comparison_bce_weight,
        epochs=args.comparison_epochs,
        batch_size=args.comparison_batch_size,
        lr=args.comparison_lr,
    )


def default_name(cfg, args: argparse.Namespace, prefix: str = "ovr") -> str:
    ratio = args.ratio.replace(":", "-")
    return (
        f"{prefix}_{cfg.data.dataset}_e{cfg.data.n_experiences}_d{cfg.model.new_feature_dim}"
        f"_r{ratio}_m{args.memory_per_class}_s{cfg.seed}"
    )


def run_experiment(cfg, args: argparse.Namespace, out_dir: Path, *, make_plots: bool = True):
    """Train and score the Skill ensemble; save the result under ``out_dir``."""
    skill_cfg = skill_config_from(cfg, args)
    comparison = comparison_config_from(args)

    def build(cfg_, backbone, device):
        return OneVsRestSkillModel(
            backbone, config=skill_cfg, seed=cfg_.seed, comparison=comparison
        )

    return run_ovr_experiment(
        cfg,
        out_dir,
        build,
        describe=f"ratio={args.ratio} memory/class={args.memory_per_class}"
        + (f" comparison_weight={comparison.weight:g}" if comparison.enabled else ""),
        config_extra={
            "skill_config": asdict(skill_cfg),
            "comparison_config": asdict(comparison),
            "ratio": args.ratio,
        },
        make_plots=make_plots,
    )


# ---------------------------------------------------------------------- comparisons
def _cell(m: dict, k: str) -> str:
    return f"{m[k]:7.3f}" if m.get(k) is not None else "    n/a"


def _mean(rows: list[dict]) -> dict:
    return {
        k: float(np.mean([r[k] for r in rows if r.get(k) is not None] or [np.nan]))
        for k in _COLUMNS
    }


def _print_table(title: str, names: list[str], rows_by_name: dict[str, list[dict]], ids) -> None:
    head = " ".join(f"{h:>7}" for h in _HEADERS)
    print(f"\n{title}")
    print(f"{'class':>6} | " + " | ".join(f"{n:^39}" for n in names))
    print(f"{'':>6} | " + " | ".join(head for _ in names))
    for i, c in enumerate(ids):
        print(
            f"{c:>6} | "
            + " | ".join(" ".join(_cell(rows_by_name[n][i], k) for k in _COLUMNS) for n in names)
        )
    print(
        f"{'mean':>6} | "
        + " | ".join(" ".join(_cell(_mean(rows_by_name[n]), k) for k in _COLUMNS) for n in names)
    )


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
        rows.append(
            {"class": k, "skill": per_skill[k], "target_run": run.experiences[-1]["target_metrics"]}
        )
    _print_table(
        "final experience, target-vs-rest over all seen classes (Skill vs target mode):",
        ["skill", "target_run"],
        {"skill": [r["skill"] for r in rows], "target_run": [r["target_run"] for r in rows]},
        targets,
    )
    return rows


def _vs_sklearn(cfg, args, skill_result, protocols: list[str], out_dir: Path) -> dict:
    from .sklearn_control import build_sklearn_model

    only = args.vs_target or None
    skill_cfg = skill_config_from(cfg, args)
    final_skill = skill_result.experiences[-1]["per_skill_metrics"]
    ids = sorted(only) if only else sorted(final_skill)
    rows_by_name = {"skill": [final_skill[c] for c in ids]}
    summary = {}
    for protocol in protocols:
        c = copy.deepcopy(cfg)
        c.name = f"sklearn_{protocol}_{cfg.name}"

        def build(cfg_, backbone, device, protocol=protocol):
            return build_sklearn_model(
                protocol,
                backbone,
                skill_cfg,
                cfg_.seed,
                C=args.sklearn_c,
                max_iter=args.sklearn_max_iter,
                only_classes=only if protocol == "cumulative" else None,
            )

        res = run_ovr_experiment(
            c,
            out_dir / "vs_sklearn" / protocol,
            build,
            describe=f"[sklearn {protocol}]",
            config_extra={"sklearn_protocol": protocol, "C": args.sklearn_c},
            make_plots=False,
        )
        per = res.experiences[-1]["per_skill_metrics"]
        rows_by_name[f"sklearn_{protocol}"] = [per[k] for k in ids]
        summary[protocol] = {k: per[k] for k in ids}
    _print_table(
        "final experience, target-vs-rest over all seen classes (Skill vs scikit-learn):",
        list(rows_by_name),
        rows_by_name,
        ids,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
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
        "--skill-max-iter",
        type=int,
        default=100,
        help="L-BFGS iterations per update (only with --set train.optimizer=lbfgs)",
    )
    parser.add_argument(
        "--skill-expansion",
        choices=EXPANSIONS,
        default="first",
        help="when a skill grows its feature space by --new-feature-dim: once at its first "
        "update (first), before every update (every), before every update except the first "
        "(later, like the target-mode trainer) or never",
    )
    parser.add_argument(
        "--comparison-weight",
        type=float,
        default=0.0,
        help="weight of the softmax cross-entropy over skills in the joint comparison phase "
        "(0 = off: skills stay independent)",
    )
    parser.add_argument("--comparison-bce-weight", type=float, default=1.0)
    parser.add_argument("--comparison-epochs", type=int, default=1)
    parser.add_argument("--comparison-batch-size", type=int, default=64)
    parser.add_argument("--comparison-lr", type=float, default=None)
    parser.add_argument(
        "--vs-target",
        type=int,
        nargs="+",
        metavar="CLASS",
        help="also run train_target (IncrementalFeatureMapModel) for these classes and compare",
    )
    parser.add_argument(
        "--vs-sklearn",
        nargs="+",
        choices=("replay", "cumulative"),
        metavar="PROTOCOL",
        help="also run the scikit-learn controls (replay: the Skills' data protocol; cumulative: "
        "all seen data, only for the --vs-target classes if given)",
    )
    parser.add_argument("--sklearn-c", type=float, default=1.0, help="LogisticRegression C")
    parser.add_argument("--sklearn-max-iter", type=int, default=200)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.memory_per_class < 0:
        parser.error("--memory-per-class must be >= 0")
    cfg = config_from_args(args, "target", "cifar100")
    cfg.model.freeze_backbone = True
    cfg.name = cfg.name or default_name(cfg, args)
    out_dir = Path(cfg.output_dir) / cfg.name
    result = run_experiment(cfg, args, out_dir, make_plots=not args.no_plots)

    final = result.experiences[-1]["per_skill_metrics"]
    if args.vs_target:
        rows = _vs_target(cfg, final, args.vs_target, out_dir)
        (out_dir / "vs_target.json").write_text(json.dumps(rows, indent=2, default=float))
    if args.vs_sklearn:
        summary = _vs_sklearn(cfg, args, result, args.vs_sklearn, out_dir)
        (out_dir / "vs_sklearn.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
