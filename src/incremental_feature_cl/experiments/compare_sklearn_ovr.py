# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Standalone scikit-learn control for the Skill experiment (``pip install scikit-learn``).

    python -m incremental_feature_cl.experiments.compare_sklearn_ovr --dataset cifar100 \
        --backbone identity --n-experiences 5 --protocol replay --ratio 1:5 --memory-per-class 100

One one-vs-rest ``LogisticRegression`` per class on the same frozen features, same class-
incremental stream and the same per-skill target-vs-rest scoring as ``compare_ovr``.

``--protocol replay`` (default) uses the Skills' data protocol (replay memory, ``--ratio``,
class-balanced old + new negatives) and refits each class on every refresh: only the learner
differs from a Skill.  ``--protocol cumulative`` refits each class on all data seen so far (not
continual; the strongest control).  ``--only-classes`` limits the fitted classes (a cumulative fit
of 100 classes on raw CIFAR pixels is slow).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.result_store import RunResult  # noqa: F401  (re-export for convenience)
from .common import add_common_args, config_from_args
from .compare_ovr import default_name, skill_config_from
from .ovr_runner import run_ovr_experiment
from .sklearn_control import build_sklearn_model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    add_common_args(parser, "cifar100", ratio_default="1:5")
    parser.add_argument("--protocol", choices=("replay", "cumulative"), default="replay")
    parser.add_argument("--memory-per-class", type=int, default=100)
    parser.add_argument("--skill-pos-weight", default="balanced")
    parser.add_argument("--skill-expansion", default="first", help=argparse.SUPPRESS)
    parser.add_argument("--skill-max-iter", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument(
        "--C", dest="sklearn_c", type=float, default=1.0, help="LogisticRegression C"
    )
    parser.add_argument("--max-iter", dest="sklearn_max_iter", type=int, default=200)
    parser.add_argument("--only-classes", type=int, nargs="+", metavar="CLASS")
    args = parser.parse_args(argv)

    cfg = config_from_args(args, "target", "cifar100")
    cfg.model.freeze_backbone = True
    cfg.name = cfg.name or default_name(cfg, args, prefix=f"sklearn_{args.protocol}")
    skill_cfg = skill_config_from(cfg, args)

    def build(cfg_, backbone, device):
        return build_sklearn_model(
            args.protocol,
            backbone,
            skill_cfg,
            cfg_.seed,
            C=args.sklearn_c,
            max_iter=args.sklearn_max_iter,
            only_classes=args.only_classes,
        )

    run_ovr_experiment(
        cfg,
        Path(cfg.output_dir) / cfg.name,
        build,
        describe=f"[sklearn {args.protocol}] ratio={args.ratio} memory/class={args.memory_per_class}",
        config_extra={
            "sklearn_protocol": args.protocol,
            "C": args.sklearn_c,
            "ratio": args.ratio,
            "skill_config": {
                "target_to_negatives": skill_cfg.target_to_negatives,
                "memory_per_class": skill_cfg.memory_per_class,
            },
        },
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
