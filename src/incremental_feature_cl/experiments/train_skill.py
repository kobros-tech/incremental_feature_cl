# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Skill: one persistent target-vs-rest learner that keeps learning as the classes arrive.

    python -m incremental_feature_cl.experiments.train_skill \
        --dataset cifar100 --target-class 17 --n-experiences 5 --ratio 1:5 --memory-per-class 100 \
        --backbone simplemlp --backbone-kwargs '{"hidden_size": 256}' --warmup-epochs 5 \
        --freeze-backbone --vs-sklearn

The class stream is class-incremental and the target class arrives in experience 0 (its data is
NOT presented again).  In every later experience the skill is refreshed on its replayed positive
exemplars against old negatives (replayed) and new negatives (the experience); the log shows
``negs(old a/b new c/d)`` as proof.  The skill sees the frozen features ``h(x)`` of any external
backbone (``--backbone simplemlp | resnet32 | dotted.path``; ``--warmup-epochs`` /
``--backbone-weights`` make the frozen features informative).

Scored like ``train_feature_map`` (target_acc = recall, neg_acc, bal_acc = their mean, auc), so the
two experiments are directly comparable (``--vs-feature-map`` runs it for you).  ``--vs-sklearn``
swaps the learner for ``LogisticRegression`` under the identical data protocol.  The skill is
configured with the same flags as ``IncrementalFeatureMapModel`` (``--new-feature-dim``,
``--activation``, ``--initialization`` ...) and the same explicit ``--optimizer`` (sgd by default; lbfgs is opt-in).
"""

from __future__ import annotations

import argparse
import copy
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ..data import build_class_incremental_stream
from ..evaluation import binary_target_metrics
from ..evaluation.result_store import RunResult
from ..models import Skill, SkillBase
from ..models.skill import cap_samples
from ..utils.reproducibility import collect_environment, resolve_device, seed_everything
from .common import (
    add_common_args,
    add_skill_args,
    build_backbone_from,
    check_training_contract,
    config_from_args,
    finish_skill_run,
    get_data,
    materialize,
    maybe_warm_up,
    print_comparison,
    skill_config_from,
    training_contract,
)
from .common import (
    run_experiment as run_feature_map,
)

SkillFactory = Callable[[int, int, object, int], SkillBase]


def run_skill(
    cfg,
    skill_cfg,
    target: int,
    out_dir: Path,
    factory: SkillFactory,
    *,
    label="skill",
    make_plots: bool = True,
) -> RunResult:
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.model.freeze_backbone = True  # a skill replays features: the representation is fixed
    device = torch.device(cfg.train.device)
    seed_everything(cfg.seed)

    train, test, num_classes, shape = get_data(cfg)
    n_exp = cfg.data.n_experiences
    base = build_class_incremental_stream(train, test, n_exp, cfg.seed, cfg.data.class_order)
    order = [target] + [c for c in base.class_order if c != target]  # target arrives first
    stream = build_class_incremental_stream(train, test, n_exp, cfg.seed, order)

    backbone = build_backbone_from(cfg, shape).to(device)
    maybe_warm_up(cfg, backbone, stream.train[0])
    backbone.requires_grad_(False).eval()

    def features(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([backbone(c.to(device)).cpu() for c in x.split(512)])

    x_test, y_test = materialize(test)
    h_test, labels = features(x_test), y_test.numpy()

    skill = factory(target, backbone.out_dim, skill_cfg, cfg.seed * 100_003 + target).to(device)
    exemplars: dict[int, torch.Tensor] = {}
    records = []
    cfg.name = cfg.name or (
        f"{label}_t{target}_{cfg.data.dataset}_e{n_exp}_d{cfg.model.new_feature_dim}_s{cfg.seed}"
    )
    print(
        f"[{cfg.name}] dataset={cfg.data.dataset} backbone={cfg.model.backbone} "
        f"out_dim={backbone.out_dim} device={device} memory/class={skill_cfg.memory_per_class}",
        flush=True,
    )

    for exp in stream.train:
        started = time.time()
        x, y = materialize(exp.dataset)
        h = features(x)
        by_class = {int(c): h[y == c] for c in y.unique()}
        seen = stream.seen_classes(exp.index)
        if exp.index == 0:  # nothing to replay yet
            skill.remember(negatives={})
        info = skill.update(
            by_class.get(target),
            {c: v for c, v in by_class.items() if c != target},
            step_seed=exp.index,
            device=device,
        )
        # commit this experience's exemplars (after training, so all updates see old memory only)
        cap = skill_cfg.memory_per_class
        new_ex = {c: cap_samples(v, cap, cfg.seed + 31 * c) for c, v in by_class.items()}
        for c, v in new_ex.items():
            exemplars[c] = (
                cap_samples(torch.cat([exemplars[c], v]), cap, cfg.seed + 31 * c)
                if c in exemplars
                else v
            )
        skill.remember(
            positive=new_ex.get(target), negatives={c: v for c, v in new_ex.items() if c != target}
        )
        train_time = time.time() - started

        with torch.no_grad():
            scores = skill.decision_function(h_test.to(device)).cpu().numpy()
        pred = scores > 0
        negatives = [c for c in seen if c != target]
        metrics = binary_target_metrics(pred, labels, target, negatives, scores)
        correct = np.where(labels == target, pred, ~pred)
        old_neg = {c for c in seen if c != target} - set(exp.new_classes)
        new_neg = set(exp.new_classes) - {target}
        used = set(info.get("negatives_used_per_class", {}))
        coverage = {
            "old_negative_classes": len(old_neg),
            "old_covered": len(used & old_neg),
            "new_negative_classes": len(new_neg),
            "new_covered": len(used & new_neg),
        }
        coverage["complete"] = coverage["old_covered"] == len(old_neg) and coverage[
            "new_covered"
        ] == len(new_neg)
        records.append(
            {
                "index": exp.index,
                "classes": exp.classes,
                "new_classes": exp.new_classes,
                "seen_classes": seen,
                "train_loss": info.get("train_loss"),
                "train_time_s": train_time,
                "per_class_accuracy": [
                    float(correct[labels == c].mean())
                    if c in seen and (labels == c).any()
                    else None
                    for c in range(num_classes)
                ],
                "accuracy_seen": float(correct[np.isin(labels, seen)].mean()),
                "accuracy_all": float(correct.mean()),
                "target_metrics": metrics,
                "negative_coverage": coverage,
                "n_positive": info.get("n_positive"),
                "n_negative": info.get("n_negative"),
                "feature_dim": skill.total_feature_dim,
                "parameter_count": sum(p.numel() for p in skill.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in skill.parameters() if p.requires_grad
                ),
            }
        )
        m = metrics
        print(
            f"  exp {exp.index:>2} dim={skill.total_feature_dim} pos/neg={info.get('n_positive')}/"
            f"{info.get('n_negative')} target_acc={m['target_accuracy']:.3f} "
            f"neg_acc={m['negative_accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
            f"auc={m['auc']:.3f} negs(old {coverage['old_covered']}/{coverage['old_negative_classes']}"
            f" new {coverage['new_covered']}/{coverage['new_negative_classes']}"
            f"{'' if coverage['complete'] else ' INCOMPLETE'})",
            flush=True,
        )

    result = RunResult(
        {**cfg.to_dict(), "skill_config": vars(skill_cfg)},
        collect_environment(),
        "target",
        num_classes,
        stream.class_order,
        stream.class_first_experience(),
        records,
        target_class=target,
    )
    finish_skill_run(result, out_dir, make_plots)
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p, "cifar100", ratio_default="1:5")
    add_skill_args(p)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument(
        "--vs-feature-map",
        action="store_true",
        help="also run train_feature_map (IncrementalFeatureMapModel) for this class",
    )
    a = p.parse_args(argv)
    cfg = config_from_args(a, "target", "cifar100")
    cfg.model.freeze_backbone = True
    check_training_contract(p, a, cfg, skills=True)
    skill_cfg = skill_config_from(cfg, a)
    print(f"[contract] {training_contract(skill_cfg)}", flush=True)
    out = Path(cfg.output_dir)
    name = cfg.name
    run_dir = out / (name or f"skill_t{a.target_class}_{cfg.data.dataset}_s{cfg.seed}")
    result = run_skill(cfg, skill_cfg, a.target_class, run_dir, Skill, make_plots=not a.no_plots)
    final = {"skill": result.experiences[-1]["target_metrics"]}

    if a.vs_sklearn:
        from ..evaluation.sklearn_skill import SklearnSkill

        c = copy.deepcopy(cfg)
        c.name = f"sklearn_{result.config['name']}"
        sk = run_skill(
            c,
            skill_cfg,
            a.target_class,
            run_dir / "vs_sklearn",
            lambda t, d, cf, s: SklearnSkill(t, d, cf, s, C=a.sklearn_c),
            label="sklearn",
            make_plots=False,
        )
        final["sklearn"] = sk.experiences[-1]["target_metrics"]
    if a.vs_feature_map:
        c = copy.deepcopy(cfg)
        c.mode, c.target_class, c.name = "target", a.target_class, ""
        c.name = c.auto_name()
        fm = run_feature_map(c, out_dir=run_dir / "vs_feature_map", make_plots=False, verbose=False)
        final["feature_map"] = fm.experiences[-1]["target_metrics"]
    if len(final) > 1:
        print_comparison(
            f"final experience, class {a.target_class}, target-vs-rest over all seen classes:",
            {k: [v] for k, v in final.items()},
            [a.target_class],
        )


if __name__ == "__main__":
    main()
