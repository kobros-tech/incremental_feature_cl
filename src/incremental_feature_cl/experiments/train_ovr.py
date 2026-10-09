# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""OneVsRestSkillModel: one Skill per class, learning continuously over a class-incremental stream.

    python -m incremental_feature_cl.experiments.train_ovr \
        --dataset cifar100 --n-experiences 5 --ratio 1:5 --memory-per-class 100 \
        --backbone simplemlp --backbone-kwargs '{"hidden_size": 256}' --warmup-epochs 5 \
        --freeze-backbone --vs-sklearn --vs-feature-map 0 17 50

After each experience ALL skills are refreshed: the new classes become negatives of every old
skill (and the old classes negatives of every new skill) on top of each skill's replay memory.
The log line proves it: ``refreshed 80/80 old skills (negatives: new classes 100%, old classes
100% ...)``.  The backbone is any external ``nn.Module`` (see ``models/backbones.py``).

Each skill is scored like a ``train_feature_map`` run with that class as target (its own decision
``logit > 0`` on its class vs. the other seen classes): the headline numbers are means over skills
of ``recall``, ``neg_acc``, ``bal_acc`` = (recall + neg_acc) / 2, ``f1`` and ``auc``.
``argmax_acc`` (multiclass accuracy of the arg-max over skill logits) is only a diagnostic:
independent skills are not trained to be compared.  ``--comparison-weight W`` (> 0) adds a joint
comparison phase that trains all skills together on the replay exemplars so that arg-max becomes
meaningful.

``--vs-feature-map K ...`` runs IncrementalFeatureMapModel for those classes and ``--vs-sklearn``
the scikit-learn control (same data protocol) and prints everything side by side.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..data import build_class_incremental_stream
from ..evaluation import argmax_accuracy, macro_metrics, per_skill_metrics
from ..evaluation.result_store import RunResult
from ..models import ComparisonConfig, OneVsRestSkillModel
from ..utils.reproducibility import collect_environment, resolve_device, seed_everything
from .common import (
    add_common_args,
    add_skill_args,
    build_backbone_from,
    config_from_args,
    finish_skill_run,
    get_data,
    materialize,
    maybe_warm_up,
    print_comparison,
    skill_config_from,
)
from .common import run_experiment as run_feature_map


def refresh_report(info: dict, seen: list[int], new_classes: list[int]) -> dict:
    """Did this experience refresh the old skills against old AND new negatives?"""
    old = list(info["updated_old_skills"])
    trained = [c for c in old if info["skills"][c].get("trained")]
    new_cov, old_cov = [], []
    for c in trained:
        used = set(info["skills"][c].get("negatives_used_per_class", {}))
        want_new = set(new_classes) - {c}
        want_old = (set(seen) - set(new_classes)) - {c}
        new_cov.append(len(used & want_new) / len(want_new) if want_new else 1.0)
        old_cov.append(len(used & want_old) / len(want_old) if want_old else 1.0)
    return {
        "old_skills": len(old),
        "old_skills_refreshed": len(trained),
        "new_class_coverage": float(np.mean(new_cov)) if new_cov else None,
        "old_class_coverage": float(np.mean(old_cov)) if old_cov else None,
        "mean_positives": float(np.mean([info["skills"][c]["n_positive"] for c in trained]))
        if trained
        else None,
        "mean_negatives": float(np.mean([info["skills"][c]["n_negative"] for c in trained]))
        if trained
        else None,
    }


def run_ovr(
    cfg,
    skill_cfg,
    comparison: ComparisonConfig,
    out_dir: Path,
    build_model=None,
    make_plots: bool = True,
    label: str = "ovr",
) -> RunResult:
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.model.freeze_backbone = True  # skills replay features: the representation is fixed
    device = torch.device(cfg.train.device)
    seed_everything(cfg.seed)

    train, test, num_classes, shape = get_data(cfg)
    stream = build_class_incremental_stream(
        train, test, cfg.data.n_experiences, cfg.seed, cfg.data.class_order
    )
    backbone = build_backbone_from(cfg, shape).to(device)
    maybe_warm_up(cfg, backbone, stream.train[0])
    model = (build_model or OneVsRestSkillModel)(
        backbone,
        config=skill_cfg,
        seed=cfg.seed,
        **({} if build_model else {"comparison": comparison}),
    ).to(device)

    x_test, y_test = materialize(test)
    h_test = model.extract(x_test.to(device)).cpu()  # frozen representation: computed once
    labels = y_test.numpy()
    cfg.name = cfg.name or (
        f"{label}_{cfg.data.dataset}_e{cfg.data.n_experiences}_d{cfg.model.new_feature_dim}"
        f"_m{skill_cfg.memory_per_class}_s{cfg.seed}"
    )
    print(
        f"[{cfg.name}] dataset={cfg.data.dataset} backbone={cfg.model.backbone} "
        f"out_dim={backbone.out_dim} device={device} memory/class={skill_cfg.memory_per_class}"
        + (f" comparison_weight={comparison.weight:g}" if comparison.enabled else ""),
        flush=True,
    )

    records = []
    for exp in stream.train:
        started = time.time()
        x, y = materialize(exp.dataset)
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
            if v.get("trained")
            and v.get("train_loss") is not None
            and not math.isnan(v["train_loss"])
        ]
        dims = {c: s.total_feature_dim for c, s in model.skills.items()}
        rec = {
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
            "target_metrics": macro,  # mean over skills (headline)
            "per_skill_metrics": per_skill,
            "accuracy_seen": argmax_seen,  # arg-max over skill logits: diagnostic only
            "accuracy_all": float((predicted == labels).mean()),
            "refresh": refresh,
            "comparison": info.get("comparison"),
            "feature_dim": max(dims.values(), default=model.feature_dim),
            "number_of_skills": info["number_of_skills"],
            "new_skills": info["new_skills"],
            "updated_old_skills": info["updated_old_skills"],
            "parameter_count": model.parameter_count(),
            "trainable_parameter_count": info["trainable_parameter_count"],
        }
        records.append(rec)
        auc = f"{macro['auc']:.3f}" if macro["auc"] is not None else "n/a"
        row = (
            f"  exp {exp.index:>2} skills={rec['number_of_skills']:<3} "
            f"bal_acc={macro['balanced_accuracy']:.3f} recall={macro['target_recall']:.3f} "
            f"neg_acc={macro['negative_accuracy']:.3f} f1={macro['target_f1']:.3f} auc={auc} "
            f"| argmax_acc={argmax_seen:.3f}"
        )
        if refresh["old_skills"]:
            row += (
                f" | refreshed {refresh['old_skills_refreshed']}/{refresh['old_skills']} old skills"
            )
            if refresh["old_skills_refreshed"]:
                row += (
                    f" (negatives: new classes {refresh['new_class_coverage']:.0%}, old classes "
                    f"{refresh['old_class_coverage']:.0%}; pos/neg per update "
                    f"{refresh['mean_positives']:.0f}/{refresh['mean_negatives']:.0f})"
                )
            else:
                row += "  WARNING: old skills cannot be refreshed without --memory-per-class > 0"
        print(row, flush=True)

    result = RunResult(
        {**cfg.to_dict(), "skill_config": vars(skill_cfg)},
        collect_environment(),
        "target",
        num_classes,
        stream.class_order,
        stream.class_first_experience(),
        records,
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
    p.add_argument(
        "--comparison-weight",
        type=float,
        default=0.0,
        help="weight of the softmax cross-entropy over skills in the joint comparison "
        "phase (0 = off: skills stay independent)",
    )
    p.add_argument("--comparison-epochs", type=int, default=1)
    p.add_argument(
        "--vs-feature-map",
        type=int,
        nargs="+",
        metavar="CLASS",
        help="also run IncrementalFeatureMapModel for these classes and compare",
    )
    a = p.parse_args(argv)
    cfg = config_from_args(a, "target", "cifar100")
    cfg.model.freeze_backbone = True
    skill_cfg = skill_config_from(cfg, a)
    comparison = ComparisonConfig(weight=a.comparison_weight, epochs=a.comparison_epochs)
    run_dir = Path(cfg.output_dir) / (cfg.name or f"ovr_{cfg.data.dataset}_s{cfg.seed}")
    result = run_ovr(cfg, skill_cfg, comparison, run_dir, make_plots=not a.no_plots)
    final = result.experiences[-1]["per_skill_metrics"]
    ids = sorted(a.vs_feature_map) if a.vs_feature_map else sorted(final)
    rows = {"skill": [final[c] for c in ids]}

    if a.vs_sklearn:
        from ..evaluation.sklearn_skill import SklearnOneVsRest

        c = copy.deepcopy(cfg)
        c.name = f"sklearn_{result.config['name']}"
        sk = run_ovr(
            c,
            skill_cfg,
            comparison,
            run_dir / "vs_sklearn",
            build_model=lambda bb, config, seed: SklearnOneVsRest(
                bb, config=config, seed=seed, C=a.sklearn_c
            ),
            make_plots=False,
            label="sklearn",
        )
        per = sk.experiences[-1]["per_skill_metrics"]
        rows["sklearn"] = [per[c_] for c_ in ids]
    if a.vs_feature_map:
        fm = []
        for k in ids:
            c = copy.deepcopy(cfg)
            c.mode, c.target_class, c.name = "target", k, ""
            c.name = c.auto_name()
            r = run_feature_map(
                c,
                out_dir=run_dir / "vs_feature_map" / f"target_{k}",
                make_plots=False,
                verbose=False,
            )
            fm.append(r.experiences[-1]["target_metrics"])
        rows["feature_map"] = fm
    if len(rows) > 1:
        print_comparison("final experience, target-vs-rest over all seen classes:", rows, ids)


if __name__ == "__main__":
    main()
