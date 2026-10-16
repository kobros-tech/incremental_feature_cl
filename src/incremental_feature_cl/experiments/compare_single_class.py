# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Is the gap between the Skill and the feature-map results the learner or the protocol?

Part 1 - *matched learner comparison* (one target class, one update).  The **same frozen feature
vectors**, the **same training rows** (the Skill's own target:negative draw), the same validation
rows, the same positive weighting, the **same initial weights**, the same optimizer and budget
(``--optimizer sgd|adam``, ``--train-epochs``, ``--train-mb-size``, ``--lr``) and the same test rows
and decision thresholds are given to

* route ``skill``: ``Skill._fit`` (what ``train_ovr`` / ``train_skill`` run), and
* route ``feature_map``: ``ContinualTrainer`` training an ``IncrementalFeatureMapModel`` (what
  ``train_feature_map`` runs).

If the two still diverge, the learner / update implementation differs.  If they agree (try
``--train-mb-size 100000`` for exactly the same full-batch steps), the earlier gap lies in the data
protocol, the replay or the evaluation, not in the classifier.  The feature-map trainer has no
L-BFGS, so this part needs ``sgd`` or ``adam``.

Part 2 - *continual vs joint* (``--n-experiences K`` > 1).  The Skill is refreshed through the
class-incremental stream exactly as ``train_skill`` does, and the final skill is compared with
skills trained **at once** on all data under the same learner, ratio, features, validation rows,
test rows and threshold rule:

* ``joint_all_positives``: every positive row of the target;
* ``joint_memory_positives``: only as many positives as the continual skill may replay
  (``--memory-per-class``).

``continual`` vs ``joint_memory_positives`` isolates the effect of learning continually;
``joint_memory_positives`` vs ``joint_all_positives`` isolates the replay budget.

    python -m incremental_feature_cl.experiments.compare_single_class \\
        --dataset cifar100 --backbone identity --target-class 0 --n-experiences 5 \\
        --ratio 1:5 --memory-per-class 100 --train-epochs 10 --seed 1

Scores are logits of one target-vs-rest classifier, evaluated over all test rows of the seen
classes; no ``argmax`` is used anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import build_class_incremental_stream
from ..data.datasets import ArrayDataset
from ..data.streams import Experience
from ..evaluation import best_f1_threshold, binary_target_metrics
from ..models import IdentityBackbone, IncrementalFeatureMapModel, Skill, SkillConfig
from ..models.skill import cap_samples
from ..training import ContinualTrainer, ExpansionPolicy, TrainerConfig
from ..utils.reproducibility import resolve_device, seed_everything
from .common import (
    add_common_args,
    add_skill_args,
    build_backbone_from,
    check_training_contract,
    config_from_args,
    get_data,
    materialize,
    maybe_warm_up,
    skill_config_from,
    training_contract,
)
from .train_ovr import split_validation

KEYS = (
    "auc",
    "balanced_accuracy",
    "target_recall",
    "negative_accuracy",
    "target_precision",
    "target_f1",
)


# ---------------------------------------------------------------------- #
# metrics
# ---------------------------------------------------------------------- #
def score_metrics(
    scores: np.ndarray, labels: np.ndarray, target: int, threshold: float = 0.0
) -> dict[str, float]:
    """Target-vs-rest metrics of one score vector on every row of ``labels`` (no argmax)."""
    negatives = sorted(set(labels.tolist()) - {target})
    m = binary_target_metrics(scores > threshold, labels, target, negatives, scores)
    return {k: m[k] for k in KEYS}


def evaluate_route(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    target: int,
) -> dict[str, Any]:
    """Metrics at threshold 0 and at the F1-maximising threshold of the validation rows."""
    threshold = 0.0
    if len(val_scores):
        threshold, _ = best_f1_threshold(val_scores, val_labels == target)
    return {
        "threshold_0": score_metrics(test_scores, test_labels, target, 0.0),
        "validated": score_metrics(test_scores, test_labels, target, threshold),
        "validated_threshold": threshold,
    }


# ---------------------------------------------------------------------- #
# Part 1: the two routes on identical tensors
# ---------------------------------------------------------------------- #
def build_feature_map_route(
    skill_cfg: SkillConfig, feature_dim: int, seed: int, target: int, device: torch.device
) -> tuple[IncrementalFeatureMapModel, ContinualTrainer]:
    """``IncrementalFeatureMapModel`` + trainer built exactly like a ``Skill``'s inner model."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = IncrementalFeatureMapModel(
            backbone=IdentityBackbone(feature_dim),
            backbone_dim=feature_dim,
            num_outputs=1,
            activation=skill_cfg.activation,
            initialization=skill_cfg.initialization,
            output_init=skill_cfg.output_init,
            freeze_backbone=True,
            freeze_old_blocks=skill_cfg.freeze_old_blocks,
            seed=seed,
        )
    policy = ExpansionPolicy(skill_cfg.feature_expansion_dim, skill_cfg.initialization)
    policy.single_output = True
    tcfg = TrainerConfig(
        mode="target",
        optimizer=skill_cfg.optimizer,
        lr=skill_cfg.lr,
        momentum=skill_cfg.momentum,
        weight_decay=skill_cfg.weight_decay,
        train_epochs=skill_cfg.epochs,
        train_mb_size=skill_cfg.batch_size,
        device=str(device),
        pos_weight=skill_cfg.pos_weight,
        probe_size=0,
        seed=seed,
    )
    trainer = ContinualTrainer(model, policy, tcfg)
    trainer._target = int(target)
    return model, trainer


def matched_comparison(
    pos: torch.Tensor,
    neg: torch.Tensor,
    val: tuple[torch.Tensor, np.ndarray],
    test: tuple[torch.Tensor, np.ndarray],
    target: int,
    skill_cfg: SkillConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train both routes on the identical ``pos`` / ``neg`` rows and evaluate them identically."""
    if skill_cfg.optimizer == "lbfgs":
        raise ValueError("the feature-map trainer has no L-BFGS: use --optimizer sgd or adam")
    d = pos.shape[1]
    skill = Skill(target, d, skill_cfg, seed).to(device)
    model, trainer = build_feature_map_route(skill_cfg, d, seed, target, device)
    same_init = all(
        torch.equal(a, b)
        for a, b in zip(skill.model.state_dict().values(), model.state_dict().values())
    )

    skill._fit(pos, neg, step_seed=0, device=device)

    if skill_cfg.feature_expansion_dim > 0:
        # exactly what ``Skill._fit`` does before its first update (same block, same seed)
        model.expand_feature_space(
            new_dim=skill_cfg.feature_expansion_dim, seed=seed * 100_003 + 0 + 1
        )
    x = torch.cat([pos, neg])
    y = torch.cat([torch.full((len(pos),), target), torch.full((len(neg),), -1)])
    exp = Experience(0, ArrayDataset(x, y), [target], [target])
    pos_weight_trainer = trainer._pos_weight(exp)
    pos_weight_skill = skill._pos_weight(len(pos), len(neg), device)
    trainer._train_experience(exp)

    out: dict[str, Any] = {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "same_initial_weights": bool(same_init),
        "pos_weight": {
            "skill": None if pos_weight_skill is None else float(pos_weight_skill),
            "feature_map": None if pos_weight_trainer is None else float(pos_weight_trainer),
        },
    }
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fn in (
        ("skill", lambda h: skill.decision_function(h.to(device))),
        ("feature_map", lambda h: model(h.to(device)).squeeze(-1)),
    ):
        with torch.no_grad():
            model.eval()
            skill.eval()
            scores[name] = (fn(val[0]).cpu().numpy(), fn(test[0]).cpu().numpy())
        out[name] = evaluate_route(scores[name][0], val[1], scores[name][1], test[1], target)
    diff = np.abs(scores["skill"][1] - scores["feature_map"][1])
    out["score_agreement"] = {
        "max_abs_difference": float(diff.max()),
        "mean_abs_difference": float(diff.mean()),
        "correlation": float(np.corrcoef(scores["skill"][1], scores["feature_map"][1])[0, 1])
        if scores["skill"][1].std() > 0 and scores["feature_map"][1].std() > 0
        else None,
    }
    return out


# ---------------------------------------------------------------------- #
# Part 2: continual vs joint
# ---------------------------------------------------------------------- #
def run_continual_skill(
    exps: list[dict[int, torch.Tensor]],
    target: int,
    skill_cfg: SkillConfig,
    seed: int,
    device: torch.device,
    d: int,
    evaluate,
) -> tuple[Skill, list[dict]]:
    """Refresh one Skill through the stream exactly like ``train_skill`` (then ``evaluate``)."""
    skill = Skill(target, d, skill_cfg, seed * 100_003 + target).to(device)
    exemplars: dict[int, torch.Tensor] = {}
    cap, records = skill_cfg.memory_per_class, []
    for i, by_class in enumerate(exps):
        if i == 0:
            skill.remember(negatives={})
        info = skill.update(
            by_class.get(target),
            {c: v for c, v in by_class.items() if c != target},
            step_seed=i,
            device=device,
        )
        new_ex = {c: cap_samples(v, cap, seed + 31 * c) for c, v in by_class.items()}
        for c, v in new_ex.items():
            exemplars[c] = (
                cap_samples(torch.cat([exemplars[c], v]), cap, seed + 31 * c)
                if c in exemplars
                else v
            )
        skill.remember(
            positive=new_ex.get(target), negatives={c: v for c, v in new_ex.items() if c != target}
        )
        records.append({"experience": i, **evaluate(skill, i), "train": info})
    return skill, records


def joint_skill(
    pos: torch.Tensor,
    negatives: dict[int, torch.Tensor],
    target: int,
    skill_cfg: SkillConfig,
    seed: int,
    device: torch.device,
) -> Skill:
    """A skill trained once on all given rows, with the Skill's own target:negative protocol."""
    s = Skill(target, pos.shape[1], skill_cfg, seed * 100_003 + target).to(device)
    s.remember(negatives={})
    s.update(pos, negatives, step_seed=0, device=device)
    return s


# ---------------------------------------------------------------------- #
def _table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"  {'route':<22}" + "".join(f"{k[:11]:>12}" for k in KEYS))
    for name, m in rows.items():
        print(f"  {name:<22}" + "".join(f"{m[k]:>12.3f}" for k in KEYS))


def main(argv=None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p, "cifar100", ratio_default="1:5")
    add_skill_args(p)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1, help="validation rows per class")
    a = p.parse_args(argv)
    if not 0.0 < a.val_fraction < 0.5:
        p.error("--val-fraction must be in (0, 0.5)")
    cfg = config_from_args(a, "target", "cifar100")
    cfg.model.freeze_backbone = True
    check_training_contract(p, a, cfg, skills=False)  # part 1 needs the feature-map trainer
    skill_cfg = skill_config_from(cfg, a)
    print(f"[contract] {training_contract(skill_cfg)}", flush=True)

    cfg.train.device = str(resolve_device(cfg.train.device))
    device, seed, target = torch.device(cfg.train.device), cfg.seed, a.target_class
    seed_everything(seed)
    train, test, _, shape = get_data(cfg)
    n_exp = cfg.data.n_experiences
    base = build_class_incremental_stream(train, test, n_exp, seed, cfg.data.class_order)
    order = [target] + [c for c in base.class_order if c != target]
    stream = build_class_incremental_stream(train, test, n_exp, seed, order)
    backbone = build_backbone_from(cfg, shape).to(device)
    maybe_warm_up(cfg, backbone, stream.train[0])
    backbone.requires_grad_(False).eval()

    def features(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([backbone(c.to(device)).cpu() for c in x.split(512)])

    # per-experience feature rows, split once into training / validation rows
    exps: list[dict[int, torch.Tensor]] = []
    val_h, val_y = [], []
    for exp in stream.train:
        x, y = materialize(exp.dataset)
        (x, y), (xv, yv) = split_validation(x, y, a.val_fraction, seed * 1009 + exp.index)
        h = features(x)
        exps.append({int(c): h[y == c] for c in y.unique()})
        val_h.append(features(xv))
        val_y.append(yv)
    x_test, y_test = materialize(test)
    h_test, labels = features(x_test), y_test.numpy()
    h_val_all, y_val_all = torch.cat(val_h), torch.cat(val_y).numpy()
    d = h_test.shape[1]
    report: dict[str, Any] = {
        "target": target,
        "feature_dim": d,
        "contract": training_contract(skill_cfg),
    }

    # ---------------- part 1: identical tensors through both learners ----------------
    pos_all = torch.cat([e[target] for e in exps if target in e])
    neg_by_class: dict[int, list[torch.Tensor]] = {}
    for e in exps:
        for c, v in e.items():
            if c != target:
                neg_by_class.setdefault(c, []).append(v)
    negs = {c: torch.cat(v) for c, v in neg_by_class.items()}
    probe = Skill(target, d, skill_cfg, seed * 100_003 + target)
    pos, neg, info = probe.training_set(pos_all, negs, step_seed=0)
    report["part1_data"] = {k: v for k, v in info.items() if k != "negatives_used_per_class"}
    part1 = matched_comparison(
        pos, neg, (h_val_all, y_val_all), (h_test, labels), target, skill_cfg,
        seed * 100_003 + target, device,
    )  # fmt: skip
    report["part1"] = part1
    print(
        f"\nPART 1 - same {part1['n_positive']} positives / {part1['n_negative']} negatives, "
        f"same initial weights: {part1['same_initial_weights']}, pos_weight skill/feature_map = "
        f"{part1['pos_weight']['skill']}/{part1['pos_weight']['feature_map']}"
    )
    for view, title in (("threshold_0", "threshold 0"), ("validated", "validated threshold")):
        _table(
            f"class {target} vs rest, {title}:",
            {r: part1[r][view] for r in ("skill", "feature_map")},
        )
    sa = part1["score_agreement"]
    print(
        f"\n  score agreement: max|diff|={sa['max_abs_difference']:.4g} "
        f"mean|diff|={sa['mean_abs_difference']:.4g} corr={sa['correlation']}"
    )

    # ---------------- part 2: continual vs joint ----------------
    if n_exp > 1:

        def evaluate(skill: Skill, i: int) -> dict[str, Any]:
            seen = set(stream.seen_classes(i))
            tm, vm = np.isin(labels, list(seen)), np.isin(y_val_all, list(seen))
            # validation rows of the classes seen so far (features are frozen, so reusable)
            with torch.no_grad():
                sv = skill.decision_function(h_val_all[vm].to(device)).cpu().numpy()
                st = skill.decision_function(h_test[tm].to(device)).cpu().numpy()
            return evaluate_route(sv, y_val_all[vm], st, labels[tm], target)

        _, records = run_continual_skill(exps, target, skill_cfg, seed, device, d, evaluate)
        report["continual"] = records
        cap = skill_cfg.memory_per_class
        joint = {
            "joint_all_positives": joint_skill(pos_all, negs, target, skill_cfg, seed, device),
            "joint_memory_positives": joint_skill(
                cap_samples(pos_all, cap, seed + 31 * target) if cap > 0 else pos_all,
                negs, target, skill_cfg, seed, device,
            ),
        }  # fmt: skip
        final = {"continual": records[-1]}
        for name, s in joint.items():
            final[name] = evaluate(s, n_exp - 1)
        report["part2_final"] = {
            k: {"threshold_0": v["threshold_0"], "validated": v["validated"]}
            for k, v in final.items()
        }
        for view, title in (("threshold_0", "threshold 0"), ("validated", "validated threshold")):
            _table(
                f"PART 2 - final, class {target} vs rest over all {len(stream.seen_classes(n_exp - 1))} classes, {title}:",
                {k: v[view] for k, v in final.items()},
            )
        print("\n  continual per experience (AUC / recall / precision / F1 at threshold 0):")
        for r in records:
            m = r["threshold_0"]
            print(
                f"    exp {r['experience']}: auc={m['auc']:.3f} recall={m['target_recall']:.3f} "
                f"prec={m['target_precision']:.3f} f1={m['target_f1']:.3f} "
                f"(replay pos={r['train'].get('replay_positive')}, "
                f"neg={r['train'].get('n_negative')})"
            )

    out = Path(cfg.output_dir) / (cfg.name or f"single_class_t{target}_{cfg.data.dataset}_s{seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {out / 'comparison.json'}")
    return report


if __name__ == "__main__":
    main()
