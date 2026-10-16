# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Matched Skill vs feature-map comparison, and the continual update's data (third review)."""

import json

import numpy as np
import pytest
import torch

from incremental_feature_cl.experiments import compare_single_class as cmp
from incremental_feature_cl.models import IdentityBackbone, OneVsRestSkillModel, Skill, SkillConfig

D = 6


def _blob(n, center, seed):
    g = torch.Generator().manual_seed(seed)
    return center + 0.8 * torch.randn(n, D, generator=g)


@pytest.fixture(scope="module")
def tensors():
    pos = _blob(30, 1.0, 1)
    neg = _blob(90, -1.0, 2)
    val_x = torch.cat([_blob(10, 1.0, 3), _blob(30, -1.0, 4)])
    val_y = np.array([0] * 10 + [1] * 30)
    test_x = torch.cat([_blob(20, 1.0, 5), _blob(60, -1.0, 6)])
    test_y = np.array([0] * 20 + [1] * 60)
    return pos, neg, (val_x, val_y), (test_x, test_y)


# ---------------------------------------------------------------- part 1
@pytest.mark.parametrize(
    "cfg",
    [
        SkillConfig(optimizer="sgd", epochs=4, batch_size=10_000, lr=0.05),
        SkillConfig(optimizer="adam", epochs=4, batch_size=10_000, lr=0.01),
        SkillConfig(optimizer="sgd", epochs=3, batch_size=10_000, feature_expansion_dim=4),
        SkillConfig(optimizer="sgd", epochs=3, batch_size=10_000, pos_weight=None),
    ],
    ids=["sgd", "adam", "sgd+expansion", "no-pos-weight"],
)
def test_skill_and_feature_map_agree_under_matched_conditions(tensors, cfg):
    pos, neg, val, test = tensors
    r = cmp.matched_comparison(pos, neg, val, test, 0, cfg, seed=3, device=torch.device("cpu"))
    assert r["same_initial_weights"]
    assert r["pos_weight"]["skill"] == r["pos_weight"]["feature_map"]
    assert r["score_agreement"]["max_abs_difference"] < 1e-4  # same learner, same steps
    for view in ("threshold_0", "validated"):
        for k, v in r["skill"][view].items():
            assert v == pytest.approx(r["feature_map"][view][k], abs=1e-6)


def test_matched_comparison_refuses_lbfgs(tensors):
    pos, neg, val, test = tensors
    with pytest.raises(ValueError, match="L-BFGS"):
        cmp.matched_comparison(
            pos, neg, val, test, 0, SkillConfig(optimizer="lbfgs"), 0, torch.device("cpu")
        )


def test_matched_comparison_detects_a_real_difference(tensors):
    """The harness is not vacuous: a different optimization budget makes the routes diverge."""
    pos, neg, val, test = tensors
    cfg = SkillConfig(optimizer="sgd", epochs=4, batch_size=10_000)
    r = cmp.matched_comparison(pos, neg, val, test, 0, cfg, 3, torch.device("cpu"))
    # train the feature-map route for a different number of epochs by hand
    d = pos.shape[1]
    skill = Skill(0, d, cfg, 3)
    skill._fit(pos, neg, 0, "cpu")
    short = Skill(0, d, SkillConfig(optimizer="sgd", epochs=1, batch_size=10_000), 3)
    short._fit(pos, neg, 0, "cpu")
    gap = (skill.decision_function(test[0]) - short.decision_function(test[0])).abs().max()
    assert float(gap) > 100 * r["score_agreement"]["max_abs_difference"]


def test_end_to_end_outputs(tmp_path):
    rep = cmp.main(
        [
            "--dataset", "synthetic", "--backbone", "identity", "--n-experiences", "3",
            "--ratio", "1:2", "--memory-per-class", "6", "--train-epochs", "3",
            "--train-mb-size", "100000", "--seed", "1", "--device", "cpu",
            "--set", "data.synthetic.n_classes=9", "--output-dir", str(tmp_path), "--name", "c",
        ]
    )  # fmt: skip
    assert rep["part1"]["same_initial_weights"]
    assert rep["part1"]["score_agreement"]["max_abs_difference"] < 1e-4
    assert set(rep["part2_final"]) == {"continual", "joint_all_positives", "joint_memory_positives"}
    assert len(rep["continual"]) == 3
    saved = json.loads((tmp_path / "c" / "comparison.json").read_text())
    assert saved["target"] == 0


def test_comparison_rejects_lbfgs_and_conflicting_flags():
    for argv in (["--optimizer", "lbfgs"], ["--skill-max-iter", "5"]):
        with pytest.raises(SystemExit) as e:
            cmp.main(["--dataset", "synthetic", *argv])
        assert e.value.code == 2


# ---------------------------------------------------------------- part 3: the continual update
def _class_data(classes, n, seed):
    """Feature 0 carries the class id (+ small noise) so every row's class is recoverable."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        x = 0.3 * torch.randn(n, D, generator=g)
        x[:, 0] = c + 0.05 * torch.randn(n, generator=g)
        xs.append(x)
        ys.append(torch.full((n,), c))
    return torch.cat(xs), torch.cat(ys)


def _cls(rows):
    return torch.round(rows[:, 0]).long()


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    original = Skill._fit

    def spy(self, pos, neg, step_seed, device):
        calls.append({"target": self.target_class, "pos": pos.clone(), "neg": neg.clone()})
        return original(self, pos, neg, step_seed, device)

    monkeypatch.setattr(Skill, "_fit", spy)
    return calls


def test_old_and_new_skills_receive_the_intended_positives_and_negatives(recorded):
    cap, n = 5, 12
    cfg = SkillConfig(memory_per_class=cap, epochs=1, target_to_negatives=0.5)
    m = OneVsRestSkillModel(IdentityBackbone(D), config=cfg, seed=0)
    m.update(*_class_data([0, 1], n, 0))
    recorded.clear()
    m.update(*_class_data([2, 3], n, 1))
    by_target = {c["target"]: c for c in recorded}
    assert set(by_target) == {0, 1, 2, 3}  # every skill, old and new, is refreshed

    for old in (0, 1):  # old skills: replayed positives, replayed old + new negatives
        pos, neg = by_target[old]["pos"], by_target[old]["neg"]
        assert (_cls(pos) == old).all() and len(pos) == cap  # memory budget, not the full class
        classes = set(_cls(neg).tolist())
        assert old not in classes and {2, 3} <= classes  # sees the newly arrived classes
        assert ({0, 1} - {old}) <= classes  # and keeps its replayed old negative
        assert len(neg) == 2 * len(pos)  # 1:2 target:negative ratio
    for new in (2, 3):  # new skills: all their fresh positives, replayed old classes as negatives
        pos, neg = by_target[new]["pos"], by_target[new]["neg"]
        assert (_cls(pos) == new).all() and len(pos) == n
        classes = set(_cls(neg).tolist())
        assert new not in classes and {0, 1} <= classes and ({2, 3} - {new}) <= classes


def test_replay_memory_respects_budget_and_only_holds_true_class_rows():
    cap = 4
    m = OneVsRestSkillModel(
        IdentityBackbone(D), config=SkillConfig(memory_per_class=cap, epochs=1), seed=0
    )
    for step, classes in enumerate(([0, 1], [2, 3], [4, 5])):
        m.update(*_class_data(classes, 10, step))
    for c in range(6):
        skill = m._skills[str(c)]
        assert 0 < len(skill.positive_memory) <= cap
        assert (_cls(skill.positive_memory) == c).all()
        assert c not in skill.negative_memory
        for k, rows in skill.negative_memory.items():
            assert len(rows) <= cap and (_cls(rows) == k).all()
        # a skill remembers every other class it has met (here: all five, none dropped)
        assert set(skill.negative_memory) == set(range(6)) - {c}
