# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""The optimizer is chosen explicitly and the configured options really control the training."""

import argparse
import math

import pytest
import torch

from incremental_feature_cl.experiments import common, train_ovr, train_skill
from incremental_feature_cl.models import Skill, SkillConfig
from incremental_feature_cl.models import skill as skill_module

D = 5


def _rows(n, shift, seed):
    g = torch.Generator().manual_seed(seed)
    return shift + torch.randn(n, D, generator=g)


def _fit(cfg: SkillConfig, n_pos=10, n_neg=30):
    s = Skill(target_class=0, feature_dim=D, config=cfg, seed=0)
    s._fit(_rows(n_pos, 1.0, 1), _rows(n_neg, -1.0, 2), step_seed=0, device="cpu")
    return s


@pytest.fixture
def loss_calls(monkeypatch):
    calls = {"n": 0}
    real = skill_module.F.binary_cross_entropy_with_logits

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(skill_module.F, "binary_cross_entropy_with_logits", counted)
    return calls


# ---------------------------------------------------------------- sgd / adam
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize(("epochs", "mb"), [(1, 10), (3, 10), (2, 7)])
def test_epochs_and_minibatch_control_minibatch_training(optimizer, epochs, mb, loss_calls):
    _fit(SkillConfig(optimizer=optimizer, epochs=epochs, batch_size=mb))
    assert loss_calls["n"] == epochs * math.ceil(40 / mb)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_max_iter_does_not_change_minibatch_training(optimizer):
    a = _fit(SkillConfig(optimizer=optimizer, epochs=2, batch_size=8, max_iter=1))
    b = _fit(SkillConfig(optimizer=optimizer, epochs=2, batch_size=8, max_iter=500))
    x = _rows(6, 0.0, 9)
    assert torch.equal(a.decision_function(x), b.decision_function(x))


# ---------------------------------------------------------------- lbfgs
def test_lbfgs_ignores_epochs_and_batch_size_but_obeys_max_iter(loss_calls):
    x = _rows(6, 0.0, 9)
    a = _fit(SkillConfig(optimizer="lbfgs", epochs=1, batch_size=4, max_iter=30))
    b = _fit(SkillConfig(optimizer="lbfgs", epochs=9, batch_size=999, max_iter=30))
    assert torch.equal(a.decision_function(x), b.decision_function(x))
    short = _fit(SkillConfig(optimizer="lbfgs", max_iter=1))
    assert not torch.allclose(short.decision_function(x), a.decision_function(x))
    loss_calls["n"] = 0
    _fit(SkillConfig(optimizer="lbfgs", max_iter=3))
    assert 0 < loss_calls["n"] <= 3 * 1.25 + 5  # bounded by the iteration limit (plus line search)


# ---------------------------------------------------------------- the CLI contract
def _parse(argv, *, skills=True):
    p = argparse.ArgumentParser(allow_abbrev=False)
    common.add_common_args(p, "synthetic", ratio_default="1:5")
    common.add_skill_args(p)
    a = p.parse_args(argv)
    cfg = common.config_from_args(a, "target", "synthetic")
    common.check_training_contract(p, a, cfg, skills=skills)
    return a, cfg


def test_the_default_is_one_explicit_optimizer_with_no_hidden_weight_decay():
    a, cfg = _parse([])
    sc = common.skill_config_from(cfg, a)
    assert sc.optimizer == cfg.train.optimizer == "sgd"
    assert sc.weight_decay == cfg.train.weight_decay == 0.0
    assert sc.epochs == cfg.train.train_epochs and sc.batch_size == cfg.train.train_mb_size
    # opting in to lbfgs changes only the optimizer: no special weight decay appears
    a, cfg = _parse(["--optimizer", "lbfgs", "--skill-max-iter", "40"])
    sc = common.skill_config_from(cfg, a)
    assert sc.optimizer == "lbfgs" and sc.max_iter == 40 and sc.weight_decay == 0.0
    a, cfg = _parse(["--optimizer", "lbfgs", "--weight-decay", "0.001"])
    assert common.skill_config_from(cfg, a).weight_decay == 0.001


@pytest.mark.parametrize(
    "argv",
    [
        ["--skill-max-iter", "50"],  # sgd: max-iter would be ignored
        ["--optimizer", "adam", "--skill-max-iter", "50"],
        ["--optimizer", "lbfgs", "--train-epochs", "3"],  # lbfgs: epochs would be ignored
        ["--optimizer", "lbfgs", "--train-mb-size", "8"],
        ["--optimizer", "lbfgs", "--vs-feature-map", "0"],  # the feature-map trainer has no lbfgs
    ],
)
def test_options_that_would_be_silently_ignored_are_rejected(argv):
    with pytest.raises(SystemExit) as e:
        _parse(argv)
    assert e.value.code == 2


def test_lbfgs_is_rejected_on_the_feature_map_routes():
    with pytest.raises(SystemExit):
        _parse(["--optimizer", "lbfgs"], skills=False)


def test_contract_line_states_what_controls_the_training():
    a, cfg = _parse([])
    assert "epochs=" in common.training_contract(common.skill_config_from(cfg, a))
    a, cfg = _parse(["--optimizer", "lbfgs", "--skill-max-iter", "7"])
    line = common.training_contract(common.skill_config_from(cfg, a))
    assert "iterations<=7" in line and "unused" in line


@pytest.mark.parametrize("module", [train_ovr, train_skill])
def test_skill_entry_points_reject_conflicting_flags(module):
    with pytest.raises(SystemExit) as e:
        module.main(["--dataset", "synthetic", "--optimizer", "lbfgs", "--train-epochs", "2"])
    assert e.value.code == 2
