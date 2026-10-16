# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Validated thresholds and the argmax-free firing report (second PR #5 review)."""

import csv
import inspect

import numpy as np
import pytest
import torch

from incremental_feature_cl.evaluation import (
    best_f1_threshold,
    decision_report,
    per_skill_metrics,
    select_thresholds,
    skill_decisions,
)
from incremental_feature_cl.evaluation.forgetting import (
    forgetting_per_class,
    mean_running_forgetting,
    running_forgetting,
)
from incremental_feature_cl.evaluation.result_store import RunResult
from incremental_feature_cl.experiments import train_ovr
from incremental_feature_cl.models import IdentityBackbone, OneVsRestSkillModel, SkillConfig


# ---------------------------------------------------------------- thresholds
def test_best_f1_threshold_separates_a_shifted_score_distribution():
    # a skill whose positives sit at ~2 and negatives at ~1: threshold 0 fires on everything
    scores = np.array([2.1, 2.0, 1.9, 1.2, 1.1, 1.0, 0.9, 0.8])
    pos = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    t, f1 = best_f1_threshold(scores, pos)
    assert 1.2 < t < 1.9 and f1 == pytest.approx(1.0)
    # at the default threshold 0 the same skill has precision 3/8
    assert (scores > 0).sum() == 8


def test_best_f1_threshold_degenerate_cases_fall_back_to_zero():
    assert best_f1_threshold(np.array([1.0, 2.0]), np.array([False, False])) == (0.0, None)
    assert best_f1_threshold(np.array([1.0, 2.0]), np.array([True, True])) == (0.0, None)


def test_best_f1_threshold_handles_tied_scores():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    t, _ = best_f1_threshold(scores, np.array([True, False, True, False]))
    assert t < 1.0  # fires on the tie group (recall 1) rather than on none of it


def test_select_thresholds_uses_only_seen_classes_and_validation_scores():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2], 20)
    scores = rng.normal(0, 1, (60, 3)) + 3 * np.eye(3)[labels]
    out = select_thresholds(scores, labels, [0, 1, 2], seen=[0, 1])
    assert set(out) == {0, 1}  # class 2 is not seen yet


# ---------------------------------------------------------------- decision report
def _toy():
    # 3 skills, 6 samples (2 per class); thresholds 0
    labels = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array(
        [
            [2.0, -1.0, -1.0],  # exact
            [2.0, 3.0, -1.0],  # true fires but skill 1 too (multi) - argmax would be WRONG
            [-1.0, -1.0, -1.0],  # nothing fires (abstain) - argmax would answer anyway
            [-1.0, 2.0, -1.0],  # exact
            [-1.0, -1.0, 1.0],  # exact
            [1.0, -1.0, -1.0],  # a single skill fires, but the wrong one
        ]
    )
    return scores, labels


def test_decision_report_counts_abstentions_ambiguity_and_exact_matches():
    scores, labels = _toy()
    r = decision_report(scores, labels, [0, 1, 2], [0, 1, 2])
    assert r["n_samples"] == 6
    assert r["none_fired_rate"] == pytest.approx(1 / 6)
    assert r["multi_fired_rate"] == pytest.approx(1 / 6)
    assert r["single_fired_rate"] == pytest.approx(4 / 6)
    assert r["exact_match_rate"] == pytest.approx(3 / 6)
    assert r["precision_when_single"] == pytest.approx(3 / 4)
    assert r["true_skill_fires_rate"] == pytest.approx(4 / 6)
    # wrong skills fired: sample 2 (skill 1 alongside the true one) and sample 6 (the wrong one)
    assert r["mean_wrong_skills_fired"] == pytest.approx(2 / 6)


def test_decision_report_refuses_credit_that_an_argmax_would_give():
    labels = np.array([0, 0])
    ambiguous = np.array([3.0, 2.0, -1.0])  # argmax -> 0 (right), but skill 1 also fires
    abstain = np.array([-1.0, -2.0, -3.0])  # argmax -> 0 (right), but no skill recognises it
    scores = np.stack([ambiguous, abstain])
    assert (scores.argmax(1) == labels).all()  # an argmax readout would score 100 %
    r = decision_report(scores, labels, [0, 1, 2], [0, 1, 2])
    assert r["exact_match_rate"] == 0.0
    assert r["multi_fired_rate"] == 0.5 and r["none_fired_rate"] == 0.5


def test_decision_report_firing_matrix_and_per_class_numbers():
    scores, labels = _toy()
    r = decision_report(scores, labels, [0, 1, 2], [0, 1, 2])
    m = np.array(r["firing_matrix"])
    assert m.shape == (3, 3)
    np.testing.assert_allclose(m[0], [1.0, 0.5, 0.0])  # class 0: skill 0 always, skill 1 half
    assert r["per_class"][2]["recall"] == pytest.approx(0.5)
    assert r["per_class"][0]["false_positive_rate"] == pytest.approx(1 / 4)  # fires on a class-2 x
    assert r["per_class"][0]["precision"] == pytest.approx(2 / 3)


def test_decision_report_respects_thresholds_and_seen_classes():
    scores, labels = _toy()
    low = decision_report(scores, labels, [0, 1, 2], [0, 1, 2], {0: 0.0, 1: 0.0, 2: 0.0})
    high = decision_report(scores, labels, [0, 1, 2], [0, 1, 2], {0: 5.0, 1: 5.0, 2: 5.0})
    assert high["none_fired_rate"] == 1.0 and low["none_fired_rate"] < 1.0
    only01 = decision_report(scores, labels, [0, 1, 2], [0, 1])
    assert only01["n_samples"] == 4 and set(only01["per_class"]) == {0, 1}


def test_per_skill_metrics_default_threshold_is_zero_and_custom_is_applied():
    scores, labels = _toy()
    base = per_skill_metrics(scores, labels, [0, 1, 2], [0, 1, 2])
    assert all(m["threshold"] == 0.0 for m in base.values())
    strict = per_skill_metrics(scores, labels, [0, 1, 2], [0, 1, 2], {0: 1.5})
    assert strict[0]["threshold"] == 1.5
    assert strict[0]["target_recall"] == pytest.approx(1.0)  # both class-0 scores are 2.0
    assert strict[0]["false_positive_rate"] == 0.0 < base[0]["false_positive_rate"]
    assert strict[0]["auc"] == base[0]["auc"]  # AUC is threshold free


def test_the_decision_module_never_calls_argmax():
    code = [
        line.split("#")[0]
        for line in inspect.getsource(skill_decisions).splitlines()
        if not line.lstrip().startswith(('"', "*", "``"))
    ]
    stripped = "\n".join(code)
    # the module docstring mentions the word; executable code must not call it
    body = stripped[stripped.index("from __future__") :]
    assert ".argmax(" not in body and "np.argmax(" not in body.replace("np.argmin", "")


# ---------------------------------------------------------------- forgetting
def test_running_forgetting_matches_standard_forgetting_at_the_end():
    v = np.array([[0.9, np.nan, np.nan], [0.7, 0.8, np.nan], [0.8, 0.5, 0.6], [0.6, 0.6, 0.4]])
    first = {0: 0, 1: 1, 2: 2}
    r = running_forgetting(v, first)
    assert r[0, 0] == 0.0 and np.isnan(r[0, 1])
    assert r[1, 0] == pytest.approx(0.2) and r[2, 1] == pytest.approx(0.3)
    np.testing.assert_allclose(r[-1], forgetting_per_class(v, first), atol=1e-12)
    assert mean_running_forgetting(v, first).shape == (4,)


# ---------------------------------------------------------------- validation split
def test_split_validation_is_disjoint_stratified_and_deterministic():
    y = torch.tensor([0] * 20 + [1] * 10 + [2])  # class 2 has a single sample
    x = torch.arange(len(y)).float().unsqueeze(1)
    (xt, yt), (xv, yv) = train_ovr.split_validation(x, y, 0.2, seed=3)
    assert set(xt.flatten().tolist()).isdisjoint(xv.flatten().tolist())
    assert len(xt) + len(xv) == len(x)
    assert int((yv == 0).sum()) == 4 and int((yv == 1).sum()) == 2
    assert int((yt == 2).sum()) == 1 and int((yv == 2).sum()) == 0  # singleton stays in training
    _, (xv2, _) = train_ovr.split_validation(x, y, 0.2, seed=3)
    assert torch.equal(xv, xv2)
    with pytest.raises(ValueError):
        train_ovr.split_validation(x, y, 0.0, seed=0)


# ---------------------------------------------------------------- diagnostic (Finding 5)
D = 6


def _data(classes, n=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        center = torch.zeros(D)
        center[c % D] = 3.0
        xs.append(center + 0.3 * torch.randn(n, D, generator=g))
        ys.append(torch.full((n,), c))
    return torch.cat(xs), torch.cat(ys)


def test_newly_known_negatives_is_computed_before_memory_is_committed():
    m = OneVsRestSkillModel(
        IdentityBackbone(D), config=SkillConfig(memory_per_class=5, epochs=1), seed=0
    )
    first = m.update(*_data([0, 1]))
    assert first["skills"][0]["newly_known_negative_classes"] == [1]
    second = m.update(*_data([2, 3]))
    # skill.update() never writes memory (remember() runs afterwards), so these are the classes
    # the old skills learn about *in this experience*, not an always-empty list
    assert second["skills"][0]["newly_known_negative_classes"] == [2, 3]
    assert second["skills"][1]["newly_known_negative_classes"] == [2, 3]
    # a new skill already knows the old classes (replay) and only learns the other new one
    assert second["skills"][2]["newly_known_negative_classes"] == [3]
    # the same classes again: nothing is newly known any more
    third = m.update(*_data([2, 3], seed=5))
    assert third["skills"][0]["newly_known_negative_classes"] == []


# ---------------------------------------------------------------- experiment output
ARGS = [
    "--dataset", "synthetic", "--backbone", "identity", "--n-experiences", "3",
    "--train-epochs", "2", "--lr", "0.05", "--device", "cpu", "--no-plots", "--ratio", "1:2",
    "--memory-per-class", "8", "--set", "data.synthetic.n_classes=12", "--seed", "1",
]  # fmt: skip


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ovr")
    train_ovr.main([*ARGS, "--val-fraction", "0.25", "--output-dir", str(out), "--name", "t"])
    return out / "t"


def test_run_reports_thresholds_decisions_and_per_skill_forgetting(run_dir):
    r = RunResult.load(run_dir / "results.json")
    last = r.experiences[-1]
    assert last["n_validation"] > 0
    assert set(map(int, last["thresholds"])) == set(last["seen_classes"])
    for key in ("decision_report", "decision_report_tuned", "target_metrics_tuned"):
        assert key in last
    for e in r.experiences:
        assert "firing_matrix" in e["decision_report"] or e is not last
        assert len(e["per_skill_auc"]) == r.num_classes
    for k in (
        "average_skill_forgetting",
        "average_skill_auc_forgetting",
        "average_skill_tuned_recall_forgetting",
        "final_tuned_target_f1",
        "final_tuned_exact_match_rate",
        "final_exact_match_rate",
    ):
        assert k in r.summary, k
    for name in (
        "skill_auc.csv",
        "skill_auc_forgetting.csv",
        "skill_target_recall_tuned_forgetting.csv",
        "skill_thresholds.csv",
        "firing_matrix_tuned.csv",
        "firing_matrix_threshold0.csv",
    ):
        assert (run_dir / name).exists(), name
    # nothing in the saved record is an argmax-derived accuracy
    assert "final_accuracy_seen" not in r.summary and "accuracy_seen" not in last


def test_validation_samples_are_never_used_for_training(run_dir):
    r = RunResult.load(run_dir / "results.json")
    n_per_class = 20  # make_synthetic_dataset default (n_train_per_class)
    held_out = [e["n_validation"] for e in r.experiences]
    assert held_out == sorted(held_out)  # accumulates with the classes seen so far
    # 25 % of every class's training data, for the 12 classes seen by the end
    assert held_out[-1] == pytest.approx(0.25 * 12 * n_per_class, abs=12)


def test_val_fraction_zero_reports_only_the_threshold_zero_decision(tmp_path):
    train_ovr.main([*ARGS, "--val-fraction", "0", "--output-dir", str(tmp_path), "--name", "z"])
    r = RunResult.load(tmp_path / "z" / "results.json")
    assert "thresholds" not in r.experiences[-1] and "decision_report" in r.experiences[-1]
    assert not (tmp_path / "z" / "skill_thresholds.csv").exists()
    with (tmp_path / "z" / "skill_auc.csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == 3
