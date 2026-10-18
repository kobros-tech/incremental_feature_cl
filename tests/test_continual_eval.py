# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Tests for the continual-learning evaluation protocol."""

import numpy as np
import pytest

from experiments import continual_eval as ce


def _step(per_class):
    return {
        c: {"recall": r, "accuracy": r, "specificity": 1.0, "auc": 1.0}
        for c, r in per_class.items()
    }


def test_summarise_matches_hand_computation():
    # classes 0 (arrives in 0), 1 (arrives in 1), 2 (arrives in 2); 3 experiences
    metrics = [
        _step({0: 1.0}),
        _step({0: 0.8, 1: 0.9}),
        _step({0: 0.4, 1: 0.9, 2: 0.7}),
    ]
    arrival = {0: 0, 1: 1, 2: 2}
    s = ce.summarise([{int(k): v for k, v in m.items()} for m in metrics], arrival, 3)
    assert s["final_recall"] == pytest.approx((0.4 + 0.9 + 0.7) / 3)
    assert s["recall_at_arrival"] == pytest.approx((1.0 + 0.9 + 0.7) / 3)
    # forgetting: class 0 peaked at 1.0 -> 0.4; class 1 peaked at 0.9 -> 0.9 (class 2 excluded)
    assert s["recall_forgetting"] == pytest.approx((0.6 + 0.0) / 2)
    assert s["recall_backward_transfer"] == pytest.approx((0.4 - 1.0 + 0.9 - 0.9) / 2)
    assert s["average_incremental_recall"] == pytest.approx(
        np.mean([1.0, (0.8 + 0.9) / 2, (0.4 + 0.9 + 0.7) / 3])
    )


def test_stream_only_reports_seen_classes_and_arrival_is_correct():
    r = ce.run_stream("digits", "replay", seed=0, n_experiences=5, n_epochs=1)
    arrival = {int(c): a for c, a in r["arrival"].items()}
    for e, step in enumerate(r["metrics"]):
        assert {int(c) for c in step} == {c for c, a in arrival.items() if a <= e}
    assert sorted(arrival.values()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_runs_are_reproducible_and_seed_shuffles_class_order():
    a = ce.run_stream("digits", "replay", seed=1, n_epochs=1)
    b = ce.run_stream("digits", "replay", seed=1, n_epochs=1)
    c = ce.run_stream("digits", "replay", seed=2, n_epochs=1)
    assert a["summary"] == b["summary"]
    assert a["arrival"] != c["arrival"]


def test_replay_retains_recall_better_than_naive_on_a_small_stream():
    kw = {"n_experiences": 5, "n_epochs": 2}
    naive = ce.run_stream("digits", "naive", seed=0, **kw)["summary"]
    replay = ce.run_stream("digits", "replay", seed=0, **kw)["summary"]
    assert replay["recall_forgetting"] < naive["recall_forgetting"]
    assert replay["final_recall"] > naive["final_recall"]


def test_verdict_flags_a_method_that_does_not_beat_naive():
    def fake(**v):
        base = {
            "recall_forgetting": 0.5, "final_recall": 0.5, "final_accuracy": 0.5,
            "final_balanced_accuracy": 0.7, "balanced_accuracy_at_arrival": 0.8,
            "recall_at_arrival": 0.9, "auc_at_arrival": 0.9, "replay_memory_fraction_of_train": 0.2,
        }  # fmt: skip
        return {"summary": {**base, **v}}

    runs = {"naive": [fake(), fake(), fake()], "replay": [fake(), fake(), fake()]}
    out = ce.verdict(runs, plasticity_margin=0.05, joint_margin=0.15)
    assert out["all_passed"] is False  # identical to naive: no evidence of benefit
    assert out["retention_recall_forgetting_lower_than_naive"]["passed"] is False


def test_default_replay_budget_is_a_fraction_of_the_training_set_and_is_reported():
    r = ce.run_stream("digits", "replay", seed=0, n_epochs=1, memory_fraction=0.2)
    s = r["summary"]
    assert s["replay_memory_rows"] <= r["replay_mem_size_per_class_model"] * 10
    assert s["replay_memory_fraction_of_train"] == pytest.approx(
        s["replay_memory_rows"] / r["n_train_rows"]
    )
    assert s["replay_memory_fraction_of_train"] <= 0.2 + 1e-9
    assert (
        ce.run_stream("digits", "naive", seed=0, n_epochs=1)["summary"]["replay_memory_rows"] == 0
    )


def test_explicit_per_model_memory_overrides_the_fraction():
    r = ce.run_stream("digits", "replay", seed=0, n_epochs=1, replay_mem_size=10)
    assert r["replay_mem_size_per_class_model"] == 10
    assert r["summary"]["replay_memory_rows"] <= 100


def test_offline_methods_report_only_final_metrics_not_arrival_or_incremental():
    for method in ("joint", "joint_matched"):
        r = ce.run_stream("digits", method, seed=0, n_epochs=1, n_experiences=3)
        assert r["offline"] is True
        assert len(r["metrics"]) == 1  # a single final evaluation, nothing copied per experience
        s = r["summary"]
        assert np.isfinite(s["final_recall"]) and np.isfinite(s["final_balanced_accuracy"])
        for key in (
            "recall_at_arrival", "accuracy_at_arrival", "auc_at_arrival",
            "average_incremental_recall", "average_incremental_accuracy",
            "recall_forgetting", "accuracy_forgetting", "recall_backward_transfer",
            "balanced_accuracy_at_arrival",
        ):  # fmt: skip
            assert np.isnan(s[key]), key
        assert s["replay_memory_fraction_of_train"] == 1.0
    stream = ce.run_stream("digits", "naive", seed=0, n_epochs=1, n_experiences=3)
    assert stream["offline"] is False and len(stream["metrics"]) == 3


@pytest.mark.parametrize("n_experiences", [3, 4, 5])
@pytest.mark.parametrize("n_epochs", [1, 2, 5])
def test_every_method_gets_the_same_current_sample_training_budget(n_epochs, n_experiences):
    runs = {
        m: ce.run_stream("digits", m, seed=1, n_epochs=n_epochs, n_experiences=n_experiences)
        for m in ce.METHODS
    }
    n_rows = runs["naive"]["n_train_rows"]
    for m, r in runs.items():  # each row is presented n_epochs times in total, whatever the split
        assert r["summary"]["train_presentations_current"] == n_epochs * n_rows, m
    assert runs["joint_matched"]["summary"]["train_presentations_replay"] == 0
    assert runs["naive"]["summary"]["train_presentations_replay"] == 0
    # replay is the only method with extra (replayed) presentations, and they are reported
    assert runs["replay"]["summary"]["train_presentations_replay"] > 0


def test_matched_joint_has_the_same_expansion_schedule_as_the_stream(monkeypatch):
    from incremental_feature_cl import IncrementalFeatureMapModel as Model

    widths = {}
    original = Model.expand_features

    def spy(self, new_dim=None):
        out = original(self, new_dim)
        widths.setdefault(self.__class__, []).append(out)
        return out

    counts = {}
    for method in ("naive", "joint_matched"):
        widths.clear()
        monkeypatch.setattr(Model, "expand_features", spy)
        ce.run_stream("digits", method, seed=0, n_epochs=1, n_experiences=4, new_feature_dim=3)
        counts[method] = sorted(set(widths[Model]))
    # 3 expansions after the first stage: input width 64 + 3, 6, 9 appended features
    assert counts["naive"] == counts["joint_matched"] == [67, 70, 73]


def test_verdict_fails_when_replay_stores_the_whole_training_set():
    def run(frac):
        return {
            "summary": {
                "recall_forgetting": 0.0, "final_recall": 1.0, "final_accuracy": 1.0,
                "final_balanced_accuracy": 1.0, "balanced_accuracy_at_arrival": 1.0,
                "recall_at_arrival": 1.0, "auc_at_arrival": 1.0,
                "replay_memory_fraction_of_train": frac,
            }
        }  # fmt: skip

    def naive():
        r = run(0.0)
        r["summary"].update(recall_forgetting=0.9, final_recall=0.1, final_accuracy=0.1)
        r["summary"].update(final_balanced_accuracy=0.2)
        return r

    runs = {"naive": [naive() for _ in range(3)], "replay": [run(1.4) for _ in range(3)]}
    out = ce.verdict(runs, plasticity_margin=0.05, joint_margin=0.15)
    assert out["replay_stores_less_than_the_training_set"]["passed"] is False
    assert out["all_passed"] is False


def test_negatives_come_only_from_seen_classes():
    class Stub:
        classes_ = np.array([0, 1, 2])

        def decision_function(self, X):
            # class 0 fires on rows of class 0 and (wrongly) on every row of the unseen class 2
            y = X[:, 0]
            return np.column_stack(
                [
                    np.where((y == 0) | (y == 2), 1.0, -1.0),
                    np.where(y == 1, 1.0, -1.0),
                    -np.ones(len(y)),
                ]
            )

    y = np.repeat([0, 1, 2], 10)
    X = y[:, None].astype(float)
    out = ce._evaluate(Stub(), X, y, seen=[0, 1])
    assert out[0]["specificity"] == 1.0  # the unseen class-2 rows must not count as negatives
    assert out[0]["auc"] == 1.0
    assert out[0]["recall"] == 1.0


def test_no_seen_negatives_gives_nan_not_an_error():
    class Stub:
        classes_ = np.array([0, 1])

        def decision_function(self, X):
            return np.column_stack([np.ones(len(X)), -np.ones(len(X))])

    y = np.repeat([0, 1], 5)
    out = ce._evaluate(Stub(), y[:, None].astype(float), y, seen=[0])
    assert out[0]["recall"] == 1.0
    assert np.isnan(out[0]["specificity"]) and np.isnan(out[0]["auc"])
