# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Regression tests for target exposure timing in class-incremental runs."""

import warnings

import numpy as np
import pytest

from experiments import run_target


def _stream(monkeypatch):
    rng = np.random.RandomState(7)
    y_train = np.repeat(np.arange(4), 24)
    y_test = np.repeat(np.arange(4), 8)
    X_train = np.eye(4)[y_train] + rng.normal(scale=0.05, size=(len(y_train), 4))
    X_test = np.eye(4)[y_test] + rng.normal(scale=0.05, size=(len(y_test), 4))
    monkeypatch.setattr(
        run_target, "load_dataset", lambda dataset, seed: (X_train, X_test, y_train, y_test)
    )


def _run(target_class, n_experiences, flag, **kw):
    return run_target.run_experiment(
        dataset="synthetic",
        target_class=target_class,
        n_experiences=n_experiences,
        seed=3,
        new_feature_dim=0,
        n_epochs=1,
        target_every_experience=flag,
        **kw,
    )


def test_enabled_and_disabled_modes_differ_only_after_natural_arrival(monkeypatch):
    _stream(monkeypatch)
    on = _run(1, 4, True, replay_mem_size=0)
    off = _run(1, 4, False, replay_mem_size=0)
    assert on["target_reintroduction_effective"] is True
    assert off["target_reintroduction_effective"] is False
    for mode in (on, off):  # identical up to and including natural arrival
        assert mode["first_training_exposure_experience"] == 1
    for e in (0, 1):
        for key in ("n_target_train", "n_target_injected", "classes_in_training_batch"):
            assert on["history"][e][key] == off["history"][e][key]
        assert on["history"][e]["target_added_to_experience"] is False
    assert on["history"][1]["n_target_natural"] > 0
    for e in (2, 3):
        assert on["history"][e]["target_added_to_experience"] is True
        assert on["history"][e]["n_target_injected"] > 0
        assert on["history"][e]["n_target_natural"] == 0
        assert off["history"][e]["target_added_to_experience"] is False
        assert off["history"][e]["n_target_train"] == 0
        assert off["history"][e]["n_target_injected"] == 0


def test_target_in_last_experience_makes_flag_a_noop_and_warns(monkeypatch):
    _stream(monkeypatch)
    with pytest.warns(UserWarning, match="no effect"):
        on = _run(3, 4, True, replay_mem_size=0)
    off = _run(3, 4, False, replay_mem_size=0)
    assert on["target_reintroduction_effective"] is False
    assert [h["n_target_train"] for h in on["history"]] == [
        h["n_target_train"] for h in off["history"]
    ]
    assert (
        on["final_target_recall_at_zero_threshold"] == off["final_target_recall_at_zero_threshold"]
    )


def test_injected_target_rows_train_but_do_not_enter_replay_memory(monkeypatch):
    _stream(monkeypatch)
    result = _run(1, 4, True, replay_mem_size=40)
    h = result["history"]
    natural = h[1]["fit_diagnostics"]
    assert natural["memory_target_added"] == natural["current_target_after_ratio"] > 0
    for e in (2, 3):
        fit = h[e]["fit_diagnostics"]
        assert fit["current_target_after_ratio"] == h[e]["n_target_injected"] > 0
        assert fit["current_not_offered_to_memory"] == h[e]["n_target_injected"]
        assert fit["memory_target_added"] == 0
        assert h[e]["memory_after_update"]["target"] == h[1]["memory_after_update"]["target"]
        # current-batch counts and replay counts are reported separately
        assert fit["replay_target_drawn_total"] > 0
        assert "replay_samples_drawn_total" in fit


def test_target_arriving_in_last_experience_with_reintroduction_off_is_silent(monkeypatch):
    _stream(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(3, 4, False, replay_mem_size=0)


def test_target_reintroduced_only_after_natural_class_arrival(monkeypatch):
    rng = np.random.RandomState(7)
    y_train = np.repeat(np.arange(4), 24)
    y_test = np.repeat(np.arange(4), 8)
    X_train = np.eye(4)[y_train] + rng.normal(scale=0.05, size=(len(y_train), 4))
    X_test = np.eye(4)[y_test] + rng.normal(scale=0.05, size=(len(y_test), 4))

    def fake_load_dataset(dataset, seed):
        assert dataset == "synthetic"
        return X_train, X_test, y_train, y_test

    monkeypatch.setattr(run_target, "load_dataset", fake_load_dataset)

    result = run_target.run_experiment(
        dataset="synthetic",
        target_class=1,
        n_experiences=4,
        seed=3,
        new_feature_dim=0,
        n_epochs=1,
        replay_mem_size=0,
        target_every_experience=True,
    )

    history = result["history"]
    assert result["natural_target_arrival_experience"] == 1
    assert result["first_training_exposure_experience"] == 1

    # Experience 0 must remain target-free: it precedes the target's natural arrival.
    assert history[0]["classes_arriving"] == [0]
    assert history[0]["classes_in_training_batch"] == [0]
    assert history[0]["n_target_train"] == 0
    assert history[0]["target_added_to_experience"] is False
    assert history[0]["first_training_exposure_experience"] is None

    # The target arrives naturally in experience 1, so it is not an injected duplicate.
    assert history[1]["classes_arriving"] == [1]
    assert history[1]["n_target_train"] > 0
    assert history[1]["target_added_to_experience"] is False
    assert history[1]["first_training_exposure_experience"] == 1

    # Only later experiences reintroduce the target samples.
    for experience in history[2:]:
        assert experience["target_added_to_experience"] is True
        assert 1 in experience["classes_in_training_batch"]
        assert experience["n_target_train"] > 0
        assert experience["first_training_exposure_experience"] == 1
