# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Regression tests for target exposure timing in class-incremental runs."""

import numpy as np

from experiments import run_target


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
