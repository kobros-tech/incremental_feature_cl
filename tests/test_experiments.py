# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pickle
import tarfile

import numpy as np
import pytest

from experiments.data import class_incremental_indices, load_dataset, resolve_dataset
from experiments.math_reference import polynomial_features
from experiments.metrics import select_balanced_accuracy_threshold, target_metrics
from experiments.run_target import run_experiment


def test_class_incremental_stream_partitions_every_sample_once():
    y = np.repeat(np.arange(6), [3, 4, 5, 6, 7, 8])
    streams = class_incremental_indices(y, 3)
    joined = np.concatenate(streams)
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(y)))
    assert len(set(joined.tolist())) == len(y)
    assert all(len(stream) > 0 for stream in streams)


def test_class_incremental_stream_rejects_invalid_experience_count():
    with pytest.raises(ValueError, match="n_experiences"):
        class_incremental_indices(np.repeat(np.arange(3), 2), 4)


def test_target_metrics_handles_missing_target_in_test_set():
    metrics = target_metrics(np.zeros(5, dtype=bool), np.full(5, -1.0))
    assert np.isnan(metrics["target_recall"])
    assert metrics["negative_specificity"] == 1.0
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["roc_auc"])


def test_run_target_returns_one_record_per_experience():
    result = run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=1,
        new_feature_dim=2,
        n_epochs=1,
        replay_mem_size=20,
        target_to_negatives=0.5,
    )
    assert len(result["history"]) == 3
    assert result["history"][-1]["feature_dim"] == 32 + 2 * 2
    assert result["history"][-1]["target_recall_forgetting"] is not None
    assert result["final_target_recall_forgetting"] is not None
    assert result["validation_size"] > 0
    assert (
        result["threshold_calibration_method"]
        == "validation_balanced_accuracy_seen_classes_each_experience"
    )
    assert (
        result["fixed_threshold_calibration_method"]
        == "first_valid_seen_class_validation_balanced_accuracy"
    )
    history = result["history"]
    assert all(
        row["threshold_calibration"]["method"] == "validation_balanced_accuracy_seen_classes"
        for row in history
    )
    assert all(
        "balanced_accuracy" in row["threshold_calibration"]["validation"]
        and "balanced_accuracy" in row["threshold_calibration"]["test"]
        for row in history
    )
    assert all(
        set(row["threshold_calibration"]["validation_classes"]).issubset(
            set(row["threshold_calibration"]["calibration_allowed_classes"])
        )
        for row in history
    )
    assert result["natural_target_arrival_experience"] == 0
    assert result["first_training_exposure_experience"] == 0
    assert result["fixed_threshold_calibration_experience"] == 0
    assert all("roc_auc" in row for row in history)
    assert all(
        row["fixed_threshold_calibration"]["threshold"] is not None
        and "balanced_accuracy" in row["fixed_threshold_calibration"]["test"]
        for row in history
    )
    fixed_thresholds = {row["fixed_threshold_calibration"]["threshold"] for row in history}
    assert len(fixed_thresholds) == 1
    arrival = next(row for row in history if row["target_recall_at_arrival"] is not None)
    assert arrival["target_recall_change_since_arrival"] == 0.0
    assert result["final_target_recall_change_since_arrival"] == pytest.approx(
        history[-1]["target_recall"] - arrival["target_recall"]
    )


def test_polynomial_features_match_degree_two_kernel():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 3))
    Z = rng.normal(size=(5, 3))
    actual = polynomial_features(X) @ polynomial_features(Z).T
    expected = (1.0 + X @ Z.T) ** 2
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _write_fake_cifar(root, nested):
    data_dir = root / "cifar-100-python" if nested else root
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data = np.arange(4 * 3072, dtype=np.uint8).reshape(4, 3072)
    test_data = np.arange(2 * 3072, dtype=np.uint8).reshape(2, 3072)
    for name, data, labels in (
        ("train", train_data, [0, 1, 0, 1]),
        ("test", test_data, [0, 1]),
    ):
        with (data_dir / name).open("wb") as file:
            pickle.dump(
                {b"data": data, b"fine_labels": labels},
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


@pytest.mark.parametrize("nested", [False, True])
def test_load_cifar100_from_extracted_directory(tmp_path, nested):
    _write_fake_cifar(tmp_path, nested)

    X_train, X_test, y_train, y_test = load_dataset(str(tmp_path), seed=7)

    assert X_train.shape == (4, 3072)
    assert X_test.shape == (2, 3072)
    np.testing.assert_array_equal(np.sort(y_train), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.sort(y_test), [0, 1])
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0


@pytest.mark.parametrize("name", ["digits", "synthetic", "cifar100"])
def test_resolve_dataset_accepts_names(name):
    assert resolve_dataset(name) == (name, None)


def test_resolve_dataset_turns_a_directory_into_cifar100(tmp_path):
    _write_fake_cifar(tmp_path, nested=True)
    assert resolve_dataset(tmp_path) == ("cifar100", tmp_path)
    assert resolve_dataset(str(tmp_path)) == ("cifar100", tmp_path)


def test_resolve_dataset_rejects_directory_without_batch_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="train/test files"):
        resolve_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="train/test files"):
        load_dataset(tmp_path)


def test_resolve_dataset_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_dataset("mnist")
    with pytest.raises(ValueError, match="Unsupported dataset"):
        resolve_dataset(tmp_path / "missing")


def test_run_experiment_accepts_a_cifar_directory_as_dataset(tmp_path):
    _write_fake_cifar(tmp_path, nested=False)
    result = run_experiment(
        dataset=str(tmp_path), target_class=0, n_experiences=2, n_epochs=1, new_feature_dim=2
    )
    assert result["dataset"] == "cifar100"
    assert result["data_dir"] == str(tmp_path)
    assert len(result["history"]) == 2
    assert result["history"][-1]["feature_dim"] == 3072 + 2


def test_load_cifar100_from_local_archive(tmp_path):
    source = tmp_path / "source"
    _write_fake_cifar(source, nested=True)
    archive_path = tmp_path / "fake-cifar.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source / "cifar-100-python" / "train", arcname="cifar-100-python/train")
        archive.add(source / "cifar-100-python" / "test", arcname="cifar-100-python/test")

    assert resolve_dataset(archive_path) == ("cifar100", archive_path)
    X_train, X_test, y_train, y_test = load_dataset(archive_path, seed=7)
    assert X_train.shape == (4, 3072)
    assert X_test.shape == (2, 3072)
    np.testing.assert_array_equal(np.sort(y_train), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.sort(y_test), [0, 1])


def test_run_experiment_records_and_uses_inference_batch_size():
    result = run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=2,
        new_feature_dim=2,
        n_epochs=1,
        inference_batch_size=7,
    )

    assert result["inference_batch_size"] == 7
    assert len(result["history"]) == 3


def test_run_target_examples_in_every_experience_by_default():
    result = run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=1,
        new_feature_dim=2,
        n_epochs=1,
        replay_mem_size=20,
    )

    history = result["history"]
    assert result["target_every_experience"] is True
    assert all(row["n_target_train"] > 0 for row in history)
    assert all(0 in row["classes_in_training_batch"] for row in history)
    assert [row["target_added_to_experience"] for row in history] == [False, True, True]
    assert history[0]["target_recall_change_since_arrival"] == 0.0
    assert all(row["fit_diagnostics"]["memory_target"] <= 10 for row in history)


def test_run_target_can_keep_strict_class_once_stream():
    result = run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=6,
        seed=1,
        new_feature_dim=2,
        n_epochs=1,
        replay_mem_size=20,
        target_every_experience=False,
    )

    history = result["history"]
    assert result["target_every_experience"] is False
    assert sum(row["n_target_train"] > 0 for row in history) == 1
    assert history[0]["target_recall_at_arrival"] is not None
    assert history[0]["target_recall_change_since_arrival"] == 0.0
    calibrated = next(
        row for row in history if row["fixed_threshold_calibration"]["threshold"] is not None
    )
    assert calibrated["fixed_threshold_calibration"]["validation_at_calibration"] is not None
    assert all(
        row["fixed_threshold_calibration"]["threshold"] is None
        for row in history[: history.index(calibrated)]
    )
    assert result["natural_target_arrival_experience"] == 0
    assert result["first_training_exposure_experience"] == 0
    assert result["fixed_threshold_calibration_experience"] == calibrated["experience"]


def test_target_metrics_counts_zero_score_as_positive():
    metrics = target_metrics(
        np.array([True, False]),
        np.array([0.0, 0.0]),
    )
    assert metrics["target_recall"] == 1.0
    assert metrics["negative_specificity"] == 0.0
    assert metrics["balanced_accuracy"] == 0.5


def test_validation_threshold_selection_maximizes_balanced_accuracy():
    y_validation = np.array([True, True, False, False])
    validation_scores = np.array([2.0, 1.0, 0.8, -2.0])

    threshold, metrics = select_balanced_accuracy_threshold(y_validation, validation_scores)

    assert threshold == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    calibrated_metrics = target_metrics(y_validation, validation_scores, threshold)
    assert calibrated_metrics["balanced_accuracy"] == 1.0


def test_threshold_calibration_rejects_single_class_validation_data():
    with pytest.raises(ValueError, match="both target and negative"):
        select_balanced_accuracy_threshold(np.array([True, True]), np.array([1.0, 2.0]))


def test_strict_stream_does_not_calibrate_using_future_target_validation():
    import experiments.run_target as runner

    target_class = int(np.max(runner.load_dataset("synthetic", seed=3)[2]))
    result = runner.run_experiment(
        dataset="synthetic",
        target_class=target_class,
        n_experiences=3,
        seed=3,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=False,
    )
    arrival = result["natural_target_arrival_experience"]
    assert arrival == result["first_training_exposure_experience"]
    assert arrival > 0
    for row in result["history"][:arrival]:
        assert row["threshold_calibration"]["threshold"] is None
        assert row["fixed_threshold_calibration"]["threshold"] is None
        assert target_class not in row["threshold_calibration"]["calibration_allowed_classes"]
    assert result["history"][arrival]["threshold_calibration"]["threshold"] is not None


def test_run_never_trains_or_stores_validation_or_test_rows(monkeypatch):
    import experiments.run_target as runner
    from incremental_feature_cl.model import IncrementalFeatureMapModel

    X_train = np.column_stack([np.arange(18, dtype=float), np.arange(18, dtype=float) ** 2])
    y_train = np.repeat(np.arange(3), 6)
    X_test = np.column_stack([1000 + np.arange(6), 2000 + np.arange(6)]).astype(float)
    y_test = np.array([0, 1, 2, 0, 1, 2])
    original_split = runner.train_test_split
    captured = {}

    def capture_split(*args, **kwargs):
        result = original_split(*args, **kwargs)
        captured["fit_indices"], captured["validation_indices"] = result
        return result

    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda dataset, seed: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
        ),
    )
    monkeypatch.setattr(runner, "train_test_split", capture_split)
    trained_rows = []
    stored_rows = []
    original_partial_fit = IncrementalFeatureMapModel.partial_fit
    original_update = IncrementalFeatureMapModel._update_memory

    def spy_partial_fit(self, X, y, classes=None, **kwargs):
        trained_rows.extend(np.asarray(X).copy())
        return original_partial_fit(self, X, y, classes=classes, **kwargs)

    def spy_update(self, X, pos):
        stored_rows.extend(np.asarray(X).copy())
        return original_update(self, X, pos)

    monkeypatch.setattr(IncrementalFeatureMapModel, "partial_fit", spy_partial_fit)
    monkeypatch.setattr(IncrementalFeatureMapModel, "_update_memory", spy_update)
    runner.run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=9,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=True,
    )

    fit_rows = {tuple(row) for row in X_train[captured["fit_indices"]]}
    validation_rows = {tuple(row) for row in X_train[captured["validation_indices"]]}
    test_rows = {tuple(row) for row in X_test}
    assert trained_rows and stored_rows
    assert all(tuple(row) in fit_rows for row in trained_rows)
    assert not ({tuple(row) for row in trained_rows} & validation_rows)
    assert not ({tuple(row) for row in trained_rows} & test_rows)
    assert all(tuple(row) in fit_rows for row in stored_rows)
    assert not ({tuple(row) for row in stored_rows} & validation_rows)
    assert not ({tuple(row) for row in stored_rows} & test_rows)


def test_evaluation_does_not_mutate_model_or_replay_memory():
    from incremental_feature_cl.model import IncrementalFeatureMapModel

    X = np.array([[-2.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model = (
        IncrementalFeatureMapModel(new_feature_dim=2, n_epochs=1, replay_mem_size=4, random_state=2)
        .partial_fit(X, y, classes=[0, 1])
        .partial_fit(X, y)
    )
    coef = model.coef_.copy()
    intercept = model.intercept_.copy()
    blocks = [(block.A.copy(), block.c.copy()) for block in model.blocks_]
    memory = [[row.copy() for row in rows] for rows in model._mem]
    predictions = model.predict(X)

    model.decision_function(X[::-1])
    model.predict_proba(X[::-1])
    model.predict(X[::-1])

    np.testing.assert_array_equal(model.coef_, coef)
    np.testing.assert_array_equal(model.intercept_, intercept)
    np.testing.assert_array_equal(model.predict(X), predictions)
    for block, (A, c) in zip(model.blocks_, blocks):
        np.testing.assert_array_equal(block.A, A)
        np.testing.assert_array_equal(block.c, c)
    for actual, expected in zip(model._mem, memory):
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected):
            np.testing.assert_array_equal(actual_row, expected_row)


def test_test_labels_do_not_change_training_or_calibration(monkeypatch):
    import experiments.run_target as runner

    X_train = np.column_stack([np.arange(30, dtype=float), np.arange(30, dtype=float) ** 2])
    y_train = np.repeat(np.arange(3), 10)
    X_test = np.column_stack([100 + np.arange(9), 200 + np.arange(9)]).astype(float)
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda dataset, seed: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
        ),
    )
    first = runner.run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=4,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=True,
    )
    flipped_test_labels = np.where(y_test == 0, 1, 0)
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda dataset, seed: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            flipped_test_labels.copy(),
        ),
    )
    second = runner.run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=4,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=True,
    )

    for left, right in zip(first["history"], second["history"]):
        np.testing.assert_allclose(
            left["coefficients_after_update"]["input_coefficient_l2"],
            right["coefficients_after_update"]["input_coefficient_l2"],
        )
        assert (
            left["threshold_calibration"]["threshold"]
            == right["threshold_calibration"]["threshold"]
        )
        assert (
            left["fixed_threshold_calibration"]["threshold"]
            == right["fixed_threshold_calibration"]["threshold"]
        )
        assert left["coefficients_after_update"] == right["coefficients_after_update"]
        assert left["memory_after_update"] == right["memory_after_update"]
        assert left["fit_diagnostics"] == right["fit_diagnostics"]
    assert not np.array_equal(y_test == 0, flipped_test_labels == 0)


def test_repeated_target_exposure_is_distinct_from_natural_class_arrival():
    import experiments.run_target as runner

    # Class 2 arrives in the middle experience, leaving a later experience in which
    # target examples should be reintroduced without leaking into earlier experiences.
    target_class = 2
    result = runner.run_experiment(
        dataset="synthetic",
        target_class=target_class,
        n_experiences=3,
        seed=5,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=True,
    )
    natural_arrival = result["natural_target_arrival_experience"]
    assert natural_arrival == 1
    assert result["first_training_exposure_experience"] == natural_arrival
    assert result["fixed_threshold_calibration_experience"] == natural_arrival

    pre_arrival_row = result["history"][0]
    assert target_class not in pre_arrival_row["classes_in_training_batch"]
    assert pre_arrival_row["target_added_to_experience"] is False
    assert pre_arrival_row["first_training_exposure_experience"] is None

    arrival_row = result["history"][natural_arrival]
    assert target_class in arrival_row["classes_arriving"]
    assert target_class in arrival_row["classes_in_training_batch"]
    assert arrival_row["target_added_to_experience"] is False
    assert result["target_recall_at_natural_class_arrival"] == arrival_row["target_recall"]

    later_row = result["history"][natural_arrival + 1]
    assert target_class not in later_row["classes_arriving"]
    assert target_class in later_row["classes_in_training_batch"]
    assert later_row["target_added_to_experience"] is True


def test_future_class_validation_features_do_not_change_earlier_threshold(monkeypatch):
    import experiments.run_target as runner

    X_train = np.column_stack([np.arange(18, dtype=float), np.arange(18, dtype=float) ** 2])
    y_train = np.repeat(np.arange(3), 6)
    X_test = np.column_stack([1000 + np.arange(6), 2000 + np.arange(6)]).astype(float)
    y_test = np.array([0, 1, 2, 0, 1, 2])

    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda dataset, seed: (
            X_train.copy(),
            X_test.copy(),
            y_train.copy(),
            y_test.copy(),
        ),
    )
    baseline = runner.run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=9,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=False,
    )

    changed_train = X_train.copy()
    changed_train[y_train == 2] += 10000.0
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda dataset, seed: (changed_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()),
    )
    changed = runner.run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=9,
        n_epochs=1,
        new_feature_dim=1,
        target_every_experience=False,
    )

    # Class 2 has not arrived by experience 1, so its validation features cannot
    # affect calibration at that point.
    baseline_row = baseline["history"][1]
    changed_row = changed["history"][1]
    assert 2 not in baseline_row["threshold_calibration"]["calibration_allowed_classes"]
    assert (
        baseline_row["threshold_calibration"]["threshold"]
        == changed_row["threshold_calibration"]["threshold"]
    )


def test_recall_diagnostics_separate_zero_calibrated_and_frozen_thresholds():
    result = run_experiment(
        dataset="synthetic",
        target_class=0,
        n_experiences=3,
        seed=11,
        new_feature_dim=1,
        n_epochs=1,
        replay_mem_size=20,
    )
    history = result["history"]
    for row in history:
        assert row["target_recall_at_zero_threshold"] == row["target_recall"]
        assert row["target_recall_forgetting_at_zero_threshold"] == row["target_recall_forgetting"]
        if row["best_target_recall"] is not None and np.isfinite(row["target_recall"]):
            assert row["target_recall_forgetting_at_zero_threshold"] == pytest.approx(
                max(0.0, row["best_target_recall"] - row["target_recall_at_zero_threshold"])
            )
        assert (
            row["target_recall_at_validation_selected_threshold"]
            == row["threshold_calibration"]["test"]["target_recall"]
        )
        assert (
            row["target_recall_at_fixed_threshold"]
            == row["fixed_threshold_calibration"]["test"]["target_recall"]
        )

    calibration_exp = result["fixed_threshold_calibration_experience"]
    calibration_row = history[calibration_exp]
    assert calibration_row["fixed_threshold_target_recall_change_since_calibration"] == 0.0
    assert result["final_fixed_threshold_target_recall_change_since_calibration"] == pytest.approx(
        history[-1]["target_recall_at_fixed_threshold"]
        - calibration_row["target_recall_at_fixed_threshold"]
    )
    assert result["final_target_recall_at_zero_threshold"] == history[-1]["target_recall"]
    assert (
        result["final_target_recall_at_validation_selected_threshold"]
        == history[-1]["threshold_calibration"]["test"]["target_recall"]
    )
