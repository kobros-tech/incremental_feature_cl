# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pickle

import numpy as np
import pytest

from experiments.data import class_incremental_indices, load_dataset
from experiments.math_reference import polynomial_features
from experiments.metrics import target_metrics
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


def test_polynomial_features_match_degree_two_kernel():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(8, 3))
    Z = rng.normal(size=(5, 3))
    actual = polynomial_features(X) @ polynomial_features(Z).T
    expected = (1.0 + X @ Z.T) ** 2
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nested", [False, True])
def test_load_cifar100_from_extracted_directory(tmp_path, nested):
    data_dir = tmp_path / "cifar-100-python" if nested else tmp_path
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

    X_train, X_test, y_train, y_test = load_dataset("cifar100", seed=7, data_dir=tmp_path)

    assert X_train.shape == (4, 3072)
    assert X_test.shape == (2, 3072)
    np.testing.assert_array_equal(np.sort(y_train), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.sort(y_test), [0, 1])
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0


def test_load_cifar100_rejects_directory_without_batch_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="train/test files"):
        load_dataset("cifar100", data_dir=tmp_path)
