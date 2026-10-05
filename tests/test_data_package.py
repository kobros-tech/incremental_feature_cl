# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest

from incremental_feature_cl.data import (
    build_class_incremental_stream,
    build_target_vs_rest_stream,
    make_synthetic_dataset,
)


def test_synthetic_dataset_and_class_stream():
    train, test = make_synthetic_dataset(
        n_classes=6, n_train_per_class=4, n_test_per_class=2, seed=3
    )
    stream = build_class_incremental_stream(
        train, test, 3, seed=0, class_order=list(range(6))
    )
    assert [e.classes for e in stream.train] == [[0, 1], [2, 3], [4, 5]]
    assert stream.seen_classes(1) == [0, 1, 2, 3]
    assert stream.class_first_experience() == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
    }


def test_target_stream_repeats_target_and_partitions_negatives():
    train, test = make_synthetic_dataset(
        n_classes=10, n_train_per_class=20, n_test_per_class=10, seed=0
    )
    stream = build_target_vs_rest_stream(train, test, 3, 3, seed=0)
    all_neg = []
    for exp in stream.train:
        assert 3 in exp.classes
        assert int((exp.labels == 3).sum()) == 20
        neg = sorted(set(exp.labels.tolist()) - {3})
        assert not set(neg) & set(all_neg)
        all_neg.extend(neg)
    assert sorted(all_neg) == [c for c in range(10) if c != 3]


def test_invalid_class_split():
    train, test = make_synthetic_dataset(n_classes=6)
    with pytest.raises(ValueError):
        build_class_incremental_stream(train, test, 4)
    with pytest.raises(ValueError):
        build_target_vs_rest_stream(train, test, 3, 10)
