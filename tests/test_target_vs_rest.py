# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone
from incremental_feature_cl.data import binary_labels, build_target_vs_rest_stream
from incremental_feature_cl.evaluation import binary_target_metrics
from incremental_feature_cl.training import ContinualTrainer, ExpansionPolicy, TrainerConfig


def test_binary_labels():
    y = torch.tensor([3, 7, 3, 0])
    assert binary_labels(y, 3).tolist() == [1, 0, 1, 0]
    assert binary_labels(np.array([3, 7, 3, 0]), 7).tolist() == [0, 1, 0, 0]


def test_stream_composition(synth):
    train, test = synth
    st = build_target_vs_rest_stream(train, test, target_class=3, n_experiences=3, seed=0)
    all_neg: list[int] = []
    for e in st.train:
        assert 3 in e.classes and 3 in set(e.labels.tolist())
        assert (e.labels == 3).sum() == 20  # all target samples every experience
        negs = sorted(set(e.labels.tolist()) - {3})
        assert set(all_neg).issubset(negs)  # negatives accumulate across experiences
        new_negs = set(negs) - set(all_neg)
        assert new_negs  # each experience introduces new negative classes
        all_neg = negs
    assert all_neg == [c for c in range(10) if c != 3]
    assert st.target_class == 3
    assert set(st.seen_negatives(0)) < set(st.seen_negatives(1))
    assert set(st.seen_negatives(1)) < set(st.seen_negatives(2))
    assert 3 not in st.seen_negatives(2)


def test_target_only_in_first_experience(synth):
    train, test = synth
    st = build_target_vs_rest_stream(train, test, 3, 3, seed=0, target_in_every_experience=False)
    assert 3 in set(st.train[0].labels.tolist())
    assert all(3 not in set(e.labels.tolist()) for e in st.train[1:])


def test_invalid_target_and_split(synth):
    train, test = synth
    with pytest.raises(ValueError):
        build_target_vs_rest_stream(train, test, 99, 3)
    with pytest.raises(ValueError):
        build_target_vs_rest_stream(train, test, 3, 10)  # only 9 negatives


def test_target_metrics_hand_computed():
    labels = np.array([1, 1, 1, 1, 0, 0, 2, 2, 5])  # target=1; negatives seen: 0,2 (class 5 unseen)
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1])
    m = binary_target_metrics(pred, labels, 1, [0, 2])
    assert (m["target_recall"], m["target_accuracy"]) == (0.75, 0.75)
    assert m["target_precision"] == 0.75  # TP=3, FP=1 (class 5 ignored)
    assert m["negative_accuracy"] == 0.75 and m["false_positive_rate"] == 0.25
    assert m["false_negative_rate"] == 0.25 and m["overall_binary_accuracy"] == 6 / 8
    assert m["target_f1"] == pytest.approx(0.75)


def test_trainer_in_target_mode(synth):
    train, test = synth
    st = build_target_vs_rest_stream(train, test, 3, 3, seed=0)
    m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=1)
    tr = ContinualTrainer(
        m,
        ExpansionPolicy(4, "zero"),
        TrainerConfig(mode="target", train_epochs=2, lr=0.05, train_mb_size=16),
    )
    recs = tr.fit(st)
    assert m.num_outputs == 1 and m.feature_dim == 64 + 2 * 4
    for r in recs:
        for k in (
            "target_accuracy",
            "target_precision",
            "target_recall",
            "target_f1",
            "negative_accuracy",
            "false_positive_rate",
            "false_negative_rate",
            "overall_binary_accuracy",
        ):
            assert k in r["target_metrics"]
        assert len(r["per_class_accuracy"]) == 10
        assert r["per_class_accuracy"][3] == pytest.approx(r["target_metrics"]["target_recall"])
