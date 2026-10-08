# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone
from incremental_feature_cl.data import (
    binary_labels,
    build_target_vs_rest_stream,
    make_synthetic_dataset,
    negatives_for_ratio,
    ratio_report,
)
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


# --------------------------------------------------------------------------- #
# target_to_negatives: explicit target:negative sampling ratio
# --------------------------------------------------------------------------- #
TARGET = 3


@pytest.fixture(scope="module")
def big():
    """21 classes x 10 train samples: 10 targets, 5 negative classes (= 50 samples) per experience."""
    return make_synthetic_dataset(n_classes=21, n_train_per_class=10, n_test_per_class=4, seed=0)


def _counts(stream):
    """[(n_target, n_negative)] per experience."""
    return [
        (int((e.labels == TARGET).sum()), int((e.labels != TARGET).sum())) for e in stream.train
    ]


def _big_stream(big, ratio, n_exp=4, seed=0, **kw):
    train, test = big
    return build_target_vs_rest_stream(
        train, test, TARGET, n_exp, seed, target_to_negatives=ratio, **kw
    )


def test_ratio_none_preserves_cumulative_pool(synth):
    train, test = synth
    omitted = build_target_vs_rest_stream(train, test, TARGET, 3, seed=0)
    explicit = build_target_vs_rest_stream(train, test, TARGET, 3, seed=0, target_to_negatives=None)
    # 20 targets + 20 samples for each of the 3 / 6 / 9 negative classes seen so far
    assert _counts(omitted) == [(20, 60), (20, 120), (20, 180)]
    for a, b in zip(omitted.train, explicit.train):
        assert torch.equal(a.dataset.x, b.dataset.x)
        assert torch.equal(a.dataset.targets, b.dataset.targets)
    # exact content/order: target samples first, then every seen negative class in class order
    for e in omitted.train:
        negs = sorted(set(e.labels.tolist()) - {TARGET})
        want = torch.cat(
            [train.x[train.targets == TARGET]] + [train.x[train.targets == c] for c in negs]
        )
        assert torch.equal(e.dataset.x, want)


def test_ratio_one_gives_equal_counts(synth):
    train, test = synth
    st = build_target_vs_rest_stream(train, test, TARGET, 3, seed=0, target_to_negatives=1.0)
    assert _counts(st) == [(20, 20)] * 3  # every pool (60/120/180) has >= 20 negatives


def test_ratio_point_two_is_one_to_five(big):
    assert _counts(_big_stream(big, 0.2)) == [(10, 50)] * 4  # pools: 50 / 100 / 150 / 200


def test_ratio_point_one_is_one_to_ten(big):
    # pools are 50 / 100 / 150 / 200: early pools are smaller than requested -> all kept
    assert _counts(_big_stream(big, 0.1)) == [(10, 50), (10, 100), (10, 100), (10, 100)]


def test_ratio_two_is_two_to_one(synth):
    train, test = synth
    st = build_target_vs_rest_stream(train, test, TARGET, 3, seed=0, target_to_negatives=2.0)
    assert _counts(st) == [(20, 10)] * 3  # fewer negatives than targets; targets untouched


def test_small_negative_pool_is_kept_whole(big):
    st = _big_stream(big, 0.01)  # asks for 1000 negatives; at most 200 exist
    ref = _big_stream(big, None)
    assert _counts(st) == _counts(ref) == [(10, 50), (10, 100), (10, 150), (10, 200)]
    for a, b in zip(st.train, ref.train):
        assert torch.equal(a.dataset.x, b.dataset.x)


@pytest.mark.parametrize("ratio", [None, 0.1, 0.2, 1.0, 2.0, 10.0])
def test_target_samples_are_never_subsampled_or_duplicated(big, ratio):
    train, _ = big
    want = train.x[train.targets == TARGET]
    for e in _big_stream(big, ratio).train:
        got = e.dataset.x[e.dataset.targets == TARGET]
        assert torch.equal(got, want)


def test_ratio_subsample_is_drawn_without_replacement_from_cumulative_pool(big):
    train, _ = big
    st = _big_stream(big, 0.2)
    for e in st.train:
        neg = e.dataset.x[e.dataset.targets != TARGET]
        assert len(torch.unique(neg.flatten(1), dim=0)) == len(neg)  # no duplicates
        assert set(e.labels.tolist()) - {TARGET} <= set(st.seen_negatives(e.index))
        pool = train.x[torch.isin(train.targets, torch.tensor(st.seen_negatives(e.index)))]
        assert len(neg) <= len(pool)
    # experiences >= 1 draw from classes introduced earlier too (cumulative, not just new classes)
    assert set(st.train[3].labels.tolist()) - {TARGET} > set(st.train[3].new_classes)


def test_ratio_sampling_is_deterministic_for_same_seed(big):
    a, b = _big_stream(big, 0.2, seed=5), _big_stream(big, 0.2, seed=5)
    for x, y in zip(a.train, b.train):
        assert torch.equal(x.dataset.x, y.dataset.x)
        assert torch.equal(x.dataset.targets, y.dataset.targets)


def test_ratio_sampling_changes_with_seed(big):
    order = list(range(21))  # fix the class order so only the sampling seed differs
    a = _big_stream(big, 0.2, seed=0, class_order=order)
    b = _big_stream(big, 0.2, seed=1, class_order=order)
    # experience 3 draws 50 of 200 negatives: different seeds give different draws
    assert a.train[3].classes == b.train[3].classes
    assert not torch.equal(a.train[3].dataset.x, b.train[3].dataset.x)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, float("nan"), float("inf")])
def test_non_positive_or_non_finite_ratio_raises(synth, bad):
    train, test = synth
    with pytest.raises(ValueError):
        build_target_vs_rest_stream(train, test, TARGET, 3, target_to_negatives=bad)


def test_ratio_requesting_zero_negatives_raises(synth):
    train, test = synth
    with pytest.raises(ValueError, match="negatives"):
        build_target_vs_rest_stream(train, test, TARGET, 3, target_to_negatives=1000.0)


def test_negatives_for_ratio_is_floor_with_float_guard():
    assert negatives_for_ratio(500, 0.2) == 2500
    assert negatives_for_ratio(20, 0.1) == 200
    assert negatives_for_ratio(29, 0.29) == 100  # 29 / 0.29 = 99.99999999999999 in floats
    assert negatives_for_ratio(7, 2.0) == 3  # floor, never rounds up


def test_ratio_with_target_only_in_first_experience(big):
    st = _big_stream(big, 0.2, target_in_every_experience=False)
    assert st.train[0].classes[0] == TARGET and _counts(st)[0] == (10, 50)
    for e in st.train[1:]:
        assert TARGET not in set(e.labels.tolist())  # no target -> no ratio to enforce
    # negatives are still the whole cumulative pool there
    assert [c[1] for c in _counts(st)[1:]] == [100, 150, 200]
    ref = _big_stream(big, None, target_in_every_experience=False)
    assert [c[1] for c in _counts(ref)] == [50, 100, 150, 200]


# --------------------------------------------------------------------------- #
# requested vs realized ratio
# --------------------------------------------------------------------------- #
def test_ratio_report_requested_vs_realized():
    full = ratio_report(10, 50, 0.2)  # 1:5 requested, 50 negatives available -> satisfied
    assert full["n_requested_negative"] == 50 and full["ratio_satisfied"] is True
    assert full["realized_target_to_negatives"] == pytest.approx(0.2)
    assert full["realized_ratio_label"] == "1:5" and full["requested_target_to_negatives"] == 0.2

    short = ratio_report(12, 36, 0.2)  # asked for 60 negatives, only 36 exist -> realized 1:3
    assert short["n_requested_negative"] == 60 and short["ratio_satisfied"] is False
    assert short["realized_ratio_label"] == "1:3"
    assert short["realized_target_to_negatives"] == pytest.approx(1 / 3)

    over = ratio_report(10, 4, 2.0)  # 2:1 asks for 5 negatives, 4 available
    assert over["n_requested_negative"] == 5 and over["ratio_satisfied"] is False

    cumulative = ratio_report(10, 50, None)  # nothing to enforce
    assert cumulative["ratio_satisfied"] is None and cumulative["n_requested_negative"] is None
    assert cumulative["realized_ratio_label"] == "1:5"

    no_target = ratio_report(0, 100, 0.2)  # target absent: no ratio to enforce
    assert no_target["ratio_satisfied"] is None and no_target["realized_target_to_negatives"] == 0
    assert no_target["realized_ratio_label"] is None
    assert ratio_report(5, 0, 1.0)["realized_target_to_negatives"] is None  # no negatives at all


def test_trainer_records_requested_and_realized_ratio(big):
    """1:10 on the 4-experience stream: pools are 50/100/150/200, 100 negatives requested."""
    st = _big_stream(big, 0.1)
    assert st.target_to_negatives == 0.1
    m = IncrementalFeatureMapModel(build_backbone("mlp", in_dim=3 * 8 * 8), num_outputs=1)
    tr = ContinualTrainer(
        m,
        ExpansionPolicy(2, "zero"),
        TrainerConfig(mode="target", train_epochs=1, lr=0.05, train_mb_size=16, probe_size=0),
    )
    recs = tr.fit(st)
    assert [r["n_train_negative"] for r in recs] == [50, 100, 100, 100]
    assert [r["n_requested_negative"] for r in recs] == [100] * 4
    assert [r["ratio_satisfied"] for r in recs] == [False, True, True, True]
    assert [r["realized_ratio_label"] for r in recs] == ["1:5", "1:10", "1:10", "1:10"]
    assert all(r["requested_target_to_negatives"] == 0.1 for r in recs)


def test_trainer_cumulative_reports_nothing_to_satisfy(big):
    st = _big_stream(big, None)
    m = IncrementalFeatureMapModel(build_backbone("mlp", in_dim=3 * 8 * 8), num_outputs=1)
    tr = ContinualTrainer(
        m, ExpansionPolicy(0, "zero"), TrainerConfig(mode="target", train_mb_size=16, probe_size=0)
    )
    recs = tr.fit(st)
    assert all(r["ratio_satisfied"] is None and r["n_requested_negative"] is None for r in recs)
    assert [r["realized_ratio_label"] for r in recs] == ["1:5", "1:10", "1:15", "1:20"]


def test_trainer_handles_experiences_without_target_samples(big):
    st = _big_stream(big, 0.2, target_in_every_experience=False)
    m = IncrementalFeatureMapModel(build_backbone("mlp", in_dim=3 * 8 * 8), num_outputs=1)
    tr = ContinualTrainer(
        m,
        ExpansionPolicy(0, "zero"),
        TrainerConfig(mode="target", train_mb_size=16, probe_size=0, pos_weight=None),
    )
    recs = tr.fit(st)
    assert recs[0]["ratio_satisfied"] is True and recs[0]["realized_ratio_label"] == "1:5"
    for r in recs[1:]:  # no target samples: nothing to enforce, no label, no crash
        assert r["n_train_target"] == 0 and r["ratio_satisfied"] is None
        assert r["realized_ratio_label"] is None and r["n_requested_negative"] is None
