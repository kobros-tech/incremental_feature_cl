# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche integration of the feature-expansion model (restored; backbones are test-only)."""

import pytest
import torch

pytest.importorskip("avalanche")

from _backbones import build_backbone

from incremental_feature_cl import IncrementalFeatureMapModel
from incremental_feature_cl.avalanche import (
    FeatureExpansionPlugin,
    IncrementalFeatureSpaceStrategy,
    make_avalanche_baseline,
    make_tensor_benchmark,
)
from incremental_feature_cl.data import make_synthetic_dataset
from incremental_feature_cl.training import ExpansionPolicy


@pytest.fixture(scope="module")
def bm():
    tr, te = make_synthetic_dataset(n_classes=6, n_train_per_class=16, n_test_per_class=8)
    return make_tensor_benchmark(tr, te, 3, seed=0, class_order=list(range(6)))


def test_strategy_train_eval_and_growth(bm):
    m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=1)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    s = IncrementalFeatureSpaceStrategy(
        model=m,
        optimizer=opt,
        policy=ExpansionPolicy(4, "zero"),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=32,
    )
    for i, exp in enumerate(bm.train_stream):
        s.train(exp)
        res = s.eval(bm.test_stream[: i + 1])
        assert m.feature_dim == 64 + 4 * i and m.num_outputs == 2 * (i + 1)
        assert any(k.startswith("Top1_Acc_Stream") for k in res)  # one multiclass head
    # the (rebuilt) optimizer must contain every trainable parameter, including newest block
    in_opt = {id(p) for g in s.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in in_opt for p in m.parameters() if p.requires_grad)
    assert len(s.expansion_plugin.records) == 3


def test_plugin_works_inside_a_different_avalanche_strategy(bm):
    from avalanche.training.supervised import Naive

    m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=1)
    plugin = FeatureExpansionPlugin(ExpansionPolicy(8, "zero"))
    s = Naive(
        model=m,
        optimizer=torch.optim.SGD(m.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=32,
        plugins=[plugin],
    )
    for exp in bm.train_stream:
        s.train(exp)
    assert m.feature_dim == 64 + 2 * 8
    assert m.classifier_block_norms()[-1] >= 0.0


@pytest.mark.parametrize("name", ["naive", "replay"])
def test_avalanche_baselines_run(bm, name):
    m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=6)
    s = make_avalanche_baseline(name, m, train_epochs=1, mem_size=20)
    s.train(bm.train_stream[0])
