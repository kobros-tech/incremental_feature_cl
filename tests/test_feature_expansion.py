# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import pytest
import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone


def test_feature_dim_increases_exactly_as_requested(conv_model):
    d0 = conv_model.feature_dim
    for new in (4, 16, 1):
        before = conv_model.feature_dim
        rec = conv_model.expand_feature_space(new)
        assert conv_model.feature_dim == before + new
        assert rec.old_feature_dim == before and rec.new_feature_dim == new
        assert rec.total_feature_dim == conv_model.feature_dim
    assert conv_model.block_dims == [d0, 4, 16, 1]


def test_zero_expansion_is_noop(conv_model, x_batch):
    y0, n0 = conv_model(x_batch), conv_model.parameter_count()
    rec = conv_model.expand_feature_space(0)
    assert rec.new_feature_dim == 0 and rec.number_of_new_parameters == 0
    assert conv_model.parameter_count() == n0 and torch.equal(conv_model(x_batch), y0)


def test_old_parameters_preserved_and_not_recreated(conv_model):
    old_modules = (conv_model.backbone, conv_model.classifier, conv_model.classifier.weights[0])
    snap = {n: (p, p.detach().clone()) for n, p in conv_model.named_parameters()}
    conv_model.expand_feature_space(8)
    conv_model.expand_feature_space(8)
    assert old_modules == (
        conv_model.backbone,
        conv_model.classifier,
        conv_model.classifier.weights[0],
    )
    params = dict(conv_model.named_parameters())
    for n, (obj, val) in snap.items():
        assert params[n] is obj, f"{n} was recreated"
        assert torch.equal(params[n].detach(), val), f"{n} was modified"


def test_expansion_record_bookkeeping(conv_model):
    n0 = conv_model.parameter_count()
    rec = conv_model.expand_feature_space(16)
    # psi: Linear(64->16) = 64*16+16 ; classifier block: 3 outputs x 16 weights
    assert rec.number_of_new_parameters == 64 * 16 + 16 + 3 * 16
    assert rec.old_parameter_count == n0
    assert (
        rec.total_parameter_count
        == conv_model.parameter_count()
        == n0 + rec.number_of_new_parameters
    )
    assert conv_model.expansion_log[-1] is rec


def test_features_concatenate_old_then_new(conv_model, x_batch):
    old = conv_model.features(x_batch)
    conv_model.expand_feature_space(5)
    new = conv_model.features(x_batch)
    assert new.shape[1] == old.shape[1] + 5
    assert torch.equal(new[:, : old.shape[1]], old)  # Phi_{t+1} = [Phi_t, psi]


def test_freezing_flags():
    m = IncrementalFeatureMapModel(
        build_backbone("smallconv"), freeze_backbone=True, freeze_old_blocks=True
    )
    m.expand_feature_space(4)
    m.expand_feature_space(4)
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert not any(p.requires_grad for p in m.feature_blocks[0].parameters())
    assert all(p.requires_grad for p in m.feature_blocks[1].parameters())


def test_negative_dim_rejected(conv_model):
    with pytest.raises(ValueError):
        conv_model.expand_feature_space(-1)
