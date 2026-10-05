# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Feature expansion and output expansion are independent operations."""

import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone


def test_output_dim_independent_of_feature_dim(conv_model, x_batch):
    assert conv_model.num_outputs == 3
    conv_model.expand_feature_space(10)
    assert conv_model.num_outputs == 3 and conv_model(x_batch).shape == (7, 3)
    conv_model.expand_outputs(4)
    assert conv_model.feature_dim == 64 + 10 and conv_model(x_batch).shape == (7, 7)


def test_output_expansion_preserves_old_logits_and_zero_rows(conv_model, x_batch):
    before = conv_model(x_batch)
    rec = conv_model.expand_outputs(2)
    after = conv_model(x_batch)
    torch.testing.assert_close(after[:, :3], before, rtol=0, atol=1e-7)
    assert torch.count_nonzero(after[:, 3:]) == 0  # zero rows + zero bias => logit 0
    assert rec.kind == "output" and rec.new_feature_dim == 0
    assert rec.old_num_outputs == 3 and rec.new_num_outputs == 5


def test_parameter_identity_kept_under_output_expansion(conv_model):
    objs = {n: p for n, p in conv_model.named_parameters()}
    conv_model.expand_outputs(3)
    for n, p in conv_model.named_parameters():
        assert objs[n] is p


def test_both_orders_commute(x_batch):
    def build(order):
        torch.manual_seed(0)
        m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=2, seed=5).eval()
        for op in order:
            m.expand_feature_space(6) if op == "f" else m.expand_outputs(3)
        return m

    a, b = build("fo"), build("of")
    assert a.feature_dim == b.feature_dim and a.num_outputs == b.num_outputs == 5
    torch.testing.assert_close(a(x_batch), b(x_batch))


def test_expand_outputs_after_feature_expansion_keeps_block_shapes(conv_model):
    conv_model.expand_feature_space(8)
    conv_model.expand_outputs(2)
    for w, d in zip(conv_model.classifier.weights, conv_model.block_dims):
        assert tuple(w.weight.shape) == (5, d)
    assert conv_model.classifier.bias.shape == (5,)


def test_ensure_outputs(conv_model):
    assert conv_model.ensure_outputs(2) is None
    conv_model.ensure_outputs(9)
    assert conv_model.num_outputs == 9
