# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Central invariant:  f_{t+1}(x) == f_t(x)  immediately after expansion (zero init)."""

import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone


def test_predictions_unchanged_after_expansion(conv_model, x_batch):
    before = conv_model(x_batch)
    conv_model.expand_feature_space(16, initialization="zero")
    after = conv_model(x_batch)
    torch.testing.assert_close(before, after, rtol=0, atol=1e-7)
    assert torch.equal(before.argmax(1), after.argmax(1))


def test_invariant_with_batchnorm_backbone_and_repeated_expansions(x_batch):
    m = IncrementalFeatureMapModel(build_backbone("slimresnet18", nf=4), num_outputs=5).eval()
    ref = m(x_batch)
    for _ in range(3):
        m.expand_feature_space(8)
        torch.testing.assert_close(m(x_batch), ref, rtol=0, atol=1e-6)


def test_new_classifier_weights_start_at_zero_and_bias_untouched(conv_model):
    bias = conv_model.classifier.bias.detach().clone()
    conv_model.expand_feature_space(8)
    assert torch.count_nonzero(conv_model.classifier.weights[-1].weight) == 0
    assert torch.equal(conv_model.classifier.bias.detach(), bias)
    assert conv_model.classifier_block_norms()[-1] == 0.0
    # psi itself is NOT zero: only its classifier contribution is
    assert conv_model.feature_blocks[-1].linear.weight.abs().sum() > 0


def test_random_initialization_changes_predictions(conv_model, x_batch):
    before = conv_model(x_batch)
    conv_model.expand_feature_space(16, initialization="random")
    assert (conv_model(x_batch) - before).abs().max() > 1e-4


def test_expansion_is_seed_deterministic(x_batch):
    def build():
        m = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=2, seed=3)
        m.expand_feature_space(6, "random")
        return m

    a, b = build(), build()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if n.startswith(("feature_blocks", "classifier.weights.1")):
            assert torch.equal(pa, pb), n
