# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Zero init != frozen: the new block must receive gradients and become non-zero."""

import torch
import torch.nn.functional as F

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone


def _xor_like():
    # x in {-1,0,1}, label 1 iff |x| = 1: not linearly separable in x (lecture 6 example)
    x = torch.tensor([[-1.0], [0.0], [1.0]]).repeat(8, 1)
    y = (x.abs().squeeze(1) > 0.5).long()
    return x, y


def _fit(model, x, y, steps=400, lr=0.05):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def _acc(model, x, y):
    return float((model(x).argmax(1) == y).float().mean())


def test_new_weights_zero_then_become_nonzero_when_needed():
    torch.manual_seed(0)
    x, y = _xor_like()
    m = IncrementalFeatureMapModel(build_backbone("identity", in_dim=1), num_outputs=2)
    _fit(m, x, y)
    assert _acc(m, x, y) < 0.9  # linear in x cannot solve it

    m.expand_feature_space(8, "zero")
    assert m.classifier_block_norms()[-1] == 0.0  # initially zero
    assert all(p.requires_grad for p in m.new_block_parameters())
    _fit(m, x, y)
    assert m.classifier_block_norms()[-1] > 0.0  # became non-zero
    assert _acc(m, x, y) == 1.0  # and was actually useful


def test_gradient_flow_at_zero_init():
    torch.manual_seed(0)
    x, y = _xor_like()
    m = IncrementalFeatureMapModel(build_backbone("identity", in_dim=1), num_outputs=2)
    m.expand_feature_space(8, "zero")
    F.cross_entropy(m(x), y).backward()
    w_new = m.classifier.weights[-1].weight
    assert w_new.grad is not None and w_new.grad.abs().sum() > 0  # W_new moves on step 1
    # psi's own parameters have exactly zero gradient until W_new != 0 (documented saddle)
    assert m.feature_blocks[-1].linear.weight.grad.abs().sum() == 0
    with torch.no_grad():
        w_new.add_(0.1 * torch.randn_like(w_new))
    m.zero_grad()
    F.cross_entropy(m(x), y).backward()
    assert m.feature_blocks[-1].linear.weight.grad.abs().sum() > 0  # ...then psi learns too
