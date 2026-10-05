# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""psi_t: a learnable feature block appended to the representation."""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "identity": nn.Identity}


class FeatureBlock(nn.Module):
    """psi(h) = act(Linear(h)) with h the backbone feature vector.

    Its own parameters are randomly initialised (so psi is a non-trivial
    function); only its *classifier contribution* starts at zero.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = _ACTIVATIONS[activation]()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(h))
