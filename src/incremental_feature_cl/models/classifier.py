# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Block-wise linear classifier with *independent* feature and output expansion.

    logits(x) = b + sum_k  W_k  phi_k(x)

``W_k`` has shape ``(num_outputs, d_k)`` for feature block ``k``.

* ``add_block(d, init)``     grows the feature side: a new ``W`` (zeros by default).
* ``expand_outputs(n, init)`` grows the output side: new rows in every ``W_k`` and ``b``.

The two operations touch different axes and never interact.  Expanding outputs
swaps ``.data`` so the ``nn.Parameter`` *objects* keep their identity; the
optimizer must nevertheless be rebuilt afterwards because shapes changed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .expandable import check_initialization


def _fresh_rows(n: int, d: int, init: str, seed: int | None, like: torch.Tensor) -> torch.Tensor:
    if init == "zero":
        return torch.zeros(n, d, device=like.device, dtype=like.dtype)
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        w = torch.empty(n, d)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))  # nn.Linear default
    return w.to(device=like.device, dtype=like.dtype)


class ExpandableLinearClassifier(nn.Module):
    def __init__(self, block_dims: Sequence[int], num_outputs: int):
        super().__init__()
        if num_outputs < 1:
            raise ValueError("num_outputs must be >= 1")
        self.weights = nn.ModuleList([nn.Linear(d, num_outputs, bias=False) for d in block_dims])
        self.bias = nn.Parameter(torch.zeros(num_outputs))

    @property
    def num_outputs(self) -> int:
        return self.bias.numel()

    @property
    def block_dims(self) -> list[int]:
        return [w.in_features for w in self.weights]

    def forward(self, blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(blocks) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} feature blocks, got {len(blocks)}")
        out = self.bias
        for w, b in zip(self.weights, blocks):
            out = out + w(b)
        return out

    def block_contributions(self, blocks: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Per-block logit contribution W_k phi_k (bias excluded)."""
        return [w(b) for w, b in zip(self.weights, blocks)]

    @torch.no_grad()
    def add_block(
        self, dim: int, initialization: str = "zero", seed: int | None = None
    ) -> nn.Linear:
        init = check_initialization(initialization)
        ref = self.bias
        lin = nn.Linear(dim, self.num_outputs, bias=False).to(device=ref.device, dtype=ref.dtype)
        lin.weight.copy_(_fresh_rows(self.num_outputs, dim, init, seed, ref))
        self.weights.append(lin)
        return lin

    @torch.no_grad()
    def expand_outputs(
        self, n_new: int, initialization: str = "zero", seed: int | None = None
    ) -> None:
        if n_new < 0:
            raise ValueError("n_new must be >= 0")
        if n_new == 0:
            return
        init = check_initialization(initialization)
        for k, w in enumerate(self.weights):
            rows = _fresh_rows(
                n_new, w.in_features, init, None if seed is None else seed + k, w.weight
            )
            w.weight.data = torch.cat([w.weight.data, rows], dim=0)
            w.out_features += n_new
        self.bias.data = torch.cat(
            [self.bias.data, torch.zeros(n_new, device=self.bias.device, dtype=self.bias.dtype)]
        )
