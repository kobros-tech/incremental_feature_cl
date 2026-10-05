# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""A deliberately minimal class-balanced replay buffer for *controlled ablations*.

It exists so replay can be toggled on/off in the same trainer as the expansion
experiments.  Comparisons against real CL methods (Replay, ER-ACE, iCaRL, EWC)
use Avalanche's implementations (see ``avalanche/baselines.py``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.store: dict[int, list[torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self.store.values())

    def update(self, dataset: Dataset) -> None:
        """Add this experience's data (per-class replace), then enforce capacity // n_classes."""
        fresh: dict[int, list[torch.Tensor]] = {}
        for i in range(len(dataset)):
            x, y = dataset[i][0], int(dataset[i][1])
            fresh.setdefault(y, []).append(x)
        self.store.update(fresh)
        per_class = max(1, self.capacity // max(1, len(self.store)))
        for c, xs in self.store.items():
            if len(xs) > per_class:
                keep = self.rng.choice(len(xs), per_class, replace=False)
                self.store[c] = [xs[k] for k in keep]

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not len(self):
            return None
        flat = [(x, c) for c, xs in self.store.items() for x in xs]
        idx = self.rng.choice(len(flat), min(n, len(flat)), replace=False)
        return (
            torch.stack([flat[i][0] for i in idx]),
            torch.tensor([flat[i][1] for i in idx], dtype=torch.long),
        )
