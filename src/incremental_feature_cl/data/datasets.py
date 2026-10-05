# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Small, dependency-light dataset adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        if x.ndim < 2:
            raise ValueError("x must contain a batch dimension")
        y = torch.as_tensor(targets, dtype=torch.long)
        if len(x) != len(y):
            raise ValueError("x and targets must have the same length")
        self.x, self.targets, self.mean, self.std = x, y, mean, std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.x[index]
        x = x.float().div(255.0) if x.dtype == torch.uint8 else x.float()
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x, self.targets[index]


def _subset(dataset, indices):
    return ArrayDataset(
        dataset.x[indices],
        dataset.targets[indices],
        mean=dataset.mean,
        std=dataset.std,
    )


def make_synthetic_dataset(
    n_classes=10,
    n_train_per_class=20,
    n_test_per_class=10,
    seed=0,
    *,
    shape=(3, 8, 8),
):
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if n_train_per_class < 1 or n_test_per_class < 1:
        raise ValueError("samples per class must be >= 1")

    g = torch.Generator().manual_seed(seed)
    dim = int(np.prod(shape))
    base = torch.randn(n_classes, dim, generator=g)
    base = base / base.norm(dim=1, keepdim=True).clamp_min(1e-8)

    xs, ys = [], []
    for c in range(n_classes):
        total = n_train_per_class + n_test_per_class
        x = base[c].unsqueeze(0) + 0.25 * torch.randn(total, dim, generator=g)
        xs.append(x.reshape(total, *shape).float())
        ys.append(torch.full((total,), c, dtype=torch.long))

    x, y = torch.cat(xs), torch.cat(ys)
    train_idx, test_idx = [], []
    for c in range(n_classes):
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        train_idx.append(idx[:n_train_per_class])
        test_idx.append(idx[n_train_per_class:])

    full = ArrayDataset(x, y)
    return _subset(full, torch.cat(train_idx)), _subset(full, torch.cat(test_idx))


def _load_torchvision(name, root):
    try:
        from torchvision import datasets
    except (ImportError, OSError, RuntimeError) as exc:
        raise ImportError(
            "CIFAR datasets require torchvision; install the [vision] extra"
        ) from exc

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    tr = cls(root=str(root), train=True, download=True)
    te = cls(root=str(root), train=False, download=True)

    tx = torch.from_numpy(np.asarray(tr.data)).permute(0, 3, 1, 2).contiguous()
    vx = torch.from_numpy(np.asarray(te.data)).permute(0, 3, 1, 2).contiguous()
    train = ArrayDataset(tx, np.asarray(tr.targets))
    test = ArrayDataset(vx, np.asarray(te.targets))
    return train, test, (10 if name == "cifar10" else 100), (3, 32, 32)


def load_dataset(name, root="./data", seed=0, **synthetic_kwargs):
    key = name.lower()
    if key == "synthetic":
        train, test = make_synthetic_dataset(seed=seed, **synthetic_kwargs)
        return train, test, int(train.targets.max().item()) + 1, tuple(train.x.shape[1:])
    if key in {"cifar10", "cifar100"}:
        return _load_torchvision(key, root)
    raise ValueError(
        "unknown dataset {!r}; choose synthetic, cifar10, or cifar100".format(name)
    )
