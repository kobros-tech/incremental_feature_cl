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
        raise ImportError("CIFAR datasets require torchvision; install the [vision] extra") from exc

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


def _digits_arrays():
    """sklearn's bundled 8x8 digits (1797 images, no download), fixed stratified 70/30 split."""
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'digits' datasets need scikit-learn (pip install scikit-learn)"
        ) from exc

    bunch = load_digits()
    images = torch.from_numpy(bunch.images).float() / 16.0  # [N, 8, 8] in [0, 1]
    labels = torch.from_numpy(bunch.target).long()
    rng = np.random.default_rng(0)  # the split is part of the dataset, not of the experiment seed
    train_idx, test_idx = [], []
    for c in range(10):
        idx = np.flatnonzero(labels.numpy() == c)
        rng.shuffle(idx)
        cut = round(0.7 * len(idx))
        train_idx.append(idx[:cut])
        test_idx.append(idx[cut:])
    return images, labels, np.concatenate(train_idx), np.concatenate(test_idx)


def _render(images: torch.Tensor, upscale: int, channels: int) -> torch.Tensor:
    """[N, H, W] -> [N, channels, H*upscale, W*upscale] (nearest-neighbour upscale)."""
    x = images.unsqueeze(1)
    if upscale > 1:
        x = x.repeat_interleave(upscale, dim=2).repeat_interleave(upscale, dim=3)
    return x.repeat(1, channels, 1, 1).contiguous()


def load_digits_dataset(
    *,
    pairs: bool = False,
    upscale: int = 1,
    channels: int = 1,
    noise: float = 0.0,
    n_train_per_class: int = 30,
    n_test_per_class: int = 12,
):
    """Small offline stand-ins for CIFAR-style experiments.

    ``pairs=False``: the 10 digit classes (8x8 images).
    ``pairs=True``: 100 classes ``10*a+b``, each image is digit ``a`` next to digit ``b`` (8x16).
    The 1% positive prevalence of a target-vs-rest problem then resembles CIFAR-100, while
    train and test pairs are built from disjoint source images.
    ``upscale`` / ``channels`` enlarge the input (e.g. ``upscale=4, channels=3`` gives the
    3072-dim input scale of CIFAR when ``pairs=False``), which matters for optimisation stability.
    ``noise`` adds i.i.d. Gaussian pixel noise (fixed seed) to make the problem harder, like raw
    CIFAR pixels, so the target-vs-rest decision threshold is genuinely sensitive.
    """
    images, labels, train_idx, test_idx = _digits_arrays()
    noise_gen = torch.Generator().manual_seed(1234)

    def render(imgs: torch.Tensor) -> torch.Tensor:
        x = _render(imgs, upscale, channels)
        return x + noise * torch.randn(x.shape, generator=noise_gen) if noise > 0 else x

    if not pairs:
        tr = ArrayDataset(render(images[train_idx]), labels[train_idx])
        te = ArrayDataset(render(images[test_idx]), labels[test_idx])
        return tr, te, 10, tuple(tr.x.shape[1:])

    rng = np.random.default_rng(0)
    by_digit = {
        "train": {d: train_idx[labels[train_idx].numpy() == d] for d in range(10)},
        "test": {d: test_idx[labels[test_idx].numpy() == d] for d in range(10)},
    }

    def build(split: str, per_class: int):
        xs, ys = [], []
        for a in range(10):
            for b in range(10):
                ia = rng.choice(by_digit[split][a], per_class)
                ib = rng.choice(by_digit[split][b], per_class)
                xs.append(torch.cat([images[ia], images[ib]], dim=2))  # [n, 8, 16]
                ys.append(torch.full((per_class,), 10 * a + b, dtype=torch.long))
        return ArrayDataset(render(torch.cat(xs)), torch.cat(ys))

    tr, te = build("train", n_train_per_class), build("test", n_test_per_class)
    return tr, te, 100, tuple(tr.x.shape[1:])


def load_dataset(name, root="./data", seed=0, **synthetic_kwargs):
    key = name.lower()
    if key == "synthetic":
        train, test = make_synthetic_dataset(seed=seed, **synthetic_kwargs)
        return train, test, int(train.targets.max().item()) + 1, tuple(train.x.shape[1:])
    if key in {"cifar10", "cifar100"}:
        return _load_torchvision(key, root)
    if key in {"digits", "digitpairs"}:
        return load_digits_dataset(pairs=key == "digitpairs", **synthetic_kwargs)
    raise ValueError(
        f"unknown dataset {name!r}; choose synthetic, digits, digitpairs, cifar10, or cifar100"
    )
