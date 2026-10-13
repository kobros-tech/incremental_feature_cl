# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche benchmark builders (imports Avalanche lazily)."""

from __future__ import annotations

from collections.abc import Sequence


def make_split_cifar100(
    n_experiences: int = 20,
    seed: int = 1,
    class_order: Sequence[int] | None = None,
    dataset_root: str | None = None,
    shuffle: bool = True,
):
    """Avalanche ``SplitCIFAR100`` (the primary benchmark)."""
    from avalanche.benchmarks.classic import SplitCIFAR100

    return SplitCIFAR100(
        n_experiences=n_experiences,
        seed=seed,
        shuffle=shuffle,
        fixed_class_order=list(class_order) if class_order is not None else None,
        dataset_root=dataset_root,
    )


def make_tensor_benchmark(
    train, test, n_experiences: int, seed: int = 0, class_order: Sequence[int] | None = None
):
    """Class-incremental Avalanche benchmark from our ``ArrayDataset`` objects (tests / synthetic runs)."""
    import torch
    from avalanche.benchmarks import nc_benchmark
    from avalanche.benchmarks.utils.classification_dataset import (
        _make_taskaware_tensor_classification_dataset as make_ds,
    )

    def to_ds(d):
        x = d.x.float() / 255.0 if d.x.dtype == torch.uint8 else d.x
        if d.mean is not None:
            x = (x - d.mean) / d.std
        return make_ds(x, d.targets)

    return nc_benchmark(
        to_ds(train),
        to_ds(test),
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        fixed_class_order=list(class_order) if class_order is not None else None,
    )
