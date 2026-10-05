# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Deterministic class-incremental and target-vs-rest streams."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import numpy as np
import torch
from .datasets import ArrayDataset

def binary_labels(labels, target):
    return (torch.as_tensor(labels) == int(target)).long()

@dataclass
class Experience:
    index: int
    dataset: ArrayDataset
    classes: list[int]
    new_classes: list[int]
    @property
    def labels(self): return self.dataset.targets.cpu().numpy()

class Stream:
    def __init__(self, train, test_dataset, num_classes, class_order, *, target_class=None):
        self.train, self.test_dataset = train, test_dataset
        self.num_classes, self.class_order, self.target_class = num_classes, class_order, target_class
        self._first = {}
        for exp in train:
            for c in exp.new_classes: self._first.setdefault(int(c), exp.index)
    def seen_classes(self, exp_index):
        return [c for c in self.class_order if c in self._first and self._first[c] <= exp_index]
    def seen_negatives(self, exp_index):
        return [c for c in self.seen_classes(exp_index) if c != self.target_class]
    def class_first_experience(self): return dict(self._first)

def _validate_split(num_classes, n_experiences):
    if n_experiences < 1: raise ValueError("n_experiences must be >= 1")
    if n_experiences > num_classes: raise ValueError("n_experiences cannot exceed the number of classes")
    if num_classes % n_experiences: raise ValueError(f"number of classes ({num_classes}) must be divisible by n_experiences ({n_experiences})")

def _class_order(num_classes, seed, class_order):
    if class_order is None:
        order = list(range(num_classes)); np.random.default_rng(seed).shuffle(order); return order
    order = [int(c) for c in class_order]
    if sorted(order) != list(range(num_classes)): raise ValueError("class_order must be a permutation of all class ids")
    return order

def _select_classes(dataset, classes):
    wanted = torch.as_tensor(list(classes), dtype=torch.long)
    idx = torch.nonzero(torch.isin(dataset.targets, wanted), as_tuple=False).flatten()
    return ArrayDataset(dataset.x[idx], dataset.targets[idx], mean=dataset.mean, std=dataset.std)

def build_class_incremental_stream(train, test, n_experiences, seed=0, class_order=None):
    num_classes = int(torch.max(torch.cat([train.targets, test.targets])).item()) + 1
    order = _class_order(num_classes, seed, class_order); _validate_split(num_classes, n_experiences)
    per_exp = num_classes // n_experiences
    exps = []
    for i in range(n_experiences):
        classes = order[i*per_exp:(i+1)*per_exp]
        exps.append(Experience(i, _select_classes(train, classes), classes, classes.copy()))
    return Stream(exps, test, num_classes, order)

def build_target_vs_rest_stream(train, test, target_class, n_experiences, seed=0, class_order=None, target_in_every_experience=True):
    num_classes = int(torch.max(torch.cat([train.targets, test.targets])).item()) + 1
    if not 0 <= target_class < num_classes: raise ValueError(f"target_class must be in [0, {num_classes-1}]")
    order = _class_order(num_classes, seed, class_order)
    negatives = [c for c in order if c != target_class]
    if n_experiences > len(negatives): raise ValueError("n_experiences cannot exceed the number of negative classes")
    exps = []
    for i, chunk in enumerate(np.array_split(negatives, n_experiences)):
        negs = [int(c) for c in chunk.tolist()]
        classes = ([target_class] if target_in_every_experience or i == 0 else []) + negs
        new_classes = ([target_class] if i == 0 else []) + negs
        parts = []
        if target_in_every_experience or i == 0: parts.append(_select_classes(train, [target_class]))
        if negs: parts.append(_select_classes(train, negs))
        x, y = torch.cat([p.x for p in parts]), torch.cat([p.targets for p in parts])
        exps.append(Experience(i, ArrayDataset(x, y, mean=train.mean, std=train.std), classes, new_classes))
    return Stream(exps, test, num_classes, [target_class] + [c for c in order if c != target_class], target_class=target_class)
