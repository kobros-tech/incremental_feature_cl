# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Deterministic class-incremental and target-vs-rest streams."""

from __future__ import annotations

import math
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
    def labels(self):
        return self.dataset.targets.cpu().numpy()


class Stream:
    def __init__(
        self,
        train,
        test_dataset,
        num_classes,
        class_order,
        *,
        target_class=None,
        target_to_negatives=None,
    ):
        self.train, self.test_dataset = train, test_dataset
        self.num_classes = num_classes
        self.class_order = class_order
        self.target_class = target_class
        # the *requested* ratio (targets per negative); the realized one is per experience,
        # see ``ratio_report``
        self.target_to_negatives = target_to_negatives
        self._first = {}
        for exp in train:
            for c in exp.new_classes:
                self._first.setdefault(int(c), exp.index)

    def seen_classes(self, exp_index):
        return [c for c in self.class_order if c in self._first and self._first[c] <= exp_index]

    def seen_negatives(self, exp_index):
        return [c for c in self.seen_classes(exp_index) if c != self.target_class]

    def class_first_experience(self):
        return dict(self._first)


def _validate_split(num_classes, n_experiences):
    if n_experiences < 1:
        raise ValueError("n_experiences must be >= 1")
    if n_experiences > num_classes:
        raise ValueError("n_experiences cannot exceed the number of classes")
    if num_classes % n_experiences:
        raise ValueError(
            f"number of classes ({num_classes}) must be divisible by "
            f"n_experiences ({n_experiences})"
        )


def _class_order(num_classes, seed, class_order):
    if class_order is None:
        order = list(range(num_classes))
        np.random.default_rng(seed).shuffle(order)
        return order

    order = [int(c) for c in class_order]
    if sorted(order) != list(range(num_classes)):
        raise ValueError("class_order must be a permutation of all class ids")
    return order


def _select_classes(dataset, classes):
    wanted = torch.as_tensor(list(classes), dtype=torch.long)
    idx = torch.nonzero(torch.isin(dataset.targets, wanted), as_tuple=False).flatten()
    return ArrayDataset(
        dataset.x[idx],
        dataset.targets[idx],
        mean=dataset.mean,
        std=dataset.std,
    )


def build_class_incremental_stream(train, test, n_experiences, seed=0, class_order=None):
    num_classes = int(torch.max(torch.cat([train.targets, test.targets])).item()) + 1
    order = _class_order(num_classes, seed, class_order)
    _validate_split(num_classes, n_experiences)
    per_exp = num_classes // n_experiences

    exps = []
    for i in range(n_experiences):
        classes = order[i * per_exp : (i + 1) * per_exp]
        exps.append(
            Experience(
                i,
                _select_classes(train, classes),
                classes,
                classes.copy(),
            )
        )
    return Stream(exps, test, num_classes, order)


def validate_target_to_negatives(target_to_negatives: float | None) -> float | None:
    """Return the ratio as a float (or ``None``); raise ``ValueError`` unless it is ``None`` or > 0."""
    if target_to_negatives is None:
        return None
    ratio = float(target_to_negatives)
    if not (math.isfinite(ratio) and ratio > 0):
        raise ValueError(
            f"target_to_negatives must be None or a finite number > 0, got {target_to_negatives!r}"
        )
    return ratio


def parse_ratio(text: str) -> float | None:
    """Parse a target:negative ratio such as ``5:1``, ``1:5``, ``0.2`` or ``cumulative``.

    Returns *targets per negative* (``5:1`` -> 5.0, ``1:5`` -> 0.2) or ``None`` for ``cumulative``/``none``.
    """
    t = str(text).strip().lower()
    if t in {"cumulative", "none", "null"}:
        return None
    try:
        if ":" in t:
            a, b = t.split(":")
            value = float(a) / float(b)
        else:
            value = float(t)
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f"cannot parse ratio {text!r}: use 'target:negative' (e.g. 1:5), a number, or 'cumulative'"
        ) from None
    return validate_target_to_negatives(value)


def format_ratio(target_to_negatives: float | None) -> str:
    """Human label: ``5.0`` -> ``5:1``, ``0.2`` -> ``1:5``, ``None`` -> ``cumulative``."""
    if target_to_negatives is None:
        return "cumulative"
    r = float(target_to_negatives)
    return f"{r:.3g}:1" if r >= 1 else f"1:{1 / r:.3g}"


def negatives_for_ratio(n_target: int, target_to_negatives: float) -> int:
    """Number of negatives for ``n_target`` targets: ``floor(n_target / ratio)``.

    ``ratio`` is *targets per negative*, so ``0.2`` means 1 target : 5 negatives. The tiny
    epsilon only guards against float error (e.g. ``29 / 0.29 = 99.99999...``).
    """
    return math.floor(n_target / target_to_negatives + 1e-9)


def ratio_report(
    n_target: int, n_negative: int, target_to_negatives: float | None
) -> dict[str, float | int | str | bool | None]:
    """Requested vs realized target:negative ratio of one training experience.

    ``target_to_negatives`` is a requested *upper bound on negatives per target*: negatives are
    drawn without replacement from the cumulative pool and never duplicated, so while the pool is
    small the realized ratio is less negative-heavy than requested (e.g. requested 1:5, realized
    1:3).  All ratios are *targets per negative* (``5:1`` -> 5.0, ``1:5`` -> 0.2).

    Returns ``requested_target_to_negatives``, ``n_requested_negative``,
    ``realized_target_to_negatives`` (``None`` without negatives), ``realized_ratio_label`` (``None``
    unless both targets and negatives are present) and
    ``ratio_satisfied`` (``True`` when the requested negatives were available; ``None`` when
    there is nothing to enforce: cumulative mode or an experience without target samples).
    """
    realized = n_target / n_negative if n_negative > 0 else None
    out: dict[str, float | int | str | bool | None] = {
        "requested_target_to_negatives": target_to_negatives,
        "n_requested_negative": None,
        "realized_target_to_negatives": realized,
        # no label without both classes present (``format_ratio`` is for positive ratios)
        "realized_ratio_label": format_ratio(realized) if realized else None,
        "ratio_satisfied": None,
    }
    if target_to_negatives is not None and n_target > 0:
        n_requested = negatives_for_ratio(n_target, target_to_negatives)
        out["n_requested_negative"] = n_requested
        out["ratio_satisfied"] = n_negative >= n_requested
    return out


def build_target_vs_rest_stream(
    train,
    test,
    target_class,
    n_experiences,
    seed=0,
    class_order=None,
    target_in_every_experience=True,
    target_to_negatives=None,
):
    """Target-vs-rest stream over a growing set of negative classes.

    Experience ``i`` trains on the target class (every experience, or only experience 0 when
    ``target_in_every_experience`` is false) plus the *cumulative* pool of all negative classes
    introduced so far.

    ``target_to_negatives`` controls the target:negative sampling ratio of that training data:

    * ``None`` (default): keep the whole cumulative negative pool, so the negative/positive
      ratio grows with the experience index.
    * a number ``> 0``: keep **all** target samples and draw
      ``floor(n_target / target_to_negatives)`` negatives *without replacement* from the cumulative
      pool (``1.0`` = 1:1, ``0.2`` = 1 target : 5 negatives, ``0.1`` = 1:10, ``2.0`` = 2:1).
      Targets are never subsampled or duplicated. Negatives are never duplicated either: if the
      pool holds fewer negatives than requested, all of them are kept, so the requested ratio is an
      *upper bound on negatives per target* and the realized ratio can be less negative-heavy in
      early experiences (see ``ratio_report``). The draw is deterministic given ``seed`` and the
      experience index.
      Experiences without target samples (``target_in_every_experience=False``, ``i > 0``) have no
      ratio to enforce and keep the whole pool.
    """
    ratio = validate_target_to_negatives(target_to_negatives)
    num_classes = int(torch.max(torch.cat([train.targets, test.targets])).item()) + 1
    if not 0 <= target_class < num_classes:
        raise ValueError(f"target_class must be in [0, {num_classes - 1}]")

    order = _class_order(num_classes, seed, class_order)
    negatives = [c for c in order if c != target_class]
    if n_experiences > len(negatives):
        raise ValueError("n_experiences cannot exceed the number of negative classes")

    n_negative = None
    if ratio is not None:
        n_target = int((train.targets == target_class).sum())
        n_negative = negatives_for_ratio(n_target, ratio)
        if n_negative < 1:
            raise ValueError(
                f"target_to_negatives={ratio:g} with {n_target} target samples requests "
                f"{n_negative} negatives; use a smaller ratio"
            )

    negative_chunks = np.array_split(negatives, n_experiences)

    exps = []
    seen_negs: list[int] = []
    for i, chunk in enumerate(negative_chunks):
        new_negs = [int(c) for c in chunk.tolist()]
        seen_negs.extend(new_negs)

        has_target = target_in_every_experience or i == 0
        classes = ([target_class] if has_target else []) + seen_negs
        new_classes = ([target_class] if i == 0 else []) + new_negs

        parts = []
        if has_target:
            parts.append(_select_classes(train, [target_class]))

        negative_data = _select_classes(train, seen_negs)
        if n_negative is not None and has_target and len(negative_data) > n_negative:
            # seed + i: deterministic per (seed, experience); kept stable so earlier runs reproduce
            rng = np.random.default_rng(seed + i)
            indices = torch.as_tensor(
                rng.choice(len(negative_data), size=n_negative, replace=False),
                dtype=torch.long,
            )
            negative_data = ArrayDataset(
                negative_data.x[indices],
                negative_data.targets[indices],
                mean=negative_data.mean,
                std=negative_data.std,
            )
        parts.append(negative_data)

        x = torch.cat([p.x for p in parts])
        y = torch.cat([p.targets for p in parts])
        dataset = ArrayDataset(x, y, mean=train.mean, std=train.std)
        exps.append(Experience(i, dataset, classes, new_classes))

    stream_order = [target_class] + [c for c in order if c != target_class]
    return Stream(
        exps,
        test,
        num_classes,
        stream_order,
        target_class=target_class,
        target_to_negatives=ratio,
    )
