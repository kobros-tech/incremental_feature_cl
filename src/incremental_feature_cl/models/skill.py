# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""One persistent class-specific target-vs-rest learner.

A :class:`Skill` only knows: *"I am the learner for class K."*  It has no notion of
the other classes' identities beyond the labels of the negatives it remembers, no
experience ids, no Avalanche, and no global prediction.

Learning mechanism = the repository's existing binary target learner, reused rather
than copied:

* the data ratio (``target_to_negatives`` = targets per negative, the ``--ratio`` flag: ``0.2`` is
  1:5) uses ``data.streams.negatives_for_ratio`` / ``ratio_report`` (requested vs realized
  target:negative ratio, negatives never duplicated).  The positives counted are the replayed
  exemplars plus any new positives, so a refresh of an old skill (no new positives) trains on
  ``memory_per_class`` positives and ``ratio`` times as many negatives;
* the loss is the same ``binary_cross_entropy_with_logits`` with the same
  ``pos_weight`` conventions as ``ContinualTrainer`` in target mode
  (``"balanced"`` | float | ``None``).

The skill works on *feature vectors* ``h(x)`` produced by a frozen shared
representation, so its memory stores features, not images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.streams import negatives_for_ratio, ratio_report
from .backbones import IdentityBackbone
from .incremental_model import IncrementalFeatureMapModel


@dataclass
class SkillConfig:
    """Explicit learning configuration of a skill (nothing is hidden)."""

    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 32
    feature_expansion_dim: int = 0
    # targets per negative as a number (1.0 = 1:1, 0.2 = 1:5, 0.1 = 1:10); None = use every
    # available negative (``--ratio cumulative``)
    target_to_negatives: float | None = None
    pos_weight: Any = "balanced"  # "balanced" | float | None  (as in target mode)
    memory_per_class: int = 20  # replay budget per remembered class (0 = no replay)


def cap_samples(x: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    """At most ``cap`` rows of ``x`` (deterministic). Returns ``x`` itself if nothing is dropped."""
    if cap <= 0:
        return x[:0]
    if len(x) <= cap:
        return x
    idx = np.random.default_rng(seed).choice(len(x), cap, replace=False)
    return x[torch.as_tensor(np.sort(idx), dtype=torch.long)]


def stratified_draw(
    pools: dict[int, torch.Tensor], n: int, seed: int
) -> tuple[torch.Tensor, dict[int, int]]:
    """Draw ``n`` rows without replacement, spread as evenly as possible over the classes.

    Guarantees that every negative class is represented whenever ``n`` >= #classes, so a
    freshly arrived class can never be sampled away.  Returns (rows, per-class counts).
    """
    classes = sorted(pools)
    if not classes:
        raise ValueError("stratified_draw needs at least one class pool")
    total = sum(len(pools[c]) for c in classes)
    if n >= total:
        return torch.cat([pools[c] for c in classes]), {c: len(pools[c]) for c in classes}
    rng = np.random.default_rng(seed)
    quota = {c: 0 for c in classes}
    order = list(rng.permutation(classes))  # deterministic tie-break
    remaining = n
    while remaining > 0:  # water-filling: +1 per class per round until budget/pool is exhausted
        progressed = False
        for c in order:
            if remaining == 0:
                break
            if quota[c] < len(pools[c]):
                quota[c] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    parts = []
    for c in classes:
        if quota[c]:
            idx = rng.choice(len(pools[c]), quota[c], replace=False)
            parts.append(pools[c][torch.as_tensor(np.sort(idx), dtype=torch.long)])
    return torch.cat(parts), {c: q for c, q in quota.items() if q}


class Skill(nn.Module):
    def __init__(
        self,
        target_class: int,
        feature_dim: int,
        config: SkillConfig | None = None,
        seed: int = 0,
    ):
        super().__init__()

        self.target_class = int(target_class)
        self.feature_dim = int(feature_dim)
        self.config = config or SkillConfig()
        self.seed = int(seed)

        if self.config.feature_expansion_dim < 0:
            raise ValueError("feature_expansion_dim must be >= 0")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = IncrementalFeatureMapModel(
                backbone=IdentityBackbone(self.feature_dim),
                backbone_dim=self.feature_dim,
                num_outputs=1,
                activation="relu",
                initialization="zero",
                output_init="zero",
                freeze_backbone=True,
                freeze_old_blocks=False,
                seed=self.seed,
            )

        self.positive_memory: torch.Tensor | None = None
        self.negative_memory: dict[int, torch.Tensor] = {}
        self.n_updates = 0

    # ------------------------------------------------------------------ #
    @property
    def known_negatives(self) -> list[int]:
        return sorted(self.negative_memory)

    def decision_function(self, h: torch.Tensor) -> torch.Tensor:
        """Return target-vs-rest logits with shape [N]."""
        return self.model(h).squeeze(-1)

    forward = decision_function

    def _check_features(self, x: torch.Tensor | None, what: str) -> None:
        if x is not None and len(x) and (x.ndim != 2 or x.shape[1] != self.feature_dim):
            raise ValueError(
                f"{what}: expected features of shape [N, {self.feature_dim}], got {tuple(x.shape)}"
            )

    # ------------------------------------------------------------------ #
    # memory
    # ------------------------------------------------------------------ #
    def remember(
        self,
        positive: torch.Tensor | None = None,
        negatives: dict[int, torch.Tensor] | None = None,
    ) -> None:
        """Store exemplars (capped at ``memory_per_class`` each). Never stores its own class as negative."""
        cap = self.config.memory_per_class
        self._check_features(positive, "remember(positive)")
        for c, x in (negatives or {}).items():
            self._check_features(x, f"remember(negatives[{c}])")
        if positive is not None and len(positive):
            merged = (
                positive
                if self.positive_memory is None
                else torch.cat([self.positive_memory, positive])
            )
            self.positive_memory = cap_samples(merged, cap, self.seed)
        for c, x in (negatives or {}).items():
            c = int(c)
            if c == self.target_class:
                raise ValueError("a skill cannot hold its own class as a negative")
            merged = x if c not in self.negative_memory else torch.cat([self.negative_memory[c], x])
            # cap == 0 keeps the class as a *known* negative with an empty exemplar set
            self.negative_memory[c] = cap_samples(merged, cap, self.seed + 1 + c)

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def update(
        self,
        new_positive: torch.Tensor | None,
        new_negatives: dict[int, torch.Tensor],
        *,
        step_seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Train on (new data + replayed memory) with the configured target:negative ratio.

        Does NOT modify memory; the caller commits exemplars via :meth:`remember` afterwards,
        so that every skill of one experience trains on the same pre-experience memory.
        """
        cfg = self.config
        self._check_features(new_positive, "update(new_positive)")
        for c, x in new_negatives.items():
            self._check_features(x, f"update(new_negatives[{c}])")
        pos_parts = [p for p in (self.positive_memory, new_positive) if p is not None and len(p)]
        n_replay_pos = 0 if self.positive_memory is None else len(self.positive_memory)
        if not pos_parts:
            return {"target": self.target_class, "trained": False, "reason": "no positives"}
        pos = torch.cat(pos_parts)

        pools: dict[int, torch.Tensor] = {c: x for c, x in self.negative_memory.items() if len(x)}
        replay_neg = {c: len(x) for c, x in pools.items()}
        for c, x in new_negatives.items():
            if int(c) == self.target_class or not len(x):
                continue
            pools[int(c)] = torch.cat([pools[int(c)], x]) if int(c) in pools else x
        if not pools:
            return {"target": self.target_class, "trained": False, "reason": "no negatives"}

        n_available = sum(len(x) for x in pools.values())
        if cfg.target_to_negatives is None:
            n_neg = n_available
        else:
            n_neg = min(n_available, negatives_for_ratio(len(pos), cfg.target_to_negatives))
        neg, per_class = stratified_draw(pools, n_neg, self.seed * 7919 + step_seed)

        loss = self._fit(pos, neg, step_seed, device)
        self.n_updates += 1
        report = ratio_report(len(pos), len(neg), cfg.target_to_negatives)
        return {
            "target": self.target_class,
            "trained": True,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "replay_positive": n_replay_pos,
            "replay_negative": int(sum(replay_neg.values())),
            "negatives_used_per_class": {int(c): int(n) for c, n in per_class.items()},
            "known_negatives": self.known_negatives,
            "train_loss": loss,
            **report,
        }

    def _fit(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        step_seed: int,
        device: torch.device | str,
    ) -> float:
        cfg = self.config
        self.model.to(device)

        # Expand once, when this skill is first trained.
        # Replay updates train the existing feature blocks and classifier
        # instead of adding more parameters on every experience.
        if cfg.feature_expansion_dim > 0 and self.n_updates == 0:
            self.model.expand_feature_space(
                new_dim=cfg.feature_expansion_dim,
                seed=self.seed * 100_003 + step_seed + 1,
            )

        x = torch.cat([pos, neg]).to(device)
        y = torch.cat(
            [
                torch.ones(len(pos)),
                torch.zeros(len(neg)),
            ]
        ).to(device)

        if cfg.pos_weight == "balanced":
            pos_weight = torch.tensor(
                max(1, len(neg)) / max(1, len(pos)),
                device=device,
            )
        elif cfg.pos_weight is not None:
            pos_weight = torch.tensor(
                float(cfg.pos_weight),
                device=device,
            )
        else:
            pos_weight = None

        # Rebuild the optimizer after expansion so it includes the new
        # feature-block and classifier parameters.
        opt = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

        generator = torch.Generator().manual_seed(self.seed * 10_007 + step_seed)

        total_loss = 0.0
        n_batches = 0

        self.model.train()

        for _ in range(cfg.epochs):
            permutation = torch.randperm(
                len(x),
                generator=generator,
            ).to(device)

            for start in range(0, len(x), cfg.batch_size):
                indices = permutation[start : start + cfg.batch_size]

                logits = self.model(x[indices]).squeeze(-1)

                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y[indices],
                    pos_weight=pos_weight,
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss += float(loss.detach())
                n_batches += 1

        return total_loss / n_batches if n_batches and not math.isnan(total_loss) else float("nan")
