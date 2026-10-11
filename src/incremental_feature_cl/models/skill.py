# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""One persistent class-specific target-vs-rest learner.

A skill only knows: *"I am the learner for class K."*  It has no notion of the other classes'
identities beyond the labels of the negatives it remembers, no experience ids, no Avalanche and
no global prediction.

Two layers, so that different learners can be compared on exactly the same data:

* :class:`SkillBase` owns the **data protocol**: the replay memory (capped feature exemplars per
  class), the target:negative ratio and the class-balanced draw of negatives.  Every refresh of
  a skill trains the positive class against *old* negatives (replayed exemplars) and *new*
  negatives (the current experience).
* :class:`Skill` is the torch learner.  It wraps an :class:`IncrementalFeatureMapModel` over the
  frozen shared features and is configured through :class:`SkillConfig`, whose model / optimizer
  fields have the same names and meaning as ``ModelConfig`` / ``TrainConfig`` of the target-vs-rest
  experiments (``activation``, ``initialization``, ``output_init``, ``freeze_old_blocks``,
  ``feature_expansion_dim``, ``optimizer``, ``lr`` ...).  Nothing about the model is hardcoded.

The data ratio (``target_to_negatives`` = targets per negative, the ``--ratio`` flag: ``0.2`` is
1:5) uses ``data.streams.negatives_for_ratio`` / ``ratio_report``.  The positives counted are the
replayed exemplars plus any new positives, so a refresh of an old skill (no new positives) trains
on ``memory_per_class`` positives and ``ratio`` times as many negatives.  The loss is the same
``binary_cross_entropy_with_logits`` with the same ``pos_weight`` conventions as
``ContinualTrainer`` in target mode (``"balanced"`` | float | ``None``).

Skills work on *feature vectors* ``h(x)`` of a frozen shared representation, so their memory
stores features, not images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.streams import negatives_for_ratio, ratio_report, stratified_quota
from .backbones import IdentityBackbone
from .expandable import check_initialization
from .incremental_model import IncrementalFeatureMapModel

OPTIMIZERS = ("sgd", "adam", "lbfgs")
# When a skill grows its feature space (only if ``feature_expansion_dim > 0``):
#   first : once, when the skill is first trained        (default)
#   every : before every update                          (like the target-mode trainer, but also
#                                                         at the skill's first update)
#   later : before every update except the first one     (``ExpansionPolicy`` default)
#   never : the skill stays a (non)linear model of fixed size
EXPANSIONS = ("first", "every", "later", "never")


@dataclass
class SkillConfig:
    """Explicit configuration of a skill (nothing is hidden)."""

    # --- learner (names as in ``TrainConfig``) ---
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 32
    # --- data protocol ---
    feature_expansion_dim: int = 0  # new_feature_dim of ``ModelConfig``; 0 = linear skill
    # targets per negative as a number (1.0 = 1:1, 0.2 = 1:5, 0.1 = 1:10); None = use every
    # available negative (``--ratio cumulative``)
    target_to_negatives: float | None = None
    pos_weight: Any = "balanced"  # "balanced" | float | None  (as in target mode)
    memory_per_class: int = 20  # replay budget per remembered class (0 = no replay)
    # --- model (names as in ``ModelConfig`` / ``IncrementalFeatureMapModel``) ---
    activation: str = "relu"
    initialization: str = "zero"
    output_init: str = "zero"
    freeze_old_blocks: bool = False
    expansion: str = "first"  # see EXPANSIONS
    # "sgd" | "adam" (as in ``TrainConfig``) | "lbfgs": full-batch L-BFGS on the (small) training
    # set of a refresh.  It converges deterministically, with no learning rate / epoch count to
    # tune and no last-minibatch noise on the decision threshold; ``weight_decay`` is then an
    # L2 penalty and ``max_iter`` the iteration cap (``lr``, ``momentum``, ``epochs`` and
    # ``batch_size`` are unused).
    optimizer: str = "sgd"
    max_iter: int = 100  # L-BFGS iterations per update

    def __post_init__(self) -> None:
        if self.feature_expansion_dim < 0:
            raise ValueError("feature_expansion_dim must be >= 0")
        if self.memory_per_class < 0:
            raise ValueError("memory_per_class must be >= 0")
        if self.expansion not in EXPANSIONS:
            raise ValueError(f"unknown expansion {self.expansion!r}; choose from {EXPANSIONS}")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; choose from {OPTIMIZERS}")
        check_initialization(self.initialization)
        check_initialization(self.output_init)


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
    quota = stratified_quota({c: len(pools[c]) for c in classes}, n, rng)
    parts = []
    for c in classes:
        if quota[c]:
            idx = rng.choice(len(pools[c]), quota[c], replace=False)
            parts.append(pools[c][torch.as_tensor(np.sort(idx), dtype=torch.long)])
    return torch.cat(parts), {c: q for c, q in quota.items() if q}


class SkillBase(nn.Module):
    """Replay memory + data protocol of a skill; subclasses implement the learner.

    Subclasses provide :meth:`_fit` (train on ``pos`` vs ``neg`` feature rows) and
    :meth:`decision_function` (logit-like score, ``> 0`` means "this is my class").
    """

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
        self.positive_memory: torch.Tensor | None = None
        self.negative_memory: dict[int, torch.Tensor] = {}
        self.n_updates = 0

    # ------------------------------------------------------------------ #
    @property
    def known_negatives(self) -> list[int]:
        return sorted(self.negative_memory)

    @property
    def total_feature_dim(self) -> int:
        """Dimension of the skill's own feature space (grows for expanding skills)."""
        return self.feature_dim

    def decision_function(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit(
        self, pos: torch.Tensor, neg: torch.Tensor, step_seed: int, device: torch.device | str
    ) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

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
    # data protocol
    # ------------------------------------------------------------------ #
    def training_set(
        self,
        new_positive: torch.Tensor | None,
        new_negatives: dict[int, torch.Tensor],
        step_seed: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        """Positives and negatives of one refresh: new data + replayed memory, ratio-limited.

        Returns ``(pos, neg, info)``; ``pos`` / ``neg`` are ``None`` (and ``info["trained"]`` is
        False with a ``reason``) when there is nothing to learn from.  Negatives are drawn
        class-balanced over *all* known negative classes: replayed old ones and the new ones.
        """
        cfg = self.config
        self._check_features(new_positive, "update(new_positive)")
        for c, x in new_negatives.items():
            self._check_features(x, f"update(new_negatives[{c}])")
        pos_parts = [p for p in (self.positive_memory, new_positive) if p is not None and len(p)]
        n_replay_pos = 0 if self.positive_memory is None else len(self.positive_memory)
        if not pos_parts:
            return (
                None,
                None,
                {"target": self.target_class, "trained": False, "reason": "no positives"},
            )
        pos = torch.cat(pos_parts)

        pools: dict[int, torch.Tensor] = {c: x for c, x in self.negative_memory.items() if len(x)}
        replay_neg = {c: len(x) for c, x in pools.items()}
        for c, x in new_negatives.items():
            if int(c) == self.target_class or not len(x):
                continue
            pools[int(c)] = torch.cat([pools[int(c)], x]) if int(c) in pools else x
        if not pools:
            return (
                None,
                None,
                {"target": self.target_class, "trained": False, "reason": "no negatives"},
            )

        n_available = sum(len(x) for x in pools.values())
        if cfg.target_to_negatives is None:
            n_neg = n_available
        else:
            n_neg = min(n_available, negatives_for_ratio(len(pos), cfg.target_to_negatives))
        neg, per_class = stratified_draw(pools, n_neg, self.seed * 7919 + step_seed)
        info = {
            "target": self.target_class,
            "trained": True,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "replay_positive": n_replay_pos,
            "replay_negative": int(sum(replay_neg.values())),
            "negatives_used_per_class": {int(c): int(n) for c, n in per_class.items()},
            "known_negatives": self.known_negatives,
            **ratio_report(len(pos), len(neg), cfg.target_to_negatives),
        }
        return pos, neg, info

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
        pos, neg, info = self.training_set(new_positive, new_negatives, step_seed)
        if pos is None:
            return info
        info["train_loss"] = self._fit(pos, neg, step_seed, device)
        self.n_updates += 1
        return info


class Skill(SkillBase):
    """Torch skill: an :class:`IncrementalFeatureMapModel` over frozen shared features.

    Equivalent to ``IncrementalFeatureMapModel(backbone=Identity, num_outputs=1,
    freeze_backbone=True, ...)`` with the model options taken from ``config``.
    """

    def __init__(
        self,
        target_class: int,
        feature_dim: int,
        config: SkillConfig | None = None,
        seed: int = 0,
    ):
        super().__init__(target_class, feature_dim, config, seed)
        cfg = self.config
        # Building the model draws from the global RNG (default ``nn.Linear`` init of the base
        # head); fork it so every skill is deterministic and the caller's RNG stream is untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = IncrementalFeatureMapModel(
                backbone=IdentityBackbone(self.feature_dim),
                backbone_dim=self.feature_dim,
                num_outputs=1,
                activation=cfg.activation,
                initialization=cfg.initialization,
                output_init=cfg.output_init,
                freeze_backbone=True,  # skills replay features, so the representation is fixed
                freeze_old_blocks=cfg.freeze_old_blocks,
                seed=self.seed,
            )

    @property
    def total_feature_dim(self) -> int:
        return self.model.feature_dim

    def decision_function(self, h: torch.Tensor) -> torch.Tensor:
        """Return target-vs-rest logits with shape [N]."""
        return self.model(h).squeeze(-1)

    forward = decision_function

    def _should_expand(self) -> bool:
        cfg = self.config
        if cfg.feature_expansion_dim <= 0 or cfg.expansion == "never":
            return False
        if cfg.expansion == "first":
            return self.n_updates == 0
        if cfg.expansion == "later":
            return self.n_updates > 0
        return True  # "every"

    def _fit_lbfgs(self, x: torch.Tensor, y: torch.Tensor, pos_weight) -> float:
        """Full-batch L-BFGS (strong-Wolfe line search) on ``BCE + weight_decay/2 * ||theta||^2``."""
        cfg = self.config
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(
            params, lr=1.0, max_iter=cfg.max_iter, history_size=20, line_search_fn="strong_wolfe"
        )
        self.model.train()
        last = {"loss": float("nan")}

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                self.model(x).squeeze(-1), y, pos_weight=pos_weight
            )
            last["loss"] = float(loss.detach())
            if cfg.weight_decay > 0:
                loss = loss + 0.5 * cfg.weight_decay * sum((p**2).sum() for p in params)
            loss.backward()
            return loss

        opt.step(closure)
        return last["loss"]

    def _make_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        params = [p for p in self.model.parameters() if p.requires_grad]
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )

    def _pos_weight(self, n_pos: int, n_neg: int, device) -> torch.Tensor | None:
        w = self.config.pos_weight
        if w == "balanced":
            return torch.tensor(max(1, n_neg) / max(1, n_pos), device=device)
        return torch.tensor(float(w), device=device) if w is not None else None

    def _fit(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor,
        step_seed: int,
        device: torch.device | str,
    ) -> float:
        cfg = self.config
        self.model.to(device)
        if self._should_expand():
            self.model.expand_feature_space(
                new_dim=cfg.feature_expansion_dim,
                seed=self.seed * 100_003 + step_seed + 1,
            )

        x = torch.cat([pos, neg]).to(device)
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)
        pos_weight = self._pos_weight(len(pos), len(neg), device)
        if cfg.optimizer == "lbfgs":
            return self._fit_lbfgs(x, y, pos_weight)
        # The optimizer is rebuilt every update so it covers freshly expanded parameters.
        opt = self._make_optimizer()
        generator = torch.Generator().manual_seed(self.seed * 10_007 + step_seed)

        total_loss, n_batches = 0.0, 0
        self.model.train()
        for _ in range(cfg.epochs):
            permutation = torch.randperm(len(x), generator=generator).to(device)
            for start in range(0, len(x), cfg.batch_size):
                idx = permutation[start : start + cfg.batch_size]
                logits = self.model(x[idx]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
                n_batches += 1
        return total_loss / n_batches if n_batches and not math.isnan(total_loss) else float("nan")
