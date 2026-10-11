# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche adapters for the two OvR models (core models stay Avalanche-free).

Avalanche provides the benchmark, the experience stream, device handling and metrics.  The models
provide everything class-specific:

* ``OneVsRestHeadsStrategy`` - conventional OvR control (``PyTorchOneVsRestNet``): ordinary
  minibatch SGD on independent binary heads, plain BCE, no replay.
* ``OneVsRestSkillStrategy`` - Skill OvR (``OneVsRestSkillModel``).  Avalanche's generic replay is
  deliberately NOT used: every skill owns its target/negative memory.  Per experience the strategy
  hands the model the experience data once; the model creates new skills, updates *every*
  existing skill with the new classes as negatives, and replays per-skill memory.  Skill epochs,
  batch size and learning rate live in ``SkillConfig`` (so ``train_epochs`` is fixed to 1 here).
"""

from __future__ import annotations

import torch
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader

from ..models import OneVsRestSkillModel
from .strategy import quiet_evaluator


class _SeenClassesPlugin(SupervisedPlugin):
    def before_training_exp(self, strategy, **kwargs):
        strategy.model.mark_seen([int(c) for c in strategy.experience.classes_in_this_experience])


class OneVsRestSkillStrategy(SupervisedTemplate):
    """Avalanche strategy around :class:`OneVsRestSkillModel`; ``strategy.train(exp)`` updates skills."""

    def __init__(
        self,
        *,
        model: OneVsRestSkillModel,
        num_classes: int,
        eval_mb_size: int = 256,
        device: str = "cpu",
        plugins: list | None = None,
        evaluator=None,
        eval_every: int = -1,
    ):
        self.num_classes = int(num_classes)
        self.skill_records: list[dict] = []
        dummy = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)  # never stepped
        super().__init__(
            model=model,
            optimizer=dummy,
            criterion=torch.nn.CrossEntropyLoss(),  # unused: the skills own their binary losses
            train_mb_size=eval_mb_size,
            train_epochs=1,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins or [],
            evaluator=evaluator or quiet_evaluator(),
            eval_every=eval_every,
        )

    def make_optimizer(self, *args, **kwargs):  # skills carry their own optimisers
        pass

    def forward(self):
        """Scores laid out by class id so Avalanche's argmax-based metrics see class ids."""
        return self.model.class_id_scores(self.mb_x, self.num_classes)

    def training_epoch(self, **kwargs):
        xs, ys = [], []
        for batch in DataLoader(self.adapted_dataset, batch_size=512, shuffle=False):
            xs.append(batch[0])
            ys.append(batch[1])
        self.skill_records.append(
            self.model.update(torch.cat(xs), torch.cat(ys), device=self.device)
        )
