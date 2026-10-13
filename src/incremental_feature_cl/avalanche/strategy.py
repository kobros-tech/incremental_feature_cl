# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche strategy around :class:`OneVsRestSkillModel` (the core models stay Avalanche-free).

Avalanche provides the benchmark, the experience stream, device handling and metrics.  The model
provides everything class-specific.  Avalanche's generic replay is deliberately NOT used: every
skill owns its target/negative memory.  Per experience the strategy hands the model the
experience data once; the model creates new skills, updates *every* existing skill with the new
classes as negatives, and replays the per-skill memory.  Skill epochs, batch size and learning
rate live in ``SkillConfig`` (so ``train_epochs`` is fixed to 1 here).

Use any Avalanche benchmark, and any Avalanche model as the (frozen) backbone::

    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.models import SimpleMLP
    from incremental_feature_cl.models import FeatureBackbone, OneVsRestSkillModel, SkillConfig

    backbone = FeatureBackbone(SimpleMLP(input_size=3072), input_shape=(3, 32, 32))
    model = OneVsRestSkillModel(backbone, config=SkillConfig(feature_expansion_dim=16))
    strategy = OneVsRestSkillStrategy(model=model, num_classes=100)
    for experience in SplitCIFAR100(n_experiences=5).train_stream:
        strategy.train(experience)
"""

from __future__ import annotations

import warnings

import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader

from ..models import OneVsRestSkillModel


def quiet_evaluator() -> EvaluationPlugin:
    """Return an evaluator with stream accuracy and no loggers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvaluationPlugin(accuracy_metrics(stream=True), loggers=[])


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
        """Return scores laid out by global class ID.

        The output columns correspond to class IDs, but this mapping does not
        calibrate or make independently trained skill logits comparable.
        Do not interpret argmax over these scores as validated multiclass accuracy.
        """
        return self.model.class_id_scores(self.mb_x, self.num_classes)

    def training_epoch(self, **kwargs):
        xs, ys = [], []
        for batch in DataLoader(self.adapted_dataset, batch_size=512, shuffle=False):
            xs.append(batch[0])
            ys.append(batch[1])
        self.skill_records.append(
            self.model.update(torch.cat(xs), torch.cat(ys), device=self.device)
        )
