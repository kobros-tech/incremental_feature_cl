# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche adapter.

The model and the expansion policy live in the core package; this module only
connects them to Avalanche's experience boundaries:

* ``FeatureExpansionPlugin``   - attach to *any* Avalanche strategy.
* ``IncrementalFeatureSpaceStrategy`` - ``SupervisedTemplate`` + that plugin.

Note: evaluate on ``test_stream[: t + 1]`` (seen experiences). Output heads are
grown on demand, so unseen classes have no output unit yet.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate

from ..models import IncrementalFeatureMapModel
from ..training.policy import ExpansionPolicy


def quiet_evaluator() -> EvaluationPlugin:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No loggers specified" is intended here
        return EvaluationPlugin(accuracy_metrics(stream=True), loggers=[])


class FeatureExpansionPlugin(SupervisedPlugin):
    """Expand the feature space (and output head) at each experience boundary.

    Runs after Avalanche has prepared the experience; if the model grew, the
    optimizer is rebuilt so new parameters are optimised.  Works with any
    strategy whose ``model`` is an :class:`IncrementalFeatureMapModel`.
    """

    def __init__(self, policy: ExpansionPolicy):
        super().__init__()
        self.policy = policy
        self.records: list[dict[str, Any]] = []

    def before_training_exp(self, strategy, **kwargs):
        exp = strategy.experience
        model = strategy.model
        if not isinstance(model, IncrementalFeatureMapModel):
            raise TypeError("FeatureExpansionPlugin needs an IncrementalFeatureMapModel")
        recs = self.policy.on_experience_start(
            model, exp.current_experience, list(exp.classes_in_this_experience)
        )
        self.records.append(
            {"index": exp.current_experience, "expansions": [r.to_dict() for r in recs]}
        )
        if recs:
            model.to(strategy.device)
            strategy.make_optimizer(reset_optimizer_state=True)  # shapes/params changed


class IncrementalFeatureSpaceStrategy(SupervisedTemplate):
    """Standard Avalanche strategy: ``strategy.train(exp)`` / ``strategy.eval(stream)``."""

    def __init__(
        self,
        *,
        model: IncrementalFeatureMapModel,
        optimizer: torch.optim.Optimizer,
        policy: ExpansionPolicy | None = None,
        criterion=None,
        train_mb_size: int = 32,
        train_epochs: int = 1,
        eval_mb_size: int = 256,
        device: str = "cpu",
        plugins: list | None = None,
        evaluator=None,
        eval_every: int = -1,
    ):
        self.expansion_plugin = FeatureExpansionPlugin(policy or ExpansionPolicy())
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion or torch.nn.CrossEntropyLoss(),
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=[self.expansion_plugin, *(plugins or [])],
            evaluator=evaluator or quiet_evaluator(),
            eval_every=eval_every,
        )
