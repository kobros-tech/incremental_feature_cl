# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche adapters.  The core models stay Avalanche-free; this module only connects them.

* :class:`FeatureExpansionPlugin` / :class:`IncrementalFeatureSpaceStrategy` - expand an
  :class:`IncrementalFeatureMapModel` at every experience boundary (attach the plugin to *any*
  Avalanche strategy).  The model has one multiclass head, so Avalanche's argmax accuracy is a
  meaningful metric here (:func:`quiet_evaluator`).
* :class:`OneVsRestSkillStrategy` - ``strategy.train(exp)`` updates the persistent skills of a
  :class:`OneVsRestSkillModel`.  Its skills are *independent* target-vs-rest learners, so
  Avalanche's argmax-based accuracy is deliberately NOT part of its default evaluator (see
  :func:`skill_evaluator`); use :meth:`OneVsRestSkillStrategy.evaluate_skills` instead.

Avalanche provides the benchmark, the experience stream, device handling and metrics.  Avalanche's
generic replay is deliberately not used for the skills: every skill owns its target/negative memory.
Per experience the strategy hands the model the experience data once; the model creates new skills,
updates *every* existing skill with the new classes as negatives, and replays the per-skill memory.
Skill epochs, batch size and learning rate live in ``SkillConfig`` (so ``train_epochs`` is fixed to
1 here).

Use any Avalanche benchmark, and any Avalanche model as the (frozen) backbone::

    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.models import SimpleMLP
    from incremental_feature_cl.models import FeatureBackbone, OneVsRestSkillModel, SkillConfig

    backbone = FeatureBackbone(SimpleMLP(input_size=3072), input_shape=(3, 32, 32))
    model = OneVsRestSkillModel(backbone, config=SkillConfig(feature_expansion_dim=16))
    strategy = OneVsRestSkillStrategy(model=model, num_classes=100)
    for experience in SplitCIFAR100(n_experiences=5).train_stream:
        strategy.train(experience)
    per_skill, decisions = strategy.evaluate_skills(benchmark.test_stream)
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader

from ..evaluation.skill_decisions import decision_report
from ..evaluation.skill_metrics import per_skill_metrics
from ..models import IncrementalFeatureMapModel, OneVsRestSkillModel
from ..training.policy import ExpansionPolicy


def quiet_evaluator() -> EvaluationPlugin:
    """Evaluator with stream accuracy and no loggers (for single multiclass-head models)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No loggers specified" is intended here
        return EvaluationPlugin(accuracy_metrics(stream=True), loggers=[])


def skill_evaluator() -> EvaluationPlugin:
    """Evaluator without metrics, for independent skills.

    Avalanche's accuracy takes the ``argmax`` over the score columns.  Independently trained
    target-vs-rest logits are not comparable across skills and ``argmax`` always answers, so that
    number would be a misleading multiclass accuracy.  Per-skill binary metrics and the firing
    based decision report come from :meth:`OneVsRestSkillStrategy.evaluate_skills`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EvaluationPlugin(loggers=[])


# ---------------------------------------------------------------------- #
# single multiclass-head model with feature expansion
# ---------------------------------------------------------------------- #
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
    """Standard Avalanche strategy: ``strategy.train(exp)`` / ``strategy.eval(stream)``.

    Evaluate on ``test_stream[: t + 1]`` (seen experiences): output heads are grown on demand, so
    unseen classes have no output unit yet.
    """

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


# ---------------------------------------------------------------------- #
# one-vs-rest skills
# ---------------------------------------------------------------------- #
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
            evaluator=evaluator or skill_evaluator(),
            eval_every=eval_every,
        )

    def make_optimizer(self, *args, **kwargs):  # skills carry their own optimisers
        pass

    def forward(self):
        """Return scores laid out by global class ID (columns = class ids).

        The layout only maps a score column to its class.  It does not calibrate or make the
        independently trained skill logits comparable, so ``argmax`` over these scores is not a
        validated multiclass prediction (and none of this package's reports use it).
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

    @torch.no_grad()
    def evaluate_skills(
        self,
        experiences,
        thresholds: Mapping[int, float] | None = None,
    ) -> tuple[dict[int, dict], dict[str, Any]]:
        """Per-skill binary metrics and the firing-based decision report (no ``argmax``).

        ``experiences`` is an iterable of Avalanche experiences (e.g. ``benchmark.test_stream``);
        only samples of learned classes are scored.  ``thresholds`` maps class id -> decision
        threshold (``0`` when omitted), see :func:`evaluation.skill_decisions.select_thresholds`.
        Returns ``(per_skill_metrics, decision_report)``.
        """
        scores, labels = [], []
        self.model.eval()
        for exp in experiences:
            for batch in DataLoader(exp.dataset, batch_size=512, shuffle=False):
                x, y = batch[0].to(self.device), batch[1]
                scores.append(self.model.decision_function(x).cpu().numpy())
                labels.append(np.asarray(y))
        s, y = np.concatenate(scores), np.concatenate(labels)
        ids = self.model.class_ids
        known = [c for c in ids if (y == c).any()]
        keep = np.isin(y, ids)
        return (
            per_skill_metrics(s[keep], y[keep], ids, known, thresholds),
            decision_report(s[keep], y[keep], ids, known, thresholds, include_matrix=False),
        )
