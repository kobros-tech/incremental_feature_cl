# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""scikit-learn control for the Skills (``pip install scikit-learn``).

:class:`SklearnSkill` shares the Skills' *data protocol* (replay memory, ``--ratio``, class-balanced
draw of old + new negatives, balanced positive weight; all inherited from ``SkillBase``) and
refits a ``LogisticRegression`` from scratch on every refresh.  Only the learner differs from a
:class:`Skill`, so the comparison is like-for-like.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from ..models import OneVsRestSkillModel, SkillBase, SkillConfig


class SklearnSkill(SkillBase):
    def __init__(
        self,
        target_class: int,
        feature_dim: int,
        config: SkillConfig | None = None,
        seed: int = 0,
        *,
        C: float = 1.0,
        max_iter: int = 200,
    ):
        super().__init__(target_class, feature_dim, config, seed)
        self.C, self.max_iter = float(C), int(max_iter)
        self.clf = None

    def decision_function(self, h: torch.Tensor) -> torch.Tensor:
        if self.clf is None:
            return torch.zeros(len(h))
        return torch.from_numpy(self.clf.decision_function(h.detach().cpu().numpy())).float()

    def _fit(self, pos, neg, step_seed: int, device) -> float:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("the sklearn control needs scikit-learn") from exc
        x = torch.cat([pos, neg]).cpu().numpy()
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        w_pos = self.config.pos_weight
        w_pos = max(1, len(neg)) / max(1, len(pos)) if w_pos == "balanced" else (w_pos or 1.0)
        weights = np.where(y == 1, float(w_pos), 1.0)
        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.seed + step_seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lbfgs ConvergenceWarning on raw pixels is expected
            clf.fit(x, y, sample_weight=weights)
        self.clf = clf
        margin = clf.decision_function(x) * (2 * y - 1)
        return float(np.average(np.logaddexp(0.0, -margin), weights=weights))


class SklearnOneVsRest(OneVsRestSkillModel):
    """The Skill ensemble's protocol with sklearn learners."""

    def __init__(self, backbone, feature_dim=None, config=None, seed=0, *, C=1.0, max_iter=200):
        def make(target_class, dim, cfg, skill_seed):
            return SklearnSkill(target_class, dim, cfg, skill_seed, C=C, max_iter=max_iter)

        super().__init__(backbone, feature_dim, config, seed, skill_factory=make)

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            0 if sk.clf is None else int(sk.clf.coef_.size + sk.clf.intercept_.size)
            for sk in self.skills.values()
        )

    def skill_parameters(self):
        return iter(())
