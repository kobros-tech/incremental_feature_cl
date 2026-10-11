# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""scikit-learn controls for the Skill ensemble (``pip install scikit-learn``).

Both controls are one-vs-rest ``LogisticRegression`` learners over the same frozen features and
are scored by exactly the same code as the Skills (:mod:`experiments.ovr_runner`):

* ``replay``: :class:`SklearnSkill` shares the Skills' *data protocol* (same replay memory,
  ``--ratio``, class-balanced draw of old + new negatives, balanced positive weight) and refits
  from scratch on every refresh.  Only the learner differs, so this is the like-for-like control.
* ``cumulative``: after every experience each class is refit on ALL training data of all classes
  seen so far (no memory limit, no ratio, class-balanced).  It is not continual (it re-reads every
  old sample), so it is the strongest control / an upper-bound reference.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..models import OneVsRestSkillModel, SkillBase, SkillConfig


def _logistic_regression(C: float, max_iter: int, seed: int):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the sklearn controls need scikit-learn (pip install scikit-learn)"
        ) from exc
    return LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=seed)


class SklearnSkill(SkillBase):
    """A skill whose learner is ``sklearn.linear_model.LogisticRegression`` (refit each update)."""

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

    @property
    def n_parameters(self) -> int:
        return 0 if self.clf is None else int(self.clf.coef_.size + self.clf.intercept_.size)

    def decision_function(self, h: torch.Tensor) -> torch.Tensor:
        if self.clf is None:
            return torch.zeros(len(h))
        return torch.from_numpy(self.clf.decision_function(h.detach().cpu().numpy())).float()

    def _fit(
        self, pos: torch.Tensor, neg: torch.Tensor, step_seed: int, device: torch.device | str
    ) -> float:
        x = torch.cat([pos, neg]).cpu().numpy()
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        w_pos = self.config.pos_weight
        w_pos = max(1, len(neg)) / max(1, len(pos)) if w_pos == "balanced" else (w_pos or 1.0)
        weights = np.where(y == 1, float(w_pos), 1.0)
        clf = _logistic_regression(self.C, self.max_iter, self.seed + step_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lbfgs ConvergenceWarning on raw pixels is expected
            clf.fit(x, y, sample_weight=weights)
        self.clf = clf
        margin = clf.decision_function(x) * (2 * y - 1)
        return float(np.average(np.logaddexp(0.0, -margin), weights=weights))


def _skill_factory(C: float, max_iter: int):
    def make(target_class: int, feature_dim: int, config: SkillConfig, seed: int) -> SkillBase:
        return SklearnSkill(target_class, feature_dim, config, seed, C=C, max_iter=max_iter)

    return make


class SklearnReplayOvR(OneVsRestSkillModel):
    """The Skill ensemble's protocol with sklearn learners (see the module docstring)."""

    def __init__(self, backbone, feature_dim=None, config=None, seed=0, *, C=1.0, max_iter=200):
        super().__init__(
            backbone, feature_dim, config, seed, skill_factory=_skill_factory(C, max_iter)
        )

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(sk.n_parameters for sk in self.skills.values())

    def skill_parameters(self):
        return iter(())


class CumulativeSklearnOvR(SklearnReplayOvR):
    """Refit every selected class on all data seen so far (non-continual reference).

    ``only_classes`` limits the evaluated skills (the fit of one class is a full pass over the
    cumulative data, which is slow for 100 classes on raw pixels); ``None`` fits every class.
    """

    def __init__(
        self,
        backbone,
        feature_dim=None,
        config=None,
        seed=0,
        *,
        C=1.0,
        max_iter=200,
        only_classes: Sequence[int] | None = None,
    ):
        super().__init__(backbone, feature_dim, config, seed, C=C, max_iter=max_iter)
        self.only_classes = None if only_classes is None else {int(c) for c in only_classes}
        self._h = torch.empty(0, self.feature_dim)
        self._y = torch.empty(0, dtype=torch.long)

    @torch.no_grad()
    def update(self, x, y, device=None) -> dict[str, Any]:
        device = torch.device(device) if device is not None else self._device_ref.device
        y = torch.as_tensor(y, dtype=torch.long).cpu().reshape(-1)
        step = len(self.history)
        self._h = torch.cat([self._h, self.extract(x.to(device)).cpu()])
        self._y = torch.cat([self._y, y])
        seen = sorted(int(c) for c in self._y.unique())
        selected = [c for c in seen if self.only_classes is None or c in self.only_classes]

        old_classes = list(self.class_ids)
        new_skills = [c for c in selected if str(c) not in self._skills]
        for c in new_skills:
            skill = self.skill_factory(c, self.feature_dim, self.config, self.seed * 100_003 + c)
            self._skills[str(c)] = skill
            self.class_ids.append(c)

        per_skill: dict[int, dict[str, Any]] = {}
        counts = {c: int((self._y == c).sum()) for c in seen}
        for c in self.class_ids:
            skill = self._skills[str(c)]
            positive = self._h[self._y == c]
            negative = self._h[self._y != c]
            loss = skill._fit(positive, negative, step, device)
            skill.n_updates += 1
            # placeholders: the control re-reads all data, it keeps no exemplar memory
            skill.negative_memory = {k: self._h[:0] for k in seen if k != c}
            per_skill[c] = {
                "target": c,
                "trained": True,
                "is_new_skill": c in new_skills,
                "n_positive": len(positive),
                "n_negative": len(negative),
                "negatives_used_per_class": {k: n for k, n in counts.items() if k != c},
                "train_loss": loss,
            }
        rec = {
            "index": step,
            "classes": sorted(int(c) for c in y.unique()),
            "new_skills": new_skills,
            "updated_old_skills": old_classes,
            "number_of_skills": self.num_skills,
            "known_negatives": {c: self._skills[str(c)].known_negatives for c in self.class_ids},
            "replay_count": {c: len(self._h) for c in self.class_ids},
            "skills": per_skill,
            "comparison": {"ran": False, "reason": "sklearn control"},
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(),
        }
        self.history.append(rec)
        return rec


def build_sklearn_model(
    protocol: str,
    backbone,
    skill_cfg: SkillConfig,
    seed: int,
    *,
    C: float = 1.0,
    max_iter: int = 200,
    only_classes: Sequence[int] | None = None,
) -> OneVsRestSkillModel:
    if protocol == "replay":
        return SklearnReplayOvR(backbone, config=skill_cfg, seed=seed, C=C, max_iter=max_iter)
    if protocol == "cumulative":
        return CumulativeSklearnOvR(
            backbone, config=skill_cfg, seed=seed, C=C, max_iter=max_iter, only_classes=only_classes
        )
    raise ValueError(f"unknown sklearn protocol {protocol!r}; choose replay or cumulative")
