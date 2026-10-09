# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Continual One-vs-Rest Skill model.

    x -> frozen shared representation h(x) -> Skill(c) for every learned class c
                                           -> decision scores [N, C] -> argmax

Invariant: after experience ``t`` with learned classes ``C_t``, every ``c in C_t`` owns exactly
one persistent :class:`Skill` that represents ``c`` vs ``C_t - {c}``.  Arriving classes create
new skills *and* are added as negatives to every existing skill.

The model is a plain ``nn.Module`` (no Avalanche imports).  Because skills replay *feature
vectors*, the shared representation must be frozen; the constructor enforces this.
Scores are raw independent logits, never softmax-calibrated probabilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .skill import Skill, SkillConfig, cap_samples


class OneVsRestSkillModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int | None = None,
        config: SkillConfig | None = None,
        seed: int = 0,
    ):
        super().__init__()
        feature_dim = feature_dim if feature_dim is not None else getattr(backbone, "out_dim", None)
        if feature_dim is None:
            raise ValueError("feature_dim is required when the backbone has no `out_dim`")
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)  # skills store features, so the representation is fixed
        self.backbone.eval()
        self.register_buffer("_device_ref", torch.zeros(()), persistent=False)
        self.feature_dim = int(feature_dim)
        self.config = config or SkillConfig()
        self.seed = int(seed)
        self._skills = nn.ModuleDict()  # keys are str(class id); order = discovery order
        self.class_ids: list[int] = []  # explicit, append-only column order of the score matrix
        # Latest capped feature exemplars per class. Skills may hold per-skill replay
        # tensors derived from these after merging new samples with their existing memories.
        self.exemplars: dict[int, torch.Tensor] = {}
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen representation: fixed BatchNorm statistics
        return self

    @property
    def skills(self) -> dict[int, Skill]:
        """``{class id: Skill}`` in discovery order (the same persistent objects every time)."""
        return {c: self._skills[str(c)] for c in self.class_ids}

    @property
    def num_skills(self) -> int:
        return len(self.class_ids)

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ------------------------------------------------------------------ #
    # representation / scoring
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """Frozen shared representation ``h(x)``, shape ``[N, feature_dim]``."""
        self.backbone.eval()
        out = []
        for i in range(0, len(x), batch_size):
            h = self.backbone(x[i : i + batch_size])
            out.append(h.flatten(1) if h.ndim > 2 else h)
        return torch.cat(out) if out else torch.empty(0, self.feature_dim, device=x.device)

    def decision_function_features(self, h: torch.Tensor) -> torch.Tensor:
        """Scores from features: column ``j`` is the skill of ``class_ids[j]``. Shape ``[N, C]``."""
        if not self.class_ids:
            raise RuntimeError("no skills yet: call update() first")
        return torch.stack([self._skills[str(c)].decision_function(h) for c in self.class_ids], 1)

    @torch.no_grad()
    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Anonymous scoring: depends on ``x`` only (no target class / experience / task id)."""
        return self.decision_function_features(self.extract(x))

    forward = decision_function

    def scores_to_predictions(self, scores: torch.Tensor) -> torch.Tensor:
        """argmax over columns, mapped to *class ids* through the explicit ``class_ids`` order."""
        ids = torch.as_tensor(self.class_ids, device=scores.device)
        return ids[scores.argmax(dim=1)]

    @torch.no_grad()
    def class_id_scores(
        self, x: torch.Tensor, num_classes: int, fill: float = -1e9
    ) -> torch.Tensor:
        """Scores laid out by class id ``[N, num_classes]`` (unlearned classes get ``fill``).

        Lets generic evaluators that expect ``argmax == class id`` (e.g. Avalanche metrics) work
        without knowing the discovery-ordered ``class_ids``.
        """
        s = self.decision_function(x)
        out = torch.full((len(x), num_classes), fill, dtype=s.dtype, device=s.device)
        out[:, torch.as_tensor(self.class_ids, device=s.device)] = s
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.scores_to_predictions(self.decision_function(x))

    # ------------------------------------------------------------------ #
    # continual update
    # ------------------------------------------------------------------ #
    def update(
        self, x: torch.Tensor, y: torch.Tensor, device: torch.device | str | None = None
    ) -> dict[str, Any]:
        """Learn one experience ``(x, y)``.

        1. discover new classes -> create their skills (seeded, never replacing existing ones);
        2. give every new skill the exemplars of all previously seen classes as negatives;
        3. update EVERY skill (old and new) on new data + its own replay memory;
        4. commit exemplars of this experience's classes to every skill's replay memory.
        """
        device = torch.device(device) if device is not None else self._device_ref.device
        y = torch.as_tensor(y, dtype=torch.long).cpu()
        step = len(self.history)
        h = self.extract(x.to(device)).cpu()
        classes = sorted(int(c) for c in y.unique())
        by_class = {c: h[y == c] for c in classes}

        # 1-2: new skills
        new_classes = [c for c in classes if str(c) not in self._skills]
        old_classes = list(self.class_ids)
        for c in new_classes:
            skill = Skill(c, self.feature_dim, self.config, seed=self.seed * 100_003 + c)
            self._skills[str(c)] = skill.to(device)
            self.class_ids.append(c)
            skill.remember(negatives={k: self.exemplars[k] for k in old_classes})

        # 3: train all skills on the pre-experience memory
        per_skill: dict[int, dict[str, Any]] = {}
        for c in self.class_ids:
            skill = self._skills[str(c)]
            negs = {k: v for k, v in by_class.items() if k != c}
            per_skill[c] = skill.update(by_class.get(c), negs, step_seed=step, device=device)
            per_skill[c]["is_new_skill"] = c in new_classes
            # This counts newly known negative classes, not every class whose current
            # examples were used. ``negatives_used_per_class`` records actual training use.
            per_skill[c]["newly_known_negative_classes"] = sorted(
                k for k in negs if k not in skill.negative_memory
            )

        # 4: commit capped exemplars. A new exemplar tensor is shared on first assignment,
        # Later memory merges can create per-skill tensors; replay storage is not globally shared.
        cap = self.config.memory_per_class
        new_ex = {c: cap_samples(by_class[c], cap, self.seed + 31 * c) for c in classes}
        for c in classes:
            self.exemplars[c] = new_ex[c]
        for c in self.class_ids:
            skill = self._skills[str(c)]
            skill.remember(
                positive=new_ex.get(c),
                negatives={k: v for k, v in new_ex.items() if k != c},
            )

        rec = {
            "index": step,
            "classes": classes,
            "new_skills": new_classes,
            "updated_old_skills": [c for c in old_classes],
            "number_of_skills": self.num_skills,
            "known_negatives": {c: self._skills[str(c)].known_negatives for c in self.class_ids},
            "replay_count": {
                c: int(
                    (
                        0
                        if self._skills[str(c)].positive_memory is None
                        else len(self._skills[str(c)].positive_memory)
                    )
                    + sum(len(v) for v in self._skills[str(c)].negative_memory.values())
                )
                for c in self.class_ids
            },
            "skills": per_skill,
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": sum(p.numel() for p in self.skill_parameters()),
        }
        self.history.append(rec)
        return rec

    def skill_parameters(self) -> Iterable[nn.Parameter]:
        return self._skills.parameters()
