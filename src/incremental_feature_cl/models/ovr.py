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

Scores are raw logits of independently trained target-vs-rest skills, so by themselves they are
not comparable across skills and ``argmax`` over them is only a weak multiclass readout.  The
optional :class:`ComparisonConfig` adds a *comparison phase* after the skill updates: all skills
are trained jointly on the replay exemplars of every class so that their scores become
comparable (softmax cross-entropy over the skills) while a per-skill binary loss keeps each
score a calibrated target-vs-rest logit.  Its default is off.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .skill import Skill, SkillBase, SkillConfig, cap_samples

SkillFactory = Callable[[int, int, SkillConfig, int], SkillBase]


@dataclass
class ComparisonConfig:
    """Joint training of all skills on the replay exemplars (see the module docstring).

    ``weight`` scales the softmax cross-entropy over the skills' scores (``0`` disables the whole
    phase).  ``bce_weight`` scales a per-skill binary cross-entropy on the same batches (every
    skill against all other classes, ``pos_weight = C - 1`` so positives and negatives weigh
    equally) that keeps each score a target-vs-rest logit.  The optimizer, momentum and weight
    decay are those of the skills; ``lr`` defaults to the skills' ``lr``.
    """

    weight: float = 0.0
    bce_weight: float = 1.0
    epochs: int = 1
    batch_size: int = 64
    lr: float | None = None

    def __post_init__(self) -> None:
        if self.weight < 0 or self.bce_weight < 0:
            raise ValueError("comparison weights must be >= 0")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("comparison epochs and batch_size must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.weight > 0


class OneVsRestSkillModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int | None = None,
        config: SkillConfig | None = None,
        seed: int = 0,
        *,
        comparison: ComparisonConfig | None = None,
        skill_factory: SkillFactory | None = None,
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
        self.comparison = comparison or ComparisonConfig()
        self.skill_factory: SkillFactory = skill_factory or Skill
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
        3. update EVERY skill (old and new) on new data + its own replay memory, so each skill
           trains its class against old negatives (replay) and new negatives (this experience);
        4. commit exemplars of this experience's classes to every skill's replay memory;
        5. optional comparison phase (``ComparisonConfig.weight > 0``): joint training of all
           skills on the exemplars so that their scores can be compared.
        """
        device = torch.device(device) if device is not None else self._device_ref.device
        y = torch.as_tensor(y, dtype=torch.long).cpu().reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
        if len(y) == 0:
            raise ValueError("update() needs at least one sample")
        step = len(self.history)
        h = self.extract(x.to(device)).cpu()
        classes = sorted(int(c) for c in y.unique())
        by_class = {c: h[y == c] for c in classes}

        # 1-2: new skills
        new_classes = [c for c in classes if str(c) not in self._skills]
        old_classes = list(self.class_ids)
        for c in new_classes:
            skill = self.skill_factory(c, self.feature_dim, self.config, self.seed * 100_003 + c)
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
        # later memory merges can create per-skill tensors; replay storage is not globally shared.
        cap = self.config.memory_per_class
        new_ex = {c: cap_samples(by_class[c], cap, self.seed + 31 * c) for c in classes}
        for c in classes:
            # a class that re-appears keeps its old exemplars (merged and capped, like a skill's)
            self.exemplars[c] = (
                cap_samples(torch.cat([self.exemplars[c], new_ex[c]]), cap, self.seed + 31 * c)
                if c in self.exemplars
                else new_ex[c]
            )
        for c in self.class_ids:
            self._skills[str(c)].remember(
                positive=new_ex.get(c),
                negatives={k: v for k, v in new_ex.items() if k != c},
            )

        # 5: comparison phase
        comparison = self._comparison_phase(device, step)

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
            "comparison": comparison,
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": sum(p.numel() for p in self.skill_parameters()),
        }
        self.history.append(rec)
        return rec

    def skill_parameters(self) -> Iterable[nn.Parameter]:
        return self._skills.parameters()

    # ------------------------------------------------------------------ #
    # comparison phase
    # ------------------------------------------------------------------ #
    def _comparison_phase(self, device: torch.device, step: int) -> dict[str, Any]:
        """Jointly train all skills so that their scores are comparable (see ComparisonConfig)."""
        cmp = self.comparison
        if not cmp.enabled:
            return {"ran": False, "reason": "disabled"}
        skills = [self._skills[str(c)] for c in self.class_ids]
        if not all(isinstance(sk, Skill) for sk in skills):
            return {"ran": False, "reason": "skills are not torch skills"}
        if len(skills) < 2:
            return {"ran": False, "reason": "needs at least two skills"}
        pools = [
            (j, self.exemplars[c]) for j, c in enumerate(self.class_ids) if len(self.exemplars[c])
        ]
        if len({j for j, _ in pools}) < 2:
            return {"ran": False, "reason": "no replay exemplars (memory_per_class=0)"}
        xs = torch.cat([e for _, e in pools])
        ys = torch.cat([torch.full((len(e),), j, dtype=torch.long) for j, e in pools])
        n_skills = len(skills)

        params = [p for sk in skills for p in sk.model.parameters() if p.requires_grad]
        lr = cmp.lr if cmp.lr is not None else self.config.lr
        if self.config.optimizer == "adam":
            opt = torch.optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)
        else:
            opt = torch.optim.SGD(
                params, lr=lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay
            )
        pos_weight = torch.tensor(float(n_skills - 1), device=device)
        generator = torch.Generator().manual_seed(self.seed * 10_009 + step)

        def joint_loss(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch_scores = torch.stack(
                [sk.decision_function(xs[idx].to(device)) for sk in skills], dim=1
            )
            target = ys[idx].to(device)
            ce = F.cross_entropy(batch_scores, target)
            bce = F.binary_cross_entropy_with_logits(
                batch_scores,
                F.one_hot(target, n_skills).float(),
                pos_weight=pos_weight,
            )
            return cmp.weight * ce + cmp.bce_weight * bce, ce.detach()

        for sk in skills:
            sk.model.train()
        first_ce = last_ce = float("nan")
        for epoch in range(cmp.epochs):
            permutation = torch.randperm(len(xs), generator=generator)
            for start in range(0, len(xs), cmp.batch_size):
                loss, ce = joint_loss(permutation[start : start + cmp.batch_size])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if epoch == 0 and start == 0:
                    first_ce = float(ce)
                last_ce = float(ce)
        return {
            "ran": True,
            "n_samples": len(xs),
            "n_skills": n_skills,
            "cross_entropy_first_batch": first_ce,
            "cross_entropy_last_batch": last_ce,
        }
