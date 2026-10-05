# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""When and how the model grows.  Shared by the plain trainer and the Avalanche plugin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..models import ExpansionRecord, IncrementalFeatureMapModel


@dataclass
class ExpansionPolicy:
    """Grow the feature space by ``new_feature_dim`` at every experience boundary.

    ``new_feature_dim=0`` disables feature growth (the fixed-dimension baseline).
    Feature growth starts at experience 1 (experience 0 trains the initial
    representation) unless ``expand_first=True``.
    Output growth (``grow_outputs``) is independent of feature growth.
    """

    new_feature_dim: int = 0
    initialization: str = "zero"
    expand_first: bool = False
    grow_outputs: bool = True
    single_output: bool = False  # target-vs-rest: one binary logit, never grown

    def should_expand_features(self, exp_index: int) -> bool:
        return self.new_feature_dim > 0 and (exp_index > 0 or self.expand_first)

    def expand_features(
        self, model: IncrementalFeatureMapModel, exp_index: int
    ) -> ExpansionRecord | None:
        if not self.should_expand_features(exp_index):
            return None
        return model.expand_feature_space(self.new_feature_dim, self.initialization)

    def expand_outputs(
        self, model: IncrementalFeatureMapModel, classes: Sequence[int]
    ) -> ExpansionRecord | None:
        if not self.grow_outputs or self.single_output or not len(classes):
            return None
        return model.ensure_outputs(max(int(c) for c in classes) + 1)

    def on_experience_start(
        self, model: IncrementalFeatureMapModel, exp_index: int, classes: Sequence[int]
    ) -> list[ExpansionRecord]:
        recs = [self.expand_features(model, exp_index), self.expand_outputs(model, classes)]
        return [r for r in recs if r is not None]
