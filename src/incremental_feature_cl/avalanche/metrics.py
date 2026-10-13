# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Avalanche plugin logging feature growth / block usage after every training experience."""

from __future__ import annotations

from typing import Any

from avalanche.training.plugins import SupervisedPlugin


class FeatureSpaceLogger(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def after_training_exp(self, strategy, **kwargs):
        m = strategy.model
        self.rows.append(
            {
                "experience": strategy.experience.current_experience,
                "feature_dim": m.feature_dim,
                "num_outputs": m.num_outputs,
                "parameter_count": m.parameter_count(),
                "classifier_block_norms": m.classifier_block_norms(),
            }
        )
