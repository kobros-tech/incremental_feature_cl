# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Incremental feature-space continual learning.

Core object::

    Phi_{t+1}(x) = [Phi_t(x), psi_{t+1}(x)],     W_{t+1} = [W_t, 0]

Three models, one question: can they learn continuously?

* ``IncrementalFeatureMapModel`` - expandable feature map + classifier (``train_feature_map``);
* ``Skill``                      - one persistent target-vs-rest learner (``train_skill``);
* ``OneVsRestSkillModel``        - one Skill per class (``train_ovr``, Avalanche strategy in
  ``avalanche``).

The shared representation is any external ``nn.Module`` (``models.backbones``), e.g. Avalanche's
``SimpleMLP`` / ``resnet32``.
"""

__version__ = "0.1.0"

from .models import (
    ExpansionRecord,
    FeatureBackbone,
    IncrementalFeatureMapModel,
    OneVsRestSkillModel,
    Skill,
    SkillConfig,
    build_backbone,
)

__all__ = [
    "ExpansionRecord",
    "FeatureBackbone",
    "IncrementalFeatureMapModel",
    "OneVsRestSkillModel",
    "Skill",
    "SkillConfig",
    "__version__",
    "build_backbone",
]
