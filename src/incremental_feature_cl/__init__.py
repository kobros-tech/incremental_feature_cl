# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Incremental feature-space continual learning.

Core object::

    Phi_{t+1}(x) = [Phi_t(x), psi_{t+1}(x)],     W_{t+1} = [W_t, 0]

Layout: mathematical reference (``math``), model (``models``), plain-PyTorch
training (``training``), optional Avalanche adapter (``avalanche``),
evaluation, plotting and experiments.
"""

__version__ = "0.1.0"

from .models import ExpansionRecord, IncrementalFeatureMapModel, build_backbone

__all__ = ["ExpansionRecord", "IncrementalFeatureMapModel", "__version__", "build_backbone"]
