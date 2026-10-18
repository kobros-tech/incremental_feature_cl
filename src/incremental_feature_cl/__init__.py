# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Binary linear classifier with zero-initialised incremental feature expansion."""

from .model import FeatureBlock, IncrementalFeatureMapModel

__version__ = "0.3.0"
__all__ = ["FeatureBlock", "IncrementalFeatureMapModel", "__version__"]
