# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Optional Avalanche adapter.  Importing this subpackage requires ``avalanche-lib``."""

from .baselines import AVALANCHE_BASELINES, make_avalanche_baseline
from .benchmark import make_split_cifar100, make_tensor_benchmark
from .metrics import FeatureSpaceLogger
from .strategy import FeatureExpansionPlugin, IncrementalFeatureSpaceStrategy, quiet_evaluator

__all__ = [
    "AVALANCHE_BASELINES",
    "FeatureExpansionPlugin",
    "FeatureSpaceLogger",
    "IncrementalFeatureSpaceStrategy",
    "make_avalanche_baseline",
    "make_split_cifar100",
    "make_tensor_benchmark",
    "quiet_evaluator",
]
