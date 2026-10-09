# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Optional Avalanche adapter.  Importing this subpackage requires ``avalanche-lib``."""

from .strategy import OneVsRestSkillStrategy, quiet_evaluator

__all__ = ["OneVsRestSkillStrategy", "quiet_evaluator"]
