# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Shared types for expansion bookkeeping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

INITIALIZATIONS = ("zero", "random")


def check_initialization(name: str) -> str:
    if name not in INITIALIZATIONS:
        raise ValueError(f"initialization must be one of {INITIALIZATIONS}, got {name!r}")
    return name


@dataclass
class ExpansionRecord:
    """One expansion event (feature space *or* classifier outputs).

    ``kind='feature'``: ``old_feature_dim -> old_feature_dim + new_feature_dim``.
    ``kind='output'`` : ``old_num_outputs -> new_num_outputs`` (feature dim unchanged).
    """

    event: int
    kind: str
    initialization: str
    old_feature_dim: int
    new_feature_dim: int
    total_feature_dim: int
    old_num_outputs: int
    new_num_outputs: int
    number_of_new_parameters: int
    old_parameter_count: int
    total_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Expandable(Protocol):
    """Anything whose feature space can grow without being rebuilt."""

    @property
    def feature_dim(self) -> int: ...

    def expand_feature_space(
        self, new_dim: int, initialization: str | None = None
    ) -> ExpansionRecord: ...
