# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Forgetting and summary metrics from the (experience x class) accuracy matrix."""

from __future__ import annotations

import numpy as np


def forgetting_per_class(acc: np.ndarray, first_exp: dict[int, int]) -> np.ndarray:
    """forgetting[c] = max_{first(c) <= t < T-1} acc[t,c] - acc[T-1,c]   (NaN if undefined)."""
    T, C = acc.shape
    out = np.full(C, np.nan)
    for c, f in first_exp.items():
        if f < T - 1 and not np.isnan(acc[T - 1, c]):
            prev = acc[f : T - 1, c]
            prev = prev[~np.isnan(prev)]
            if len(prev):
                out[c] = prev.max() - acc[T - 1, c]
    return out


def average_forgetting(acc: np.ndarray, first_exp: dict[int, int]) -> float:
    f = forgetting_per_class(acc, first_exp)
    return float(np.nanmean(f)) if np.any(~np.isnan(f)) else float("nan")
