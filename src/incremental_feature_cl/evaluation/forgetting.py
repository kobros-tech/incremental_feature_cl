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


def running_forgetting(values: np.ndarray, first_exp: dict[int, int]) -> np.ndarray:
    """``out[t, c] = max_{first(c) <= s <= t} values[s, c] - values[t, c]`` (NaN if undefined).

    The per-experience version of :func:`forgetting_per_class` (whose last row it equals when the
    final experience is excluded from the peak).  ``0`` right after a class appears and when a
    class is at its best; positive when it has lost ground since its best experience.
    """
    T, C = values.shape
    out = np.full((T, C), np.nan)
    for c, f in first_exp.items():
        peak = -np.inf
        for t in range(f, T):
            v = values[t, c]
            if np.isnan(v):
                continue
            peak = max(peak, v)
            out[t, c] = peak - v
    return out


def mean_running_forgetting(values: np.ndarray, first_exp: dict[int, int]) -> np.ndarray:
    """Mean over classes of :func:`running_forgetting` for every experience (NaN if none)."""
    r = running_forgetting(values, first_exp)
    return np.array([np.nanmean(row) if np.any(~np.isnan(row)) else np.nan for row in r])
