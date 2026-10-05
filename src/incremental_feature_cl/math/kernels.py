# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Kernel functions. All take row-sample matrices X (n,d), Z (m,d) and return (n,m).

Note: the lecture notebooks write ``X @ Z`` for single vectors; here matrices are
row-sample, so the Gram matrix is ``X @ Z.T``.
"""

from __future__ import annotations

import numpy as np


def linear_kernel(X, Z) -> np.ndarray:
    return np.atleast_2d(X) @ np.atleast_2d(Z).T


def polynomial_kernel(X, Z, degree: int = 2, coef0: float = 1.0) -> np.ndarray:
    """K(x,z) = (coef0 + x.z) ** degree  (lecture: coef0 = 1)."""
    return (coef0 + linear_kernel(X, Z)) ** degree


def rbf_kernel(X, Z, gamma: float = 1.0) -> np.ndarray:
    """K(x,z) = exp(-gamma * ||x - z||^2)."""
    X, Z = np.atleast_2d(X), np.atleast_2d(Z)
    sq = (X**2).sum(1)[:, None] + (Z**2).sum(1)[None, :] - 2.0 * X @ Z.T
    return np.exp(-gamma * np.maximum(sq, 0.0))
