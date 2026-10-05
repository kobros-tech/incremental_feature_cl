# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Explicit polynomial feature maps (MIT 6.86x Lecture 6).

With ``kernel_scaled=True`` the map satisfies exactly

    phi(x) . phi(z) = (1 + x . z) ** degree

(the lecture's degree-2 map is ``(1, sqrt2 x1, sqrt2 x2, x1^2, x2^2, sqrt2 x1 x2)``).
With ``kernel_scaled=False`` it returns plain monomials, e.g. ``x -> [x, x^2, x^3]``.

The map is *block structured by degree*::

    Phi_p(x) = [Phi_{p-1}(x), psi_p(x)],   psi_p = all monomials of exactly degree p

which is the same incremental structure used by the neural model: the old
representation is kept and a new block is appended.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial, sqrt

import numpy as np


def _exponent_tuples(n_features: int, k: int) -> list[tuple[int, ...]]:
    """All exponent vectors alpha with |alpha| == k (deterministic order)."""
    out = []
    for combo in combinations_with_replacement(range(n_features), k):
        alpha = [0] * n_features
        for i in combo:
            alpha[i] += 1
        out.append(tuple(alpha))
    return out


class PolynomialFeatureMap:
    """Explicit polynomial feature map up to ``degree``.

    Args:
        degree: maximum monomial degree (>= 1).
        include_bias: include the constant feature (degree-0 block).
        kernel_scaled: scale monomials by sqrt(multinomial coefficient) so that
            ``phi(x) . phi(z) == (1 + x . z) ** degree`` (requires ``include_bias``).
    """

    def __init__(self, degree: int, include_bias: bool = True, kernel_scaled: bool = False):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if kernel_scaled and not include_bias:
            raise ValueError("kernel_scaled=True requires include_bias=True")
        self.degree = degree
        self.include_bias = include_bias
        self.kernel_scaled = kernel_scaled

    def _coef(self, alpha: tuple[int, ...], p: int) -> float:
        if not self.kernel_scaled:
            return 1.0
        k = sum(alpha)
        denom = factorial(p - k)
        for a in alpha:
            denom *= factorial(a)
        return sqrt(factorial(p) / denom)

    def degree_block(self, X, k: int) -> np.ndarray:
        """psi_k(X): monomials of exactly degree ``k`` (k=0 -> constant column)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2:
            raise ValueError("X must be (n_samples, n_features)")
        n, d = X.shape
        if k == 0:
            return np.ones((n, 1))
        cols = []
        for alpha in _exponent_tuples(d, k):
            mono = np.ones(n)
            for j, a in enumerate(alpha):
                if a:
                    mono = mono * X[:, j] ** a
            cols.append(self._coef(alpha, self.degree) * mono)
        return np.stack(cols, axis=1)

    def blocks(self, X) -> list[np.ndarray]:
        start = 0 if self.include_bias else 1
        return [self.degree_block(X, k) for k in range(start, self.degree + 1)]

    def __call__(self, X) -> np.ndarray:
        """Phi(X) = [psi_0, psi_1, ..., psi_degree] concatenated."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:  # 1-D inputs: a vector of scalars
            X = X[:, None]
        return np.concatenate(self.blocks(X), axis=1)

    def output_dim(self, n_features: int) -> int:
        start = 0 if self.include_bias else 1
        return sum(
            len(_exponent_tuples(n_features, k)) if k else 1 for k in range(start, self.degree + 1)
        )


def quadratic_map_1d(x) -> np.ndarray:
    """Lecture 6 demo: phi(x) = (x, x^2)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, x**2])
