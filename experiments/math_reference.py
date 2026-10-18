# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Small NumPy checks for zero-initialised expansion and a polynomial feature map."""

from __future__ import annotations

import numpy as np


def polynomial_features(X: np.ndarray) -> np.ndarray:
    """Explicit degree-two feature map whose dot product equals (1 + x dot z)^2."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    quadratic = np.column_stack(
        [np.sqrt(2.0) * X[:, i] * X[:, j] if i != j else X[:, i] ** 2 for i, j in pairs]
    )
    return np.column_stack([np.ones(n), np.sqrt(2.0) * X, quadratic])


def main() -> None:
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    old_weights = rng.normal(size=X.shape[1])
    old_bias = 0.2
    new_features = rng.normal(size=(len(X), 5))
    expanded_weights = np.r_[old_weights, np.zeros(new_features.shape[1])]
    old_scores = X @ old_weights + old_bias
    expanded_scores = np.column_stack([X, new_features]) @ expanded_weights + old_bias
    max_diff = float(np.max(np.abs(old_scores - expanded_scores)))
    if max_diff > 1e-12:
        raise AssertionError(f"Expansion changed scores: max_abs_diff={max_diff}")

    Z = rng.normal(size=(20, 3))
    lhs = polynomial_features(X) @ polynomial_features(Z).T
    rhs = (1.0 + X @ Z.T) ** 2
    kernel_error = float(np.max(np.abs(lhs - rhs)))
    if kernel_error > 1e-10:
        raise AssertionError(f"Polynomial identity failed: max_abs_error={kernel_error}")
    print(f"zero-initialised expansion max_abs_diff={max_diff:.3g}")
    print(f"degree-two polynomial feature identity max_abs_error={kernel_error:.3g}")


if __name__ == "__main__":
    main()
