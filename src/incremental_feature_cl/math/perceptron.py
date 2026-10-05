# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Reference perceptrons from Lecture 6.

* ``Perceptron``       - feature-space perceptron whose feature space can be
                         *expanded*: theta_{t+1} = [theta_t, 0].  This is the
                         exact mathematical analogue of the neural model.
* ``KernelPerceptron`` - the dual (kernel) form; never builds phi explicitly.

Labels are in {-1, +1}; a score >= 0 predicts +1 (as in the lecture notebooks).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

FeatureFn = Callable[[np.ndarray], np.ndarray]


def _identity_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


class Perceptron:
    """Perceptron over a block-structured feature space ``[phi_0(x), phi_1(x), ...]``.

    ``add_feature_block(fn)`` appends ``fn(x)`` to the representation and
    appends *zeros* to ``theta`` so ``decision_function`` is unchanged.
    """

    def __init__(self, feature_fn: FeatureFn | None = None, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.feature_blocks: list[FeatureFn] = [feature_fn or _identity_features]
        self.thetas: list[np.ndarray | None] = [None]  # lazily sized
        self.intercept = 0.0
        self.n_updates = 0

    # --- representation ------------------------------------------------- #
    def _blocks(self, X) -> list[np.ndarray]:
        return [np.asarray(f(X), dtype=float) for f in self.feature_blocks]

    def features(self, X) -> np.ndarray:
        return np.concatenate(self._blocks(X), axis=1)

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([t for t in self.thetas if t is not None])

    def add_feature_block(self, fn: FeatureFn, X_probe) -> int:
        """Expand: Phi' = [Phi, fn(x)], theta' = [theta, 0]. Returns new block dim."""
        dim = np.asarray(fn(np.asarray(X_probe)[:1]), dtype=float).shape[1]
        self.feature_blocks.append(fn)
        self.thetas.append(np.zeros(dim))
        return dim

    # --- learning ---------------------------------------------------------- #
    def decision_function(self, X) -> np.ndarray:
        blocks = self._blocks(X)
        score = np.full(len(blocks[0]), self.intercept, dtype=float)
        for b, t in zip(blocks, self.thetas):
            if t is not None:
                score = score + b @ t
        return score

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def fit(self, X, y, epochs: int = 100) -> Perceptron:
        y = np.asarray(y)
        blocks = self._blocks(X)
        for i, (b, t) in enumerate(zip(blocks, self.thetas)):
            if t is None:
                self.thetas[i] = np.zeros(b.shape[1])
        for _ in range(epochs):
            mistakes = 0
            for i in range(len(y)):
                score = self.intercept + sum(b[i] @ t for b, t in zip(blocks, self.thetas))
                if y[i] * score <= 0:  # lecture rule: update when y * score <= 0
                    for b, t in zip(blocks, self.thetas):
                        t += y[i] * b[i]
                    if self.fit_intercept:
                        self.intercept += y[i]
                    mistakes += 1
                    self.n_updates += 1
            if mistakes == 0:
                break
        return self


class KernelPerceptron:
    """Dual perceptron: score(x) = sum_j alpha_j y_j K(x_j, x)  (no intercept; as in the lecture)."""

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.kernel = kernel
        self.alpha: np.ndarray | None = None
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X, y, epochs: int = 100) -> KernelPerceptron:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        K = self.kernel(X, X)
        alpha = np.zeros(len(y), dtype=int)
        for _ in range(epochs):
            mistakes = 0
            for i in range(len(y)):
                score = np.sum(alpha * y * K[:, i])
                if y[i] * score <= 0:
                    alpha[i] += 1
                    mistakes += 1
            if mistakes == 0:
                break
        self.alpha, self.X_, self.y_ = alpha, X, y
        return self

    def decision_function(self, X) -> np.ndarray:
        assert self.alpha is not None, "call fit first"
        return self.kernel(self.X_, np.asarray(X, dtype=float)).T @ (self.alpha * self.y_)

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)
