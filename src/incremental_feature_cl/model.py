# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Binary linear classifier over an expandable feature space (scikit-learn estimator).

    Phi_t(x) = [x, psi_1(x), ..., psi_t(x)]        score(x) = b + theta_t . Phi_t(x)

``expand_features()`` appends a block ``psi_{t+1}`` and zeros to ``theta``:

    Phi_{t+1} = [Phi_t, psi_{t+1}],   theta_{t+1} = [theta_t, 0]

so ``decision_function`` is unchanged at the expansion boundary.

One ``partial_fit`` call is one experience.  After the first call it

 1. expands the feature space by ``new_feature_dim`` (zero-initialised),
 2. keeps ``floor(n_target / target_to_negatives)`` negatives of the call's data,
 3. trains ``n_epochs`` passes of minibatch SGD with momentum on the weighted logistic loss; every
    minibatch of ``batch_size`` current samples is accompanied by up to ``batch_size`` samples
    replayed from the memory,
 4. stores the data it trained on in the replay memory.

``fit`` trains once without expansion (its data still enters the memory).  The positive class is
``classes_[1]``.  Multiclass problems use scikit-learn::

    OneVsRestClassifier(IncrementalFeatureMapModel(new_feature_dim=16))
"""

from __future__ import annotations

import math
import numbers
from typing import ClassVar

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class FeatureBlock:
    """psi(x) = relu(x A + c), ``A`` and ``c`` drawn once from U(-1/sqrt(in), 1/sqrt(in)).

    This is ``torch.nn.Linear``'s default initialisation.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.RandomState):
        bound = 1.0 / math.sqrt(in_dim)
        self.A = rng.uniform(-bound, bound, (in_dim, out_dim))
        self.c = rng.uniform(-bound, bound, out_dim)

    @property
    def out_dim(self) -> int:
        return self.c.shape[0]

    def pre(self, X: np.ndarray) -> np.ndarray:
        return X @ self.A + self.c

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(self.pre(X), 0.0)


class IncrementalFeatureMapModel(ClassifierMixin, BaseEstimator):
    """Binary classifier with an incrementally expandable, zero-initialised feature space.

    Parameters
    ----------
    new_feature_dim : int, default=0
        Width of the feature block appended before every ``partial_fit`` call after the first.
        ``0`` keeps the feature space fixed.
    target_to_negatives : float or None, default=None
        Targets per negative in the call's data (``1.0`` = 1:1, ``0.2`` = 1 target : 5
        negatives, ``2.0`` = 2:1).  Negatives are drawn without replacement, so the ratio is an
        upper bound on negatives per target: ``floor(n_target / ratio)`` negatives are kept, or all
        of them if fewer exist.  ``None`` keeps all negatives.  Applied to each call's own data
        (before replay); a call without any target sample keeps all its negatives.
    pos_weight : "balanced", float or None, default="balanced"
        Weight of a target sample in the loss (negatives weigh 1).  ``"balanced"`` uses
        ``n_negative / n_target`` of the call's data after the ratio is applied; the same weight is
        used for replayed targets.  Scores then reflect the re-weighted problem: ``predict_proba``
        is not calibrated to the original class prior.
    replay_mem_size : int, default=200
        Total replay memory, split equally between target and non-target samples (the binary form
        of a class-balanced buffer).  Each class keeps a reservoir sample, so every sample trained
        on so far is equally likely to be stored.  From the second ``partial_fit`` call on, every
        minibatch of ``batch_size`` current samples is accompanied by ``min(batch size, memory)``
        memory samples drawn without replacement.  ``0`` switches replay off, otherwise at least 2.
    train_blocks : bool, default=True
        Train the appended feature blocks (their ``A`` and ``c``).  ``False`` keeps them as fixed
        random features.
    learning_rate_init, momentum : float, default=0.01, 0.9
        SGD step size and momentum.  The momentum buffers start from zero in every call.
    batch_size : int, default=32
        Minibatch size of the current data.
    alpha : float, default=0.0
        L2 penalty (weight decay) on every trained parameter.
    n_epochs : int, default=5
        Passes over the call's data.
    random_state : int, RandomState or None, default=None
        Seeds the feature blocks, negative subsampling, replay and minibatch shuffling.

    Notes
    -----
    Scale the features (e.g. ``StandardScaler``) before use; the optimiser is plain SGD.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    blocks_ : list of FeatureBlock
    feature_dim_ : int
        Current width of ``Phi_t`` (input features plus all appended blocks).
    coef_ : ndarray of shape (1, feature_dim_)
    intercept_ : ndarray of shape (1,)
    """

    _parameter_constraints: ClassVar[dict] = {
        "new_feature_dim": [Interval(numbers.Integral, 0, None, closed="left")],
        "target_to_negatives": [Interval(numbers.Real, 0, None, closed="neither"), None],
        "pos_weight": [
            StrOptions({"balanced"}),
            Interval(numbers.Real, 0, None, closed="neither"),
            None,
        ],
        "replay_mem_size": [
            Interval(numbers.Integral, 2, None, closed="left"),
            Options(numbers.Integral, {0}),
        ],
        "train_blocks": ["boolean"],
        "learning_rate_init": [Interval(numbers.Real, 0, None, closed="neither")],
        "momentum": [Interval(numbers.Real, 0, 1, closed="left")],
        "batch_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "n_epochs": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        new_feature_dim=0,
        target_to_negatives=None,
        pos_weight="balanced",
        replay_mem_size=200,
        train_blocks=True,
        learning_rate_init=0.01,
        momentum=0.9,
        batch_size=32,
        alpha=0.0,
        n_epochs=5,
        random_state=None,
    ):
        self.new_feature_dim = new_feature_dim
        self.target_to_negatives = target_to_negatives
        self.pos_weight = pos_weight
        self.replay_mem_size = replay_mem_size
        self.train_blocks = train_blocks
        self.learning_rate_init = learning_rate_init
        self.momentum = momentum
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False  # binary only; use OneVsRestClassifier
        return tags

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Train from scratch: one stage, no feature expansion."""
        self._reset()
        return self._partial_fit(X, y, classes=None, first=True)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None):
        """Train on one experience.  Expands by ``new_feature_dim`` unless it is the first call."""
        first = not hasattr(self, "classes_")
        if first:
            self._reset()
        return self._partial_fit(X, y, classes=classes, first=first)

    def expand_features(self, new_dim: int | None = None) -> int:
        """Append a block of ``new_dim`` features and zeros to ``coef_``; return the new width."""
        check_is_fitted(self, "classes_")
        d = self.new_feature_dim if new_dim is None else int(new_dim)
        if d < 0:
            raise ValueError("new_dim must be >= 0")
        if d > 0:
            self.blocks_.append(FeatureBlock(self.n_features_in_, d, self._rng))
            self.coef_ = np.hstack([self.coef_, np.zeros((1, d))])
            self.feature_dim_ += d
        return self.feature_dim_

    def _reset(self) -> None:
        for name in (
            "classes_",
            "blocks_",
            "coef_",
            "intercept_",
            "feature_dim_",
            "_rng",
            "_mem",
            "_mem_seen",
            "_balanced_pos_weight_",
        ):
            self.__dict__.pop(name, None)

    def _partial_fit(self, X, y, classes, first):
        X, y = validate_data(self, X, y, reset=first, dtype=np.float64)
        check_classification_targets(y)
        if first:
            self._rng = check_random_state(self.random_state)
            self.classes_ = np.unique(y if classes is None else classes)
            if len(self.classes_) != 2:
                raise ValueError(
                    "Only binary classification is supported. The type of the target is "
                    f"multiclass ({len(self.classes_)} classes); wrap the estimator in "
                    "sklearn.multiclass.OneVsRestClassifier."
                )
            self.blocks_ = []
            self._mem = [[], []]  # raw samples of [non-target, target]
            self._mem_seen = [0, 0]  # how many samples each reservoir has been offered
            self.coef_ = np.zeros((1, self.n_features_in_))  # zero initialisation
            self.intercept_ = np.zeros(1)
            self.feature_dim_ = self.n_features_in_
        if not np.isin(y, self.classes_).all():
            raise ValueError(f"y contains labels outside classes_={self.classes_.tolist()}")

        pos = y == self.classes_[1]
        keep = self._select_negatives(pos)
        X, pos = X[keep], pos[keep]
        weight = self._pos_weight(pos)

        if not first:
            self.expand_features()
        self._train(X, pos, weight, *self._replay_pool())
        self._update_memory(X, pos)
        return self

    # ------------------------------------------------------------------ #
    # optimisation (plain NumPy: minibatch SGD with momentum on the weighted logistic loss)
    # ------------------------------------------------------------------ #
    def _train(self, X, pos, weight, Xm, pm):
        params = self._params()
        velocity = [np.zeros_like(p) for p in params]
        n, b = len(X), self.batch_size
        for _ in range(self.n_epochs):
            order = self._rng.permutation(n)
            for start in range(0, n, b):
                idx = order[start : start + b]
                xb, pb = X[idx], pos[idx]
                if Xm is not None:  # up to as many replayed samples as current ones
                    j = self._rng.choice(len(Xm), min(len(idx), len(Xm)), replace=False)
                    xb, pb = np.vstack([xb, Xm[j]]), np.concatenate([pb, pm[j]])
                grads = self._gradients(xb, pb, weight)
                for p, v, g in zip(params, velocity, grads):
                    v *= self.momentum
                    v += g
                    p -= self.learning_rate_init * v

    def _params(self) -> list[np.ndarray]:
        """Arrays updated in place, in the order ``_gradients`` returns their gradients."""
        params = [self.coef_[0], self.intercept_]
        if self.train_blocks:
            params += [a for blk in self.blocks_ for a in (blk.A, blk.c)]
        return params

    def _gradients(self, X, pos, weight) -> list[np.ndarray]:
        """Gradient of mean_i w_i * logloss_i (+ alpha/2 |p|^2) w.r.t. ``_params()``."""
        w = self.coef_[0]
        pres = [blk.pre(X) for blk in self.blocks_]
        Phi = np.hstack([X] + [np.maximum(p, 0.0) for p in pres])
        sw = 1.0 if weight is None else np.where(pos, weight, 1.0)
        dz = sw * (expit(Phi @ w + self.intercept_[0]) - pos) / len(X)
        grads = [Phi.T @ dz, np.array([dz.sum()])]
        if self.train_blocks:
            offset = self.n_features_in_
            for blk, pre in zip(self.blocks_, pres):
                dpre = np.outer(dz, w[offset : offset + blk.out_dim]) * (pre > 0)
                grads += [X.T @ dpre, dpre.sum(0)]
                offset += blk.out_dim
        if self.alpha:
            grads = [g + self.alpha * p for g, p in zip(grads, self._params())]
        return grads

    # ------------------------------------------------------------------ #
    # ratio, weights, replay memory
    # ------------------------------------------------------------------ #
    def _select_negatives(self, pos: np.ndarray) -> np.ndarray:
        """Indices kept after applying ``target_to_negatives`` (``floor(n_pos / ratio)`` negatives)."""
        n_pos = int(pos.sum())
        if self.target_to_negatives is None or n_pos == 0:
            return np.arange(len(pos))
        neg_idx = np.flatnonzero(~pos)
        n_req = math.floor(n_pos / self.target_to_negatives + 1e-9)
        if len(neg_idx) and n_req == 0:
            raise ValueError(
                "target_to_negatives requests zero negatives for this batch; use a smaller "
                "ratio or include more target samples."
            )
        if len(neg_idx) > n_req:
            neg_idx = np.sort(self._rng.choice(neg_idx, n_req, replace=False))
        return np.sort(np.concatenate([np.flatnonzero(pos), neg_idx]))

    def _pos_weight(self, pos: np.ndarray):
        if self.pos_weight is None:
            return None
        if isinstance(self.pos_weight, str):  # "balanced"
            n_pos = int(pos.sum())
            n_neg = int((~pos).sum())
            if n_pos > 0 and n_neg > 0:
                self._balanced_pos_weight_ = n_neg / n_pos
                return self._balanced_pos_weight_
            # Before both classes have appeared together there is no useful ratio. Once
            # learned, retain the last valid weight so replayed positives stay weighted
            # during later negative-only experiences.
            return getattr(self, "_balanced_pos_weight_", 1.0)
        return float(self.pos_weight)

    def _replay_pool(self):
        """Raw samples and labels of the replay memory, or ``(None, None)``."""
        rows = [np.asarray(m) for m in self._mem]
        if not any(len(r) for r in rows):
            return None, None
        X = np.vstack([r for r in rows if len(r)])
        pos = np.concatenate([np.full(len(r), k == 1) for k, r in enumerate(rows) if len(r)])
        return X, pos

    def _update_memory(self, X, pos):
        """Reservoir sampling per class: every sample trained on is equally likely to be kept."""
        cap = self.replay_mem_size // 2
        if cap == 0:
            return
        for k, rows in enumerate((X[~pos], X[pos])):
            mem, seen = self._mem[k], self._mem_seen[k]
            free = max(0, cap - len(mem))
            mem.extend(rows[:free])
            seen += min(free, len(rows))
            rest = rows[free:]
            if len(rest):
                j = (self._rng.random_sample(len(rest)) * (seen + 1 + np.arange(len(rest)))).astype(
                    int
                )
                for i in np.flatnonzero(j < cap):
                    mem[j[i]] = rest[i]
                seen += len(rest)
            self._mem_seen[k] = seen

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def _features(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X] + [blk(X) for blk in self.blocks_])

    def decision_function(self, X) -> np.ndarray:
        """Score over ``Phi_t(x)``; positive means ``classes_[1]``."""
        check_is_fitted(self, "classes_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._features(X) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "classes_")
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
