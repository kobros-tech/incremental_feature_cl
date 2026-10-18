# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from incremental_feature_cl import FeatureBlock
from incremental_feature_cl import IncrementalFeatureMapModel as Model


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=0)
    return StandardScaler().fit_transform(X), y


def record_batches(model):
    """Wrap ``_gradients`` and record (n samples, n targets) of every minibatch step."""
    steps, orig = [], model._gradients
    model._gradients = lambda X, pos, w: (steps.append((len(X), int(pos.sum()))), orig(X, pos, w))[
        1
    ]
    return steps


def stream(n_exp=4, n=100, d=5, seed=0):
    """Experience i has X[:, 0] == i, so the memory shows which experiences it holds."""
    r = np.random.RandomState(seed)
    out = []
    for i in range(n_exp):
        X = r.standard_normal((n, d))
        X[:, 0] = i
        out.append((X, (r.rand(n) < 0.3).astype(int)))
    return out


# ---------------------------------------------------------------------- #
# learning, expansion, zero initialisation
# ---------------------------------------------------------------------- #
def test_learns(binary):
    X, y = binary
    assert Model(random_state=0).fit(X, y).score(X, y) > 0.75


@pytest.mark.parametrize("train_blocks", [True, False])
def test_expansion_is_zero_initialised_and_continuous(binary, train_blocks):
    X, y = binary
    m = Model(new_feature_dim=8, train_blocks=train_blocks, random_state=0).fit(X, y)
    m.partial_fit(X, y)  # now has one trained block
    before, d0 = m.decision_function(X), m.feature_dim_
    m.expand_features()
    assert m.feature_dim_ == d0 + 8 and m.coef_.shape == (1, d0 + 8)
    assert np.all(m.coef_[0, d0:] == 0)
    np.testing.assert_allclose(m.decision_function(X), before)


def test_partial_fit_expands_after_first_call(binary):
    X, y = binary
    m = Model(new_feature_dim=4, random_state=0)
    for i, (a, b) in enumerate([(0, 150), (150, 300), (300, 400)]):
        m.partial_fit(X[a:b], y[a:b], classes=[0, 1])
        assert m.feature_dim_ == 10 + 4 * i
    assert m.score(X, y) > 0.7


def test_fit_resets_expansion(binary):
    X, y = binary
    m = Model(new_feature_dim=4, random_state=0)
    m.partial_fit(X, y).partial_fit(X, y)
    m.fit(X, y)
    assert m.feature_dim_ == 10 and m.blocks_ == []


def test_feature_block_uses_linear_default_init():
    b = FeatureBlock(100, 4000, np.random.RandomState(0))
    bound = 1 / np.sqrt(100)
    assert np.abs(b.A).max() <= bound and np.abs(b.c).max() <= bound
    assert abs(b.A.std() - bound / np.sqrt(3)) < 1e-3  # uniform, like nn.Linear


def test_blocks_train_only_when_asked(binary):
    X, y = binary

    def run(train_blocks):
        m = Model(new_feature_dim=6, train_blocks=train_blocks, random_state=0)
        m.partial_fit(X, y).partial_fit(X, y)
        before = m.blocks_[0].A.copy()
        m.partial_fit(X, y)
        return np.abs(m.blocks_[0].A - before).max()

    assert run(True) > 0
    assert run(False) == 0


# ---------------------------------------------------------------------- #
# optimiser
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("train_blocks", [True, False])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_gradients_match_finite_differences(binary, train_blocks, alpha):
    X, y = binary
    X, pos = X[:40], y[:40] == 1
    m = Model(
        new_feature_dim=5, train_blocks=train_blocks, alpha=alpha, replay_mem_size=0, random_state=0
    )
    m.fit(X, y[:40]).expand_features()
    m.expand_features(3)
    r = np.random.RandomState(1)
    m.coef_[:] = r.randn(*m.coef_.shape) * 0.5
    m.intercept_[:] = 0.3
    for b in m.blocks_:
        b.A += r.randn(*b.A.shape) * 0.3
        b.c += r.randn(*b.c.shape) * 0.3
    weight = 2.5

    def loss():
        p = 1 / (1 + np.exp(-(m._features(X) @ m.coef_[0] + m.intercept_[0])))
        w = np.where(pos, weight, 1.0)
        data = np.mean(w * -(pos * np.log(p) + (~pos) * np.log(1 - p)))
        return data + alpha / 2 * sum((q**2).sum() for q in m._params())

    for p, g in zip(m._params(), m._gradients(X, pos, weight)):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in range(min(len(flat), 6)):
            old, h = flat[i], 1e-6
            flat[i] = old + h
            up = loss()
            flat[i] = old - h
            down = loss()
            flat[i] = old
            assert abs((up - down) / (2 * h) - gflat[i]) < 1e-6


def test_first_step_is_a_plain_gradient_step(binary):
    X, y = binary
    m = Model(
        replay_mem_size=0, batch_size=len(X), n_epochs=1, learning_rate_init=0.05, random_state=0
    )
    m._reset()
    # emulate fit() up to the first step: zero parameters, balanced weight
    m.fit(X[:2], y[:2])  # any fit just to allocate attributes; then overwrite to a known state
    m.coef_[:] = 0.0
    m.intercept_[:] = 0.0
    pos = y == 1
    weight = (~pos).sum() / pos.sum()
    g = [x.copy() for x in m._gradients(X, pos, weight)]
    m._train(X, pos, weight, None, None)
    np.testing.assert_allclose(m.coef_[0], -0.05 * g[0], atol=1e-12)  # momentum buffer starts at 0
    np.testing.assert_allclose(m.intercept_, -0.05 * g[1], atol=1e-12)


def test_deterministic_given_random_state(binary):
    X, y = binary
    kw = {"target_to_negatives": 1.0, "new_feature_dim": 4, "random_state": 3}
    a = Model(**kw).partial_fit(X, y).partial_fit(X, y)
    b = Model(**kw).partial_fit(X, y).partial_fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_)


# ---------------------------------------------------------------------- #
# ratio and pos_weight
# ---------------------------------------------------------------------- #
def test_ratio_keeps_floor_n_pos_over_ratio_negatives():
    pos = np.r_[np.ones(10, bool), np.zeros(100, bool)]
    m = Model(target_to_negatives=0.2)  # 1 target : 5 negatives
    m._rng = np.random.RandomState(0)
    keep = m._select_negatives(pos)
    assert pos[keep].sum() == 10 and (~pos[keep]).sum() == 50


def test_ratio_is_an_upper_bound_on_negatives():
    pos = np.r_[np.ones(10, bool), np.zeros(20, bool)]
    m = Model(target_to_negatives=0.2)  # asks for 50, only 20 exist
    m._rng = np.random.RandomState(0)
    assert len(m._select_negatives(pos)) == 30


def test_ratio_none_keeps_everything():
    pos = np.r_[np.ones(3, bool), np.zeros(7, bool)]
    assert len(Model()._select_negatives(pos)) == 10


def test_ratio_reaches_training(binary):
    X, y = binary
    n_pos = int((y == 1).sum())
    m = Model(target_to_negatives=1.0, n_epochs=2, replay_mem_size=0, random_state=0)
    steps = record_batches(m)
    m.fit(X, y)
    assert (y == 0).sum() > n_pos
    assert sum(n for n, _ in steps) == 2 * n_pos * 2  # 1:1 -> 2 * n_pos samples, 2 epochs


def test_invalid_ratio_rejected(binary):
    X, y = binary
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            Model(target_to_negatives=bad).fit(X, y)


def test_pos_weight_modes_and_validation(binary):
    pos = np.array([True, False, False, False])
    assert Model()._pos_weight(pos) == 3.0  # balanced: n_neg / n_pos
    assert Model(pos_weight=2.5)._pos_weight(pos) == 2.5
    assert Model(pos_weight=None)._pos_weight(pos) is None
    X, y = binary
    for bad in (0, -1.0, "auto"):
        with pytest.raises(ValueError):
            Model(pos_weight=bad).fit(X, y)


# ---------------------------------------------------------------------- #
# replay
# ---------------------------------------------------------------------- #
def test_replay_default_is_on():
    assert Model().get_params()["replay_mem_size"] == 200


def test_replay_can_be_turned_off_but_not_set_to_one():
    X, y = make_classification(n_samples=60, random_state=0)
    for bad in (1, -5):
        with pytest.raises(ValueError):
            Model(replay_mem_size=bad).fit(X, y)
    m = Model(replay_mem_size=0, random_state=0)
    steps = record_batches(m)
    for Xi, yi in stream(n_exp=3, n=100):
        m.partial_fit(Xi, yi, classes=[0, 1])
    assert m._mem == [[], []] and m._mem_seen == [0, 0]
    assert sum(n for n, _ in steps) == 3 * 5 * 100  # only current samples, never replayed ones


def test_memory_is_bounded_and_holds_both_classes():
    m = Model(replay_mem_size=20, random_state=0)
    for X, y in stream():
        m.partial_fit(X, y, classes=[0, 1])
        assert [len(r) for r in m._mem] == [10, 10]
    assert sum(m._mem_seen) == 400


def test_memory_keeps_old_experiences():
    m = Model(replay_mem_size=200, random_state=0)
    for X, y in stream(n_exp=4, n=300):
        m.partial_fit(X, y, classes=[0, 1])
    stored = np.array([r[0] for r in m._mem[0]])  # experience id of every stored non-target
    counts = np.bincount(stored.astype(int), minlength=4) / len(stored)
    assert (counts > 0.1).all()  # reservoir: ~1/4 each, the first experience is not forgotten


def test_replay_rate_is_per_minibatch():
    """Each minibatch of 32 current samples gets min(32, memory) replayed ones."""
    m = Model(replay_mem_size=40, n_epochs=1, random_state=0)
    (X0, y0), (X1, y1) = stream(n_exp=2, n=100)
    steps = record_batches(m)
    m.partial_fit(X0, y0, classes=[0, 1])
    assert [n for n, _ in steps] == [32, 32, 32, 4]  # first call: nothing to replay
    steps.clear()
    m.partial_fit(X1, y1)
    # memory holds 40; batches of 32, 32, 32, 4 current samples -> 32, 32, 32, 4 replayed
    assert [n for n, _ in steps] == [64, 64, 64, 8]


def test_small_memory_replays_all_of_it_each_time():
    m = Model(replay_mem_size=4, n_epochs=1, random_state=0)
    (X0, y0), (X1, y1) = stream(n_exp=2, n=100)
    m.partial_fit(X0, y0, classes=[0, 1])
    steps = record_batches(m)
    m.partial_fit(X1, y1)
    assert [n for n, _ in steps] == [36, 36, 36, 8]  # 32/32/32/4 current + the 4-sample memory


def test_memory_stores_the_data_that_was_trained_on():
    m = Model(replay_mem_size=40, target_to_negatives=1.0, random_state=0)
    data = stream(n_exp=3, n=100)
    for X, y in data:
        m.partial_fit(X, y, classes=[0, 1])
    # negatives were cut to 1:1 *before* entering the memory; replayed samples never re-enter
    assert m._mem_seen[1] == sum(int(y.sum()) for _, y in data)
    assert m._mem_seen[0] == m._mem_seen[1]


def test_ratio_acts_on_current_data_not_on_replay():
    m = Model(replay_mem_size=200, target_to_negatives=1.0, n_epochs=1, random_state=0)
    (X0, y0), (X1, y1) = stream(n_exp=2, n=200)
    m.partial_fit(X0, y0, classes=[0, 1])
    steps = record_batches(m)
    m.partial_fit(X1, y1)
    n_cur = 2 * int(y1.sum())  # 1:1 on this call's data
    assert sum(n for n, _ in steps) == 2 * n_cur  # replay (as many as current) is added on top


def test_fit_resets_memory_but_fills_it():
    m = Model(replay_mem_size=20, random_state=0)
    s = stream(n_exp=2)
    m.partial_fit(*s[0], classes=[0, 1]).partial_fit(*s[1])
    m.fit(*s[0])
    assert sum(m._mem_seen) == 100  # old memory dropped, this fit's data stored
    m.partial_fit(*s[1])  # ... and replayed afterwards
    assert sum(len(r) for r in m._mem) == 20


def test_replay_reduces_forgetting():
    """Class-incremental negatives: a real memory keeps the old negatives recognised (same data)."""

    def old_tnr(mem, seed, d=10, n_cls=6):
        r = np.random.RandomState(seed)
        mu = r.standard_normal((n_cls, d))
        mu *= 4 / np.linalg.norm(mu, axis=1, keepdims=True)
        test = [mu[c] + r.standard_normal((200, d)) for c in range(n_cls)]
        m = Model(replay_mem_size=mem, n_epochs=10, random_state=0)
        for t in range(n_cls // 2):
            Xn = np.vstack([mu[c] + r.standard_normal((150, d)) for c in (2 * t, 2 * t + 1)])
            Xp = r.standard_normal((100, d))
            y = np.r_[np.ones(100), np.zeros(len(Xn))]
            m.partial_fit(np.vstack([Xp, Xn]), y, classes=[0, 1])
        return np.mean([(m.predict(test[c]) == 0).mean() for c in range(4)])  # first 2 experiences

    gaps = [old_tnr(400, s) - old_tnr(0, s) for s in range(5)]
    assert np.mean(gaps) > 0.1


# ---------------------------------------------------------------------- #
# scikit-learn integration
# ---------------------------------------------------------------------- #
def test_multiclass_input_points_to_ovr():
    X, y = make_classification(n_samples=90, n_classes=3, n_informative=4, random_state=0)
    with pytest.raises(ValueError, match="OneVsRestClassifier"):
        Model().fit(X, y)


def test_one_vs_rest_fit_predict():
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=8, n_classes=4, random_state=0
    )
    X = StandardScaler().fit_transform(X)
    ovr = OneVsRestClassifier(
        Model(new_feature_dim=8, target_to_negatives=0.5, random_state=0)
    ).fit(X, y)
    assert len(ovr.estimators_) == 4
    assert ovr.predict_proba(X).shape == (600, 4)
    assert ovr.score(X, y) > 0.5


def test_one_vs_rest_partial_fit_expands_each_estimator():
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=8, n_classes=4, random_state=0
    )
    ovr = OneVsRestClassifier(Model(new_feature_dim=4, random_state=0))
    for a, b in [(0, 300), (300, 600)]:
        ovr.partial_fit(X[a:b], y[a:b], classes=np.unique(y))
    assert all(e.feature_dim_ == 16 for e in ovr.estimators_)


def test_clone_and_params():
    m = Model(new_feature_dim=3, target_to_negatives=0.5, replay_mem_size=0, train_blocks=False)
    assert clone(m).get_params() == m.get_params()


def test_pickle_round_trip_keeps_expanded_state(binary):
    X, y = binary
    m = Model(new_feature_dim=4, random_state=0)
    m.partial_fit(X, y).partial_fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(m2.decision_function(X), m.decision_function(X))
    m2.partial_fit(X, y)  # keeps expanding after loading
    assert m2.feature_dim_ == 10 + 8


@parametrize_with_checks([Model(random_state=0)])
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------- #
# edge cases found in review
# ---------------------------------------------------------------------- #
def test_balanced_pos_weight_is_finite_for_single_class_experiences():
    model = Model(pos_weight="balanced")
    assert model._pos_weight(np.ones(12, dtype=bool)) == 1.0
    assert model._pos_weight(np.zeros(12, dtype=bool)) == 1.0
    assert model._pos_weight(np.array([True, True, False, False, False])) == 1.5
    # Keep the learned class balance for later single-class experiences with replay.
    assert model._pos_weight(np.zeros(12, dtype=bool)) == 1.5


def test_first_partial_fit_rejects_labels_outside_declared_classes():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="outside classes_"):
        Model().partial_fit(X, y, classes=[0, 1])


def test_later_partial_fit_rejects_labels_outside_classes(binary):
    X, y = binary
    model = Model(random_state=0).partial_fit(X[:20], y[:20], classes=[0, 1])
    with pytest.raises(ValueError, match="outside classes_"):
        model.partial_fit(X[20:24], np.array([0, 1, 2, 2]))


def test_ratio_that_would_drop_all_negatives_is_rejected():
    model = Model(target_to_negatives=10.0)
    model._rng = np.random.RandomState(0)
    pos = np.array([True, True, False, False, False])
    with pytest.raises(ValueError, match="zero negatives"):
        model._select_negatives(pos)


@pytest.mark.parametrize(
    ("replay_pos_weight", "expected_replay_target_weight"),
    [
        ("current", 4.0),
        ("uniform", 1.0),
    ],
)
def test_replay_pos_weight_modes(replay_pos_weight, expected_replay_target_weight):
    model = Model(
        new_feature_dim=0,
        replay_mem_size=0,
        batch_size=2,
        n_epochs=1,
        random_state=0,
        replay_pos_weight=replay_pos_weight,
    )
    X = np.array([[1.0], [-1.0]])
    pos = np.array([True, False])

    # Initialize the estimator and its optimizer state.
    model.fit(X, np.array([1, 0]))

    observed = []

    def record_effective_weights(X_batch, pos_batch, weight):
        effective = np.ones(len(pos_batch)) if weight is None else np.where(pos_batch, weight, 1.0)
        observed.append((pos_batch.copy(), effective.copy()))
        return [np.zeros_like(param) for param in model._params()]

    model._gradients = record_effective_weights

    # Current batch: one target and one negative.
    # Replay batch: one target and one negative.
    model._train(
        X,
        pos,
        4.0,
        np.array([[2.0], [-2.0]]),
        np.array([True, False]),
    )

    assert len(observed) == 1
    batch_pos, effective_weights = observed[0]

    # Current target keeps weight 4; current negative stays at 1.
    np.testing.assert_array_equal(effective_weights[:2], [4.0, 1.0])

    # Replay target uses the selected mode; replay negative stays at 1.
    replay_pos = batch_pos[2:]
    replay_weights = effective_weights[2:]

    assert replay_weights[replay_pos].tolist() == [expected_replay_target_weight]
    np.testing.assert_array_equal(replay_weights[~replay_pos], [1.0])


def test_replay_pos_weight_modes_change_gradients():
    X = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    pos = np.array([True, False, True, False])
    replay_X = np.array([[3.0], [-3.0]])
    replay_pos = np.array([True, False])

    gradients = {}

    for mode in ("current", "uniform"):
        model = Model(
            new_feature_dim=0,
            replay_mem_size=0,
            batch_size=2,
            n_epochs=1,
            random_state=0,
            replay_pos_weight=mode,
        )
        model.fit(X, pos.astype(int))
        model.coef_[:] = 0.0
        model.intercept_[:] = 0.0

        # Use identical parameters, current samples, and replay samples.
        batch_X = np.vstack([X[:2], replay_X])
        batch_pos = np.concatenate([pos[:2], replay_pos])

        weights = np.array([4.0, 1.0, 4.0 if mode == "current" else 1.0, 1.0])
        gradients[mode] = model._gradients(batch_X, batch_pos, weights)

    assert any(not np.allclose(a, b) for a, b in zip(gradients["current"], gradients["uniform"]))


def test_replay_pos_weight_modes_change_training_update():
    X = np.array([[1.0], [-1.0]])
    y = np.array([1, 0])
    pos = y == 1

    replay_X = np.array([[2.0], [-2.0]])
    replay_pos = np.array([True, False])

    results = {}

    for mode in ("current", "uniform"):
        model = Model(
            new_feature_dim=0,
            replay_mem_size=0,
            batch_size=2,
            n_epochs=1,
            learning_rate_init=0.1,
            momentum=0.0,
            random_state=0,
            replay_pos_weight=mode,
        )
        model.fit(X, y)

        # Start both modes from identical parameters.
        model.coef_[:] = 0.0
        model.intercept_[:] = 0.0

        model._train(
            X,
            pos,
            4.0,
            replay_X,
            replay_pos,
        )

        results[mode] = (
            model.coef_.copy(),
            model.intercept_.copy(),
        )

    assert not np.allclose(
        results["current"][0],
        results["uniform"][0],
    )
    assert not np.allclose(
        results["current"][1],
        results["uniform"][1],
    )


def test_replay_memory_copies_rows_instead_of_retaining_input_views():
    model = Model(replay_mem_size=4, random_state=0)
    X0 = np.array([[1.0], [2.0], [3.0], [4.0]])
    y0 = np.array([0, 0, 1, 1])
    model.partial_fit(X0, y0, classes=[0, 1])

    X1 = np.array([[11.0], [12.0], [13.0], [14.0]])
    y1 = np.array([0, 0, 1, 1])
    model._update_memory(X1, y1.astype(bool))

    for memory in model._mem:
        assert all(not np.shares_memory(row, X0) for row in memory)
        assert all(not np.shares_memory(row, X1) for row in memory)

    saved = [[row.copy() for row in memory] for memory in model._mem]
    X0[:] = -100.0
    X1[:] = -200.0
    for actual, expected in zip(model._mem, saved):
        for actual_row, expected_row in zip(actual, expected):
            np.testing.assert_array_equal(actual_row, expected_row)


def test_predict_treats_zero_score_as_positive():
    model = Model(new_feature_dim=0, random_state=0).fit(
        np.array([[-1.0], [1.0]]), np.array([0, 1])
    )
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0
    np.testing.assert_array_equal(model.predict(np.array([[-2.0], [0.0], [2.0]])), [1, 1, 1])
