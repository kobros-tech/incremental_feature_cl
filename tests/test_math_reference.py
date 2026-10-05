# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Lecture 6 reference mathematics (examples taken from the 6.86x notebooks)."""

import numpy as np

from incremental_feature_cl.math import (
    KernelPerceptron,
    Perceptron,
    PolynomialFeatureMap,
    polynomial_kernel,
    quadratic_map_1d,
    rbf_kernel,
)


def test_quadratic_map_lecture_example():
    x, y = np.array([-1.0, 0.0, 1.0]), np.array([1, -1, 1])
    assert quadratic_map_1d(x).tolist() == [[-1, 1], [0, 0], [1, 1]]
    assert np.array_equal(np.where(quadratic_map_1d(x) @ np.array([0, 1.0]) >= 0.5, 1, -1), y)


def test_degree2_feature_map_matches_lecture_formula():
    f = PolynomialFeatureMap(2, kernel_scaled=True)
    x1, x2 = 1.5, -0.5
    r2 = np.sqrt(2)
    got = sorted(f(np.array([[x1, x2]]))[0])
    want = sorted([1, r2 * x1, r2 * x2, x1**2, x2**2, r2 * x1 * x2])
    np.testing.assert_allclose(got, want)


def test_kernel_trick_identity():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    for p in (1, 2, 3, 4):
        f = PolynomialFeatureMap(p, kernel_scaled=True)
        np.testing.assert_allclose(f(A) @ f(B).T, polynomial_kernel(A, B, p), rtol=1e-7, atol=1e-9)
    x, z = np.array([[1.5, -0.5]]), np.array([[-0.25, 2.0]])
    np.testing.assert_allclose(
        PolynomialFeatureMap(2, kernel_scaled=True)(x)
        @ PolynomialFeatureMap(2, kernel_scaled=True)(z).T,
        [[0.140625]],
    )


def test_polynomial_map_is_block_structured_by_degree():
    """Phi_p = [Phi_{p-1}, psi_p]: the same incremental structure as the neural model."""
    X = np.random.default_rng(1).normal(size=(4, 3))
    f3, f2 = PolynomialFeatureMap(3), PolynomialFeatureMap(2)
    np.testing.assert_allclose(f3(X), np.concatenate([f2(X), f3.degree_block(X, 3)], axis=1))
    assert f3(X).shape[1] == f3.output_dim(3)
    np.testing.assert_allclose(
        PolynomialFeatureMap(3, include_bias=False)(np.array([2.0])), [[2, 4, 8]]
    )


def test_rbf_kernel_properties():
    X = np.random.default_rng(2).normal(size=(6, 3))
    K = rbf_kernel(X, X, gamma=0.5)
    np.testing.assert_allclose(np.diag(K), 1.0)
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_kernel_perceptron_equals_feature_space_perceptron_on_xor():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    y = np.array([1, -1, -1, 1])
    phi = lambda A: np.column_stack([A[:, 0], A[:, 1], A[:, 0] * A[:, 1]])
    fs = Perceptron(phi, fit_intercept=False).fit(X, y, epochs=10)
    ks = KernelPerceptron(lambda A, B: phi(A) @ phi(B).T).fit(X, y, epochs=10)
    np.testing.assert_allclose(fs.decision_function(X), ks.decision_function(X))
    assert np.array_equal(ks.predict(X), y)


def test_polynomial_kernel_perceptron_solves_xor():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    y = np.array([1, -1, -1, 1])
    assert np.array_equal(
        KernelPerceptron(lambda A, B: polynomial_kernel(A, B, 2)).fit(X, y).predict(X), y
    )


def test_perceptron_expansion_invariant_and_learns():
    x, y = np.array([-1.0, 0.0, 1.0]), np.array([1, -1, 1])
    p = Perceptron().fit(x, y, epochs=50)
    assert not np.array_equal(p.predict(x), y)  # not separable in x
    before = p.decision_function(x).copy()
    p.add_feature_block(lambda X: np.asarray(X, float)[:, None] ** 2, x)  # theta' = [theta, 0]
    np.testing.assert_array_equal(p.decision_function(x), before)  # f_{t+1} == f_t
    p.fit(x, y, epochs=200)
    assert np.array_equal(p.predict(x), y) and p.thetas[-1][0] != 0  # new block became useful
