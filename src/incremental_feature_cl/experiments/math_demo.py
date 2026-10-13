# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Milestone 1a: the Lecture 6 mathematics with explicit feature expansion (no neural network).

    python -m incremental_feature_cl.experiments.math_demo

x = -1, 0, +1 with labels +1, -1, +1 is not linearly separable in x.  Start with phi_0(x) = x,
append the block psi(x) = x^2 with theta_new = 0: the decision function is unchanged, then
training makes theta_new non-zero and the data becomes separable.
"""

import numpy as np

from ..math import KernelPerceptron, Perceptron, PolynomialFeatureMap, polynomial_kernel


def main() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([1, -1, 1])
    p = Perceptron(fit_intercept=True).fit(x, y, epochs=50)
    print("1) linear in x          predictions:", p.predict(x), "(labels", y, ") -> not separable")
    before = p.decision_function(x).copy()
    dim = p.add_feature_block(lambda X: np.asarray(X, float)[:, None] ** 2, x)
    after = p.decision_function(x)
    print(
        f"2) expand by x^2 (+{dim} dim, theta_new=0): max |score change| = {np.abs(before - after).max():.1e}"
    )
    p.fit(x, y, epochs=200)
    print(
        "3) after training       predictions:",
        p.predict(x),
        " theta =",
        p.theta,
        " b =",
        p.intercept,
    )
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    Y = np.array([1, -1, -1, 1])
    K = KernelPerceptron(lambda A, B: polynomial_kernel(A, B, 2)).fit(X, Y)
    print("4) kernel perceptron (poly deg 2) on XOR predictions:", K.predict(X))
    f = PolynomialFeatureMap(2, kernel_scaled=True)
    a, b = np.array([[1.5, -0.5]]), np.array([[-0.25, 2.0]])
    print(
        "5) phi(x).phi(z) =",
        float((f(a) @ f(b).T)[0, 0]),
        " (1+x.z)^2 =",
        float(polynomial_kernel(a, b, 2)[0, 0]),
    )


if __name__ == "__main__":
    main()
