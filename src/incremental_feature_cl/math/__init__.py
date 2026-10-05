# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Explicit mathematical reference implementation (MIT 6.86x Lecture 6)."""

from .feature_maps import PolynomialFeatureMap, quadratic_map_1d
from .kernels import linear_kernel, polynomial_kernel, rbf_kernel
from .perceptron import KernelPerceptron, Perceptron

__all__ = [
    "KernelPerceptron",
    "Perceptron",
    "PolynomialFeatureMap",
    "linear_kernel",
    "polynomial_kernel",
    "quadratic_map_1d",
    "rbf_kernel",
]
