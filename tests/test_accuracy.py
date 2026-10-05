# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from incremental_feature_cl.evaluation import masked_accuracy, per_class_accuracy


def test_per_class_accuracy_accepts_torch_bool_and_returns_float():
    correct = torch.tensor([True, False, True, False])
    labels = np.array([0, 0, 1, 1])
    result = per_class_accuracy(correct, labels, 2)
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_per_class_accuracy_marks_absent_classes_nan():
    result = per_class_accuracy(np.array([True, False]), np.array([0, 0]), 3)
    assert result[0] == 0.5
    assert np.isnan(result[1:]).all()


def test_per_class_accuracy_rejects_mismatched_shapes_and_invalid_labels():
    with pytest.raises(ValueError):
        per_class_accuracy(np.array([True]), np.array([0, 1]), 2)
    with pytest.raises(ValueError):
        per_class_accuracy(np.array([True, False]), np.array([0, 2]), 2)


def test_masked_accuracy_accepts_torch_bool():
    correct = torch.tensor([True, False, True])
    labels = np.array([0, 0, 1])
    assert masked_accuracy(correct, labels, [0, 1]) == pytest.approx(2 / 3)
