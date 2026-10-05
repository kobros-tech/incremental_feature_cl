# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import pytest
import torch

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone
from incremental_feature_cl.data import make_synthetic_dataset


@pytest.fixture(scope="session")
def synth():
    return make_synthetic_dataset(n_classes=10, n_train_per_class=20, n_test_per_class=10, seed=0)


@pytest.fixture
def conv_model():
    torch.manual_seed(0)
    return IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=3).eval()


@pytest.fixture
def x_batch():
    return torch.randn(7, 3, 8, 8, generator=torch.Generator().manual_seed(1))
