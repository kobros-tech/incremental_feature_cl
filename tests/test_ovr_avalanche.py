# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
import pytest

pytest.importorskip("avalanche")

import torch
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import TensorDataset

from incremental_feature_cl.avalanche import OneVsRestSkillStrategy
from incremental_feature_cl.data import make_synthetic_dataset
from incremental_feature_cl.models import (
    IdentityBackbone,
    OneVsRestSkillModel,
    SkillConfig,
)

N_CLASSES, DIM = 6, 3 * 8 * 8


@pytest.fixture(scope="module")
def bm():
    tr, te = make_synthetic_dataset(
        n_classes=N_CLASSES,
        n_train_per_class=16,
        n_test_per_class=8,
    )

    class_order = [3, 1, 5, 0, 4, 2]

    def make_experiences(dataset, samples_per_class):
        x = torch.as_tensor(dataset.x)
        y = torch.as_tensor(dataset.targets)

        experiences = []
        for start in range(0, len(class_order), 2):
            classes = class_order[start : start + 2]
            mask = torch.zeros_like(y, dtype=torch.bool)

            for class_id in classes:
                mask |= y == class_id

            experiences.append(AvalancheDataset(TensorDataset(x[mask], y[mask])))

        return experiences

    return benchmark_from_datasets(
        train=make_experiences(tr, 16),
        test=make_experiences(te, 8),
    )


def test_skill_strategy_updates_skills_through_avalanche(bm):
    model = OneVsRestSkillModel(
        IdentityBackbone(DIM), config=SkillConfig(epochs=3, lr=0.05, memory_per_class=8), seed=0
    )
    s = OneVsRestSkillStrategy(model=model, num_classes=N_CLASSES, eval_mb_size=32)
    expected = [[1, 3], [1, 3, 0, 5], [1, 3, 0, 5, 2, 4]]  # sorted within an experience
    for i, exp in enumerate(bm.train_stream):
        s.train(exp)
        assert model.class_ids == expected[i]  # append-only discovery order, stable columns
        assert set(model.skills) == set(expected[i])
        res = s.eval(bm.test_stream[: i + 1])
        assert any(k.startswith("Top1_Acc_Stream") for k in res)
    assert len(s.skill_records) == 3
    assert s.skill_records[1]["updated_old_skills"] == [1, 3]  # old skills were updated
    assert model.skills[3].known_negatives == [0, 1, 2, 4, 5]
    acc = next(v for k, v in res.items() if k.startswith("Top1_Acc_Stream"))
    assert acc > 1 / N_CLASSES  # Avalanche's argmax-over-class-id metric sees real predictions
