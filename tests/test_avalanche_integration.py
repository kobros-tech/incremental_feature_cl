# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest
import torch
from torch.utils.data import TensorDataset

pytest.importorskip("avalanche")

from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset

from incremental_feature_cl.avalanche import OneVsRestSkillStrategy
from incremental_feature_cl.models import (
    IdentityBackbone,
    OneVsRestSkillModel,
    SkillConfig,
)


def _dataset(classes, samples_per_class=12, seed=0):
    """Build a deterministic feature-vector classification dataset."""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []

    for class_id in classes:
        center = torch.zeros(6)
        center[class_id % 6] = 3.0

        xs.append(
            center
            + 0.3
            * torch.randn(
                samples_per_class,
                6,
                generator=generator,
            )
        )
        ys.append(torch.full((samples_per_class,), class_id))

    return TensorDataset(torch.cat(xs), torch.cat(ys))


@pytest.fixture(scope="module")
def benchmark():
    train_datasets = [
        _dataset([0, 1], seed=0),
        _dataset([2, 3], seed=1),
        _dataset([4, 5], seed=2),
    ]
    test_datasets = [
        _dataset([0, 1], samples_per_class=6, seed=10),
        _dataset([2, 3], samples_per_class=6, seed=11),
        _dataset([4, 5], samples_per_class=6, seed=12),
    ]

    train_datasets = [AvalancheDataset(dataset) for dataset in train_datasets]
    test_datasets = [AvalancheDataset(dataset) for dataset in test_datasets]

    return benchmark_from_datasets(
        train=train_datasets,
        test=test_datasets,
    )


def _make_model():
    return OneVsRestSkillModel(
        IdentityBackbone(6),
        config=SkillConfig(
            memory_per_class=5,
            epochs=1,
            batch_size=8,
        ),
        seed=1,
    )


def test_strategy_trains_across_experiences_and_accumulates_skills(
    benchmark,
):
    model = _make_model()

    strategy = OneVsRestSkillStrategy(
        model=model,
        num_classes=6,
        eval_mb_size=16,
    )

    for index, experience in enumerate(benchmark.train_stream):
        strategy.train(experience)

        expected_classes = list(range(2 * (index + 1)))

        assert model.class_ids == expected_classes
        assert set(model.skills) == set(expected_classes)

    assert len(strategy.skill_records) == 3


def test_strategy_default_evaluator_has_no_argmax_accuracy(benchmark):
    model = _make_model()

    strategy = OneVsRestSkillStrategy(
        model=model,
        num_classes=6,
        eval_mb_size=16,
    )

    strategy.train(benchmark.train_stream[0])

    results = strategy.eval(benchmark.test_stream[:1])

    # Avalanche's accuracy is an argmax over independent skill logits: not reported by default
    assert not any(key.startswith("Top1_Acc") for key in results)
    per_skill, decisions = strategy.evaluate_skills(benchmark.test_stream[:1])
    assert set(per_skill) == set(model.class_ids)
    assert decisions["n_samples"] > 0


def test_model_predictions_are_consistent_with_class_id_mapping(
    benchmark,
):
    model = _make_model()

    strategy = OneVsRestSkillStrategy(
        model=model,
        num_classes=6,
        eval_mb_size=16,
    )

    strategy.train(benchmark.train_stream[0])

    dataset = benchmark.test_stream[0].dataset
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])

    scores = model.class_id_scores(x, num_classes=6)
    predictions = model.predict(x)

    assert scores.shape == (len(dataset), 6)
    assert predictions.shape == (len(dataset),)
    assert torch.equal(scores.argmax(dim=1), predictions)
