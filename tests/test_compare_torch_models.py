# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import csv

import numpy as np
import pytest
import torch
from incremental_feature_cl.experiments.compare_torch_models import (
    _print_skill_diagnostics,
    _ResizeToCifarInput,
    _sample_diagnostic_rows,
    _write_sample_diagnostics,
    validate_model_input,
)

from incremental_feature_cl.models import OneVsRestSkillModel, SkillConfig


def test_slim_resnet_accepts_small_rgb_images_for_resize_adapter():
    validate_model_input("slim_resnet18", (3, 8, 8))


def test_slim_resnet_accepts_expected_cifar_shape():
    validate_model_input("slim_resnet18", (3, 32, 32))


def test_slim_resnet_rejects_vector_input():
    with pytest.raises(ValueError, match="3-channel image input"):
        validate_model_input("slim_resnet18", (192,))


def test_simple_mlp_accepts_vector_input():
    validate_model_input("simple_mlp", (192,))


def test_sample_diagnostics_preserve_class_column_mapping():
    rows = _sample_diagnostic_rows(
        scores=torch.tensor([[-0.2, 0.8, 0.1]]),
        labels=np.array([2]),
        predictions=np.array([2]),
        class_ids=[17, 2, 9],
        seen_classes=[17, 2, 9],
        num_classes=20,
        skill_model=True,
    )

    row = rows[0]
    assert row["true_label"] == 2
    assert row["final_prediction"] == 2
    assert row["final_correct"] == 1
    assert row["winner_class"] == 2
    assert row["runner_up_class"] == 9
    assert row["num_binary_positive"] == 2
    assert row["binary_decision_class_17"] == 0
    assert row["binary_decision_class_2"] == 1
    assert row["binary_decision_class_9"] == 1
    assert row["score_class_0"] == ""


def test_sample_diagnostics_csv_appends_experience_and_decision_columns(tmp_path):
    row = {
        "experience": 1,
        **_sample_diagnostic_rows(
            scores=torch.tensor([[0.3, -0.4]]),
            labels=np.array([0]),
            predictions=np.array([0]),
            class_ids=[0, 1],
            seen_classes=[0, 1],
            num_classes=2,
            skill_model=True,
        )[0],
    }
    path = tmp_path / "sample_diagnostics.csv"
    _write_sample_diagnostics(path, [row], 2, append=False)

    with path.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))

    assert saved[0]["experience"] == "1"
    assert float(saved[0]["score_class_0"]) == pytest.approx(0.3)
    assert saved[0]["binary_decision_class_0"] == "1"
    assert saved[0]["binary_decision_class_1"] == "0"


def test_skill_parameter_count_increases_when_new_skill_is_created():
    model = OneVsRestSkillModel(
        torch.nn.Linear(4, 3),
        feature_dim=3,
        config=SkillConfig(epochs=1, batch_size=8, memory_per_class=4),
        seed=1,
    )
    initial_total = sum(p.numel() for p in model.parameters())
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.update(
        torch.randn(8, 4),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        device="cpu",
    )

    final_total = sum(p.numel() for p in model.parameters())
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert final_total > initial_total
    assert initial_trainable == 0
    assert final_trainable > initial_trainable


def test_resize_adapter_converts_small_rgb_images_to_32x32():
    class ShapeRecorder(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([x.shape[-2], x.shape[-1]], device=x.device)

    adapter = _ResizeToCifarInput(ShapeRecorder())
    shape = adapter(torch.randn(2, 3, 8, 8))
    assert shape.tolist() == [32, 32]


def test_skill_diagnostics_print_true_vs_elected_scores_and_skill_metrics(capsys):
    rows = _sample_diagnostic_rows(
        scores=torch.tensor([[0.2, 0.8], [0.9, -0.1], [-0.3, 0.7]]),
        labels=np.array([0, 1, 0]),
        predictions=np.array([1, 0, 1]),
        class_ids=[0, 1],
        seen_classes=[0, 1],
        num_classes=2,
        skill_model=True,
    )
    train_info = {
        "skills": {
            0: {"n_positive": 10, "n_negative": 20, "train_loss": 0.4},
            1: {"n_positive": 10, "n_negative": 20, "train_loss": 0.5},
        }
    }

    _print_skill_diagnostics(rows, train_info, [0, 1], experience=2)
    output = capsys.readouterr().out

    assert "true=  0 -> elected=  1" in output
    assert "true_score=  0.2000" in output
    assert "delta=  0.6000" in output
    assert "skill_logits=[0:0.2000*, 1:0.8000!]" in output
    assert "Confusion counts for seen true classes (true rows -> elected columns)" in output
    assert "train_pos=  10 train_neg=  20" in output
    assert "precision=" in output
    assert "pos_logit_mean/std=" in output


def test_skill_diagnostics_handles_unseen_true_class_without_score(capsys):
    rows = _sample_diagnostic_rows(
        scores=torch.tensor([[0.2, 0.8]]),
        labels=np.array([7]),
        predictions=np.array([1]),
        class_ids=[0, 1],
        seen_classes=[0, 1],
        num_classes=8,
        skill_model=True,
    )

    _print_skill_diagnostics(rows, {"skills": {}}, [0, 1], experience=0)
    output = capsys.readouterr().out

    assert "unseen_target_errors=1" in output
    assert "true=  7 -> elected=  1" in output
    assert "N/A (unseen)" in output
    assert "delta=     N/A" in output
