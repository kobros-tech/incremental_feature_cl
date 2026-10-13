# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import csv
import json

import pytest

from incremental_feature_cl.evaluation.result_store import RunResult
from incremental_feature_cl.experiments import train_ovr

ARGS = [
    "--dataset", "synthetic",
    "--backbone", "identity",
    "--n-experiences", "3",
    "--train-epochs", "2",
    "--lr", "0.05",
    "--device", "cpu",
    "--no-plots",
    "--ratio", "1:2",
    "--memory-per-class", "8",
    "--set", "data.synthetic.n_classes=12",
]  # fmt: skip


@pytest.fixture()
def out(tmp_path):
    train_ovr.main(
        [
            *ARGS,
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path),
            "--name",
            "t",
        ]
    )
    return tmp_path / "t"


def test_ovr_run_saves_skill_forgetting_without_multiclass_accuracy(out):
    result = RunResult.load(out / "results.json")

    assert len(result.experiences) == 3
    assert [row["number_of_skills"] for row in result.experiences] == [4, 8, 12]

    assert set(result.experiences[1]["updated_old_skills"]) == set(result.experiences[0]["classes"])
    assert result.experiences[-1]["feature_dim"] > 0

    assert "average_skill_forgetting" in result.summary
    assert "average_forgetting" not in result.summary
    assert "final_accuracy_seen" not in result.summary
    assert "average_accuracy_seen" not in result.summary

    assert len(result.skill_target_recall_matrix()) == 3

    for experience in result.experiences:
        seen = set(experience["seen_classes"])
        for class_id, recall in enumerate(experience["per_skill_target_recall"]):
            assert (recall is None) == (class_id not in seen)

    assert result.experiences[-1]["trainable_parameter_count"] > 0


def test_per_class_accuracy_csv_and_skill_config_are_saved(out):
    with (out / "skill_target_recall.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3
    assert "skill_11" in rows[0]

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))

    assert data["config"]["skill_config"]["target_to_negatives"] == pytest.approx(0.5)
    assert data["config"]["skill_config"]["memory_per_class"] == 8
    assert data["experiences"][-1]["number_of_skills"] == 12


def test_run_does_not_print_sample_by_sample_score_dump(tmp_path, capsys):
    train_ovr.main(
        [
            *ARGS,
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path),
            "--name",
            "quiet",
        ]
    )

    output = capsys.readouterr().out

    assert "Sample 0" not in output
    assert "avg_skill_forgetting=" in output


def test_sklearn_control(tmp_path):
    pytest.importorskip("sklearn")

    train_ovr.main(
        [
            *ARGS,
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path),
            "--name",
            "sk",
            "--vs-sklearn",
        ]
    )

    result = RunResult.load(tmp_path / "sk" / "vs_sklearn" / "results.json")

    assert len(result.experiences) == 3
    assert [row["number_of_skills"] for row in result.experiences] == [4, 8, 12]


def test_print_comparison_handles_missing_class_metrics(capsys):
    rows = {
        "skill": [
            {
                "balanced_accuracy": 0.8,
                "target_recall": 0.7,
                "negative_accuracy": 0.9,
                "target_f1": 0.6,
                "auc": 0.85,
            },
            {
                "balanced_accuracy": 0.7,
                "target_recall": 0.6,
                "negative_accuracy": 0.8,
                "target_f1": 0.5,
                "auc": 0.75,
            },
        ],
        "feature_map": [
            {
                "balanced_accuracy": 0.9,
                "target_recall": 0.8,
                "negative_accuracy": 1.0,
                "target_f1": 0.7,
                "auc": 0.95,
            },
            None,
        ],
    }

    train_ovr.print_comparison(
        "test comparison",
        rows,
        [0, 1],
    )

    output = capsys.readouterr().out

    assert "test comparison" in output
    assert "n/a" in output
    assert "mean" in output

def test_skill_config_respects_selected_optimizer_and_weight_decay():
    from argparse import Namespace

    from incremental_feature_cl.experiments.common import skill_config_from
    from incremental_feature_cl.experiments.config import ExperimentConfig

    args = Namespace(
        optimizer=None,
        weight_decay=None,
        skill_pos_weight="balanced",
        skill_expansion="first",
        skill_max_iter=100,
        memory_per_class=8,
    )
    cfg = ExperimentConfig()

    skill_cfg = skill_config_from(cfg, args)
    assert skill_cfg.optimizer == cfg.train.optimizer == "sgd"
    assert skill_cfg.weight_decay == cfg.train.weight_decay == 0.0
    assert skill_cfg.epochs == cfg.train.train_epochs
    assert skill_cfg.batch_size == cfg.train.train_mb_size

    # Explicit L-BFGS remains an option, but must not silently change weight decay.
    cfg.train.optimizer = "lbfgs"
    cfg.train.weight_decay = 0.0
    args.optimizer = "lbfgs"
    skill_cfg = skill_config_from(cfg, args)
    assert skill_cfg.optimizer == "lbfgs"
    assert skill_cfg.weight_decay == 0.0
    assert skill_cfg.max_iter == args.skill_max_iter

