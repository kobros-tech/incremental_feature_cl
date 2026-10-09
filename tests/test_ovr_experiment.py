# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import csv
import json

import pytest

from incremental_feature_cl.evaluation.result_store import RunResult
from incremental_feature_cl.experiments import compare_ovr

ARGS = [
    "--dataset", "synthetic", "--backbone", "identity", "--n-experiences", "3",
    "--train-epochs", "2", "--lr", "0.05", "--device", "cpu", "--no-plots",
    "--ratio", "1:2", "--memory-per-class", "8",
]  # fmt: skip


@pytest.fixture()
def out(tmp_path):
    compare_ovr.main([*ARGS, "--seed", "1", "--output-dir", str(tmp_path), "--name", "t"])
    return tmp_path / "t"


def test_skill_run_saves_accuracy_and_forgetting(out):
    result = RunResult.load(out / "results.json")

    assert len(result.experiences) == 3
    assert [row["number_of_skills"] for row in result.experiences] == [4, 8, 12]
    assert set(result.experiences[1]["updated_old_skills"]) == set(result.experiences[0]["classes"])
    assert result.experiences[-1]["skill_feature_dims"]
    assert {
        "final_accuracy_seen",
        "average_accuracy_seen",
        "average_forgetting",
    } <= set(result.summary)
    assert len(result.accuracy_matrix()) == 3
    for experience in result.experiences:
        seen = set(experience["seen_classes"])
        for class_id, accuracy in enumerate(experience["per_class_accuracy"]):
            assert (accuracy is None) == (class_id not in seen)
    assert result.experiences[-1]["trainable_parameter_count"] > 0


def test_per_class_accuracy_csv_and_skill_config_are_saved(out):
    with (out / "per_class_accuracy.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert "class_11" in rows[0]

    data = json.loads((out / "results.json").read_text())
    assert data["config"]["skill_config"]["target_to_negatives"] == pytest.approx(0.5)
    assert data["config"]["skill_config"]["memory_per_class"] == 8
    assert data["experiences"][-1]["number_of_classifiers"] == 12


def test_run_does_not_print_sample_by_sample_score_dump(tmp_path, capsys):
    compare_ovr.main([*ARGS, "--seed", "1", "--output-dir", str(tmp_path), "--name", "quiet"])
    output = capsys.readouterr().out
    assert "Sample 0" not in output
    assert "avg_forgetting=" in output


def test_sklearn_control(tmp_path):
    pytest.importorskip("sklearn")
    from incremental_feature_cl.experiments import compare_sklearn_ovr

    compare_sklearn_ovr.main(
        [
            "--dataset", "synthetic", "--backbone", "identity", "--n-experiences", "3",
            "--train-epochs", "2", "--device", "cpu", "--output-dir", str(tmp_path), "--name", "sk",
        ]
    )  # fmt: skip
    res = RunResult.load(tmp_path / "sk")
    assert res.experiences[-1]["number_of_classifiers"] == 12
