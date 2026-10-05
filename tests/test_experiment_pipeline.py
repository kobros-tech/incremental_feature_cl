# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""End-to-end tiny runs: preservation record, results format, plots, reproducibility, replay."""

import json

import numpy as np
import pytest

from incremental_feature_cl.evaluation import RunResult, forgetting_per_class
from incremental_feature_cl.experiments.common import run_experiment
from incremental_feature_cl.experiments.config import ExperimentConfig, apply_overrides
from incremental_feature_cl.plotting import make_all_plots


def cfg(mode="multiclass", init="zero", dim=8, replay=0, seed=1, name="t"):
    return ExperimentConfig.from_dict(
        {
            "name": name,
            "mode": mode,
            "seed": seed,
            "target_class": 2,
            "data": {
                "dataset": "synthetic",
                "n_experiences": 3,
                "synthetic": {"n_classes": 6, "n_train_per_class": 16, "n_test_per_class": 8},
            },
            "model": {"backbone": "smallconv", "new_feature_dim": dim, "initialization": init},
            "train": {
                "train_epochs": 1,
                "lr": 0.05,
                "device": "cpu",
                "train_mb_size": 16,
                "replay_mem_size": replay,
                "probe_size": 32,
            },
        }
    )


@pytest.mark.parametrize("mode", ["multiclass", "target"])
def test_run_saves_everything_and_plots(tmp_path, mode):
    run_experiment(cfg(mode), out_dir=tmp_path, verbose=False)
    for f in ("results.json", "metrics.csv", "per_class_accuracy.csv"):
        assert (tmp_path / f).exists()
    plots = {p.name for p in (tmp_path / "plots").iterdir()}
    assert {
        "A_accuracy_vs_experience.png",
        "B_accuracy_vs_class.png",
        "C_class_accuracy_heatmap.png",
        "D_class_accuracy_curves.png",
        "E_forgetting_vs_class.png",
        "F_feature_growth.png",
        "G_new_feature_utilization.png",
    } <= plots
    d = json.loads((tmp_path / "results.json").read_text())
    # reproducibility record
    assert d["config"]["seed"] == 1 and d["config"]["model"]["backbone"] == "smallconv"
    assert {"git_commit", "torch", "package_version"} <= set(d["environment"])
    assert d["class_order"] and len(d["experiences"]) == 3
    # regenerate plots from disk only
    assert make_all_plots(RunResult.load(tmp_path), tmp_path / "again")


def test_expansion_records_and_preservation_check(tmp_path):
    r = run_experiment(cfg(init="zero"), out_dir=tmp_path / "z", verbose=False, make_plots=False)
    fe = [e["feature_expansion"] for e in r.experiences]
    assert fe[0] is None and fe[1]["old_feature_dim"] == 64 and fe[1]["new_feature_dim"] == 8
    for k in ("number_of_new_parameters", "old_parameter_count", "total_parameter_count"):
        assert k in fe[1]
    assert [e["feature_dim"] for e in r.experiences] == [64, 72, 80]
    assert all(e["probe"]["logit_max_abs_diff_expansion"] == 0.0 for e in r.experiences[1:])
    assert all(e["probe"]["pred_agreement_expansion"] == 1.0 for e in r.experiences[1:])
    rr = run_experiment(cfg(init="random"), out_dir=tmp_path / "r", verbose=False, make_plots=False)
    assert all(e["probe"]["logit_max_abs_diff_expansion"] > 0 for e in rr.experiences[1:])


def test_fixed_baseline_does_not_grow(tmp_path):
    r = run_experiment(cfg(dim=0), out_dir=tmp_path, verbose=False, make_plots=False)
    assert {e["feature_dim"] for e in r.experiences} == {64}


def test_same_seed_is_reproducible(tmp_path):
    a = run_experiment(cfg(), out_dir=tmp_path / "a", verbose=False, make_plots=False)
    b = run_experiment(cfg(), out_dir=tmp_path / "b", verbose=False, make_plots=False)
    np.testing.assert_array_equal(a.accuracy_matrix(), b.accuracy_matrix())


def test_replay_reduces_forgetting_on_easy_problem(tmp_path):
    kw = {"dim": 0}
    base = run_experiment(cfg(**kw), out_dir=tmp_path / "n", verbose=False, make_plots=False)
    rep = run_experiment(
        cfg(replay=48, **kw), out_dir=tmp_path / "r", verbose=False, make_plots=False
    )
    assert rep.summary["average_forgetting"] <= base.summary["average_forgetting"]


def test_forgetting_metric():
    acc = np.array([[0.9, np.nan], [0.5, 0.8], [0.3, 0.6]])
    f = forgetting_per_class(acc, {0: 0, 1: 1})
    assert f[0] == pytest.approx(0.9 - 0.3) and f[1] == pytest.approx(0.8 - 0.6)


def test_config_overrides_and_validation():
    d = apply_overrides(ExperimentConfig().to_dict(), ["model.new_feature_dim=16", "train.lr=0.1"])
    c = ExperimentConfig.from_dict(d)
    assert c.model.new_feature_dim == 16 and c.train.lr == 0.1
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"nonsense": 1})
