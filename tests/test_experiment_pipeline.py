# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""End-to-end tiny runs: preservation record, results format, plots, reproducibility, replay."""

import argparse
import json

import numpy as np
import pytest

from incremental_feature_cl.evaluation import RunResult, forgetting_per_class
from incremental_feature_cl.experiments.common import (
    add_common_args,
    config_from_args,
    estimate_train_samples,
    run_experiment,
)
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


# --------------------------------------------------------------------------- #
# target_to_negatives plumbing: config, naming, CLI, end-to-end
# --------------------------------------------------------------------------- #
def _parse(argv, mode="target"):
    p = argparse.ArgumentParser()
    add_common_args(p, "synthetic")
    return config_from_args(p.parse_args(argv), mode, "synthetic")


def test_ratio_defaults_to_none_and_is_serialised():
    c = ExperimentConfig()
    assert c.target_to_negatives is None
    assert c.to_dict()["target_to_negatives"] is None
    assert (
        "target_to_negatives" not in c.to_dict()["train"]
    )  # data-stream control, not optimiser/loss
    c2 = ExperimentConfig.from_dict({"mode": "target", "target_to_negatives": 0.2})
    assert c2.to_dict()["target_to_negatives"] == 0.2


@pytest.mark.parametrize("bad", [0, -1, "nan"])
def test_config_rejects_invalid_ratio(bad):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"target_to_negatives": float(bad)})


def test_auto_name_distinguishes_ratios_and_keeps_old_name_for_none():
    def name(ratio, mode="target"):
        return ExperimentConfig.from_dict(
            {"mode": mode, "target_class": 17, "target_to_negatives": ratio}
        ).auto_name()

    base = name(None)
    assert base == "target_t17_synthetic_e5_d0_fixed_s1"  # unchanged for the default
    assert name(1) == "target_t17_synthetic_e5_d0_fixed_r1_s1"
    assert name(0.5).endswith("_r0.5_s1") and name(0.2).endswith("_r0.2_s1")
    assert len({base, name(1), name(0.5), name(0.2), name(0.1)}) == 5
    assert "_r" not in name(0.2, mode="multiclass")  # ratio is ignored outside target mode


def test_cli_flag_sets_ratio_and_leaves_loss_untouched():
    assert _parse([]).target_to_negatives is None
    assert _parse(["--target-to-negatives", "0.2"]).target_to_negatives == 0.2
    assert _parse(["--target-to-negatives", "1"]).target_to_negatives == 1.0
    assert _parse(["--target-to-negatives", "10"]).target_to_negatives == 10.0
    c = _parse(["--target-to-negatives", "0.2"])
    assert c.train.pos_weight == "balanced"  # the ratio never changes the loss weighting
    assert _parse(["--set", "target_to_negatives=0.5"]).target_to_negatives == 0.5
    assert _parse(["--set", "target_to_negatives=null"]).target_to_negatives is None
    with pytest.raises(ValueError):
        _parse(["--target-to-negatives", "0"])


def test_cli_flag_is_rejected_outside_target_mode():
    with pytest.raises(ValueError, match="target"):
        _parse(["--target-to-negatives", "0.2"], mode="multiclass")


def test_estimate_reflects_ratio():
    def est(ratio):
        c = cfg("target")
        c.target_to_negatives = ratio
        return estimate_train_samples(c)

    assert est(1.0) < est(None)  # 16 targets + 16 negatives per experience vs the cumulative pool


def test_target_run_with_ratio_records_it_and_changes_name(tmp_path):
    c = cfg("target", name="")
    c.target_to_negatives = 0.5
    r = run_experiment(c, out_dir=tmp_path, verbose=False, make_plots=False)
    assert r.config["target_to_negatives"] == 0.5
    assert (
        json.loads((tmp_path / "results.json").read_text())["config"]["target_to_negatives"] == 0.5
    )
    assert c.name.endswith("_r0.5_s1")
