# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""compare_ratios: parsing, aggregation, resume, realized ratios, outputs (synthetic data only)."""

import csv
import json

import numpy as np
import pytest

from incremental_feature_cl.data import format_ratio, parse_ratio
from incremental_feature_cl.evaluation.forgetting import forgetting_per_class
from incremental_feature_cl.evaluation.ratio_summary import (
    METRIC_KEYS,
    aggregate_rows,
    mean_curves,
    ratio_sort_key,
    realized_profile,
    run_metrics,
)
from incremental_feature_cl.evaluation.result_store import RunResult
from incremental_feature_cl.experiments import compare_ratios
from incremental_feature_cl.experiments.compare_ratios import (
    parse_pos_weights,
    parse_ratios,
)
from incremental_feature_cl.experiments.config import apply_overrides, parse_override_value
from incremental_feature_cl.experiments.run_sweep import main as run_sweep_main


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


SYN = "data.synthetic={n_classes: 9, n_train_per_class: 12, n_test_per_class: 6}"


def sweep_args(out, *extra, targets=("0", "3"), ratios=("5:1", "1:1", "1:5")):
    return [
        "--dataset", "synthetic", "--backbone", "mlp", "--set", SYN,
        "--target-class", *targets, "--ratios", *ratios,
        "--n-experiences", "3", "--new-feature-dim", "2", "--train-epochs", "1",
        "--lr", "0.05", "--seed", "1", "--device", "cpu",
        "--output-dir", str(out), "--name", "sw", *extra,
    ]  # fmt: skip


# --------------------------------------------------------------------------- #
# notation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,value",
    [("5:1", 5.0), ("1:5", 0.2), ("2:1", 2.0), ("1:2", 0.5), ("1:1", 1.0), ("0.2", 0.2), ("3", 3.0),
     ("cumulative", None), ("none", None), (" 1 : 10 ", 0.1)],
)  # fmt: skip
def test_parse_ratio(text, value):
    got = parse_ratio(text)
    assert got == value if value is None else got == pytest.approx(value)


@pytest.mark.parametrize("bad", ["0:1", "1:0", "-1:5", "a:b", "1:2:3", "", "nan", "inf", "0"])
def test_parse_ratio_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_ratio(bad)


def test_format_ratio_roundtrip():
    labels = {
        5.0: "5:1",
        2.0: "2:1",
        1.0: "1:1",
        0.5: "1:2",
        0.2: "1:5",
        0.1: "1:10",
        None: "cumulative",
    }
    for value, label in labels.items():
        assert format_ratio(value) == label
        got = parse_ratio(label)
        assert got == value if value is None else got == pytest.approx(value)


def test_parse_ratios_orders_5to1_down_to_1to5_with_cumulative_last():
    out = parse_ratios(["1:5", "2:1", "1:1", "5:1", "1:2"])
    assert [format_ratio(r) for r in out] == ["5:1", "2:1", "1:1", "1:2", "1:5", "cumulative"]
    assert None not in parse_ratios(["1:1", "1:5"], include_cumulative=False)
    with pytest.raises(ValueError):
        parse_ratios(["1:2", "0.5"])  # same ratio twice
    with pytest.raises(ValueError):
        parse_ratios(["1:0"])
    assert ratio_sort_key(5.0) < ratio_sort_key(0.2) < ratio_sort_key(None)


def test_parse_pos_weights():
    assert parse_pos_weights(["balanced", "none", "2.5"]) == ["balanced", None, 2.5]
    with pytest.raises(ValueError):
        parse_pos_weights(["none", "null"])


def test_yaml_would_misread_ratio_notation_so_overrides_refuse_it():
    import yaml

    assert yaml.safe_load("1:5") == 65  # the footgun being guarded against
    with pytest.raises(ValueError, match="number"):
        parse_override_value("target_to_negatives", "1:5")
    with pytest.raises(ValueError):
        apply_overrides({}, ["target_to_negatives=5:1"])
    assert parse_override_value("target_to_negatives", "0.2") == 0.2
    assert parse_override_value("other.key", "1:5") == 65  # only ratio keys are guarded
    with pytest.raises(ValueError):
        run_sweep_main(["--mode", "target", "--dataset", "synthetic", "--grid",
                        "target_to_negatives=1:5", "--dry-run"])  # fmt: skip


# --------------------------------------------------------------------------- #
# aggregation (pure functions)
# --------------------------------------------------------------------------- #
def _row(target, ratio, pw, recall):
    base = dict.fromkeys(METRIC_KEYS, 0.5)  # all metric columns, so the test cannot drift
    return {"target_class": target, "ratio": ratio, "ratio_label": format_ratio(ratio),
            "pw_unused": None, "pos_weight": pw, **base, "final_target_recall": recall}  # fmt: skip


def test_aggregate_rows_mean_std_and_order():
    rows = [_row(0, None, "balanced", 0.1), _row(0, 0.2, "balanced", 0.4), _row(1, 0.2, "balanced", 0.6),
            _row(0, 5.0, "balanced", 1.0), _row(1, 5.0, "balanced", 1.0), _row(1, None, "balanced", 0.3)]  # fmt: skip
    agg = aggregate_rows(rows)
    assert [a["ratio_label"] for a in agg] == ["5:1", "1:5", "cumulative"]
    by = {a["ratio_label"]: a for a in agg}
    assert by["1:5"]["n_targets"] == 2
    assert by["1:5"]["mean_final_target_recall"] == pytest.approx(0.5)
    assert by["1:5"]["std_final_target_recall"] == pytest.approx(0.1)
    assert by["cumulative"]["mean_final_target_recall"] == pytest.approx(0.2)
    assert by["5:1"]["std_final_target_recall"] == 0.0


def test_aggregate_rows_keeps_loss_settings_apart_and_ignores_missing():
    rows = [_row(0, 1.0, "balanced", 0.9), _row(0, 1.0, "none", 0.5)]
    rows[1]["mean_negatives_per_target"] = None
    agg = aggregate_rows(rows)
    assert {a["pos_weight"] for a in agg} == {"balanced", "none"} and len(agg) == 2
    none_row = next(a for a in agg if a["pos_weight"] == "none")
    assert none_row["mean_mean_negatives_per_target"] is None


def test_mean_curves():
    got = mean_curves([{"k": [0.0, 1.0]}, {"k": [1.0, 1.0]}])
    assert got == {"k": [0.5, 1.0]}


# --------------------------------------------------------------------------- #
# end-to-end on synthetic data
# --------------------------------------------------------------------------- #
def test_plan_only_without_yes_and_dry_run_do_not_train(tmp_path, capsys):
    for extra in ([], ["--dry-run"]):
        compare_ratios.main(sweep_args(tmp_path, *extra))
        out = capsys.readouterr().out
        assert "runs = 4 ratios x 2 targets x 1 loss settings = 8" in out
        assert "1:5" in out and "cumulative" in out
        assert not (tmp_path / "sw").exists()
    assert "Re-run with --yes" in out or "Dry run only" in out


def test_sweep_end_to_end_outputs_realized_ratios_and_resume(tmp_path, monkeypatch, capsys):
    compare_ratios.main(sweep_args(tmp_path, "--yes"))
    base = tmp_path / "sw"
    for f in ("ratio_sweep_runs.csv", "ratio_sweep_by_ratio.csv", "ratio_sweep.json",
              "ratio_sweep_final_metrics.png", "ratio_curves.png"):  # fmt: skip
        assert (base / f).exists(), f
    with open(base / "ratio_sweep_runs.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 2 and {r["ratio_label"] for r in rows} == {
        "5:1",
        "1:1",
        "1:5",
        "cumulative",
    }
    by_ratio = _read(base / "ratio_sweep_by_ratio.csv")
    assert [r["ratio_label"] for r in by_ratio] == ["5:1", "1:1", "1:5", "cumulative"]
    assert all(r["n_targets"] == "2" for r in by_ratio)
    d = json.loads((base / "ratio_sweep.json").read_text())
    assert d["targets"] == [0, 3] and d["ratios"] == ["5:1", "1:1", "1:5", "cumulative"]

    # the ratio really reached the training data: 12 targets / 5 = 2 negatives, 1:1 -> 12, 1:5 -> 60
    def counts(tag, k):
        r = json.loads((base / "runs" / tag / f"target_{k}" / "results.json").read_text())
        assert r["config"]["target_to_negatives"] == parse_ratio(
            tag[len("ratio_") :].replace("to", ":")
        )
        return [(e["n_train_target"], e["n_train_negative"]) for e in r["experiences"]]

    assert counts("ratio_5to1", 0) == [(12, 2)] * 3
    assert counts("ratio_1to1", 0) == [(12, 12)] * 3
    assert counts("ratio_1to5", 3) == [
        (12, 36),
        (12, 60),
        (12, 60),
    ]  # 8 negatives -> chunks 3/3/2 -> pools 36/72/96
    assert counts("ratio_cumulative", 3) == [(12, 36), (12, 72), (12, 96)]

    # a second invocation reuses every finished run (identical config) and recomputes nothing
    real, called = compare_ratios.run_experiment, []
    monkeypatch.setattr(
        compare_ratios, "run_experiment", lambda *a, **k: called.append(1) or real(*a, **k)
    )
    capsys.readouterr()
    compare_ratios.main(sweep_args(tmp_path, "--yes"))
    assert not called and capsys.readouterr().out.count("reused:") == 8

    # ...but a changed setting (here: learning rate) invalidates the saved runs
    compare_ratios.main([*sweep_args(tmp_path, "--yes"), "--lr", "0.04"])
    assert len(called) == 8
    # and --no-resume forces recomputation
    called.clear()
    compare_ratios.main([*sweep_args(tmp_path, "--yes"), "--lr", "0.04", "--no-resume"])
    assert len(called) == 8


def test_sweep_with_loss_weighting_axis_and_no_cumulative(tmp_path):
    compare_ratios.main(
        sweep_args(
            tmp_path,
            "--yes",
            "--no-cumulative",
            "--pos-weights",
            "balanced",
            "none",
            targets=("0",),
        )
    )
    base = tmp_path / "sw"
    rows = _read(base / "ratio_sweep_runs.csv")
    assert len(rows) == 3 * 2 and {r["pos_weight"] for r in rows} == {"balanced", "none"}
    assert "cumulative" not in {r["ratio_label"] for r in rows}
    assert (base / "ratio_curves__pw-balanced.png").exists() and (
        base / "ratio_curves__pw-none.png"
    ).exists()
    cfg = json.loads(
        (base / "runs" / "ratio_1to1__pw-none" / "target_0" / "results.json").read_text()
    )["config"]
    assert cfg["train"]["pos_weight"] is None and cfg["target_to_negatives"] == 1.0


def test_sweep_argument_errors(tmp_path):
    with pytest.raises(SystemExit):
        compare_ratios.main(sweep_args(tmp_path, "--target-to-negatives", "0.2"))
    with pytest.raises(SystemExit):
        compare_ratios.main(sweep_args(tmp_path, ratios=("1:0",)))
    with pytest.raises(SystemExit):
        compare_ratios.main(sweep_args(tmp_path, targets=("99",)))


def test_metrics_csv_has_scalar_columns_only(tmp_path):
    compare_ratios.main(
        sweep_args(tmp_path, "--yes", "--run-plots", targets=("0",), ratios=("1:1",))
    )
    run = tmp_path / "sw" / "runs" / "ratio_1to1" / "target_0"
    assert (run / "plots" / "A_accuracy_vs_experience.png").exists()  # --run-plots
    rows = _read(run / "metrics.csv")
    assert len(rows) == 3
    assert not any("{" in str(v) for r in rows for v in r.values())  # no dict reprs
    assert {"feature_expansion_new_feature_dim", "probe_logit_max_abs_diff_expansion", "n_train_target",
            "n_train_negative", "target_recall"} <= set(rows[0])  # fmt: skip
    assert "feature_expansion" not in rows[0] and "output_expansion" not in rows[0]
    assert (
        rows[1]["feature_expansion_new_feature_dim"] == "2"
        and rows[0]["feature_expansion_new_feature_dim"] == ""
    )


def test_realized_counts_recorded_for_multiclass_too():
    from incremental_feature_cl.experiments.common import run_experiment
    from incremental_feature_cl.experiments.config import ExperimentConfig

    c = ExperimentConfig.from_dict({
        "mode": "multiclass", "data": {"dataset": "synthetic", "n_experiences": 2,
                                       "synthetic": {"n_classes": 4, "n_train_per_class": 8, "n_test_per_class": 4}},
        "model": {"backbone": "mlp"}, "train": {"device": "cpu", "probe_size": 0}})  # fmt: skip
    r = run_experiment(c, out_dir="/tmp/_ratio_mc", verbose=False, make_plots=False)
    assert [e["n_train_samples"] for e in r.experiences] == [16, 16]
    assert "n_train_target" not in r.experiences[0]
    assert np.isfinite(r.summary["final_accuracy_seen"])


# --------------------------------------------------------------------------- #
# review follow-ups: requested vs realized ratio, forgetting naming, resume, resummarize
# --------------------------------------------------------------------------- #
def _run(base, tag, k):
    return RunResult.load(base / "runs" / tag / f"target_{k}" / "results.json")


def test_plan_reports_which_experiences_reach_the_requested_ratio(tmp_path, capsys):
    # 12 targets/class; 8 negative classes in chunks 3/3/2 -> pools 36/72/96 negatives
    compare_ratios.main(sweep_args(tmp_path, "--dry-run", targets=("3",), ratios=("1:1", "1:5")))
    out = capsys.readouterr().out
    assert "1:1" in out and "requested ratio reached in 3/3 experiences" in out
    assert "requested ratio reached in 2/3 experiences" in out  # 1:5 -> 60 negatives from exp 1
    assert "experience 0 has too few negatives for 1:5" in out
    assert "experience 0 trains at 1:3" in out and "first full ratio at experience 1" in out
    assert "cumulative: every negative seen so far" in out


def test_summaries_expose_realized_vs_requested_ratio(tmp_path):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("3",)))
    base = tmp_path / "sw"
    rows = {r["ratio_label"]: r for r in _read(base / "ratio_sweep_runs.csv")}
    assert rows["1:5"]["fraction_experiences_at_requested_ratio"] == str(2 / 3)
    assert rows["1:5"]["first_experience_at_requested_ratio"] == "1"
    assert rows["1:5"]["final_realized_ratio_label"] == "1:5"
    assert float(rows["1:5"]["final_negatives_per_target"]) == pytest.approx(5.0)
    assert float(rows["1:5"]["requested_negatives_per_target"]) == pytest.approx(5.0)
    assert float(rows["1:5"]["mean_negatives_per_target"]) == pytest.approx((3 + 5 + 5) / 3)
    assert rows["1:1"]["fraction_experiences_at_requested_ratio"] == "1.0"
    assert rows["cumulative"]["fraction_experiences_at_requested_ratio"] == ""  # nothing to enforce
    assert rows["cumulative"]["requested_negatives_per_target"] == ""
    d = json.loads((base / "ratio_sweep.json").read_text())
    assert "upper bound" in d["ratio_semantics"]
    assert (base / "ratio_sweep_main.png").exists()

    r = _run(base, "ratio_1to5", 3)  # per-experience fields are stored in the run itself
    assert [e["ratio_satisfied"] for e in r.experiences] == [False, True, True]
    assert [e["realized_ratio_label"] for e in r.experiences] == ["1:3", "1:5", "1:5"]
    assert [p["ratio_satisfied"] for p in realized_profile(r)] == [False, True, True]


def test_realized_profile_works_for_results_saved_before_the_fields_existed(tmp_path):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("3",), ratios=("1:5",)))
    r = _run(tmp_path / "sw", "ratio_1to5", 3)
    for e in r.experiences:  # strip what an older version never wrote
        for k in ("requested_target_to_negatives", "n_requested_negative", "ratio_satisfied",
                  "realized_target_to_negatives", "realized_ratio_label"):  # fmt: skip
            e.pop(k)
    m = run_metrics(r)
    assert m["fraction_experiences_at_requested_ratio"] == pytest.approx(2 / 3)
    assert m["first_experience_at_requested_ratio"] == 1


def test_target_recall_peak_to_final_drop_is_standard_target_forgetting(tmp_path):
    """The renamed metric is exactly forgetting_per_class of the target (recall == its accuracy)."""
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0", "3"), ratios=("1:1",)))
    for k in (0, 3):
        r = _run(tmp_path / "sw", "ratio_1to1", k)
        f = forgetting_per_class(r.accuracy_matrix(), r.class_first_experience)[k]
        assert run_metrics(r)["target_recall_peak_to_final_drop"] == pytest.approx(f)
    assert "target_recall_drop" not in METRIC_KEYS


def test_peak_to_final_drop_is_measured_from_the_peak_not_the_last_dip(tmp_path):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)))
    r = _run(tmp_path / "sw", "ratio_1to1", 0)
    for e, rec in zip(r.experiences, (0.6, 0.9, 0.7)):
        e["target_metrics"]["target_recall"] = rec
    r.experiences[-1]["target_metrics"]["target_recall"] = 0.8  # dipped to 0.7, recovered to 0.8
    assert run_metrics(r)["target_recall_peak_to_final_drop"] == pytest.approx(0.9 - 0.8)


def test_resume_config_storage_vs_dataset_identity(tmp_path):
    from incremental_feature_cl.experiments.compare_ratios import _same_config

    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)))
    cfg = _run(tmp_path / "sw", "ratio_1to1", 0).config

    moved = json.loads(json.dumps(cfg))
    moved["name"] = "different-name"
    moved["output_dir"] = "/different/output"
    assert _same_config(cfg, moved)

    moved = json.loads(json.dumps(cfg))
    moved["data"]["root"] = "/mnt/d"
    assert not _same_config(cfg, moved)

    for path, value in (
        (("train", "lr"), 0.5),
        (("seed",), 9),
        (("target_to_negatives",), 0.5),
        (("train", "pos_weight"), None),
        (("target_class",), 1),
    ):  # fmt: skip
        changed = json.loads(json.dumps(cfg))
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        assert not _same_config(cfg, changed), path


def test_moving_a_finished_sweep_does_not_recompute(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    compare_ratios.main(sweep_args(first, "--yes", targets=("0",), ratios=("1:1",)))
    (second).mkdir()
    (first / "sw").rename(second / "sw")  # new output dir => stored output_dir/name differ
    called = []
    real = compare_ratios.run_experiment
    monkeypatch.setattr(
        compare_ratios, "run_experiment", lambda *a, **k: called.append(1) or real(*a, **k)
    )
    compare_ratios.main(sweep_args(second, "--yes", targets=("0",), ratios=("1:1",)))
    assert not called


def test_other_loss_weighting_never_overwrites_default_runs(tmp_path, capsys):
    compare_ratios.main(
        sweep_args(
            tmp_path,
            "--yes",
            targets=("0",),
            ratios=("1:1",),
        )
    )
    before = (tmp_path / "sw" / "runs" / "ratio_1to1" / "target_0" / "results.json").read_text()
    compare_ratios.main([*sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)),
                         "--pos-weights", "none"])  # fmt: skip
    base = tmp_path / "sw" / "runs"
    assert (base / "ratio_1to1" / "target_0" / "results.json").read_text() == before
    assert (base / "ratio_1to1__pw-none" / "target_0" / "results.json").exists()
    assert "overwritten" not in capsys.readouterr().out


def test_overwriting_a_run_with_a_different_config_is_announced(tmp_path, capsys):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)))
    capsys.readouterr()
    compare_ratios.main([*sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)),
                         "--lr", "0.04"])  # fmt: skip
    out = capsys.readouterr().out
    assert "different config" in out and "overwritten" in out and "replaced:" in out


def test_large_sweeps_print_a_loud_warning(tmp_path, capsys):
    from incremental_feature_cl.experiments.common import large_run_warning

    assert large_run_warning(49) is None
    assert "WARNING: this will run 600 experiments" in large_run_warning(600)
    compare_ratios.main(sweep_args(tmp_path, "--dry-run", targets=tuple(str(i) for i in range(9))))
    assert "WARNING" not in capsys.readouterr().out  # 36 runs: below the threshold


def test_resummarize_rebuilds_summaries_without_training(tmp_path, monkeypatch, capsys):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0", "3")))
    base = tmp_path / "sw"
    ref_rows = _read(base / "ratio_sweep_runs.csv")
    ref_agg = _read(base / "ratio_sweep_by_ratio.csv")
    for f in base.glob("ratio_*.csv"):
        f.unlink()
    for f in base.glob("*.png"):
        f.unlink()
    monkeypatch.setattr(compare_ratios, "run_experiment", lambda *a, **k: pytest.fail("trained"))
    monkeypatch.setattr(compare_ratios, "get_data", lambda *a, **k: pytest.fail("loaded data"))
    compare_ratios.main(["--resummarize", str(base)])
    key = lambda r: (r["target_class"], r["ratio_label"])
    got = _read(base / "ratio_sweep_runs.csv")
    assert sorted(got, key=key) == sorted(ref_rows, key=key)
    assert _read(base / "ratio_sweep_by_ratio.csv") == ref_agg
    for f in ("ratio_sweep_main.png", "ratio_sweep_final_metrics.png", "ratio_curves.png"):
        assert (base / f).exists()
    assert json.loads((base / "ratio_sweep.json").read_text())["resummarized_from"] == 8
    assert "WARNING" not in capsys.readouterr().out


def test_resummarize_warns_when_a_directory_mixes_settings(tmp_path, capsys):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)))
    compare_ratios.main([*sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:5",)),
                         "--lr", "0.04", "--no-cumulative"])  # fmt: skip
    capsys.readouterr()
    compare_ratios.main(["--resummarize", str(tmp_path / "sw")])
    assert "not all produced with the same settings" in capsys.readouterr().out


def test_resummarize_empty_directory_exits(tmp_path):
    with pytest.raises(SystemExit):
        compare_ratios.main(["--resummarize", str(tmp_path)])


def test_results_json_is_written_atomically(tmp_path):
    compare_ratios.main(sweep_args(tmp_path, "--yes", targets=("0",), ratios=("1:1",)))
    run = tmp_path / "sw" / "runs" / "ratio_1to1" / "target_0"
    assert (run / "results.json").exists() and not (run / "results.json.tmp").exists()
