# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Sweep the target:negative training ratio of target-vs-rest, for one or more target classes.

Built on ``train_target``: every run is a normal target-vs-rest experiment that differs only in
``target_to_negatives``.  Ratios are written ``target:negative``; ``5:1`` = 5 targets per negative,
``1:5`` = 1 target per 5 negatives.  The default grid is 5:1, 2:1, 1:1, 1:2, 1:5 plus the original
``cumulative`` behaviour (no subsampling) as a reference.

    python -m incremental_feature_cl.experiments.compare_ratios --dataset cifar100 \\
        --target-class 17 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1 --yes
    python -m incremental_feature_cl.experiments.compare_ratios --target-class 0 17 50 99 --yes
    python -m incremental_feature_cl.experiments.compare_ratios --ratios 5:1 1:1 1:5 --no-cumulative

Note on the loss: with the default ``train.pos_weight=balanced`` the loss re-weights classes to 1:1
whatever the data ratio, so the sweep then isolates *how many / how varied* the negatives are. To also
see the effect of class imbalance in the loss, add ``--pos-weights balanced none`` (adds an axis).

Runs are resumable: a run whose saved ``results.json`` has an identical config is reused
(``--no-resume`` recomputes).  Outputs under ``<output-dir>/<name>/``:
``ratio_sweep_runs.csv`` (one row per run), ``ratio_sweep_by_ratio.csv`` (mean/std over targets),
``ratio_sweep.json``, ``ratio_sweep_final_metrics.png``, ``ratio_curves*.png`` and ``runs/<ratio>/target_<k>/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from ..data.streams import format_ratio, parse_ratio
from ..evaluation.ratio_summary import (
    aggregate_rows,
    make_row,
    mean_curves,
    ratio_sort_key,
    run_curves,
)
from ..evaluation.result_store import RunResult
from ..plotting.ratio_sweep import plot_ratio_curves, plot_ratio_final_metrics
from ..utils.reproducibility import collect_environment, resolve_device
from .common import (
    add_common_args,
    config_from_args,
    estimate_train_samples,
    get_data,
    run_experiment,
)
from .config import ExperimentConfig

DEFAULT_RATIOS = ["5:1", "2:1", "1:1", "1:2", "1:5"]


def parse_ratios(tokens: list[str], include_cumulative: bool = True) -> list[float | None]:
    """Parse ``target:negative`` tokens; de-duplicate; order 5:1 ... 1:5 with cumulative last."""
    ratios: list[float | None] = []
    for t in tokens:
        r = parse_ratio(t)
        if r in ratios:
            raise ValueError(f"ratio {t!r} given twice (equals {format_ratio(r)})")
        ratios.append(r)
    if include_cumulative and None not in ratios:
        ratios.append(None)
    if not include_cumulative and None in ratios:
        ratios.remove(None)
    if not ratios:
        raise ValueError("no ratios to run")
    return sorted(ratios, key=ratio_sort_key)


def parse_pos_weights(tokens: list[str]) -> list[Any]:
    """``balanced`` | ``none`` | a number -> values accepted by ``train.pos_weight``."""
    out: list[Any] = []
    for t in tokens:
        low = t.strip().lower()
        v: Any = (
            "balanced" if low == "balanced" else None if low in {"none", "null"} else float(low)
        )
        if v in out:
            raise ValueError(f"pos-weight {t!r} given twice")
        out.append(v)
    return out


def pos_weight_label(pw: Any) -> str:
    return "none" if pw is None else pw if isinstance(pw, str) else f"{pw:g}"


def _same_config(a: dict, b: dict) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def _load_if_same(cfg: ExperimentConfig, run_dir: Path) -> RunResult | None:
    f = run_dir / "results.json"
    if not f.exists():
        return None
    try:
        result = RunResult.load(f)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None
    return result if _same_config(result.config, cfg.to_dict()) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _print_table(aggregate: list[dict[str, Any]]) -> None:
    for pw in dict.fromkeys(a["pos_weight"] for a in aggregate):
        print(f"\n  mean over targets (pos_weight={pw}); negatives = seen negative classes")
        print(
            f"  {'ratio':<11}{'recall':>8}{'neg.acc':>9}{'bal.acc':>9}{'F1':>7}{'recall drop':>13}"
        )
        for a in (x for x in aggregate if x["pos_weight"] == pw):
            print(
                f"  {a['ratio_label']:<11}{a['mean_final_target_recall']:>8.3f}"
                f"{a['mean_final_negative_accuracy']:>9.3f}{a['mean_final_balanced_accuracy']:>9.3f}"
                f"{a['mean_final_target_f1']:>7.3f}{a['mean_target_recall_drop']:>13.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p, "cifar100")
    p.add_argument(
        "--target-class",
        nargs="+",
        default=["17"],
        metavar="K|all",
        help="one or more target classes, or 'all' (default: 17)",
    )
    p.add_argument(
        "--ratios",
        nargs="+",
        default=DEFAULT_RATIOS,
        metavar="T:N",
        help="target:negative ratios, e.g. 5:1 1:1 1:5 (also numbers = targets per "
        "negative, or 'cumulative'); default: %(default)s",
    )
    p.add_argument(
        "--no-cumulative",
        action="store_true",
        help="do not add the cumulative-negatives reference run",
    )
    p.add_argument(
        "--pos-weights",
        nargs="+",
        default=None,
        metavar="W",
        help="loss weighting axis: balanced | none | <number> (default: config value)",
    )
    p.add_argument(
        "--no-target-in-every-experience",
        action="store_true",
        help="present target samples only in experience 0",
    )
    p.add_argument("--yes", action="store_true", help="confirm and run the sweep")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument(
        "--run-plots",
        action="store_true",
        help="also make the standard A-H plots for every run (results.json is always saved)",
    )
    p.add_argument("--no-resume", action="store_true", help="recompute runs even if results exist")
    p.add_argument("--verbose", action="store_true", help="print every experience of every run")
    return p


def main(argv=None) -> None:
    p = build_parser()
    a = p.parse_args(argv)
    if a.target_to_negatives is not None:
        p.error("use --ratios for the sweep; --target-to-negatives is the single-run flag")
    try:
        ratios = parse_ratios(a.ratios, include_cumulative=not a.no_cumulative)
        cfg = config_from_args(a, "target", "cifar100")
        pws = parse_pos_weights(a.pos_weights) if a.pos_weights else [cfg.train.pos_weight]
    except ValueError as e:
        p.error(str(e))
    if a.no_target_in_every_experience:
        cfg.target_in_every_experience = False
    cfg.train.device = str(resolve_device(cfg.train.device))
    _, _, num_classes, _ = get_data(cfg)
    targets = (
        list(range(num_classes)) if a.target_class == ["all"] else [int(k) for k in a.target_class]
    )
    bad = [k for k in targets if not 0 <= k < num_classes]
    if bad:
        p.error(f"target classes {bad} outside [0, {num_classes - 1}]")
    multi_pw = len(pws) > 1

    init = cfg.model.initialization if cfg.model.new_feature_dim > 0 else "fixed"
    base = Path(cfg.output_dir) / (
        cfg.name
        or f"ratio_sweep_{cfg.data.dataset}_e{cfg.data.n_experiences}"
        f"_d{cfg.model.new_feature_dim}_{init}_s{cfg.seed}"
    )

    # ---- plan -----------------------------------------------------------------
    print("=== target:negative ratio sweep plan ===")
    print(
        f"dataset={cfg.data.dataset} n_experiences={cfg.data.n_experiences} "
        f"epochs={cfg.train.train_epochs} new_feature_dim={cfg.model.new_feature_dim} "
        f"init={init} seed={cfg.seed} device={cfg.train.device}"
    )
    print(
        f"targets ({len(targets)}): {targets if len(targets) <= 12 else str(targets[:12])[:-1] + ', ...]'}"
    )
    print(f"loss weighting (train.pos_weight): {[pos_weight_label(w) for w in pws]}")
    if not multi_pw and pws[0] == "balanced":
        print(
            "  note: 'balanced' loss weighting makes class weights 1:1 for every data ratio; "
            "add --pos-weights balanced none to also test the unweighted loss"
        )
    total_passes = 0
    for r in ratios:
        probe = copy.deepcopy(cfg)
        probe.target_class, probe.target_to_negatives = targets[0], r
        per_run = estimate_train_samples(probe)
        total_passes += per_run * len(targets) * len(pws)
        print(f"  {format_ratio(r):<11} ~{per_run:>11,} training samples per run")
    n_runs = len(ratios) * len(targets) * len(pws)
    print(
        f"runs = {len(ratios)} ratios x {len(targets)} targets x {len(pws)} loss settings = {n_runs}; "
        f"~{total_passes:,} sample-passes in total (+ a full test-set evaluation per experience)"
    )
    print(f"output: {base}")
    if a.dry_run or not a.yes:
        print("Dry run only." if a.dry_run else "Not running. Re-run with --yes to start.")
        return

    # ---- runs ---------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[tuple[float | None, str], list[dict]]] = {}
    i = 0
    for pw in pws:
        pw_lab = pos_weight_label(pw)
        for ratio in ratios:
            lab = format_ratio(ratio)
            tag = "ratio_" + lab.replace(":", "to") + (f"__pw-{pw_lab}" if multi_pw else "")
            for k in targets:
                i += 1
                c = copy.deepcopy(cfg)
                c.target_class, c.target_to_negatives, c.train.pos_weight = k, ratio, pw
                c.name = c.auto_name() + (f"_pw-{pw_lab}" if multi_pw else "")
                run_dir = base / "runs" / tag / f"target_{k}"
                result = None if a.no_resume else _load_if_same(c, run_dir)
                status = "reused"
                if result is None:
                    status = "ran"
                    print(f"[{i}/{n_runs}] target={k} ratio={lab} pos_weight={pw_lab}", flush=True)
                    result = run_experiment(
                        c, out_dir=run_dir, make_plots=a.run_plots, verbose=a.verbose
                    )
                row = make_row(result, k, ratio, pw_lab)
                rows.append(row)
                curves.setdefault(pw_lab, {}).setdefault((ratio, lab), []).append(
                    run_curves(result)
                )
                print(
                    f"[{i}/{n_runs}] {status}: target={k} ratio={lab} pw={pw_lab} "
                    f"recall={row['final_target_recall']:.3f} neg_acc={row['final_negative_accuracy']:.3f} "
                    f"balanced={row['final_balanced_accuracy']:.3f}",
                    flush=True,
                )

    # ---- summaries ---------------------------------------------------------------------
    aggregate = aggregate_rows(rows)
    base.mkdir(parents=True, exist_ok=True)
    _write_csv(base / "ratio_sweep_runs.csv", rows)
    _write_csv(base / "ratio_sweep_by_ratio.csv", aggregate)
    (base / "ratio_sweep.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "environment": collect_environment(),
                "targets": targets,
                "ratios": [format_ratio(r) for r in ratios],
                "pos_weights": [pos_weight_label(w) for w in pws],
                "runs": rows,
                "by_ratio": aggregate,
            },
            indent=2,
        )
    )
    if not a.no_plots:
        plot_ratio_final_metrics(rows, aggregate, base / "ratio_sweep_final_metrics.png")
        for pw_lab, by_ratio in curves.items():
            mean = {key: mean_curves(cs) for key, cs in by_ratio.items()}
            suffix = f"__pw-{pw_lab}" if multi_pw else ""
            plot_ratio_curves(
                mean,
                base / f"ratio_curves{suffix}.png",
                f" (mean of {len(targets)} target{'s' if len(targets) > 1 else ''})",
            )
    _print_table(aggregate)
    print(f"\nratio sweep -> {base}")


if __name__ == "__main__":
    main()
