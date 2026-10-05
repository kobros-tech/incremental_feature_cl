# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Grid sweep over config keys.  Prints the plan first; needs --yes to execute.

Preset matching the planned CIFAR-100 matrix (5/10/20 experiences x dim 0/4/16/32 x zero/random):

    python -m incremental_feature_cl.experiments.run_sweep --preset cifar100-matrix --dry-run

Custom:  --grid data.n_experiences=5,10 --grid model.new_feature_dim=0,16
Runs with new_feature_dim=0 are de-duplicated (initialisation is irrelevant there).
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path

import yaml

from .common import add_common_args, config_from_args, estimate_train_samples, run_experiment
from .config import apply_overrides

PRESETS = {
    "cifar100-matrix": {
        "data.n_experiences": [5, 10, 20],
        "model.new_feature_dim": [0, 4, 16, 32],
        "model.initialization": ["zero", "random"],
    },
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p, "cifar100")
    p.add_argument("--mode", choices=["multiclass", "target"], default="multiclass")
    p.add_argument("--target-class", type=int, default=17)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    cfg = config_from_args(a, a.mode, "cifar100")
    cfg.target_class = a.target_class
    grid = dict(PRESETS[a.preset]) if a.preset else {}
    for g in a.grid:
        k, _, v = g.partition("=")
        grid[k] = [yaml.safe_load(x) for x in v.split(",")]
    if not grid:
        p.error("give --preset or at least one --grid")

    keys = list(grid)
    jobs = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        settings = dict(zip(keys, combo))
        if (
            settings.get("model.new_feature_dim", cfg.model.new_feature_dim) == 0
            and settings.get("model.initialization", "zero")
            != grid.get("model.initialization", ["zero"])[0]
        ):
            continue  # init is irrelevant for the fixed baseline
        jobs.append(settings)
    per_run = estimate_train_samples(cfg)
    print("=== sweep plan ===")
    print(
        f"mode={a.mode} dataset={cfg.data.dataset} epochs={cfg.train.train_epochs} device={cfg.train.device}"
    )
    for j in jobs:
        print("  ", j)
    print(f"{len(jobs)} runs; ~{per_run:,} training samples per run (base config)")
    if a.dry_run or not a.yes:
        print("Not running. Re-run with --yes to start." if not a.dry_run else "Dry run only.")
        return

    base = Path(cfg.output_dir) / (cfg.name or "sweep")
    rows = []
    for j in jobs:
        d = apply_overrides(copy.deepcopy(cfg.to_dict()), [f"{k}={v}" for k, v in j.items()])
        c = type(cfg).from_dict(d)
        c.name = c.auto_name()
        r = run_experiment(c, out_dir=base / c.name, make_plots=not a.no_plots)
        rows.append({**{k: v for k, v in j.items()}, **r.summary})
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"sweep summary -> {base / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
