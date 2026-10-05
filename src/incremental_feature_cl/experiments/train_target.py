# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Mode A: target-vs-rest incremental classification.

    python -m incremental_feature_cl.experiments.train_target \\
        --dataset cifar100 --target-class 17 --n-experiences 20 --new-feature-dim 16 \\
        --train-epochs 5 --seed 1

``--target-class`` accepts several ids or ``all`` (a full sweep prints an estimate and needs ``--yes``).
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from .common import (
    add_common_args,
    config_from_args,
    estimate_train_samples,
    get_data,
    run_experiment,
)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p, "cifar100")
    p.add_argument("--target-class", nargs="+", default=["0"], metavar="K|all")
    p.add_argument("--yes", action="store_true", help="confirm a multi-run sweep")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument(
        "--no-target-in-every-experience",
        action="store_true",
        help="present target samples only in experience 0",
    )
    a = p.parse_args(argv)
    cfg = config_from_args(a, "target", "cifar100")
    if a.no_target_in_every_experience:
        cfg.target_in_every_experience = False
    _, _, num_classes, _ = get_data(cfg)
    targets = (
        list(range(num_classes)) if a.target_class == ["all"] else [int(k) for k in a.target_class]
    )

    if len(targets) > 1 or a.dry_run:
        cfg.target_class = targets[0]
        per_run = estimate_train_samples(cfg)
        print("=== target-vs-rest sweep plan ===")
        print(
            f"dataset={cfg.data.dataset} targets={len(targets)} n_experiences={cfg.data.n_experiences} "
            f"epochs={cfg.train.train_epochs} new_feature_dim={cfg.model.new_feature_dim} "
            f"init={cfg.model.initialization} device={cfg.train.device}"
        )
        print(
            f"runs={len(targets)}  training samples/run~{per_run:,}  total~{per_run * len(targets):,} "
            f"sample-passes (+ a full test-set evaluation after every experience)"
        )
        if a.dry_run or not a.yes:
            print("Not running. Re-run with --yes to start." if not a.dry_run else "Dry run only.")
            return

    user_name = cfg.name
    base = Path(cfg.output_dir) / (user_name or "target_sweep")
    rows = []
    for k in targets:
        c = copy.deepcopy(cfg)
        c.target_class, c.name = k, ""
        c.name = user_name if (user_name and len(targets) == 1) else c.auto_name()
        out = base / f"target_{k}" if len(targets) > 1 else Path(cfg.output_dir) / c.name
        r = run_experiment(c, out_dir=out, make_plots=not a.no_plots)
        rows.append({"target_class": k, **r.summary})
    if len(rows) > 1:
        base.mkdir(parents=True, exist_ok=True)
        with open(base / "sweep_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"sweep summary -> {base / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
