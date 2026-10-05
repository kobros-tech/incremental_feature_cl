# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Ablation / baseline comparison: is any gain from *expansion*, *initialisation* or *replay*?

Built-in controlled variants (same trainer, same seed, same data):
  1 fixed             new_feature_dim=0
  2 fixed+replay      new_feature_dim=0, replay
  3 expand-random     new_feature_dim=D, random init
  4 expand-zero       new_feature_dim=D, zero init       <- proposed method
  5 expand-zero+replay

Optional reference methods from Avalanche (multiclass mode, ``--avalanche-baselines naive,replay,er_ace,ewc``).
Writes comparison.csv/json and comparison.png under ``<output-dir>/<name>/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import add_common_args, config_from_args, run_experiment


def variants(replay: int, new_dim: int, with_replay: bool = True):
    v = [
        ("fixed", 0, "zero", 0),
        ("expand-random", new_dim, "random", 0),
        ("expand-zero", new_dim, "zero", 0),
    ]
    if with_replay:
        v = [
            v[0],
            ("fixed+replay", 0, "zero", replay),
            v[1],
            v[2],
            ("expand-zero+replay", new_dim, "zero", replay),
        ]
    return v


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p, "cifar100")
    p.add_argument("--mode", choices=["multiclass", "target"], default="multiclass")
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument(
        "--replay", type=int, default=200, help="replay memory size for the replay variants"
    )
    p.add_argument("--no-replay-variants", action="store_true")
    p.add_argument("--avalanche-baselines", default="", help="comma list: naive,replay,er_ace,ewc")
    a = p.parse_args(argv)
    cfg = config_from_args(a, a.mode, "cifar100")
    cfg.target_class = a.target_class
    if cfg.model.new_feature_dim == 0:
        cfg.model.new_feature_dim = 16
    D = cfg.model.new_feature_dim
    out = Path(cfg.output_dir) / (cfg.name or f"compare_{cfg.auto_name()}")
    rows = []
    for label, d, init, rep in variants(a.replay, D, not a.no_replay_variants):
        c = copy.deepcopy(cfg)
        c.model.new_feature_dim, c.model.initialization, c.train.replay_mem_size = d, init, rep
        c.name = f"{label}"
        r = run_experiment(c, out_dir=out / label, make_plots=not a.no_plots)
        rows.append({"method": label, **r.summary})
    for name in [x for x in a.avalanche_baselines.split(",") if x]:
        c = copy.deepcopy(cfg)
        c.mode, c.backend, c.name = "multiclass", "avalanche", f"avalanche_{name}"
        c.model.new_feature_dim = 0
        c.train.replay_mem_size = a.replay
        r = run_experiment(
            c, out_dir=out / f"avalanche_{name}", make_plots=not a.no_plots, avalanche_baseline=name
        )
        rows.append({"method": f"avalanche:{name}", **r.summary})

    out.mkdir(parents=True, exist_ok=True)
    keys = [
        "method",
        "final_accuracy_seen",
        "average_accuracy_seen",
        "average_forgetting",
        "final_feature_dim",
        "final_parameter_count",
        "total_train_time_s",
    ]
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2, default=float))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, k, t in zip(
        axes,
        ["final_accuracy_seen", "average_forgetting", "final_parameter_count"],
        ["final accuracy (seen)", "average forgetting", "parameters"],
    ):
        ax.barh([r["method"] for r in rows], [r[k] for r in rows])
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130)
    print(f"comparison -> {out}")
    for r in rows:
        print(
            f"  {r['method']:<22} final={r['final_accuracy_seen']:.3f} avg={r['average_accuracy_seen']:.3f} "
            f"forgetting={r['average_forgetting']:.3f} params={r['final_parameter_count']}"
        )


if __name__ == "__main__":
    main()
