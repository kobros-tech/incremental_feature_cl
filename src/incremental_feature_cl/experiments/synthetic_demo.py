# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Milestone 1b: smallest end-to-end neural experiment (synthetic data, CPU, seconds, no downloads).

    python -m incremental_feature_cl.experiments.synthetic_demo [--mode target]

Runs zero- vs random-initialised expansion and prints the preservation check
(max |logit change| caused by expansion; exactly 0 for zero initialisation).
"""

import argparse

from .common import run_experiment
from .config import ExperimentConfig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["target"], default="target")
    ap.add_argument("--output-dir", default="results")
    a = ap.parse_args(argv)
    for init in ("zero", "random"):
        cfg = ExperimentConfig.from_dict(
            {
                "mode": a.mode,
                "seed": 1,
                "output_dir": a.output_dir,
                "name": f"synthetic_{a.mode}_{init}",
                "data": {
                    "dataset": "synthetic",
                    "n_experiences": 5,
                    "synthetic": {"n_classes": 10, "n_train_per_class": 40, "n_test_per_class": 20},
                },
                "model": {"backbone": "smallconv", "new_feature_dim": 8, "initialization": init},
                "train": {"train_epochs": 3, "lr": 0.05, "device": "cpu", "train_mb_size": 16},
            }
        )
        run_experiment(cfg)


if __name__ == "__main__":
    main()
