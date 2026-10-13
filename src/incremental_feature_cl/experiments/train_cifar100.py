# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Mode B: class-incremental multiclass classification (e.g. Split CIFAR-100, 20 x 5 classes).

    python -m incremental_feature_cl.experiments.train_cifar100 \\
        --dataset cifar100 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1
"""

from __future__ import annotations

import argparse

from .common import add_common_args, config_from_args, run_experiment


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(p, "cifar100")
    p.add_argument(
        "--backend",
        choices=["torch", "avalanche"],
        default=None,
        help="torch: built-in trainer (default). avalanche: Avalanche strategy + SplitCIFAR100",
    )
    a = p.parse_args(argv)
    cfg = config_from_args(a, "multiclass", "cifar100")
    run_experiment(cfg, make_plots=not a.no_plots)


if __name__ == "__main__":
    main()
