# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Existing Avalanche CL methods used as reference baselines (not reimplemented here).

They are given a fixed-width model (``new_feature_dim=0``) with the output head
pre-allocated for all classes, which is how these methods are normally run.
iCaRL needs a feature-extractor/classifier split and is not wired up yet.
"""

from __future__ import annotations

import torch

from .strategy import quiet_evaluator

AVALANCHE_BASELINES = ("naive", "replay", "er_ace", "ewc")


def make_avalanche_baseline(
    name: str,
    model,
    *,
    lr: float = 0.01,
    momentum: float = 0.9,
    train_mb_size: int = 32,
    train_epochs: int = 1,
    eval_mb_size: int = 256,
    device: str = "cpu",
    mem_size: int = 200,
    ewc_lambda: float = 1.0,
):
    from avalanche.training.supervised import ER_ACE, EWC, Naive, Replay

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    common = {
        "model": model,
        "optimizer": opt,
        "criterion": torch.nn.CrossEntropyLoss(),
        "train_mb_size": train_mb_size,
        "train_epochs": train_epochs,
        "eval_mb_size": eval_mb_size,
        "device": device,
        "evaluator": quiet_evaluator(),
    }
    key = name.lower()
    if key == "naive":
        return Naive(**common)
    if key == "replay":
        return Replay(mem_size=mem_size, **common)
    if key == "er_ace":
        return ER_ACE(mem_size=mem_size, batch_size_mem=train_mb_size, **common)
    if key == "ewc":
        return EWC(ewc_lambda=ewc_lambda, **common)
    raise ValueError(f"unknown Avalanche baseline {name!r}; choose from {AVALANCHE_BASELINES}")
