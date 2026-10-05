# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Evaluate the current model state on the full test set (backend-independent)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from ..evaluation.accuracy import correctness, masked_accuracy, per_class_accuracy, predict_dataset
from ..evaluation.target_metrics import binary_target_metrics


def evaluate_state(
    model: torch.nn.Module,
    test_dataset: Dataset,
    num_classes: int,
    mode: str,
    seen_classes: Sequence[int],
    target_class: int | None = None,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    dec, lab = predict_dataset(model, test_dataset, mode, batch_size, device)
    ok = correctness(dec, lab, mode, target_class)
    out: dict[str, Any] = {
        "per_class_accuracy": per_class_accuracy(ok, lab, num_classes).tolist(),
        "accuracy_seen": masked_accuracy(ok, lab, seen_classes),
        "accuracy_all": float(ok.mean()),
    }
    if mode == "target":
        negs = [c for c in seen_classes if c != target_class]
        out["target_metrics"] = binary_target_metrics(dec, lab, target_class, negs)
    return out
