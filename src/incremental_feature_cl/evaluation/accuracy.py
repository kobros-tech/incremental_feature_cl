# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Prediction, correctness and per-class accuracy.  Labels are used only to *score*."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.streams import binary_labels


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    mode: str,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (decisions, labels).  The model sees ``x`` only.

    multiclass: decision = argmax logit.   target: decision = 1[logit_0 > 0] (sigmoid > 0.5).
    """
    was_training = model.training
    model.eval()
    preds, labels = [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x, y = batch[0].to(device), batch[1]
        logits = model(x)  # <- labels are never passed to the model
        preds.append(
            (logits[:, 0] > 0).long().cpu() if mode == "target" else logits.argmax(1).cpu()
        )
        labels.append(y)
    model.train(was_training)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def correctness(
    decisions: np.ndarray, labels: np.ndarray, mode: str, target_class: int | None = None
) -> np.ndarray:
    if mode == "target":
        return decisions == binary_labels(labels, target_class).numpy()
    return decisions == labels


def per_class_accuracy(correct: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """(num_classes,) accuracy; NaN for classes absent from ``labels``."""
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    if correct.shape != labels.shape:
        raise ValueError("correct and labels must have the same shape")
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("labels must be valid class indices")
    out = np.full(num_classes, np.nan, dtype=float)
    for c in np.unique(labels):
        out[c] = correct[labels == c].mean()
    return out


def masked_accuracy(correct: np.ndarray, labels: np.ndarray, classes) -> float:
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    if correct.shape != labels.shape:
        raise ValueError("correct and labels must have the same shape")
    m = np.isin(labels, list(classes))
    return float(correct[m].mean()) if m.any() else float("nan")
