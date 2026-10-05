# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plots are regenerated from results.json only (no retraining needed)."""

from __future__ import annotations

from pathlib import Path

from ..evaluation.result_store import RunResult
from .class_accuracy import plot_accuracy_heatmap, plot_accuracy_vs_class, plot_forgetting_vs_class
from .experience_accuracy import plot_accuracy_vs_experience, plot_class_accuracy_curves
from .feature_growth import plot_expansion_probe, plot_feature_growth, plot_new_feature_utilization


def make_all_plots(result: RunResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir) / "plots"
    paths = [
        plot_accuracy_vs_experience(result, out / "A_accuracy_vs_experience.png"),
        plot_accuracy_vs_class(result, out / "B_accuracy_vs_class.png"),
        plot_accuracy_heatmap(result, out / "C_class_accuracy_heatmap.png"),
        plot_class_accuracy_curves(result, out / "D_class_accuracy_curves.png"),
        plot_forgetting_vs_class(result, out / "E_forgetting_vs_class.png"),
        plot_feature_growth(result, out / "F_feature_growth.png"),
        plot_new_feature_utilization(result, out / "G_new_feature_utilization.png"),
        plot_expansion_probe(result, out / "H_expansion_probe.png"),
    ]
    return [p for p in paths if p is not None]


__all__ = [
    "make_all_plots",
    "plot_accuracy_heatmap",
    "plot_accuracy_vs_class",
    "plot_accuracy_vs_experience",
    "plot_class_accuracy_curves",
    "plot_expansion_probe",
    "plot_feature_growth",
    "plot_forgetting_vs_class",
    "plot_new_feature_utilization",
]
