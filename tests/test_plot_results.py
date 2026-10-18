# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Plotting of continual-learning results."""

import pytest

pytest.importorskip("matplotlib")

from experiments import continual_eval as ce
from experiments.plot_results import plot_continual


def test_continual_plots_are_written_from_a_saved_result(tmp_path):
    result = ce.run_comparison(
        "digits", seeds=(0, 1), n_experiences=3, n_epochs=1, new_feature_dim=2
    )
    paths = plot_continual(result, tmp_path)
    names = {p.name for p in paths}
    assert names == {
        "recall_over_experiences.png",
        "accuracy_over_experiences.png",
        "recall_matrix_naive.png",
        "recall_matrix_replay.png",
        "summary_bars.png",
    }
    assert all(p.stat().st_size > 1000 for p in paths)
