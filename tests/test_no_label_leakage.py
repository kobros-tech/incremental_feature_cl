# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Predictions may depend only on x and model state, never on (test) labels."""

import inspect

import numpy as np
import torch

from incremental_feature_cl.data import ArrayDataset
from incremental_feature_cl.evaluation import predict_dataset
from incremental_feature_cl.training import evaluate_state


def test_forward_signature_takes_only_inputs(conv_model):
    assert list(inspect.signature(conv_model.forward).parameters) == ["x"]
    assert list(inspect.signature(conv_model.features).parameters) == ["x"]


def test_labels_do_not_influence_predictions(synth, conv_model):
    _, test = synth
    conv_model.expand_outputs(7)
    for mode in ("multiclass", "target"):
        p0, _ = predict_dataset(conv_model, test, mode)
        shuffled = ArrayDataset(test.x, np.random.default_rng(0).permutation(test.targets.numpy()))
        wrong = ArrayDataset(test.x, np.zeros(len(test), dtype=int))
        for other in (shuffled, wrong):
            p1, _ = predict_dataset(conv_model, other, mode)
            assert np.array_equal(p0, p1)


def test_state_unchanged_by_evaluation(synth, conv_model):
    _, test = synth
    conv_model.expand_outputs(7)
    before = {k: v.clone() for k, v in conv_model.state_dict().items()}
    n_log = len(conv_model.expansion_log)
    evaluate_state(conv_model, test, 10, "multiclass", list(range(10)))
    assert len(conv_model.expansion_log) == n_log  # evaluation never expands / routes
    for k, v in conv_model.state_dict().items():
        assert torch.equal(before[k], v)
