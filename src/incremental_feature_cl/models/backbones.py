# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Use any external ``nn.Module`` as the shared representation ``h(x)``.

The package ships no architectures of its own.  Any model works, e.g. Avalanche's
``SimpleMLP`` or ``resnet32``::

    from avalanche.models import SimpleMLP
    backbone = FeatureBackbone(SimpleMLP(input_size=3072, hidden_size=512), input_shape=(3, 32, 32))
    model = IncrementalFeatureMapModel(backbone)           # out_dim is read from the backbone

:class:`FeatureBackbone` finds the feature path of the wrapped model in this order:

1. ``model.get_features(x)``  (Avalanche ``BaseModel`` convention, e.g. ``SimpleMLP``);
2. ``model.forward_rep(x)``   (Avalanche ``ResNet`` / ``resnet32``);
3. a final ``Linear`` named ``classifier`` / ``fc`` / ``head`` is replaced by ``Identity``
   (torchvision-style models);
4. otherwise the model is assumed to return features already.

For the Skill models the representation must be frozen (skills replay features), so it has to be
informative before it is frozen: load pretrained weights (``weights=``) or call
:func:`warm_up_backbone` once on the first experience.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_HEAD_NAMES = ("classifier", "fc", "head")
# short names for the Avalanche models; any dotted path ``package.module.Name`` works as well
ALIASES = {
    "simplemlp": "avalanche.models.SimpleMLP",
    "resnet32": "avalanche.models.resnet32.resnet32",
}


class IdentityBackbone(nn.Module):
    """Flatten only: ``h(x) = x`` (a classifier on raw inputs)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = int(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class FeatureBackbone(nn.Module):
    """Adapter that exposes an external model as ``x -> (B, out_dim)`` with an ``out_dim`` attribute."""

    def __init__(self, model: nn.Module, input_shape: Sequence[int] | None = None):
        super().__init__()
        self.model = model
        if hasattr(model, "get_features"):
            self._features = model.get_features
        elif hasattr(model, "forward_rep"):
            self._features = model.forward_rep
        else:
            for name in _HEAD_NAMES:
                if isinstance(getattr(model, name, None), nn.Linear):
                    setattr(model, name, nn.Identity())
                    break
            self._features = model
        if input_shape is None:
            raise ValueError("input_shape is required to infer the feature dimension")
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.out_dim = int(self(torch.zeros(2, *input_shape)).shape[1])
        self.train(was_training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._features(x).flatten(1)


def build_backbone(
    spec: str,
    input_shape: Sequence[int],
    kwargs: dict[str, Any] | None = None,
    weights: str | None = None,
) -> nn.Module:
    """Backbone from a name: ``identity``, an alias (``simplemlp``, ``resnet32``) or a dotted path.

    The class / factory is called with ``kwargs``; ``input_size`` is filled in from ``input_shape``
    when the callable takes one and none was given.  ``weights`` is a ``torch.save``d state dict of
    the external model (loaded with ``strict=False`` before the head is dropped).
    """
    kwargs = dict(kwargs or {})
    if spec.lower() == "identity":
        return IdentityBackbone(int(np.prod(input_shape)))
    path = ALIASES.get(spec.lower(), spec).replace(":", ".")
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"backbone {spec!r}: use 'identity', an alias or 'package.module.Name'")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import backbone {spec!r} ({exc})") from exc
    if "input_size" in inspect.signature(factory).parameters:
        kwargs.setdefault("input_size", int(np.prod(input_shape)))
    model = factory(**kwargs)
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    return FeatureBackbone(model, input_shape)


def warm_up_backbone(
    backbone: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float = 0.01,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> float:
    """Train ``backbone`` (with a temporary linear head, cross-entropy) on ``(x, y)``, then freeze it.

    Meant to be called once, on the first experience, before a frozen representation is used by
    the skills.  Returns the final mean loss.
    """
    device = torch.device(device)
    out_dim = int(backbone.out_dim)
    head = nn.Linear(out_dim, int(y.max()) + 1).to(device)
    backbone.to(device).requires_grad_(True).train()
    params = [*backbone.parameters(), *head.parameters()]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    loss_value = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            loss = F.cross_entropy(head(backbone(x[idx].to(device))), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_value = float(loss.detach())
    backbone.requires_grad_(False).eval()
    return loss_value
