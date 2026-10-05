# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Small configurable backbones.  Each exposes ``out_dim`` and returns ``(B, out_dim)``.

The feature-expansion mechanism only needs ``out_dim``; it is independent of
the backbone architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class IdentityBackbone(nn.Module):
    """Flatten only: Phi_0(x) = x.  Used for synthetic tests."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (128,), out_dim: int = 64):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class SmallConvBackbone(nn.Module):
    """Tiny CNN for fast CPU experiments / smoke tests."""

    def __init__(self, in_channels: int = 3, width: int = 16, out_dim: int = 64):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.fc = nn.Linear(2 * width, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.relu(self.c2(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.relu(self.fc(x))


class _BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class SlimResNet18(nn.Module):
    """ResNet-18 layout with width ``nf`` (default 20, as in the CL literature); out_dim = 8*nf."""

    def __init__(self, in_channels: int = 3, nf: int = 20):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        cfg = [(nf, 1), (2 * nf, 2), (4 * nf, 2), (8 * nf, 2)]
        layers, in_planes = [], nf
        for planes, stride in cfg:
            layers += [_BasicBlock(in_planes, planes, stride), _BasicBlock(planes, planes, 1)]
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.out_dim = 8 * nf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)


_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "identity": IdentityBackbone,
    "mlp": MLPBackbone,
    "smallconv": SmallConvBackbone,
    "slimresnet18": SlimResNet18,
}


def build_backbone(name: str, **kwargs) -> nn.Module:
    """Build a backbone by name; the result always has an ``out_dim`` attribute."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
