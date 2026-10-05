# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Experiment configuration (YAML-loadable, CLI-overridable, stored with every result)."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "synthetic"  # synthetic | cifar10 | cifar100
    root: str = "./data"
    n_experiences: int = 5
    class_order: list[int] | None = None
    synthetic: dict[str, Any] = field(default_factory=dict)  # kwargs for make_synthetic_dataset


@dataclass
class ModelConfig:
    backbone: str = "smallconv"  # smallconv | slimresnet18 | mlp | identity
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)
    new_feature_dim: int = 0  # 0 => fixed-dimension baseline
    initialization: str = "zero"  # zero | random   (classifier contribution of new block)
    output_init: str = "zero"
    activation: str = "relu"
    freeze_backbone: bool = False
    freeze_old_blocks: bool = False


@dataclass
class TrainConfig:
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    train_epochs: int = 1
    train_mb_size: int = 32
    eval_mb_size: int = 256
    device: str = "auto"
    replay_mem_size: int = 0
    pos_weight: Any = "balanced"
    probe_size: int = 512


@dataclass
class ExperimentConfig:
    name: str = ""
    mode: str = "multiclass"  # multiclass | target
    target_class: int = 0
    target_in_every_experience: bool = True
    backend: str = "torch"  # torch | avalanche
    seed: int = 1
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def auto_name(self) -> str:
        m, d = self.model, self.data
        init = m.initialization if m.new_feature_dim > 0 else "fixed"
        tgt = f"_t{self.target_class}" if self.mode == "target" else ""
        rep = f"_rep{self.train.replay_mem_size}" if self.train.replay_mem_size else ""
        return (
            f"{self.mode}{tgt}_{d.dataset}_e{d.n_experiences}_d{m.new_feature_dim}_{init}{rep}"
            f"_s{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        return _build(cls, d)


def _build(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        default = f.default_factory() if f.default_factory is not MISSING else f.default  # type: ignore
        v = d[f.name]
        kw[f.name] = (
            _build(type(default), v) if is_dataclass(default) and isinstance(v, dict) else v
        )
    return cls(**kw)


def apply_overrides(d: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    return ExperimentConfig.from_dict(apply_overrides(d or {}, overrides))
