# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Plain-PyTorch continual trainer (no Avalanche dependency).

Per experience ``t``:

 1. probe BEFORE expansion      (fixed diagnostic subset of already-seen classes)
 2. expand feature space        (policy)           [independent of output growth]
 3. probe AFTER expansion       -> does expansion itself change old predictions?
 4. grow classifier outputs     (new class ids)    [independent of feature growth]
 5. rebuild optimizer, train
 6. probe AFTER training        -> does optimisation change old predictions?
 7. evaluate on the full test set; log feature dim / params / block usage
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.streams import Experience, Stream, binary_labels
from ..models import IncrementalFeatureMapModel
from .evaluate import evaluate_state
from .policy import ExpansionPolicy
from .replay import ReplayBuffer


@dataclass
class TrainerConfig:
    mode: str = "multiclass"  # "multiclass" | "target"
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    train_epochs: int = 1
    train_mb_size: int = 32
    eval_mb_size: int = 256
    device: str = "cpu"
    replay_mem_size: int = 0
    pos_weight: Any = "balanced"  # target mode: "balanced" | float | None
    probe_size: int = 512
    seed: int = 0


class ContinualTrainer:
    def __init__(
        self, model: IncrementalFeatureMapModel, policy: ExpansionPolicy, cfg: TrainerConfig
    ):
        self.model, self.policy, self.cfg = model, policy, cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.buffer = (
            ReplayBuffer(cfg.replay_mem_size, cfg.seed) if cfg.replay_mem_size > 0 else None
        )

    # ------------------------------------------------------------------ #
    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        c = self.cfg
        if c.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay
            )
        if c.optimizer == "adam":
            return torch.optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)
        raise ValueError(f"unknown optimizer {c.optimizer!r}")

    def _loss(self, logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None):
        if self.cfg.mode == "target":
            tgt = binary_labels(y, self._target).float()
            return F.binary_cross_entropy_with_logits(logits[:, 0], tgt, pos_weight=pos_weight)
        return F.cross_entropy(logits, y)

    def _pos_weight(self, exp: Experience) -> torch.Tensor | None:
        pw = self.cfg.pos_weight
        if self.cfg.mode != "target" or pw is None:
            return None
        if pw == "balanced":
            n_pos = max(1, int((exp.labels == self._target).sum()))
            n_neg = max(1, int((exp.labels != self._target).sum()))
            return torch.tensor(n_neg / n_pos, device=self.device)
        return torch.tensor(float(pw), device=self.device)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _probe_logits(self, probe_x: torch.Tensor) -> torch.Tensor:
        was = self.model.training
        self.model.eval()
        out = self.model(probe_x.to(self.device)).cpu()
        self.model.train(was)
        return out

    def _make_probe(self, stream: Stream, seen: list[int]) -> torch.Tensor | None:
        """Fixed diagnostic subset of the test set restricted to already-seen classes (x only is used)."""
        if not seen or self.cfg.probe_size <= 0:
            return None
        labels = np.asarray(stream.test_dataset.targets)  # type: ignore[attr-defined]
        idx = np.nonzero(np.isin(labels, seen))[0]
        rng = np.random.default_rng(self.cfg.seed)
        idx = rng.choice(idx, min(self.cfg.probe_size, len(idx)), replace=False)
        return torch.stack([stream.test_dataset[int(i)][0] for i in idx])

    @staticmethod
    def _agreement(a: torch.Tensor, b: torch.Tensor, mode: str, n_out: int) -> float:
        """Fraction of probe predictions that agree, using only the first ``n_out`` outputs."""
        if mode == "target":
            return float(((a[:, 0] > 0) == (b[:, 0] > 0)).float().mean())
        return float((a[:, :n_out].argmax(1) == b[:, :n_out].argmax(1)).float().mean())

    # ------------------------------------------------------------------ #
    def _train_experience(self, exp: Experience) -> float:
        cfg, model = self.cfg, self.model
        opt = self._make_optimizer()
        pos_weight = self._pos_weight(exp)
        g = torch.Generator().manual_seed(cfg.seed * 10_007 + exp.index)
        loader = DataLoader(exp.dataset, batch_size=cfg.train_mb_size, shuffle=True, generator=g)
        model.train()
        total, n = 0.0, 0
        for _ in range(cfg.train_epochs):
            for batch in loader:
                x, y = batch[0], batch[1]
                if self.buffer is not None:
                    mem = self.buffer.sample(len(x))
                    if mem is not None:
                        x, y = torch.cat([x, mem[0]]), torch.cat([y, mem[1]])
                if len(x) < 2:  # BatchNorm cannot train on a single sample
                    continue
                x, y = x.to(self.device), y.to(self.device)
                loss = self._loss(model(x), y, pos_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total, n = total + float(loss.detach()), n + 1
        if self.buffer is not None:
            self.buffer.update(exp.dataset)
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    def fit(
        self, stream: Stream, callback: Callable[[dict], None] | None = None
    ) -> list[dict[str, Any]]:
        cfg, model, policy = self.cfg, self.model, self.policy
        self._target = stream.target_class
        if cfg.mode == "target":
            policy.single_output = True
            assert model.num_outputs == 1, "target mode needs a model with exactly one output"
        records: list[dict[str, Any]] = []
        for exp in stream.train:
            t0 = time.time()
            seen_before = stream.seen_classes(exp.index - 1) if exp.index > 0 else []
            n_out_before = model.num_outputs
            rec: dict[str, Any] = {
                "index": exp.index,
                "classes": exp.classes,
                "new_classes": exp.new_classes,
            }

            # (1)-(3) probes around feature expansion
            probe_x = self._make_probe(stream, seen_before) if exp.index > 0 else None
            before = self._probe_logits(probe_x) if probe_x is not None else None
            old_dim = model.feature_dim
            frec = policy.expand_features(model, exp.index)
            after = self._probe_logits(probe_x) if probe_x is not None else None
            rec["feature_expansion"] = frec.to_dict() if frec else None
            rec["feature_dim_before_expansion"] = old_dim

            # (4) output growth (independent of the above)
            orec = policy.expand_outputs(model, exp.classes if cfg.mode == "multiclass" else [])
            rec["output_expansion"] = orec.to_dict() if orec else None

            # (5) train
            rec["train_loss"] = self._train_experience(exp)
            rec["train_time_s"] = time.time() - t0

            # (6) probe after training; (7) evaluate
            if probe_x is not None:
                final = self._probe_logits(probe_x)
                n_old = n_out_before
                rec["probe"] = {
                    "n_probe": len(probe_x),
                    "logit_max_abs_diff_expansion": float(
                        (before[:, :n_old] - after[:, :n_old]).abs().max()
                    ),
                    "pred_agreement_expansion": self._agreement(before, after, cfg.mode, n_old),
                    "pred_agreement_after_training": self._agreement(after, final, cfg.mode, n_old),
                }
            seen = stream.seen_classes(exp.index)
            rec.update(
                evaluate_state(
                    model,
                    stream.test_dataset,
                    stream.num_classes,
                    cfg.mode,
                    seen,
                    stream.target_class,
                    cfg.eval_mb_size,
                    self.device,
                )
            )
            rec["seen_classes"] = seen
            rec["feature_dim"] = model.feature_dim
            rec["block_dims"] = model.block_dims
            rec["num_outputs"] = model.num_outputs
            rec["parameter_count"] = model.parameter_count()
            rec["trainable_parameter_count"] = model.parameter_count(trainable_only=True)
            rec["classifier_block_norms"] = model.classifier_block_norms()
            rec["new_block_norm"] = (
                rec["classifier_block_norms"][-1] if len(model.block_dims) > 1 else 0.0
            )
            ref = probe_x if probe_x is not None else stream.test_dataset[0][0][None]
            rec["block_contribution"] = model.block_contribution(ref.to(self.device))
            records.append(rec)
            if callback:
                callback(rec)
        return records
