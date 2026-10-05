# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""IncrementalFeatureMapModel.

    x -> backbone -> h(x) --+--> phi_0 = h(x)         (old features)
                            +--> psi_1(h), ..., psi_t(h)   (appended feature blocks)
                                      |
                        Phi_t(x) = [phi_0, psi_1, ..., psi_t]
                                      |
                       logits = b + W_0 phi_0 + sum_k W_k psi_k

``expand_feature_space(new_dim)`` appends psi_{t+1} and ``W_{t+1} = 0`` *in place*:
no module is recreated, so previously learned parameters keep their identity and
(with zero initialisation) ``f_{t+1}(x) == f_t(x)`` before any optimisation step.

Feature expansion (``expand_feature_space``) and output expansion
(``expand_outputs``) are independent operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .classifier import ExpandableLinearClassifier
from .expandable import ExpansionRecord, check_initialization
from .feature_map import FeatureBlock


class IncrementalFeatureMapModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int | None = None,
        num_outputs: int = 1,
        *,
        activation: str = "relu",
        initialization: str = "zero",
        output_init: str = "zero",
        freeze_backbone: bool = False,
        freeze_old_blocks: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        backbone_dim = (
            backbone_dim if backbone_dim is not None else getattr(backbone, "out_dim", None)
        )
        if backbone_dim is None:
            raise ValueError("backbone_dim is required when the backbone has no `out_dim`")
        self.backbone = backbone
        self.backbone_dim = int(backbone_dim)
        self.activation = activation
        self.initialization = check_initialization(initialization)
        self.output_init = check_initialization(output_init)
        self.freeze_backbone = freeze_backbone
        self.freeze_old_blocks = freeze_old_blocks
        self.seed = seed

        self.feature_blocks = nn.ModuleList()  # psi_1 ... psi_t
        self.classifier = ExpandableLinearClassifier([self.backbone_dim], num_outputs)
        self.expansion_log: list[ExpansionRecord] = []
        self._apply_freezing()

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def block_dims(self) -> list[int]:
        return [self.backbone_dim] + [b.out_dim for b in self.feature_blocks]

    @property
    def feature_dim(self) -> int:
        return sum(self.block_dims)

    @property
    def num_outputs(self) -> int:
        return self.classifier.num_outputs

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def feature_block_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[phi_0, psi_1, ..., psi_t] for a batch ``x``."""
        h = self.backbone(x)
        if h.ndim > 2:
            h = h.flatten(1)
        return [h] + [blk(h) for blk in self.feature_blocks]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Expanded feature vector Phi_t(x), shape (B, feature_dim)."""
        return torch.cat(self.feature_block_outputs(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits.  Depends only on ``x`` and the model state (never on labels)."""
        return self.classifier(self.feature_block_outputs(x))

    # ------------------------------------------------------------------ #
    # expansion
    # ------------------------------------------------------------------ #
    def _record(
        self, kind: str, init: str, old_dim: int, old_out: int, old_params: int
    ) -> ExpansionRecord:
        rec = ExpansionRecord(
            event=len(self.expansion_log),
            kind=kind,
            initialization=init,
            old_feature_dim=old_dim,
            new_feature_dim=self.feature_dim - old_dim,
            total_feature_dim=self.feature_dim,
            old_num_outputs=old_out,
            new_num_outputs=self.num_outputs,
            number_of_new_parameters=self.parameter_count() - old_params,
            old_parameter_count=old_params,
            total_parameter_count=self.parameter_count(),
        )
        self.expansion_log.append(rec)
        return rec

    @torch.no_grad()
    def expand_feature_space(
        self, new_dim: int, initialization: str | None = None, seed: int | None = None
    ) -> ExpansionRecord:
        """``old_dim -> old_dim + new_dim`` in place; new classifier block W_new = 0 (default).

        Existing modules/parameters are *not* recreated.  The caller must make
        sure the optimizer sees the new parameters (rebuild it).
        """
        if new_dim < 0:
            raise ValueError("new_dim must be >= 0")
        init = check_initialization(initialization or self.initialization)
        old_dim, old_out, old_params = self.feature_dim, self.num_outputs, self.parameter_count()
        if new_dim > 0:
            ev_seed = (
                seed if seed is not None else self.seed * 100_003 + len(self.expansion_log) + 1
            )
            ref = self.classifier.bias
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(ev_seed)
                blk = FeatureBlock(self.backbone_dim, new_dim, self.activation)
            self.feature_blocks.append(blk.to(device=ref.device, dtype=ref.dtype))
            self.classifier.add_block(new_dim, init, seed=ev_seed + 7)
            self._apply_freezing()
        return self._record("feature", init, old_dim, old_out, old_params)

    @torch.no_grad()
    def expand_outputs(self, n_new: int, initialization: str | None = None) -> ExpansionRecord:
        """Add ``n_new`` classifier outputs; feature dimension is unchanged."""
        init = check_initialization(initialization or self.output_init)
        old_dim, old_out, old_params = self.feature_dim, self.num_outputs, self.parameter_count()
        self.classifier.expand_outputs(n_new, init, seed=self.seed + len(self.expansion_log) + 13)
        return self._record("output", init, old_dim, old_out, old_params)

    def ensure_outputs(self, n: int) -> ExpansionRecord | None:
        """Grow the output head to at least ``n`` outputs (no-op if already large enough)."""
        return self.expand_outputs(n - self.num_outputs) if n > self.num_outputs else None

    # ------------------------------------------------------------------ #
    # freezing / training mode
    # ------------------------------------------------------------------ #
    def _apply_freezing(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(not self.freeze_backbone)
        last = len(self.feature_blocks) - 1
        for i, blk in enumerate(self.feature_blocks):
            for p in blk.parameters():
                p.requires_grad_(not (self.freeze_old_blocks and i < last))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep BatchNorm statistics fixed when the backbone is frozen
        return self

    # ------------------------------------------------------------------ #
    # diagnostics: how much is the newest capacity used?
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def classifier_block_norms(self) -> list[float]:
        """Frobenius norm of each classifier block W_k (index 0 = backbone block)."""
        return [float(w.weight.norm()) for w in self.classifier.weights]

    @torch.no_grad()
    def block_contribution(self, x: torch.Tensor) -> list[float]:
        """Mean |W_k phi_k(x)| per block on batch ``x`` (call in eval mode)."""
        outs = self.classifier.block_contributions(self.feature_block_outputs(x))
        return [float(o.abs().mean()) for o in outs]

    def new_block_parameters(self) -> Sequence[nn.Parameter]:
        """Parameters of the most recently added block (psi_t and W_t)."""
        if not len(self.feature_blocks):
            return []
        return [*self.feature_blocks[-1].parameters(), *self.classifier.weights[-1].parameters()]
