# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Machine-readable results: results.json (+ metrics.csv, per_class_accuracy.csv).

``results.json`` has enough information to regenerate every plot without retraining.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .forgetting import average_forgetting


def _clean(o: Any) -> Any:
    """JSON-safe: NaN -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _flat_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scalar columns for metrics.csv. A key that holds a dict in *any* record (e.g. ``probe``,
    ``feature_expansion``, which are ``None`` in experience 0) is flattened to ``<key>_<subkey>``
    columns; ``target_metrics`` keeps its unprefixed names. Lists stay in results.json only."""
    scalar = (int, float, str, bool, type(None))
    dict_keys = {k for r in records for k, v in r.items() if isinstance(v, dict)}
    dict_keys |= {"probe", "feature_expansion", "output_expansion"}  # None in some experiences
    rows = []
    for rec in records:
        row: dict[str, Any] = {}
        for k, v in rec.items():
            if k in dict_keys:
                for sk, sv in (v or {}).items():
                    if isinstance(sv, scalar):
                        row[sk if k == "target_metrics" else f"{k}_{sk}"] = _clean(sv)
            elif isinstance(v, scalar):
                row[k] = _clean(v)
        rows.append(row)
    return rows


@dataclass
class RunResult:
    config: dict[str, Any]
    environment: dict[str, Any]
    mode: str
    num_classes: int
    class_order: list[int]
    class_first_experience: dict[int, int]
    experiences: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    target_class: int | None = None

    # ------------------------------------------------------------------ #
    def accuracy_matrix(self) -> np.ndarray:
        """(n_experiences, num_classes); NaN where a class has no test samples."""
        return np.array(
            [
                [np.nan if v is None else v for v in e["per_class_accuracy"]]
                for e in self.experiences
            ],
            dtype=float,
        )

    def series(self, key: str) -> list:
        return [e.get(key) for e in self.experiences]

    def compute_summary(self) -> dict[str, Any]:
        acc = self.accuracy_matrix()
        first = {int(k): int(v) for k, v in self.class_first_experience.items()}
        last = self.experiences[-1]
        s: dict[str, Any] = {
            "final_accuracy_seen": last.get("accuracy_seen"),
            "final_accuracy_all": last.get("accuracy_all"),
            "average_accuracy_seen": float(
                np.nanmean([e["accuracy_seen"] for e in self.experiences])
            ),
            "average_forgetting": average_forgetting(acc, first),
            "final_feature_dim": last["feature_dim"],
            "final_parameter_count": last["parameter_count"],
            "total_train_time_s": float(sum(e["train_time_s"] for e in self.experiences)),
        }
        final_target = last.get("target_metrics")
        if self.mode == "target" and final_target:
            for k in (
                "target_accuracy",
                "target_f1",
                "false_positive_rate",
                "overall_binary_accuracy",
                "balanced_accuracy",
                "auc",
            ):
                s[f"final_{k}"] = final_target.get(k)
        self.summary = s
        return s

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return _clean(
            {
                "config": self.config,
                "environment": self.environment,
                "mode": self.mode,
                "target_class": self.target_class,
                "num_classes": self.num_classes,
                "class_order": self.class_order,
                "class_first_experience": self.class_first_experience,
                "summary": self.summary,
                "experiences": self.experiences,
            }
        )

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # write-then-rename: an interrupted run never leaves a truncated results.json that a
        # later resume could mistake for a finished run
        tmp = out / "results.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(out / "results.json")
        rows = _flat_rows(self.experiences)
        cols: list[str] = []
        for row in rows:
            cols += [k for k in row if k not in cols]
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in rows:
                w.writerow([row.get(k) for k in cols])
        with open(out / "per_class_accuracy.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["experience"] + [f"class_{c}" for c in range(self.num_classes)])
            for e in self.experiences:
                w.writerow([e["index"]] + [_clean(v) for v in e["per_class_accuracy"]])
        return out / "results.json"

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        p = Path(path)
        d = json.loads((p / "results.json" if p.is_dir() else p).read_text())
        return cls(
            config=d["config"],
            environment=d["environment"],
            mode=d["mode"],
            num_classes=d["num_classes"],
            class_order=d["class_order"],
            class_first_experience={int(k): v for k, v in d["class_first_experience"].items()},
            experiences=d["experiences"],
            summary=d.get("summary", {}),
            target_class=d.get("target_class"),
        )
