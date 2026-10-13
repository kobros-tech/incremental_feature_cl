# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Build -> run -> save -> plot.  One entry point for every experiment: ``run_experiment``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import build_class_incremental_stream, build_target_vs_rest_stream, load_dataset
from ..data.streams import parse_ratio, ratio_report
from ..evaluation.result_store import RunResult
from ..models import (
    IncrementalFeatureMapModel,
    SkillConfig,
    build_backbone,
    warm_up_backbone,
)
from ..models.skill import EXPANSIONS
from ..plotting import make_all_plots
from ..training import ContinualTrainer, ExpansionPolicy, TrainerConfig
from ..utils.reproducibility import collect_environment, resolve_device, seed_everything
from .config import ExperimentConfig, apply_overrides

_DATA_CACHE: dict[tuple, Any] = {}


def get_data(cfg: ExperimentConfig):
    key = (cfg.data.dataset, cfg.data.root, cfg.seed, repr(cfg.data.synthetic))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_dataset(
            cfg.data.dataset, cfg.data.root, cfg.seed, **cfg.data.synthetic
        )
    return _DATA_CACHE[key]


def build_backbone_from(cfg: ExperimentConfig, input_shape: tuple[int, ...]) -> torch.nn.Module:
    """The external backbone named by ``cfg.model`` (see ``models/backbones.py``)."""
    m = cfg.model
    return build_backbone(m.backbone, input_shape, m.backbone_kwargs, m.backbone_weights)


def build_model(cfg: ExperimentConfig, input_shape: tuple[int, ...], num_outputs: int = 1):
    m = cfg.model
    return IncrementalFeatureMapModel(
        build_backbone_from(cfg, input_shape),
        num_outputs=num_outputs,
        activation=m.activation,
        initialization=m.initialization,
        output_init=m.output_init,
        freeze_backbone=m.freeze_backbone,
        freeze_old_blocks=m.freeze_old_blocks,
        seed=cfg.seed,
    )


def materialize(dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a dataset through ``__getitem__`` (uint8 images are converted and normalized)."""
    from torch.utils.data import DataLoader

    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def maybe_warm_up(cfg: ExperimentConfig, backbone: torch.nn.Module, first_experience) -> None:
    """``--warmup-epochs N``: train the backbone on experience 0, then freeze it."""
    if cfg.model.warmup_epochs <= 0:
        return
    if not cfg.model.freeze_backbone:
        raise ValueError(
            "--warmup-epochs only makes sense with a frozen backbone (--freeze-backbone)"
        )
    x, y = materialize(first_experience.dataset)
    loss = warm_up_backbone(
        backbone,
        x,
        y,
        cfg.model.warmup_epochs,
        lr=cfg.train.lr,
        batch_size=cfg.train.train_mb_size,
        device=cfg.train.device,
        seed=cfg.seed,
    )
    print(
        f"[warm-up] backbone trained {cfg.model.warmup_epochs} epochs on experience 0 "
        f"(loss {loss:.4f}) and frozen",
        flush=True,
    )


def build_stream(cfg: ExperimentConfig, train, test):
    d = cfg.data
    if cfg.mode == "target":
        return build_target_vs_rest_stream(
            train,
            test,
            cfg.target_class,
            d.n_experiences,
            cfg.seed,
            d.class_order,
            cfg.target_in_every_experience,
            cfg.target_to_negatives,
        )
    return build_class_incremental_stream(train, test, d.n_experiences, cfg.seed, d.class_order)


def plan_stream(cfg: ExperimentConfig) -> tuple[int, list[dict[str, Any]]]:
    """Build the stream once (no training): ``(training sample-passes, per-experience ratio reports)``.

    The reports (``data.streams.ratio_report``) show *before* a sweep which experiences will get the
    requested number of negatives and which will fall short because the cumulative pool is still
    too small.  Empty list outside target mode.
    """
    train, test, *_ = get_data(cfg)
    stream = build_stream(cfg, train, test)
    passes = sum(len(e.dataset) for e in stream.train) * cfg.train.train_epochs
    if cfg.mode != "target":
        return passes, []
    reports = []
    for e in stream.train:
        n_t = int((e.labels == stream.target_class).sum())
        reports.append(ratio_report(n_t, len(e.labels) - n_t, stream.target_to_negatives))
    return passes, reports


def estimate_train_samples(cfg: ExperimentConfig) -> int:
    return plan_stream(cfg)[0]


def large_run_warning(n_runs: int, threshold: int = 50) -> str | None:
    """A loud one-line warning for big sweeps (e.g. ``--target-class all``), else ``None``."""
    if n_runs < threshold:
        return None
    return (
        f"WARNING: this will run {n_runs:,} experiments. "
        "Use --dry-run to inspect the plan first; finished runs are reused on restart."
    )


def _trainer_cfg(cfg: ExperimentConfig, device: str) -> TrainerConfig:
    t = cfg.train
    return TrainerConfig(
        mode=cfg.mode,
        optimizer=t.optimizer,
        lr=t.lr,
        momentum=t.momentum,
        weight_decay=t.weight_decay,
        train_epochs=t.train_epochs,
        train_mb_size=t.train_mb_size,
        eval_mb_size=t.eval_mb_size,
        device=device,
        replay_mem_size=t.replay_mem_size,
        pos_weight=t.pos_weight,
        probe_size=t.probe_size,
        seed=cfg.seed,
    )


def _print_row(rec: dict, mode: str) -> None:
    extra = ""
    if mode == "target":
        tm = rec["target_metrics"]
        extra = (
            f" target_acc={tm['target_accuracy']:.3f} neg_acc={tm['negative_accuracy']:.3f}"
            f" bal_acc={tm['balanced_accuracy']:.3f}"
        )
        if tm.get("auc") is not None:
            extra += f" auc={tm['auc']:.3f}"
        cov = rec.get("negative_coverage")
        if cov:
            extra += (
                f" negs(old {cov['old_covered']}/{cov['old_negative_classes']}"
                f" new {cov['new_covered']}/{cov['new_negative_classes']}"
                f"{'' if cov['complete'] else ' INCOMPLETE'})"
            )
    probe = rec.get("probe")
    pr = f" |dlogit|={probe['logit_max_abs_diff_expansion']:.1e}" if probe else ""
    print(
        f"  exp {rec['index']:>2}  dim={rec['feature_dim']:<4} params={rec['parameter_count']:<8}"
        f" acc_seen={rec['accuracy_seen']:.3f} acc_all={rec['accuracy_all']:.3f}{extra}{pr}"
        f" |W_new|={rec['new_block_norm']:.3f}",
        flush=True,
    )


def run_experiment(
    cfg: ExperimentConfig,
    out_dir: str | Path | None = None,
    make_plots: bool = True,
    verbose: bool = True,
) -> RunResult:
    """Train IncrementalFeatureMapModel on one target-vs-rest (or class-incremental) stream."""
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.name = cfg.name or cfg.auto_name()
    seed_everything(cfg.seed)
    train, test, _, shape = get_data(cfg)
    out = Path(out_dir) if out_dir else Path(cfg.output_dir) / cfg.name
    if verbose:
        print(
            f"[{cfg.name}] mode={cfg.mode} backbone={cfg.model.backbone} device={cfg.train.device}",
            flush=True,
        )

    stream = build_stream(cfg, train, test)
    model = build_model(cfg, shape, num_outputs=1)
    maybe_warm_up(cfg, model.backbone, stream.train[0])
    policy = ExpansionPolicy(cfg.model.new_feature_dim, cfg.model.initialization)
    trainer = ContinualTrainer(model, policy, _trainer_cfg(cfg, cfg.train.device))
    recs = trainer.fit(stream, (lambda r: _print_row(r, cfg.mode)) if verbose else None)
    result = RunResult(
        cfg.to_dict(),
        collect_environment(),
        cfg.mode,
        stream.num_classes,
        stream.class_order,
        stream.class_first_experience(),
        recs,
        target_class=stream.target_class,
    )
    result.compute_summary()
    result.save(out)
    if make_plots:
        make_all_plots(result, out)
    if verbose:
        s = result.summary
        tail = ""
        if s.get("final_balanced_accuracy") is not None:
            tail = f" final_bal_acc={s['final_balanced_accuracy']:.3f}"
            if s.get("final_auc") is not None:
                tail += f" final_auc={s['final_auc']:.3f}"
        print(
            f"[{cfg.name}] final_acc_seen={s['final_accuracy_seen']:.3f}{tail}"
            f" avg_acc={s['average_accuracy_seen']:.3f}"
            f" avg_forgetting={s['average_forgetting']:.3f} -> {out}",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------- #
# CLI helpers shared by the experiment scripts
# ---------------------------------------------------------------------- #
def add_common_args(
    p: argparse.ArgumentParser,
    default_dataset: str,
    *,
    include_target_ratio: bool = True,
    ratio_default: str | None = None,
) -> None:
    p.add_argument("--config", help="YAML config (CLI flags and --set override it)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="generic override, e.g. --set train.lr=0.05",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help=f"synthetic|digits|digitpairs|cifar10|cifar100 (default {default_dataset})",
    )
    p.add_argument("--root", default=None, help="dataset root")
    p.add_argument("--n-experiences", type=int)
    p.add_argument("--new-feature-dim", type=int, help="0 = fixed-dimension baseline")
    p.add_argument("--initialization", choices=["zero", "random"])
    p.add_argument(
        "--backbone",
        help="identity | simplemlp | resnet32 | dotted path of any nn.Module class/factory "
        "(default identity); the Avalanche models need avalanche-lib",
    )
    p.add_argument("--backbone-kwargs", help="JSON for the backbone, e.g. '{\"hidden_size\": 256}'")
    p.add_argument("--backbone-weights", help="state dict (.pt) to load into the external model")
    p.add_argument(
        "--warmup-epochs",
        type=int,
        help="train the backbone on experience 0, then freeze it (needs --freeze-backbone)",
    )
    p.add_argument("--freeze-backbone", action="store_true", default=None)
    p.add_argument("--freeze-old-blocks", action="store_true", default=None)
    p.add_argument("--train-epochs", type=int)
    p.add_argument("--train-mb-size", type=int)
    p.add_argument("--eval-mb-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "lbfgs"],
        help="sgd (default) | adam; lbfgs (full-batch, Skills only; the Skills' default)",
    )
    p.add_argument("--weight-decay", type=float, help="L2 (Skills with lbfgs default to 1e-3)")
    p.add_argument("--replay-mem-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto|cpu|cuda")
    p.add_argument("--output-dir")
    p.add_argument("--name")
    if include_target_ratio:
        p.add_argument(
            "--ratio",
            default=ratio_default,
            metavar="TARGET:NEGATIVE",
            help=(
                "training ratio target:negative, e.g. 1:5 (one target per five negatives), 1:1, "
                "5:1; 'cumulative' keeps every cumulative negative"
                + (f" (default {ratio_default})" if ratio_default else " (default: cumulative)")
            ),
        )
        # deprecated spelling of --ratio (targets per negative as a number); kept for old scripts
        p.add_argument("--target-to-negatives", type=float, help=argparse.SUPPRESS)
    p.add_argument("--no-plots", action="store_true")


_FLAG_TO_KEY = {
    "dataset": "data.dataset",
    "root": "data.root",
    "n_experiences": "data.n_experiences",
    "new_feature_dim": "model.new_feature_dim",
    "initialization": "model.initialization",
    "backbone": "model.backbone",
    "backbone_weights": "model.backbone_weights",
    "warmup_epochs": "model.warmup_epochs",
    "freeze_backbone": "model.freeze_backbone",
    "freeze_old_blocks": "model.freeze_old_blocks",
    "train_epochs": "train.train_epochs",
    "train_mb_size": "train.train_mb_size",
    "eval_mb_size": "train.eval_mb_size",
    "lr": "train.lr",
    "optimizer": "train.optimizer",
    "weight_decay": "train.weight_decay",
    "replay_mem_size": "train.replay_mem_size",
    "seed": "seed",
    "device": "train.device",
    "output_dir": "output_dir",
    "name": "name",
    "target_to_negatives": "target_to_negatives",
}  # (a parsed --ratio is stored in args.target_to_negatives by config_from_args)


def config_from_args(args: argparse.Namespace, mode: str, default_dataset: str) -> ExperimentConfig:
    """Precedence: defaults < YAML config < CLI flags < --set overrides."""
    d = ExperimentConfig().to_dict()
    in_cfg = False
    if args.config:
        import yaml

        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        in_cfg = "dataset" in (loaded.get("data") or {})
        for k, v in loaded.items():
            d[k] = {**d[k], **v} if isinstance(v, dict) and isinstance(d.get(k), dict) else v
    d["mode"] = mode
    ratio_text = getattr(args, "ratio", None)
    legacy_ratio = getattr(args, "target_to_negatives", None)
    if ratio_text is not None and legacy_ratio is not None:
        raise ValueError("use either --ratio or --target-to-negatives, not both")
    if mode != "target" and (ratio_text is not None or legacy_ratio is not None):
        raise ValueError("--ratio only applies to target-vs-rest (mode 'target')")
    if ratio_text is not None:
        args.target_to_negatives = parse_ratio(ratio_text)  # None for 'cumulative'
        if args.target_to_negatives is None:
            d["target_to_negatives"] = None
    if args.dataset is None and not in_cfg:
        d["data"]["dataset"] = default_dataset
    flags = [
        f"{k}={getattr(args, a)}"
        for a, k in _FLAG_TO_KEY.items()
        if getattr(args, a, None) is not None
    ]
    apply_overrides(d, flags)
    if getattr(args, "backbone_kwargs", None):
        d["model"]["backbone_kwargs"] = json.loads(args.backbone_kwargs)
    apply_overrides(d, args.overrides)
    return ExperimentConfig.from_dict(d)


# ---------------------------------------------------------------------- #
# Skill experiments (train_skill / train_ovr)
# ---------------------------------------------------------------------- #
def add_skill_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--memory-per-class",
        type=int,
        default=100,
        help="replay exemplars kept per class (default 100; 0 disables replay).  A refresh of an "
        "old skill trains on its memory positives and --ratio times as many negatives, so very "
        "small values starve it.  Not --replay-mem-size (the feature-map model's buffer).",
    )
    p.add_argument("--skill-pos-weight", default="balanced", help="balanced | none | number")
    p.add_argument(
        "--skill-expansion",
        choices=EXPANSIONS,
        default="first",
        help="when a skill grows its feature space by --new-feature-dim",
    )
    p.add_argument("--skill-max-iter", type=int, default=100, help="L-BFGS iterations per update")
    p.add_argument("--vs-sklearn", action="store_true", help="also run the scikit-learn control")
    p.add_argument("--sklearn-c", type=float, default=1.0, help="LogisticRegression C")


def skill_config_from(cfg: ExperimentConfig, args: argparse.Namespace) -> SkillConfig:
    """Map ModelConfig / TrainConfig (the flags shared with train_feature_map) onto a SkillConfig.

    Skills default to the full-batch L-BFGS solver (``--optimizer sgd|adam`` for minibatch).
    """
    optimizer = cfg.train.optimizer if args.optimizer else "lbfgs"
    weight_decay = cfg.train.weight_decay
    if args.weight_decay is None and optimizer == "lbfgs":
        weight_decay = 1e-3
    pos = args.skill_pos_weight
    pos_weight = None if pos == "none" else "balanced" if pos == "balanced" else float(pos)
    return SkillConfig(
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=weight_decay,
        epochs=cfg.train.train_epochs,
        batch_size=cfg.train.train_mb_size,
        optimizer=optimizer,
        max_iter=args.skill_max_iter,
        feature_expansion_dim=cfg.model.new_feature_dim,
        activation=cfg.model.activation,
        initialization=cfg.model.initialization,
        output_init=cfg.model.output_init,
        freeze_old_blocks=cfg.model.freeze_old_blocks,
        expansion=args.skill_expansion,
        target_to_negatives=cfg.target_to_negatives,
        pos_weight=pos_weight,
        memory_per_class=args.memory_per_class,
    )


_COLUMNS = ("balanced_accuracy", "target_recall", "negative_accuracy", "target_f1", "auc")


def print_comparison(title: str, rows_by_name: dict[str, list[dict | None]], ids) -> None:
    """Side-by-side table of per-class target-vs-rest metrics."""

    def cell(m: dict | None, k: str) -> str:
        if m is None or m.get(k) is None:
            return "    n/a"
        return f"{m[k]:7.3f}"

    def mean(rows: list[dict | None], k: str) -> dict:
        vals = [row[k] for row in rows if row is not None and row.get(k) is not None]
        return {k: float(np.mean(vals)) if vals else None}

    names = list(rows_by_name)
    head = " ".join(f"{h:>7}" for h in ("bal", "recall", "neg", "f1", "auc"))

    print(f"\n{title}")
    print(f"{'class':>6} | " + " | ".join(f"{n:^39}" for n in names))
    print(f"{'':>6} | " + " | ".join(head for _ in names))

    for i, class_id in enumerate(ids):
        print(
            f"{class_id:>6} | "
            + " | ".join(
                " ".join(cell(rows_by_name[name][i], metric) for metric in _COLUMNS)
                for name in names
            )
        )

    if len(ids) > 1:
        print(
            f"{'mean':>6} | "
            + " | ".join(
                " ".join(cell(mean(rows_by_name[name], metric), metric) for metric in _COLUMNS)
                for name in names
            )
        )


def finish_skill_run(result: RunResult, out_dir: Path, make_plots: bool) -> None:
    """Summarize, save and plot a result whose records follow the skill record layout."""
    from ..plotting import (
        plot_accuracy_heatmap,
        plot_accuracy_vs_class,
        plot_accuracy_vs_experience,
        plot_class_accuracy_curves,
        plot_feature_growth,
        plot_forgetting_vs_class,
    )

    result.compute_summary()
    result.save(out_dir)
    if make_plots:
        plots = Path(out_dir) / "plots"
        plot_accuracy_vs_experience(result, plots / "A_accuracy_vs_experience.png")
        plot_accuracy_vs_class(result, plots / "B_accuracy_vs_class.png")
        plot_accuracy_heatmap(result, plots / "C_class_accuracy_heatmap.png")
        plot_class_accuracy_curves(result, plots / "D_class_accuracy_curves.png")
        plot_forgetting_vs_class(result, plots / "E_forgetting_vs_class.png")
        plot_feature_growth(result, plots / "F_feature_growth.png")
    s = result.summary
    auc = f" final_auc={s['final_auc']:.3f}" if s.get("final_auc") is not None else ""

    if "average_skill_forgetting" in s:
        forgetting = s["average_skill_forgetting"]
        forgetting_label = "avg_skill_forgetting"
        metric_description = "skill target recall"
    else:
        # Preserve compatibility with existing skill-style runs that still
        # report class-wise accuracy forgetting.
        forgetting = s.get("average_forgetting")
        forgetting_label = "avg_forgetting"
        metric_description = "balanced accuracy per class"

    forgetting_text = (
        f"{forgetting_label}={forgetting:.3f}"
        if forgetting is not None
        else f"{forgetting_label}=n/a"
    )

    print(
        f"[{result.config.get('name', '')}] "
        f"final_bal_acc={s['final_balanced_accuracy']:.3f}{auc} "
        f"final_f1={s['final_target_f1']:.3f} "
        f"{forgetting_text} ({metric_description}) -> {out_dir}",
        flush=True,
    )
