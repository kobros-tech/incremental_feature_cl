# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Build -> run -> save -> plot.  One entry point for every experiment: ``run_experiment``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import build_class_incremental_stream, build_target_vs_rest_stream, load_dataset
from ..evaluation.result_store import RunResult
from ..models import IncrementalFeatureMapModel, build_backbone
from ..plotting import make_all_plots
from ..training import ContinualTrainer, ExpansionPolicy, TrainerConfig, evaluate_state
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


def build_model(cfg: ExperimentConfig, input_shape: tuple[int, ...], num_outputs: int = 1):
    m = cfg.model
    kw = dict(m.backbone_kwargs)
    if m.backbone in ("identity", "mlp"):
        kw.setdefault("in_dim", int(np.prod(input_shape)))
    else:
        kw.setdefault("in_channels", input_shape[0])
    backbone = build_backbone(m.backbone, **kw)
    return IncrementalFeatureMapModel(
        backbone,
        num_outputs=num_outputs,
        activation=m.activation,
        initialization=m.initialization,
        output_init=m.output_init,
        freeze_backbone=m.freeze_backbone,
        freeze_old_blocks=m.freeze_old_blocks,
        seed=cfg.seed,
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


def estimate_train_samples(cfg: ExperimentConfig) -> int:
    train, test, *_ = get_data(cfg)
    stream = build_stream(cfg, train, test)
    return sum(len(e.dataset) for e in stream.train) * cfg.train.train_epochs


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
    extra = (
        f" target_acc={rec['target_metrics']['target_accuracy']:.3f}"
        f" neg_acc={rec['target_metrics']['negative_accuracy']:.3f}"
        if mode == "target"
        else ""
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
    avalanche_baseline: str | None = None,
) -> RunResult:
    """Run one experiment. ``avalanche_baseline`` (e.g. 'replay') uses an Avalanche method instead."""
    if avalanche_baseline is not None or cfg.backend == "avalanche":
        cfg.backend = "avalanche"
    cfg.train.device = str(resolve_device(cfg.train.device))
    cfg.name = cfg.name or cfg.auto_name()
    seed_everything(cfg.seed)
    train, test, num_classes, shape = get_data(cfg)
    out = Path(out_dir) if out_dir else Path(cfg.output_dir) / cfg.name
    if verbose:
        print(
            f"[{cfg.name}] mode={cfg.mode} backend={cfg.backend} device={cfg.train.device}",
            flush=True,
        )

    if cfg.backend == "avalanche":
        result = _run_avalanche(cfg, train, test, num_classes, shape, avalanche_baseline, verbose)
    else:
        stream = build_stream(cfg, train, test)
        model = build_model(cfg, shape, num_outputs=1)
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
        print(
            f"[{cfg.name}] final_acc_seen={s['final_accuracy_seen']:.3f} avg_acc={s['average_accuracy_seen']:.3f}"
            f" avg_forgetting={s['average_forgetting']:.3f} -> {out}",
            flush=True,
        )
    return result


# ---------------------------------------------------------------------- #
# Avalanche backend (multiclass only)
# ---------------------------------------------------------------------- #
def _run_avalanche(cfg, train, test, num_classes, shape, baseline, verbose) -> RunResult:
    if cfg.mode != "multiclass":
        raise NotImplementedError(
            "the Avalanche backend supports mode='multiclass' only; "
            "use the torch backend for target-vs-rest"
        )
    from torch.utils.data import ConcatDataset

    from ..avalanche import (
        IncrementalFeatureSpaceStrategy,
        make_avalanche_baseline,
        make_split_cifar100,
        make_tensor_benchmark,
    )

    d, t, device = cfg.data, cfg.train, cfg.train.device
    if d.dataset == "cifar100":
        bm = make_split_cifar100(d.n_experiences, cfg.seed, d.class_order, d.root)
    else:
        bm = make_tensor_benchmark(train, test, d.n_experiences, cfg.seed, d.class_order)
    n_classes = int(bm.n_classes) if hasattr(bm, "n_classes") else num_classes
    policy = ExpansionPolicy(
        cfg.model.new_feature_dim if baseline is None else 0, cfg.model.initialization
    )
    model = build_model(cfg, shape, num_outputs=n_classes if baseline else 1)
    if baseline:
        strategy = make_avalanche_baseline(
            baseline,
            model,
            lr=t.lr,
            momentum=t.momentum,
            train_mb_size=t.train_mb_size,
            train_epochs=t.train_epochs,
            eval_mb_size=t.eval_mb_size,
            device=device,
            mem_size=max(t.replay_mem_size, 200),
        )
        plugin_records = None
    else:
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=t.lr, momentum=t.momentum
        )
        strategy = IncrementalFeatureSpaceStrategy(
            model=model,
            optimizer=opt,
            policy=policy,
            train_mb_size=t.train_mb_size,
            train_epochs=t.train_epochs,
            eval_mb_size=t.eval_mb_size,
            device=device,
        )
        plugin_records = strategy.expansion_plugin.records
    test_all = ConcatDataset([e.dataset for e in bm.test_stream])
    seen: list[int] = []
    first: dict[int, int] = {}
    recs = []
    for i, exp in enumerate(bm.train_stream):
        t0 = time.time()
        classes = [int(c) for c in exp.classes_in_this_experience]
        strategy.train(exp)
        dt = time.time() - t0
        new = [c for c in classes if c not in seen]
        for c in new:
            first[c] = i
        seen += new
        rec = {
            "index": i,
            "classes": classes,
            "new_classes": new,
            "train_loss": None,
            "train_time_s": dt,
        }
        exps = [x for x in (plugin_records[-1]["expansions"] if plugin_records else [])]
        rec["feature_expansion"] = next((x for x in exps if x["kind"] == "feature"), None)
        rec["output_expansion"] = next((x for x in exps if x["kind"] == "output"), None)
        rec.update(
            evaluate_state(
                model, test_all, n_classes, "multiclass", seen, None, t.eval_mb_size, device
            )
        )
        rec.update(
            seen_classes=list(seen),
            feature_dim=model.feature_dim,
            block_dims=model.block_dims,
            num_outputs=model.num_outputs,
            parameter_count=model.parameter_count(),
            trainable_parameter_count=model.parameter_count(True),
            classifier_block_norms=model.classifier_block_norms(),
        )
        rec["new_block_norm"] = (
            rec["classifier_block_norms"][-1] if len(model.block_dims) > 1 else 0.0
        )
        rec["block_contribution"] = model.block_contribution(test_all[0][0][None].to(device))
        recs.append(rec)
        if verbose:
            _print_row(rec, "multiclass")
    cfg.name = cfg.name or cfg.auto_name()
    if baseline and not cfg.name.startswith("avalanche_"):
        cfg.name = f"avalanche_{baseline}_{cfg.name}"
    return RunResult(
        cfg.to_dict(), collect_environment(), "multiclass", n_classes, seen, first, recs
    )


# ---------------------------------------------------------------------- #
# CLI helpers shared by the experiment scripts
# ---------------------------------------------------------------------- #
def add_common_args(p: argparse.ArgumentParser, default_dataset: str) -> None:
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
        "--dataset", default=None, help=f"synthetic|cifar10|cifar100 (default {default_dataset})"
    )
    p.add_argument("--root", default=None, help="dataset root")
    p.add_argument("--n-experiences", type=int)
    p.add_argument("--new-feature-dim", type=int, help="0 = fixed-dimension baseline")
    p.add_argument("--initialization", choices=["zero", "random"])
    p.add_argument("--backbone")
    p.add_argument("--freeze-backbone", action="store_true", default=None)
    p.add_argument("--freeze-old-blocks", action="store_true", default=None)
    p.add_argument("--train-epochs", type=int)
    p.add_argument("--train-mb-size", type=int)
    p.add_argument("--eval-mb-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--replay-mem-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto|cpu|cuda")
    p.add_argument("--output-dir")
    p.add_argument("--name")
    p.add_argument(
        "--target-to-negatives",
        type=float,
        help="target:negative training ratio (target mode only): 0.2 = 1 target : 5 negatives, "
        "1 = 1:1; omit to keep cumulative negative sampling",
    )
    p.add_argument("--no-plots", action="store_true")


_FLAG_TO_KEY = {
    "dataset": "data.dataset",
    "root": "data.root",
    "n_experiences": "data.n_experiences",
    "new_feature_dim": "model.new_feature_dim",
    "initialization": "model.initialization",
    "backbone": "model.backbone",
    "freeze_backbone": "model.freeze_backbone",
    "freeze_old_blocks": "model.freeze_old_blocks",
    "train_epochs": "train.train_epochs",
    "train_mb_size": "train.train_mb_size",
    "eval_mb_size": "train.eval_mb_size",
    "lr": "train.lr",
    "replay_mem_size": "train.replay_mem_size",
    "seed": "seed",
    "device": "train.device",
    "output_dir": "output_dir",
    "name": "name",
    "target_to_negatives": "target_to_negatives",
}


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
    if mode != "target" and getattr(args, "target_to_negatives", None) is not None:
        raise ValueError("--target-to-negatives only applies to target-vs-rest (mode 'target')")
    if args.dataset is None and not in_cfg:
        d["data"]["dataset"] = default_dataset
    flags = [
        f"{k}={getattr(args, a)}"
        for a, k in _FLAG_TO_KEY.items()
        if getattr(args, a, None) is not None
    ]
    apply_overrides(d, flags)
    apply_overrides(d, args.overrides)
    if getattr(args, "backend", None):
        d["backend"] = args.backend
    return ExperimentConfig.from_dict(d)
