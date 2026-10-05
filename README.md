# incremental-feature-cl

A standalone research package that tests one hypothesis:

> A continual learner can be given additional discriminative feature capacity incrementally, while the
> classifier contribution of each new block is initialised to **zero**. This guarantees continuity at the
> expansion boundary; optional freezing controls can be used when an experiment requires the previously
> learned representation itself to remain fixed.

**Status: experimental. No claim is made that the method works.** The package establishes (and tests) the
*mechanism*; whether it helps on Split CIFAR-100 is what the experiments are for.

Theory: [MIT 6.86x Lecture 6](https://github.com/kobros-tech/6.86x/tree/2025/unit_2/lecture_6) (feature maps,
kernels, kernel perceptron). Architecture is inspired by (but does **not** depend on) the OCL Survey and
Skill Memory repositories. Avalanche is an optional integration layer.

```
6.86x maths -> math/ (feature maps, kernels, perceptron) -> models/ (expandable feature model)
            -> training/ (plain PyTorch)  ->  avalanche/ (optional adapter) -> Split CIFAR-100
```

## Install

```bash
pip install -e .                      # numpy, torch, pyyaml, matplotlib
pip install -e ".[vision]"            # + torchvision (CIFAR-10/100 download)
pip install -e ".[avalanche,dev]"     # + avalanche-lib, pytest, ruff
pytest                                # 50 tests, CPU, ~15 s, no downloads
```

## Commands

```bash
# 1a. The Lecture-6 mathematics (no neural net): expansion invariance + kernel identities
python -m incremental_feature_cl.experiments.math_demo

# 1b. Smallest neural experiment (synthetic data, CPU, seconds). zero vs random expansion
python -m incremental_feature_cl.experiments.synthetic_demo            # --mode target for target-vs-rest

# 2. Split CIFAR-100, 5 experiences (first real milestone)
python -m incremental_feature_cl.experiments.train_cifar100 --dataset cifar100 \
    --n-experiences 5 --new-feature-dim 16 --train-epochs 5 --seed 1

# 3. Target class 17 vs the rest, 20 experiences
python -m incremental_feature_cl.experiments.train_target --dataset cifar100 \
    --target-class 17 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1

# 4. Full 20-experience class-incremental benchmark (20 x 5 classes)
python -m incremental_feature_cl.experiments.train_cifar100 --dataset cifar100 \
    --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1
#    same benchmark through Avalanche (SplitCIFAR100 + IncrementalFeatureSpaceStrategy):
#    add --backend avalanche

# 5. Ablation: fixed / fixed+replay / expand-random / expand-zero / expand-zero+replay (+ Avalanche refs)
python -m incremental_feature_cl.experiments.compare_baselines --dataset cifar100 --n-experiences 5 \
    --new-feature-dim 16 --replay 200 --avalanche-baselines naive,replay,er_ace,ewc

# 6. Matrix {5,10,20} x {0,4,16,32} x {zero,random}  -- prints the plan; needs --yes to run
python -m incremental_feature_cl.experiments.run_sweep --preset cifar100-matrix --dry-run

# 7. Several / all target classes (prints an estimate; needs --yes)
python -m incremental_feature_cl.experiments.train_target --target-class 0 17 50 99 --yes
python -m incremental_feature_cl.experiments.train_target --target-class all --dry-run

# Regenerate plots from a saved result, no retraining
python -m incremental_feature_cl.plotting --results results/<run>/results.json
```
Every script takes `--config configs/*.yaml`, any `--set section.key=value`, `--device auto|cpu|cuda`.
Run order recommended: synthetic -> CIFAR-10/small subset -> CIFAR-100 x5 -> x20 -> zero vs random -> baselines.

## Mathematical Motivation

**Feature maps (Lecture 6).** A linear classifier cannot separate `x = -1, 0, 1` with labels `+1, -1, +1`.
Apply a feature map `x -> Phi(x) = (x, x^2)` and the *same* linear machinery works: the decision
`theta . Phi(x) + b >= 0` is linear in feature space and nonlinear in `x`.

```
x  ->  Phi(x)  ->  linear classifier  f(x) = W Phi(x) + b
```

**Incremental expansion.** Instead of a fixed `Phi`, grow it by appending a block:

```
Phi_{t+1}(x) = [ Phi_t(x) , psi_{t+1}(x) ]          W_{t+1} = [ W_t , 0 ]
```

Then, immediately after expansion and before any optimisation step,

```
f_{t+1}(x) = W_t Phi_t(x) + 0 * psi_{t+1}(x) + b = f_t(x)        for every x
```

so expansion itself cannot damage old predictions (verified numerically: `tests/test_zero_initialization.py`,
and recorded as `logit_max_abs_diff_expansion` in every run). The gradient of the loss with respect to
`W_new` is `dL/dlogits (x) psi_new(x)^T`, which is generally non-zero, so gradient descent *can* make
`W_new` non-zero when the new capacity is useful (`tests/test_gradient_activation.py`). One honest subtlety: at
exactly `W_new = 0` the gradient w.r.t. the parameters *inside* `psi_new` is zero, so `psi` only starts to learn
after `W_new` has moved (tested and documented).

**What this is not.** It is an experimental hypothesis, **not a theorem** that zero initialisation solves
catastrophic forgetting. Zero initialisation guarantees continuity at the expansion instant only. With the
default trainable-backbone/trainable-old-blocks setting, subsequent optimisation can still change the old
representation and classifier. For strict stability experiments, use `freeze_backbone=true` and/or
`freeze_old_blocks=true`; these are explicit controls, not hidden behaviour. Results record the chosen
settings so an experiment can be audited later.

## Kernel interpretation

`K(x, z) = <Phi(x), Phi(z)>`. The lecture's degree-2 map `(1, sqrt2 x1, sqrt2 x2, x1^2, x2^2, sqrt2 x1 x2)` satisfies
`Phi(x).Phi(z) = (1 + x.z)^2`; `math.PolynomialFeatureMap(degree, kernel_scaled=True)` reproduces this exactly
and is block-structured by degree, `Phi_p = [Phi_{p-1}, psi_p]` - the same incremental shape as the neural
model. `math/` also has `polynomial_kernel`, `rbf_kernel`, a feature-space `Perceptron` that can be expanded
with `theta' = [theta, 0]`, and the dual `KernelPerceptron` (tested equal to the explicit version).

**The neural model does not implement the kernel trick.** It uses an *explicit learned* feature map
(`backbone -> psi blocks`). A kernel extension `K_{t+1} = K_t + lambda K_new` is a possible future study and is
deliberately not implemented. Raw-pixel polynomial expansion on CIFAR (3072 dims) is never attempted.

## Model components (`src/incremental_feature_cl`)

| Module | Role |
|---|---|
| `math/` | Reference maths: `PolynomialFeatureMap`, kernels, `Perceptron` (expandable), `KernelPerceptron`. |
| `models/feature_map.py` | `FeatureBlock`: `psi(h) = act(Linear(h))` on the backbone vector `h` (own params random). |
| `models/classifier.py` | `ExpandableLinearClassifier`: `logits = b + sum_k W_k phi_k`. `add_block` (feature axis) and `expand_outputs` (class axis) are independent. |
| `models/backbones.py` | `slimresnet18`, `smallconv`, `mlp`, `identity`; only `out_dim` matters to the expansion mechanism. |
| `models/incremental_model.py` | `IncrementalFeatureMapModel`: `expand_feature_space(new_dim, initialization)`, `expand_outputs(n)`, `ensure_outputs(n)`, diagnostics (`classifier_block_norms`, `block_contribution`), optional `freeze_backbone` / `freeze_old_blocks`. |
| `training/policy.py` | `ExpansionPolicy`: when/how much to grow (shared by trainer and Avalanche plugin). |
| `training/trainer.py` | `ContinualTrainer`: probe before expansion -> expand -> probe -> grow outputs -> train -> probe -> evaluate. |
| `training/replay.py` | Minimal class-balanced buffer for *controlled ablations only* (not a replacement for Avalanche's methods). |
| `data/` | Datasets (synthetic, CIFAR-10/100), `build_class_incremental_stream`, `build_target_vs_rest_stream`. |
| `evaluation/` | Per-class accuracy, forgetting, target metrics, `RunResult` (json/csv). |
| `plotting/` | Plots A-H, regenerated from `results.json`. |
| `avalanche/` | `FeatureExpansionPlugin`, `IncrementalFeatureSpaceStrategy`, benchmark builders, Avalanche baselines. |
| `experiments/` | CLIs listed above + `config.py`, `common.py`. |

Design choices worth knowing: `psi_t` takes the backbone vector `h` (not the whole `Phi_{t-1}`); one shared
classifier bias (new blocks have no own bias, so "zero bias" is automatic); the optimizer is **rebuilt at every
experience** (needed when shapes/params change; same for all variants); `new_feature_dim=0` is the fixed baseline; feature growth starts at experience 1; output growth is
zero-initialised by default and independent of feature growth. `freeze_backbone` and `freeze_old_blocks`
are explicit stability/plasticity controls and default to `false`.

## Two experiments, deliberately separate

* **Mode A - target-vs-rest** (`train_target.py`): one binary logit answers "is this class K?". Experience `t`
  trains on all target samples + the negative classes first introduced in `t` (target data re-appears each
  experience; `--no-target-in-every-experience` changes that). Earlier negatives are not re-shown (unless replay),
  but stay in the *evaluation* negative set, so the negative space grows. Loss: BCE with
  `pos_weight = n_neg / n_pos` of the experience (`train.pos_weight`; set `null` to disable).
* **Mode B - multiclass class-incremental** (`train_cifar100.py`): standard CIL, softmax head grown as classes
  appear. A binary target classifier is **not** a 100-class classifier; no one-vs-rest ensemble is implemented.

## Avalanche integration

```python
model = IncrementalFeatureMapModel(build_backbone("slimresnet18"), num_outputs=1)
strategy = IncrementalFeatureSpaceStrategy(
    model=model, optimizer=opt, policy=ExpansionPolicy(16, "zero")
)
for i, exp in enumerate(benchmark.train_stream):
    strategy.train(exp)
    strategy.eval(benchmark.test_stream[: i + 1])
```
or attach `FeatureExpansionPlugin(policy)` to *any* Avalanche strategy (`Naive`, `Replay`, ...; tested).
The plugin expands the model at the experience boundary and rebuilds the optimizer. The model is a plain
`nn.Module` and has no Avalanche imports. Evaluate on `test_stream[: t+1]` (output heads exist only for seen
classes). Avalanche's `nc_benchmark` (used for `SplitCIFAR100`) needs `n_classes % n_experiences == 0`;
the built-in streams enforce the same divisibility rule.

## Plots (`<run>/plots/`)

| File | Shows |
|---|---|
| A_accuracy_vs_experience | target accuracy (+negative acc, F1) or seen/all accuracy per experience |
| B_accuracy_vs_class | accuracy of each class after the last experience (grey = unseen) |
| C_class_accuracy_heatmap | rows = experience, columns = class, cell = class accuracy |
| D_class_accuracy_curves | every class's accuracy over experiences (drops = forgetting) |
| E_forgetting_vs_class | max previous accuracy - final accuracy per class |
| F_feature_growth | feature dimensionality and parameter count per experience |
| G_new_feature_utilization | `||W_k||` per block and mean `|W_new phi_new|` of the newest block |
| H_expansion_probe | logit change caused by expansion (0 => invariant) and old-prediction agreement after expansion / training |

## Result files

`results.json` (config, environment incl. git commit/torch version, class order, per-experience records,
summary), `metrics.csv`, `per_class_accuracy.csv`, `plots/*.png`. Per experience: feature/parameter counts, `new_parameters_this_experience`,
`cumulative_added_parameters`, `new_feature_parameters`, `cumulative_added_feature_parameters`, expansion
records (`old_feature_dim`, `new_feature_dim`, `number_of_new_parameters`, `old_parameter_count`,
`total_parameter_count`), probe diagnostics, per-class accuracy, target metrics, block norms, and train time. Comparison runs add `comparison.csv/json/png`.

## Invariants under test (`tests/`)

Old predictions unchanged after zero expansion (also with BatchNorm backbones and repeated expansions) - new
weights start at 0 - new weights become non-zero when needed - old parameters keep object identity and values -
feature dim grows exactly as requested - output dim independent of feature dim (both orders commute) -
target-vs-rest labels and streams - no label leakage (forward takes only `x`; permuted labels give identical
predictions; evaluation never mutates the model) - Avalanche integration on a tiny benchmark - Lecture-6
maths (kernel identity, kernel == feature-space perceptron) - reproducibility, results format, plots.

## Known limitations

* **CIFAR-100 remains an experiment, not a package self-test.** The repository contains the loader and
  stream implementation, but no claim of CIFAR-100 performance is encoded in the package.
* Expansion preserves old logits only at the instant of zero-initialised feature expansion; with trainable
  old representation parameters, later optimisation can still forget. Freeze controls make the stability
  choice explicit.
* The Avalanche backend supports multiclass mode only and lacks the pre/post-expansion probes; iCaRL is not
  wired (needs a feature-extractor/classifier split). No one-vs-rest ensemble, no adaptive `new_dim`, no kernel
  expansion, no data augmentation in the built-in loaders.
* The probe subset is drawn from the test set (inputs only) purely for diagnostics; it never influences training.
* Replay in `training/replay.py` is intentionally simplistic; use Avalanche for real baselines.
