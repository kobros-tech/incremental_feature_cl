# incremental-feature-cl

A small scikit-learn-compatible **binary** classifier for experiments with incremental feature expansion and replay. The installed package contains the estimator, not dataset loaders, continual-learning orchestration, plotting, or benchmark code.

The hypothesis under study is narrow: append a feature block while initialising its classifier coefficients to zero. Immediately after expansion, scores are unchanged:
`Phi_new(x) = [Phi_old(x), psi(x)]` and `theta_new = [theta_old, 0]`.
This continuity property does **not** guarantee that later training prevents forgetting.

## Install

```bash
pip install -e ".[dev]"
pytest
```

For plots, install the optional experiment dependencies:

```bash
pip install -e ".[experiments]"
```

Runtime dependencies are NumPy, SciPy, and scikit-learn. PyTorch and Avalanche are not package dependencies. The digits and synthetic experiments need no downloads. For CIFAR-100, `--dataset cifar100` downloads the official archive on first use, verifies its checksum, and caches it under `data/`; `--dataset /path/to/cifar-100-python` uses an already-extracted official Python dataset directory instead. In either case, it samples up to 100 training and 20 test images per class for a reproducible lightweight run.

## Use as a binary estimator or with OneVsRestClassifier

```python
from sklearn.multiclass import OneVsRestClassifier
from incremental_feature_cl import IncrementalFeatureMapModel

# One binary task
model = IncrementalFeatureMapModel(
    new_feature_dim=16,
    replay_mem_size=200,
    random_state=0,
)
model.fit(X, y)

# A continual stream: the first call initialises the model;
# every later partial_fit appends a zero-initialised feature block.
model = IncrementalFeatureMapModel(new_feature_dim=16, random_state=0)
model.partial_fit(X_exp0, y_exp0, classes=[0, 1])
model.partial_fit(X_exp1, y_exp1)
model.partial_fit(X_exp2, y_exp2)

# Multiclass routing is supplied by scikit-learn.
ovr = OneVsRestClassifier(IncrementalFeatureMapModel(new_feature_dim=16))
ovr.fit(X_multiclass, y_multiclass)
```

The model is binary by design. In the binary encoding, `classes_[1]` is the positive class. Scale features before fitting; the optimiser is minibatch SGD on weighted logistic loss.

## Training and replay semantics

- `fit(X, y)` resets model state and trains one stage without expansion.
- The first `partial_fit` initialises the model without expansion; each later call appends `new_feature_dim` features before training.
- `expand_features()` can append a block explicitly. Its classifier coefficients start at zero, so the decision scores must remain unchanged before further training.
- `replay_mem_size=200` enables a bounded binary replay buffer by default. `replay_mem_size=0` disables replay; values of 1 are rejected because the buffer is split between the two binary labels.
- `target_to_negatives` applies to the current batch before replay. It is a **target-samples-per-negative** ratio: `0.2` requests one target per five negatives, `1.0` requests a 1:1 ratio, and `None` keeps all negatives. If the requested ratio would discard every available negative, training raises an error rather than silently fitting a positive-only batch.
- `pos_weight="balanced"` uses the negative-to-positive ratio when both labels occur in the current batch. Before both labels have appeared together it uses 1.0; afterwards it retains the last valid weight for one-label batches so replayed positives remain weighted.

Replay memory is balanced between the binary target and non-target labels. This is **not equivalent** to a buffer balanced over the original multiclass IDs; comparisons with a multiclass replay baseline must state that distinction.

## Research experiments (outside `src/`)

These scripts are deliberately not part of the installed package API.

### Synthetic target-vs-rest stream

```bash
python -m experiments.run_target \
  --dataset synthetic --target-class 0 --n-experiences 3 \
  --new-feature-dim 8 --n-epochs 5 --replay-mem-size 200 \
  --target-to-negatives 0.2 --seed 1 \
  --output results/synthetic_target.json
```

### Digits target-vs-rest stream

```bash
python -m experiments.run_target \
  --dataset digits --target-class 5 --n-experiences 5 \
  --new-feature-dim 8 --n-epochs 5 --seed 1 \
  --output results/digits_target.json
```

### CIFAR-100 target-vs-rest stream

`--dataset` is either a name (`digits`, `synthetic`, `cifar100`) or the path of an extracted CIFAR-100 directory. The directory can itself contain `train` and `test`, or contain a `cifar-100-python/` subdirectory with those files; a path implies `cifar100`, so there is no separate data-directory flag.

```bash
# extracted copy on disk
python -m experiments.run_target \
  --dataset /path/to/cifar-100-python \
  --target-class 17 --n-experiences 20 \
  --new-feature-dim 16 --n-epochs 5 --seed 1 \
  --output results/cifar100_target.json

# or the checksum-verified download/cache
python -m experiments.run_target --dataset cifar100 --target-class 17 --n-experiences 20
```

This deliberately uses a deterministic subset of flattened raw pixels, not a learned image backbone. Treat it as an exploratory binary baseline; it is not directly comparable to the earlier PyTorch/Avalanche experiments that used a different model and training pipeline.

Each experience introduces a disjoint group of classes. At every experience, the script evaluates target recall, negative specificity, balanced accuracy, and ROC AUC on the same held-out test set. That makes the trajectory inspectable, but these small datasets are smoke tests—not evidence of performance on CIFAR-100 or a substitute for a multi-seed benchmark.

### Compare target:negative ratios across seeds

```bash
python -m experiments.compare_ratios \
  --dataset digits --target-class 5 --n-experiences 5 \
  --seeds 0 1 2 3 4 --include-all-negatives \
  --output results/ratio_comparison.json

python -m experiments.plot_results \
  --results results/ratio_comparison.json --output-dir results/ratio_plots
```

### Compare replay off versus bounded memory

```bash
python -m experiments.compare_replay \
  --dataset digits --target-class 5 --n-experiences 5 \
  --seeds 0 1 2 3 4 --replay-sizes 0 200 \
  --output results/replay_comparison.json

python -m experiments.plot_results \
  --results results/replay_comparison.json --output-dir results/replay_plots
```

The run history reports two target-recall diagnostics. `target_recall_forgetting` is the drop from the **best** recall seen since the target arrived; the maximum over a noisy series is biased upward, so a single lucky spike inflates it. `target_recall_change_since_arrival` compares with the recall measured when the target first arrived and is the safer number; read it together with `roc_auc`. If AUC is steady while recall and specificity move in opposite directions, the zero threshold is drifting along a stable ranking; that is not loss of target knowledge. Both are one-target diagnostics, not the conventional average forgetting across a multiclass benchmark. With 20 test images per class, one recall reading has a granularity of 0.05.

`--replay-pos-weight current` (the default) reuses the retained positive weight for replayed targets; `uniform` gives them weight 1. On a synthetic raw-pixel stand-in it raised balanced accuracy over `uniform` by moving the operating point (higher recall, lower specificity) without changing ROC AUC, and it did not reduce the recall swing across experiences.

**Replay memory size with scarce targets.** The buffer is split evenly between the two labels, but a label can only fill its half if enough samples have arrived. With one target class seen once (100 images) a memory of 1000 holds 100 targets and 500 negatives, and on the synthetic stand-in that lowered balanced accuracy by 0.048 ± 0.006 compared with a memory of 200. Check `last_fit_diagnostics_["memory_target"]` against `["memory_negative"]` when changing `replay_mem_size`.

**Step size and input scale.** `learning_rate_init=0.01` is plain SGD, so it interacts with the squared input norm. Flattened 32x32x3 pixels have `||x||^2` of about 850 in [0, 1] and about 3072 once standardised per pixel; on the synthetic stand-in, standardising at 0.01 was worse than raw pixels, and a smaller step (for example 0.001) was needed for the larger norm. Treat `learning_rate_init` and the feature scale as one setting.

### Plot a saved run and verify the mathematical reference

```bash
python -m experiments.plot_results \
  --results results/digits_target.json --output-dir results/digits_plots
python -m experiments.math_reference
```

The math reference checks zero-initialised expansion invariance and the explicit degree-two polynomial feature identity. Plotting reads saved JSON; it does not retrain the estimator.

## What the experiments do and do not establish

The tests cover the estimator API, deterministic behaviour, finite-difference gradient checks, zero-initialised expansion, replay, target/negative sampling, and the small experiment pipeline. A gradient check verifies that the implementation's gradients match its implemented loss; it does not establish parity with the previous PyTorch or scikit-learn implementation.

Do not compare results without recording the seed, experience construction, target class, ratio, replay capacity, training settings, and evaluation metric. In particular, zero-initialised expansion guarantees score continuity only at the expansion boundary; subsequent optimisation can still alter old predictions.
