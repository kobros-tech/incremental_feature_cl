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

Runtime dependencies are NumPy, SciPy, and scikit-learn. PyTorch and Avalanche are not package dependencies. The digits and synthetic experiments need no downloads. For CIFAR-100, `--dataset cifar100` downloads the official archive on first use, verifies its checksum, and caches it under `data/`; `--dataset /path/to/cifar-100-python` uses an already-extracted official Python dataset directory, and `--dataset /path/to/cifar-100-python.tar.gz` uses a local tar archive. The archive must contain `cifar-100-python/train` and `cifar-100-python/test`. In each case, it samples up to 100 training and 20 test images per class for a reproducible lightweight run.

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

`--dataset` accepts a name (`digits`, `synthetic`, `cifar100`), the path of an extracted CIFAR-100 directory, or a local CIFAR-100 tar archive. An extracted directory can itself contain `train` and `test`, or contain a `cifar-100-python/` subdirectory with those files; a path implies `cifar100`, so there is no separate data-directory flag.

```bash
# extracted copy or local archive on disk
python -m experiments.run_target \
  --dataset /path/to/cifar-100-python \
  --target-class 17 --n-experiences 20 \
  --new-feature-dim 16 --n-epochs 5 --seed 1 \
  --output results/cifar100_target.json

python -m experiments.run_target \
  --dataset /data/cifar-100-python.tar.gz \
  --target-class 17 --n-experiences 20

# or the checksum-verified download/cache
python -m experiments.run_target --dataset cifar100 --target-class 17 --n-experiences 20
```

This deliberately uses a deterministic subset of flattened raw pixels, not a learned image backbone. Treat it as an exploratory binary baseline; it is not directly comparable to the earlier PyTorch/Avalanche experiments that used a different model and training pipeline.

The underlying class stream introduces a disjoint group of classes at each experience. By default, the experiment also includes the target-class training examples in every experience (`--target-every-experience`), matching a target-versus-rest task in which positive examples remain available while new negative classes arrive. The target examples are reused from the training split only; held-out test data is never added to training. Their replay-memory reservoir is retained from first exposure rather than repeatedly inserting duplicate copies of the same images. Use `--no-target-every-experience` for the stricter setting where the target class appears only in its original class group and later experiences can contain only negatives. The output distinguishes `classes_arriving`, `classes_in_training_batch`, `natural_target_arrival_experience`, and `first_training_exposure_experience`; repeated target exposure means these are not necessarily the same experience. These modes answer different continual-learning questions and must not be compared without noting the mode. The script reserves a stratified validation subset from the training split (20%, with at least one validation example per class). At each experience, threshold calibration uses only validation examples whose original class labels have already been observed in the training stream through that experience; validation labels from future classes are excluded. This cumulative seen-class scope lets strict class-once mode calibrate after the target has arrived, using target and previously seen negative classes, without requiring target samples in every later training batch. In strict class-once mode, no threshold is reported before the target has arrived, and calibration waits until the permitted validation subset contains both target and negative examples. The `threshold_calibration` record recalibrates at each experience when valid; the `fixed_threshold_calibration` record freezes the first valid threshold and records its exact `calibration_experience` and `validation_classes`. Those calibration metrics use validation data only; full held-out test metrics are for evaluation only and never select thresholds. The output also reports metrics at threshold zero and ROC AUC on the full held-out test set. Threshold ties are resolved by choosing the cutoff closest to zero, then the smaller cutoff. Read all operating points alongside ROC AUC: if AUC stays stable while fixed-threshold recall/specificity shift, scores are drifting relative to the frozen cutoff; if AUC falls, ranking/separation is degrading. These small datasets are smoke tests—not evidence of performance on CIFAR-100 or a substitute for a multi-seed benchmark.

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

The run history reports target-recall diagnostics at threshold zero. `target_recall_forgetting` is the drop from the **best** recall seen since the target was first included in training; the maximum over a noisy series is biased upward, so a single lucky spike inflates it. `target_recall_change_since_arrival` compares with recall at the first training exposure (also reported as `target_recall_at_first_training_exposure`). The recall at the natural class-stream arrival is reported separately as `target_recall_at_natural_class_arrival`; these differ when `--target-every-experience` is enabled. Read them together with `roc_auc`. If AUC is steady while recall and specificity move in opposite directions, the zero threshold is drifting along a stable ranking; that is not loss of target knowledge. Both are one-target diagnostics, not the conventional average forgetting across a multiclass benchmark. With 20 test images per class, one recall reading has a granularity of 0.05.

`--replay-pos-weight current` (the default) reuses the retained positive weight for replayed targets; `uniform` gives them weight 1. On a synthetic raw-pixel stand-in it raised balanced accuracy over `uniform` by moving the operating point (higher recall, lower specificity) without changing ROC AUC, and it did not reduce the recall swing across experiences.

**Replay memory size with scarce targets.** The buffer is split evenly between the two labels, but a label can only fill its half if enough unique samples have arrived. With 100 target images in the loaded training subset, reserving 20% for validation leaves 80 unique target fitting images. A memory of 1000 can therefore hold 80 targets and 500 negatives even when target examples are included in every experience; the experiment runner does not repeatedly insert copies of those same target images into the positive reservoir. On the synthetic stand-in, that configuration lowered balanced accuracy by 0.048 ± 0.006 compared with a memory of 200. Check `last_fit_diagnostics_["memory_target"]` against `["memory_negative"]` when changing `replay_mem_size`.

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

## Experiment notes

- `--target-every-experience` re-trains on the target only in experiences *after* its natural
  arrival. If the target arrives in the last experience the flag has no effect; the runner warns
  and records `target_reintroduction_effective: false`.
- Reintroduced target rows are trained on but not stored again: `partial_fit(...,
  store_in_memory=mask)` keeps repeated copies from crowding the replay reservoir.
- Each history entry separates current-batch counts (`n_target_natural`, `n_target_injected`,
  `fit_diagnostics.current_*`) from replay counts (`fit_diagnostics.replay_*`) and memory changes
  (`memory_*_added`).
- `decision_function` accepts `float32` input and converts it chunk by chunk, so
  `inference_batch_size` also bounds input-preparation memory.
