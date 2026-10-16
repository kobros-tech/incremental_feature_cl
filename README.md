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

```text
6.86x maths -> math/ (feature maps, kernels, perceptron) -> models/ (expandable feature model)
            -> training/ (plain PyTorch)  ->  avalanche/ (optional adapter) -> Split CIFAR-100
```

## Install

```bash
pip install -e .                      # numpy, torch, pyyaml, matplotlib
pip install -e ".[vision]"            # + torchvision (CIFAR-10/100 download)
pip install -e ".[avalanche,dev]"     # + avalanche-lib, pytest, ruff
pip install -e ".[research]"          # + scikit-learn (--vs-sklearn control, digits datasets)
pytest                                # CPU, no downloads; Avalanche tests need the avalanche extra
```

## Commands

The package ships no network architectures: the shared representation is any external `nn.Module`
(`--backbone identity | simplemlp | resnet32 | package.module.Name`, see `models/backbones.py`).
`identity` feeds the flattened input (raw pixels for images); `simplemlp` / `resnet32` are Avalanche
models and need `pip install -e ".[avalanche]"`.

```bash
# 1a. The Lecture-6 mathematics (no neural net): expansion invariance + kernel identities
python -m incremental_feature_cl.experiments.math_demo

# 1b. Smallest neural experiment (synthetic data, CPU, seconds). zero vs random expansion
python -m incremental_feature_cl.experiments.synthetic_demo            # --mode target for target-vs-rest

# 2. Split CIFAR-100, 5 experiences, one multiclass head (first real milestone)
python -m incremental_feature_cl.experiments.train_cifar100 --dataset cifar100 --backbone resnet32 \
    --n-experiences 5 --new-feature-dim 16 --train-epochs 5 --seed 1

# 3. Target class 17 vs the rest, 20 experiences
python -m incremental_feature_cl.experiments.train_feature_map --dataset cifar100 --backbone resnet32 \
    --target-class 17 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1

# 3b. Same, with an explicit target:negative ratio (1:5 = 1 target : 5 negatives; omit = cumulative)
python -m incremental_feature_cl.experiments.train_feature_map --dataset cifar100 --backbone resnet32 \
    --target-class 17 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1 --ratio 1:5

# 3c. Sweep the target:negative ratio 5:1 ... 1:5 (+ cumulative reference) for one or more targets
python -m incremental_feature_cl.experiments.compare_ratios --dataset cifar100 --backbone resnet32 \
    --target-class 17 --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1 --yes
python -m incremental_feature_cl.experiments.compare_ratios --target-class 0 17 50 99 --dry-run
#     rebuild a finished sweep's tables/plots from its saved runs (no training, no dataset):
python -m incremental_feature_cl.experiments.compare_ratios --resummarize results/<sweep-dir>

# 4. Full 20-experience class-incremental benchmark (20 x 5 classes)
python -m incremental_feature_cl.experiments.train_cifar100 --dataset cifar100 --backbone resnet32 \
    --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1
#    same benchmark through Avalanche (SplitCIFAR100 + IncrementalFeatureSpaceStrategy):
#    add --backend avalanche

# 5. Ablation: fixed / fixed+replay / expand-random / expand-zero / expand-zero+replay (+ Avalanche refs)
python -m incremental_feature_cl.experiments.compare_baselines --dataset cifar100 --backbone resnet32 \
    --n-experiences 5 --new-feature-dim 16 --replay 200 --avalanche-baselines naive,replay,er_ace,ewc

# 6. Matrix {5,10,20} x {0,4,16,32} x {zero,random}  -- prints the plan; needs --yes to run
python -m incremental_feature_cl.experiments.run_sweep --preset cifar100-matrix --dry-run

# 7. Several / all target classes (prints an estimate; needs --yes)
python -m incremental_feature_cl.experiments.train_feature_map --target-class 0 17 50 99 --yes
python -m incremental_feature_cl.experiments.train_feature_map --target-class all --dry-run

# 8. One persistent Skill (class 17 vs the rest) and the Skill ensemble (one Skill per class)
python -m incremental_feature_cl.experiments.train_skill --dataset cifar100 --target-class 17 --ratio 1:5
python -m incremental_feature_cl.experiments.train_ovr --dataset cifar100 --n-experiences 5 --ratio 1:5

# Regenerate plots from a saved result, no retraining
python -m incremental_feature_cl.plotting --results results/<run>/results.json
```

Every script takes `--config configs/*.yaml`, any `--set section.key=value`, `--device auto|cpu|cuda`.
Run order recommended: synthetic -> CIFAR-10/small subset -> CIFAR-100 x5 -> x20 -> zero vs random -> baselines.

## Continual One-vs-Rest Skill experiment (`train_ovr`)

`OneVsRestSkillModel` keeps one persistent `Skill` per class (class `c` versus everything else it has
seen) on top of a **frozen** shared representation, and refreshes *every* skill whenever new classes
arrive: new classes become negatives of all old skills, old classes (replayed exemplars) become
negatives of the new skills.

```bash
# quick structural run (no downloads)
python -m incremental_feature_cl.experiments.train_ovr --dataset synthetic --backbone identity \
    --n-experiences 5 --ratio 1:5 --seed 1 --no-plots

# baseline: raw pixels (identity) on CIFAR-100, with the feature-map comparison for class 0
python -m incremental_feature_cl.experiments.train_ovr --dataset cifar100 --backbone identity \
    --n-experiences 5 --ratio 1:5 --memory-per-class 100 --train-epochs 10 --seed 1 \
    --no-plots --vs-feature-map 0

# diagnostic: a learned representation (trained on experience 0, then frozen); needs avalanche-lib
python -m incremental_feature_cl.experiments.train_ovr --dataset cifar100 --backbone simplemlp \
    --backbone-kwargs '{"hidden_size": 256}' --warmup-epochs 5 --freeze-backbone \
    --n-experiences 5 --ratio 1:5 --memory-per-class 100 --train-epochs 10 --seed 1 \
    --no-plots --vs-feature-map 0

# full-batch L-BFGS for the skills (explicit; --vs-feature-map is then unavailable)
python -m incremental_feature_cl.experiments.train_ovr --dataset cifar100 --backbone identity \
    --n-experiences 5 --ratio 1:5 --memory-per-class 100 --optimizer lbfgs --skill-max-iter 100 \
    --weight-decay 1e-3 --seed 1 --no-plots

# scikit-learn control with the same data protocol (needs the research extra)
python -m incremental_feature_cl.experiments.train_ovr --dataset synthetic --backbone identity \
    --n-experiences 5 --ratio 1:5 --vs-sklearn
```

### Is the Skill-vs-feature-map gap the learner or the protocol?

The two experiments differ in more than the model: negative sampling, replay and (before the training
contract above) the optimizer. `compare_single_class` separates them for one class:

```bash
python -m incremental_feature_cl.experiments.compare_single_class --dataset cifar100 --backbone identity \
    --target-class 0 --n-experiences 5 --ratio 1:5 --memory-per-class 100 --train-epochs 10 \
    --train-mb-size 100000 --seed 1
```

* **Part 1 (matched learner).** The same frozen features, the same training / validation / test rows, the
  same `pos_weight`, the same initial weights, the same optimizer and budget go through `Skill._fit` and
  through the feature-map trainer; it prints the score agreement. With full-batch steps
  (`--train-mb-size` larger than the training set) they agree to ~1e-7 (tested for SGD, Adam and feature
  expansion). With minibatches only the shuffle order differs. The feature-map trainer has no L-BFGS, so
  this part needs `--optimizer sgd|adam`.
* **Part 2 (continual vs joint).** The continual Skill is compared with skills trained at once on all data:
  `joint_memory_positives` (same positive budget as the replay) isolates the effect of learning
  continually, and `joint_all_positives` isolates the replay budget. All are scored on identical test rows
  with the same validated-threshold rule.

If Part 1 agrees, a gap between the earlier results is not explained by the classifier; look at the data
protocol (Part 2), the replay budget and the decision threshold instead. A high AUC with a low F1 at
threshold 0 means a ranking that is fine and a threshold that is not; it says nothing about whether the
learning procedure is correct.

### What is measured, and what deliberately is not

Skill scores are raw logits of independently trained binary classifiers; they were never trained to be
comparable with each other. **No `argmax` is used anywhere in the reports**: an `argmax` always
answers, even when no skill recognises the sample, and it silently assumes comparability. The
experiment instead asks which skills *fire* (`score > threshold`):

| Report | Meaning |
| --- | --- |
| per-skill binary metrics (`target_metrics`, `per_skill_metrics`) | recall, precision, F1, negative accuracy, balanced accuracy, AUC of every skill as a target-vs-rest classifier on the seen classes, decision `score > 0`. Headline numbers are means over skills. |
| `decision_report` (threshold 0) | over all test samples of seen classes: `none_fired_rate` (no skill recognises it), `multi_fired_rate` (several do), `exact_match_rate` (only the true skill fires; abstentions and ambiguous samples count as errors), `precision_when_single`, per-class recall / precision / false-positive rate, and the firing matrix `[true class, skill]`. |
| `target_metrics_tuned`, `decision_report_tuned` | the same with one **validated threshold per skill** (F1-maximising on held-out validation data). |

**Why thresholds matter.** With `K` seen classes only about `1/K` of the evaluated samples are positives
for a given skill, so a skill trained at a 1:5 target:negative ratio and judged at `score > 0` can show
a decent recall and a poor precision / F1 even when its ranking (AUC) is good. The validated threshold
separates *a threshold problem* from *a representation problem* (low AUC) and from *forgetting*.

**Validation protocol.** `--val-fraction` (default `0.1`; `0` disables it and reproduces the previous
training data exactly) holds out that fraction of every class's training data in every experience. The
held-out samples are **never used for training and never come from the test set**; they are only used to
choose the thresholds. The validation features of earlier classes are kept for later experiences (the
representation is frozen), which is evaluation-time calibration with access to old-class validation
data, not something a strict continual learner would have. Thresholds never feed back into training. With
very small validation sets (a handful of samples per class, as on the tiny synthetic data) the selected
thresholds are noisy and can be *worse* than `0`; the saved `n_validation` states how many samples were
used.

**Forgetting per skill.** `average_skill_forgetting` is the standard peak-to-final forgetting of each
skill's target recall at threshold 0. Because that number also moves when only a threshold drifts, the run
also reports it for the threshold-free AUC (`average_skill_auc_forgetting`) and for recall at the validated
thresholds (`average_skill_tuned_recall_forgetting`). `skill_*_forgetting.csv` holds the running forgetting
(best so far minus current) of every skill after every experience.

**Training contract (one explicit optimizer).** `--optimizer sgd|adam|lbfgs` is a single setting shared by
every route and defaults to `sgd`; it is never inferred from the model. With `sgd`/`adam`, `--train-epochs`,
`--train-mb-size` and `--lr` control the training. `lbfgs` is full-batch, available only to the Skill routes
(`train_skill`, `train_ovr`), and is bounded by `--skill-max-iter` instead. Flags the chosen optimizer would
silently ignore (`--skill-max-iter` without `lbfgs`, `--train-epochs` / `--train-mb-size` with `lbfgs`,
`--optimizer lbfgs` on a feature-map route or with `--vs-feature-map`) are rejected, and every skill run
prints a `[contract]` line stating what controls its training. Weight decay is the same
(`--weight-decay`, default 0) for every optimizer; there is no L-BFGS-specific default. The Avalanche
strategy always runs one epoch because the skills own their training loops.

**`--vs-feature-map K ...`** trains `IncrementalFeatureMapModel` once per listed class; every other class
shows `n/a` and the feature-map `mean` covers only the listed classes. One class is not evidence of a general
advantage; list several.

Outputs: `results.json`, `metrics.csv`, `skill_target_recall.csv`, `skill_auc.csv`,
`skill_target_recall_tuned.csv`, `skill_auc_forgetting.csv`, `skill_target_recall_tuned_forgetting.csv`,
`skill_thresholds.csv`, `firing_matrix_threshold0.csv`, `firing_matrix_tuned.csv` (final experience), and
plots unless `--no-plots`.

### Avalanche

```python
strategy = OneVsRestSkillStrategy(model=model, num_classes=100)
for experience in benchmark.train_stream:
    strategy.train(experience)
per_skill, decisions = strategy.evaluate_skills(benchmark.test_stream)  # no argmax
```

The strategy's default evaluator contains no Avalanche accuracy metric because that metric is an `argmax`
over the skill columns. `strategy.forward` lays the scores out by class id only to map columns to classes;
that does not make the logits comparable.

### Design responsibilities

* `IncrementalFeatureMapModel` is the expandable feature-map model used by the single-target and
  multiclass experiments. It does not itself implement class-specific replay policy.
* `Skill` is one persistent binary learner: one class versus the negatives it has observed. It owns positive
  and per-negative-class replay memory, target:negative sampling, BCE training, and one-time feature
  expansion.
* `OneVsRestSkillModel` creates exactly one `Skill` per class, updates old skills when new classes arrive,
  holds the shared frozen representation, and maps score columns back to global class IDs. The optional
  `ComparisonConfig` (`--comparison-weight`, default off) trains the skills jointly so that their scores
  become more comparable; whether it helps must be measured with the reports above, not assumed.

The shared backbone is frozen because replay stores feature vectors, not images. This differs from
`train_feature_map`'s default trainable-backbone setup, so the two results are not an apples-to-apples
comparison. The `identity` backbone gives the skills raw pixels; high results on the synthetic data show that
the continual protocol works under favourable conditions, not that a useful CIFAR-100 representation is
learned.

## Mathematical Motivation

**Feature maps (Lecture 6).** A linear classifier cannot separate `x = -1, 0, 1` with labels `+1, -1, +1`.
Apply a feature map `x -> Phi(x) = (x, x^2)` and the *same* linear machinery works: the decision
`theta . Phi(x) + b >= 0` is linear in feature space and nonlinear in `x`.

```text
x  ->  Phi(x)  ->  linear classifier  f(x) = W Phi(x) + b
```

**Incremental expansion.** Instead of a fixed `Phi`, grow it by appending a block:

```text
Phi_{t+1}(x) = [ Phi_t(x) , psi_{t+1}(x) ]          W_{t+1} = [ W_t , 0 ]
```

Then, immediately after expansion and before any optimisation step,

```text
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
| --- | --- |
| `math/` | Reference maths: `PolynomialFeatureMap`, kernels, `Perceptron` (expandable), `KernelPerceptron`. |
| `models/feature_map.py` | `FeatureBlock`: `psi(h) = act(Linear(h))` on the backbone vector `h` (own params random). |
| `models/classifier.py` | `ExpandableLinearClassifier`: `logits = b + sum_k W_k phi_k`. `add_block` (feature axis) and `expand_outputs` (class axis) are independent. |
| `models/backbones.py` | `identity` plus `FeatureBackbone`, an adapter for any external `nn.Module` (Avalanche `SimpleMLP` / `resnet32`, torchvision, ...); only `out_dim` matters to the expansion mechanism. |
| `models/incremental_model.py` | `IncrementalFeatureMapModel`: `expand_feature_space(new_dim, initialization)`, `expand_outputs(n)`, `ensure_outputs(n)`, diagnostics (`classifier_block_norms`, `block_contribution`), optional `freeze_backbone` / `freeze_old_blocks`. |
| `models/skill.py` | `Skill`: one binary target-vs-rest learner with per-class replay memory and incremental feature expansion. |
| `models/ovr.py` | `OneVsRestSkillModel`: class-to-Skill coordinator, frozen shared backbone, global class-ID score mapping. |
| `training/policy.py` | `ExpansionPolicy`: when/how much to grow (shared by trainer and Avalanche plugin). |
| `training/trainer.py` | `ContinualTrainer`: probe before expansion -> expand -> probe -> grow outputs -> train -> probe -> evaluate. |
| `training/replay.py` | Minimal class-balanced buffer for *controlled ablations only* (not a replacement for Avalanche's methods). |
| `data/` | Committed synthetic/CIFAR loaders, `ArrayDataset`, and deterministic class-incremental/target-vs-rest streams. |
| `evaluation/` | Per-class accuracy, forgetting, target metrics, per-skill metrics, validated thresholds and the argmax-free `decision_report`, `RunResult` (json/csv). |
| `plotting/` | Plots A-H, regenerated from `results.json`. |
| `avalanche/` | `FeatureExpansionPlugin`, `IncrementalFeatureSpaceStrategy`, `OneVsRestSkillStrategy`, benchmark builders, Avalanche baselines. |
| `experiments/` | CLIs listed above + `config.py`, `common.py`. |

Design choices worth knowing: `psi_t` takes the backbone vector `h` (not the whole `Phi_{t-1}`); one shared
classifier bias (new blocks have no own bias, so "zero bias" is automatic); the optimizer is **rebuilt at every
experience** (needed when shapes/params change; same for all variants); `new_feature_dim=0` is the fixed baseline; feature growth starts at experience 1; output growth is
zero-initialised by default and independent of feature growth. `freeze_backbone` and `freeze_old_blocks`
are explicit stability/plasticity controls and default to `false`.

## Two experiments, deliberately separate

* **Mode A - target-vs-rest** (`train_feature_map.py`): one binary logit answers "is this class K?". Experience `t`
  trains on all target samples + the **cumulative** pool of every negative class introduced so far
  (`0..t`; target data re-appears each experience, `--no-target-in-every-experience` changes that), so the
  classifier keeps seeing a growing negative space and the negative/positive ratio grows with `t`. Evaluation
  uses the same negative set. Loss: BCE with `pos_weight = n_neg / n_pos` of the experience
  (`train.pos_weight`; set `null` to disable). The data ratio can be controlled explicitly, see below.
* **Mode B - multiclass class-incremental** (`train_cifar100.py`): standard CIL, softmax head grown as classes
  appear. A binary target classifier is **not** a 100-class classifier; the Skill ensemble is a separate experiment (`train_ovr.py`).

### Target-vs-rest negative ratio (`target_to_negatives`)

An experimental control for studying the effect of cumulative negative imbalance. It sets how many negatives
accompany the target samples in each training experience; it changes the **data stream only** and never the loss
(`train.pos_weight` stays as configured, so data ratio and loss weighting can be varied independently).

| `target_to_negatives` | Meaning | Negatives per experience |
| --- | --- | --- |
| `None` (default) | cumulative behaviour, no subsampling | the whole pool (grows with `t`) |
| `1.0` | 1 target : 1 negative | `n_target` |
| `0.5` | 1 target : 2 negatives | `2 * n_target` |
| `0.2` | 1 target : 5 negatives | `5 * n_target` |
| `0.1` | 1 target : 10 negatives | `10 * n_target` |
| `2.0` | 2 targets : 1 negative | `n_target / 2` |

Rules: the value is *targets per negative*, so `n_negative = floor(n_target / target_to_negatives)`. All target
samples are always kept (never subsampled or duplicated); negatives are drawn without replacement from the
cumulative pool, deterministically from `seed` and the experience index; if the pool is smaller than requested
(early experiences) all of it is kept.

**Requested vs realized ratio.** `target_to_negatives` is a *requested upper bound on negatives per target*.
Negatives are never duplicated (duplicating them would add a second confound), so while the cumulative negative
pool is smaller than requested an experience trains on fewer negatives than asked: requested 1:5 may train at
1:3 in experience 0 and at 1:5 from experience 1 on. Both are recorded for every experience in `results.json` /
`metrics.csv`: `requested_target_to_negatives`, `n_requested_negative`, `n_train_target`, `n_train_negative`,
`realized_target_to_negatives` (targets per negative), `realized_ratio_label` (e.g. `1:3`) and `ratio_satisfied`
(`null` when there is nothing to enforce: cumulative mode, or an experience without target samples). Whether this
matters depends on the setup: Split CIFAR-100 with 20 experiences has 5 negative classes (2,500 images) in
experience 0, so 1:5 is reached from the very first experience (exactly), 1:10 from experience 1, 1:20 from
experience 3; with 5 experiences the pool is 20 classes and every ratio up to 1:20 is reached immediately.
`compare_ratios` prints this before running (see below) so a headline ratio is never misleading. With `--no-target-in-every-experience`, experiences without target samples
keep the whole pool. Non-positive/non-finite values (and ratios so large that zero negatives would be requested)
raise `ValueError`; the flag is rejected outside target mode. Experiment names get a `_r<ratio>` suffix (e.g.
`..._r0.2_s1`) when a ratio is set; `None` keeps the previous names. The value is stored in `results.json`.

```bash
python -m incremental_feature_cl.experiments.train_feature_map --dataset cifar100 --target-class 17 \
    --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1 --ratio 1:5
# sweep the ratio (null = original cumulative behaviour); prints the plan, needs --yes to run
python -m incremental_feature_cl.experiments.run_sweep --mode target --target-class 17 --dataset cifar100 \
    --grid target_to_negatives=null,0.1,0.2,0.5,1 --dry-run
```

No ratio is claimed to be optimal; this only makes the ratio an experimental variable. In earlier runs of this
repository, `0.2` corresponds to the 1:5 behaviour that used to be hardcoded (selected samples are identical).

### Sweeping the ratio (`compare_ratios`)

Runs the same target-vs-rest experiment once per ratio, for one or more target classes, and summarises
the trade-off. Ratios are written `target:negative`: `5:1` = 5 targets per negative, `1:5` = 1 target per 5
negatives. Default grid: `5:1 2:1 1:1 1:2 1:5` plus `cumulative` (the original un-subsampled behaviour) as a
reference; choose your own with `--ratios 5:1 1:1 1:5` (numbers and `cumulative` also work; `--no-cumulative`
drops the reference).

```bash
python -m incremental_feature_cl.experiments.compare_ratios --dataset cifar100 --target-class 17 \
    --n-experiences 20 --new-feature-dim 16 --train-epochs 5 --seed 1 --dry-run   # plan + sample counts
... same command with --yes to run it
... --target-class 0 17 50 99 --yes            # several targets; mean/std across targets are reported
... --target-class all --yes                   # every class (prints the plan first; large!)
... --ratios 5:1 1:1 1:5 --pos-weights balanced none --yes
... --config configs/ratio_sweep.yaml --yes
```

* **Safe by default:** like `run_sweep`, it prints the plan (runs, training samples per ratio) and only runs
  with `--yes` (`--dry-run` never runs). The plan also states, per ratio, in how many experiences the requested
  ratio is actually reached (and the realized ratio of the shortfall). Sweeps of 50+ runs print a `WARNING`:
  `--target-class all` on CIFAR-100 with the default grid is 600 runs (1,200 with two loss settings).
* **Loss weighting is a confound, so it is an axis.** With the default `train.pos_weight=balanced` the loss
  gives positives and negatives equal total weight whatever the data ratio (`pos_weight = n_neg / n_pos`, from
  the *realized* counts of each experience), so the sweep then isolates *how many / how varied* the negatives are
  while their aggregate loss contribution is normalised. `--pos-weights balanced none` runs both settings so the
  effect of class imbalance in the loss can be separated from the data ratio: it is a second experimental axis
  (results are tagged `__pw-<w>`), not an implementation detail. A non-default weighting is always part of the
  run directory name, so it can never overwrite the `balanced` runs. The loss is never changed implicitly.
* **Resumable:** a run whose saved `results.json` has an identical config is reused (`--no-resume`
  recomputes), so an interrupted CIFAR-100 sweep can be restarted with the same command. Everything that
  influences training (lr, seed, ratio, target, loss weighting, model, data, device, ...) is
  compared. Only the run's storage metadata (`name`, `output_dir`) is ignored, so a moved sweep is not recomputed. A run that is
  replaced because its config changed is announced. `results.json` is written atomically, so an interrupted
  run is never mistaken for a finished one. Keep one sweep per output directory (change `--name` /
  `--output-dir` when you change settings such as epochs or learning rate).
* **Re-summarisable:** `--resummarize <sweep-dir>` rebuilds every table and plot from the saved runs without
  training or a dataset (and warns if the directory mixes runs made with different settings).
* **Verifiable:** every experience records `n_train_target` / `n_train_negative`, so the realized ratio is in
  `results.json`/`metrics.csv` (not only the requested one).

Outputs in `<output-dir>/<name>/` (name defaults to `ratio_sweep_<dataset>_e<n>_d<dim>_<init>_s<seed>`):

| File | Content |
| --- | --- |
| `ratio_sweep_runs.csv` | one row per run: target, **requested** ratio (`ratio`, `ratio_label`), loss weighting, final/average metrics, and the realized ratio (`final_realized_ratio_label`, `final_negatives_per_target`, `mean_negatives_per_target`, `fraction_experiences_at_requested_ratio`, `first_experience_at_requested_ratio`) |
| `ratio_sweep_by_ratio.csv` | mean and std across targets for each ratio (and loss setting) |
| `ratio_sweep.json` | plan, config, environment, both tables, `ratio_semantics` |
| `ratio_sweep_main.png` | headline figure: balanced accuracy (final / mean over experiences) vs requested ratio, and realized vs requested negatives per target (points below the diagonal = pool too small) |
| `ratio_sweep_final_metrics.png` | final target recall / negative accuracy, balanced accuracy, F1 / false-positive rate, forgetting vs ratio (thin lines = individual targets) |
| `ratio_curves.png` | per-experience target recall, negative accuracy and balanced accuracy, one line per ratio (mean over targets); `ratio_curves__pw-<w>.png` per loss setting |
| `runs/ratio_<r>/target_<k>/` | the normal per-run `results.json`, `metrics.csv`, `per_class_accuracy.csv` (plots only with `--run-plots`; regenerate any time with `python -m incremental_feature_cl.plotting`) |

Metrics: *negative accuracy* is measured on the negative classes seen so far; *balanced accuracy* =
(target recall + negative accuracy) / 2 (overall binary accuracy depends on the test-set class mix, so use
balanced accuracy to compare ratios); *target recall peak->final drop* (`target_recall_peak_to_final_drop`) =
max earlier target recall - final target recall. It is exactly the standard per-class forgetting of the target
class (tested), so it is measured against the *peak*: a model that dipped and partly recovered still reports
the drop from its best earlier recall, and a negative value means the final recall is the best so far. It says
nothing about the negative classes; `average_forgetting` is the mean forgetting over *all* classes (it was
called `target_recall_drop` before). `average_*` metrics average over experiences and therefore mix experiences
that reached the requested ratio with ones that did not; use `final_*` (or check
`fraction_experiences_at_requested_ratio`) when comparing ratios.
The synthetic dataset is a plumbing check only (use `--backbone identity` there); conclusions about ratios need CIFAR-100 runs.

## Avalanche integration

```python
model = IncrementalFeatureMapModel(build_backbone("resnet32", (3, 32, 32)), num_outputs=1)
strategy = IncrementalFeatureSpaceStrategy(
    model=model, optimizer=opt, policy=ExpansionPolicy(16, "zero")
)
for i, exp in enumerate(benchmark.train_stream):
    strategy.train(exp)
    strategy.eval(benchmark.test_stream[: i + 1])
```

or attach `FeatureExpansionPlugin(policy)` to *any* Avalanche strategy (`Naive`, `Replay`, ...; tested).
The plugin expands the model at the experience boundary and rebuilds the optimizer. The model is a plain
`nn.Module` and has no Avalanche imports. Evaluate on `test_stream[: t+1]`. The model uses global class IDs, so the output head may already contain
units for unseen IDs when a high global class ID is first encountered; those units are untrained. Avalanche's `nc_benchmark` (used for `SplitCIFAR100`) needs `n_classes % n_experiences == 0`;
the built-in streams enforce the same divisibility rule.

## Plots (`<run>/plots/`)

| File | Shows |
| --- | --- |
| A_accuracy_vs_experience | target accuracy (+negative acc, F1) or seen/all accuracy per experience |
| B_accuracy_vs_class | accuracy of each class after the last experience (grey = unseen) |
| C_class_accuracy_heatmap | rows = experience, columns = class, cell = class accuracy |
| D_class_accuracy_curves | every class's accuracy over experiences (drops = forgetting) |
| E_forgetting_vs_class | max previous accuracy - final accuracy per class |
| F_feature_growth | feature dimensionality and parameter count per experience |
| G_new_feature_utilization | `norm(W_k)` per block and mean `norm(W_new phi_new)` of the newest block |
| H_expansion_probe | logit change caused by expansion (0 => invariant) and old-prediction agreement after expansion / training |

## Result files

`results.json` (config, environment incl. git commit/torch version, class order, per-experience records,
summary), `metrics.csv`, `per_class_accuracy.csv`, `plots/*.png`. Per experience: feature/parameter counts, `new_parameters_this_experience`,
`cumulative_added_parameters`, `new_feature_parameters`, `cumulative_added_feature_parameters`, expansion
records (`old_feature_dim`, `new_feature_dim`, `number_of_new_parameters`, `old_parameter_count`,
`total_parameter_count`), probe diagnostics, `n_train_samples` (and in target mode `n_train_target` / `n_train_negative`, the realized
training ratio, plus the requested-vs-realized report described above), per-class accuracy, target metrics, block norms, and train time. `metrics.csv` has scalar
columns only; nested records are flattened (`probe_*`, `feature_expansion_*`). Comparison runs add `comparison.csv/json/png`.

## Invariants under test (`tests/`)

Old predictions unchanged after zero expansion (also with BatchNorm backbones and repeated expansions) - new
weights start at 0 - new weights become non-zero when needed - old parameters keep object identity and values -
feature dim grows exactly as requested - output dim independent of feature dim (both orders commute) -
target-vs-rest labels and streams (incl. the `target_to_negatives` ratio: None keeps the cumulative pool,
exact 1:1 / 1:5 / 1:10 / 2:1 counts, targets never subsampled, deterministic and seed-dependent sampling,
invalid ratios rejected, requested-vs-realized report; `compare_ratios`: `a:b` notation, aggregation, realized
counts and shortfall reporting, resume (incl. what does / does not invalidate it), loss-weighting axis,
`--resummarize`, target forgetting == standard forgetting, CSV/plot outputs) - no label leakage (forward takes only `x`; permuted labels give identical
predictions; evaluation never mutates the model) - Avalanche integration on a tiny benchmark - Lecture-6
maths (kernel identity, kernel == feature-space perceptron) - reproducibility, results format, plots.

## Known limitations

* **CIFAR-100 remains an experiment, not a package self-test.** The repository contains the loader and
  stream implementation, but no claim of CIFAR-100 performance is encoded in the package.
* Expansion preserves old logits only at the instant of zero-initialised feature expansion; with trainable
  old representation parameters, later optimisation can still forget. Freeze controls make the stability
  choice explicit.
* The Avalanche backend supports multiclass mode only and lacks the pre/post-expansion probes; iCaRL is not
  wired (needs a feature-extractor/classifier split). The Skill OvR experiment uses a frozen backbone and raw
  independent logits, evaluated by per-skill binary metrics and which-skills-fire reports (no `argmax`); no adaptive `new_dim`, no kernel expansion, no data augmentation in the built-in loaders.
* The probe subset is drawn from the test set (inputs only) purely for diagnostics; it never influences training.
* Replay in `training/replay.py` is intentionally simplistic; use Avalanche for real baselines.
* Target-vs-rest streams hold every experience's training data in memory. The cumulative (`None`) stream for
  CIFAR-100 with 20 experiences is ~1.6 GB as uint8 (5 exps ~0.5 GB); an explicit ratio such as 1:5 needs
  ~0.2 GB. Experience subsampling uses `default_rng(seed + experience_index)`: deterministic, and kept this
  way so earlier 1:5 results reproduce, but `(seed, i)` pairs with equal sums share a stream.
* The requested ratio is an upper bound (see above): negatives are never duplicated, so very negative-heavy ratios
  are not reached in early experiences when few negative classes have been seen. Judge a ratio by its
  `final_*` metrics and `fraction_experiences_at_requested_ratio`, not by its label alone.
* Resume compares the saved training config, not the software environment (torch version, git commit are recorded
  in `results.json` but do not invalidate a run).
* The ratio sweep reports mean/std across *targets*; it runs one seed. Repeat with other `--seed` values (and
  distinct `--name`s) before reading small differences between ratios as real.
