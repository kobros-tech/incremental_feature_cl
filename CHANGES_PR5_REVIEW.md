# Changes made in response to the PR #5 reviews

Base: PR #5 head `c79a5f6` plus the local edits of the uploaded copy (`avalanche/strategy.py`
docstrings, `experiments/common.py` help text).

## Review 1

| Finding | Resolution |
| --- | --- |
| 1. Existing functionality deleted | Restored, adapted to the PR's external-backbone design (the built-in `mlp` / `smallconv` / `slimresnet18` stay removed): `FeatureExpansionPlugin`, `IncrementalFeatureSpaceStrategy`, Avalanche baselines / benchmark builders / logger, `--backend avalanche` and `avalanche_baseline` in `run_experiment`, `compare_ratios` (+ `ratio_summary`, `plotting/ratio_sweep`), `compare_baselines`, `run_sweep`, `math_demo`, `synthetic_demo`, `train_cifar100`, and their configs / tests. Configs now name `resnet32` instead of `slimresnet18`. |
| 2. README commands do not match | README commands rewritten; `tests/test_readme_commands.py` parses every documented command with the real argparse definitions. `--train-epochs` vs `--skill-max-iter` is explained. |
| 3. Scores are not comparable across skills | Reports are per-skill binary metrics plus a firing-based decision report. No `argmax` is used in any report. |
| 4. Avalanche accuracy evaluator | `OneVsRestSkillStrategy` no longer includes Avalanche's argmax accuracy by default (`skill_evaluator`); `evaluate_skills()` returns per-skill metrics and the decision report. |
| 5. Diagnostic computed too late | Not a bug: `SkillBase.update()` never writes memory, so `negative_memory` is still the pre-experience state. Comment clarified; `tests/test_skill_decisions.py::test_newly_known_negatives_is_computed_before_memory_is_committed` pins the behaviour. |

## Review 2

| Item | Resolution |
| --- | --- |
| Validation-selected thresholds | `evaluation/skill_decisions.py`: `select_thresholds` (F1-maximising, per skill, validation data only). `train_ovr --val-fraction` (default 0.1; 0 = off). Metrics are reported at threshold 0 and at the validated thresholds; thresholds are saved. |
| Multiclass evaluation **without argmax** | `decision_report`: none / single / multi-fired rates, `exact_match_rate`, `precision_when_single`, per-class recall / precision / FPR, firing matrix `[true class, skill]` (CSV). |
| Per-skill forgetting | `average_skill_auc_forgetting`, `average_skill_tuned_recall_forgetting`, running per-skill forgetting CSVs. |
| Feature-map comparison `n/a` | Documented and printed (intentional: only classes in `--vs-feature-map` are run). |
| Identity vs warmed-up backbone | Command documented in the README and verified end to end on the bundled digits dataset. CIFAR-100 itself was not run (no dataset download in the build environment). |

## Behaviour changes to be aware of

* `train_ovr` holds out 10 % of each class's training data by default (`--val-fraction 0` restores
  the previous training data exactly).
* The uploaded `common.py` had made Skills follow `cfg.train.optimizer` (SGD by default), which made
  `--skill-max-iter` a no-op. The L-BFGS default (weight decay 1e-3) was restored; the docstring /
  help-text edits were kept.
* `ExperimentConfig.backend` (`torch` | `avalanche`) is back.
