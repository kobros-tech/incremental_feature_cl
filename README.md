# incremental-feature-cl

A scikit-learn **binary** classifier: a linear model over a feature space that can grow, where every new
feature block starts with **zero** weights, so predictions are unchanged at the moment of expansion.
Plain NumPy training (minibatch SGD with momentum), no PyTorch.

```text
Phi_{t+1}(x) = [Phi_t(x), psi_{t+1}(x)]        theta_{t+1} = [theta_t, 0]
```

Earlier samples are **replayed** from a small built-in memory at every `partial_fit` (on by default,
`replay_mem_size=0` turns it off). Multiclass problems are handled by scikit-learn's `OneVsRestClassifier`, not by this package.

## Install

```bash
pip install -e ".[dev]"   # numpy, scikit-learn (+ pytest, ruff)
pytest
```

## Usage

```python
from sklearn.multiclass import OneVsRestClassifier
from incremental_feature_cl import IncrementalFeatureMapModel

clf = IncrementalFeatureMapModel(new_feature_dim=16, target_to_negatives=0.2, random_state=0)

clf.fit(X, y)  # binary: one stage, no expansion
clf.partial_fit(X_t, y_t, classes=[0, 1])  # one experience: expand, train with replay, remember
clf.expand_features()  # append a block by hand (decision_function is unchanged)

IncrementalFeatureMapModel(replay_mem_size=0)  # replay off

ovr = OneVsRestClassifier(clf).fit(X, y)  # multiclass
ovr.partial_fit(X_t, y_t, classes=all_classes)  # incremental multiclass
```

Scale the features (e.g. `StandardScaler`); pass an external backbone's features in as `X`.

## Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `new_feature_dim` | `0` | Width of the block appended before each `partial_fit` after the first. `0` = fixed feature space. |
| `target_to_negatives` | `None` | Targets per negative in the call's data: `1.0` = 1:1, `0.2` = 1 target : 5 negatives. Keeps `floor(n_target / ratio)` negatives (without replacement; all if fewer exist). `None` keeps all. Applied **before** replay. |
| `pos_weight` | `"balanced"` | Loss weight of a target sample: `"balanced"` = `n_negative / n_target` of the call's data (after the ratio), a float, or `None`. Replayed targets get the same weight. |
| `replay_mem_size` | `200` | Total replay memory, split equally between target and non-target; a reservoir sample per class of everything trained on. Each minibatch of `batch_size` current samples gets `min(batch size, memory)` memory samples without replacement. `0` = replay off, otherwise at least 2. |
| `train_blocks` | `True` | Train the appended feature blocks. `False` = fixed random features. |
| `learning_rate_init`, `momentum` | `0.01`, `0.9` | SGD step size and momentum (the momentum buffers restart in every call). |
| `batch_size` | `32` | Minibatch size of the current data. |
| `alpha` | `0.0` | L2 penalty (weight decay). |
| `n_epochs` | `5` | Passes per `fit` / `partial_fit` call. |
| `random_state` | `None` | Seeds feature blocks, negative subsampling, replay and shuffling. |

The positive class is `classes_[1]`. New weights are always zero-initialised; there is no option for random
initialisation.

## Relation to the PyTorch version

Run on identical synthetic streams (10 seeds), the final balanced accuracy of this package differs from the
PyTorch package by +0.002 ± 0.003 (replay 200, the original's cumulative stream) and -0.013 ± 0.011 (a stream with
only new negatives per experience). The mechanisms line up as follows.

| Mechanism | PyTorch package | This package |
| --- | --- | --- |
| Zero init | `W_new = 0`, `b = 0` | same (`decision_function` is unchanged by `expand_features`) |
| Feature block | `relu(Linear(x))`, `nn.Linear` default init, trained | same (`train_blocks=False` freezes it) |
| Expansion schedule | every experience after the first | every `partial_fit` after the first |
| Ratio | `floor(n_target / ratio)` negatives, before replay | same |
| Loss | BCE with `pos_weight` from the experience's data | same name and meaning |
| Optimiser | SGD + momentum, minibatch 32, restarted per experience | same |
| Replay rate | per minibatch of 32: `min(32, memory)` samples without replacement | same |
| Replay off | `replay_mem_size=0` | same |
| Replay memory | class-balanced over the **class ids**, per-class replace | class-balanced over the **two binary classes**, reservoir |

The last row is the one difference that cannot be removed: scikit-learn passes a binary `y`, so class ids are not
known. With K classes the original keeps 1/K targets in memory; this package keeps 1/2. Balanced accuracy is
similar, but this package trades target recall for old-negative recognition differently.
