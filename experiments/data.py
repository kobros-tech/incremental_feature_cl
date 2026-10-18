# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Small, deterministic datasets and class-incremental stream builders for experiments."""

from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
_CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


DATASETS = ("digits", "synthetic", "cifar100")


def _cifar_files(root: Path) -> tuple[Path, Path] | None:
    """The ``train``/``test`` files inside ``root`` or ``root/cifar-100-python``, if present."""
    for candidate in (root, root / "cifar-100-python"):
        if (candidate / "train").is_file() and (candidate / "test").is_file():
            return candidate / "train", candidate / "test"
    return None


def resolve_dataset(dataset: str | Path) -> tuple[str, Path | None]:
    """Map ``--dataset`` to ``(name, data_dir)``.

    ``dataset`` is a name from ``DATASETS`` or the path of an extracted CIFAR-100 directory
    (``train``/``test`` inside it or inside a ``cifar-100-python`` subdirectory); a path means
    ``("cifar100", that directory)``.
    """
    if str(dataset) in DATASETS:
        return str(dataset), None
    root = Path(dataset).expanduser()
    if root.is_dir():
        if _cifar_files(root) is None:
            raise FileNotFoundError(
                f"No CIFAR-100 train/test files found under {root}; expected train/test "
                "inside that directory or its cifar-100-python subdirectory."
            )
        return "cifar100", root
    raise ValueError(
        f"Unsupported dataset {str(dataset)!r}; use one of {', '.join(DATASETS)} or the path "
        "of an extracted CIFAR-100 directory."
    )


def _read_cifar_batch(file, source) -> tuple[np.ndarray, np.ndarray]:
    batch = pickle.load(file, encoding="bytes")
    if b"data" not in batch or b"fine_labels" not in batch:
        raise ValueError(f"{source} is missing CIFAR-100 data or fine_labels.")
    X = np.asarray(batch[b"data"], dtype=np.float32)
    y = np.asarray(batch[b"fine_labels"], dtype=int)
    if X.ndim != 2 or X.shape[1] != 3072 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{source} has an unexpected CIFAR-100 data/label shape.")
    return X, y


def _cifar100_archive() -> Path:
    """The checksum-verified official archive, downloaded to ``data/`` on first use."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "cifar-100-python.tar.gz"
    if (
        archive_path.exists()
        and hashlib.md5(archive_path.read_bytes()).hexdigest() != _CIFAR100_MD5
    ):
        archive_path.unlink()
    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".download")
        urllib.request.urlretrieve(_CIFAR100_URL, temporary_path)
        if hashlib.md5(temporary_path.read_bytes()).hexdigest() != _CIFAR100_MD5:
            temporary_path.unlink(missing_ok=True)
            raise ValueError("Downloaded CIFAR-100 archive failed its MD5 checksum.")
        temporary_path.replace(archive_path)
    return archive_path


def _load_cifar100(seed: int, data_dir: Path | None = None):
    """A reproducible CIFAR-100 subset (100 train / 20 test images per class), pixels in [0, 1]."""
    if data_dir is not None:
        train_path, test_path = _cifar_files(data_dir)
        with train_path.open("rb") as file:
            X_train, y_train = _read_cifar_batch(file, train_path)
        with test_path.open("rb") as file:
            X_test, y_test = _read_cifar_batch(file, test_path)
    else:
        with tarfile.open(_cifar100_archive(), "r:gz") as archive:
            members = [
                archive.extractfile(f"cifar-100-python/{name}") for name in ("train", "test")
            ]
            if any(member is None for member in members):
                raise ValueError("CIFAR-100 archive is missing its train or test data.")
            (X_train, y_train), (X_test, y_test) = (
                _read_cifar_batch(member, name) for member, name in zip(members, ("train", "test"))
            )

    rng = np.random.RandomState(seed)

    def sample_per_class(labels, per_class):
        selected = []
        for class_id in np.unique(labels):
            candidates = np.flatnonzero(labels == class_id)
            selected.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
        return np.asarray(selected, dtype=int)

    train_indices = sample_per_class(y_train, 100)
    test_indices = sample_per_class(y_test, 20)
    return (
        X_train[train_indices] / 255.0,
        X_test[test_indices] / 255.0,
        y_train[train_indices],
        y_test[test_indices],
    )


def load_dataset(dataset: str | Path, seed: int = 0):
    """Return train/test arrays for a name from ``DATASETS`` or an extracted CIFAR-100 directory."""
    name, data_dir = resolve_dataset(dataset)
    if name == "cifar100":
        return _load_cifar100(seed, data_dir)
    if name == "digits":
        X, y = load_digits(return_X_y=True)
    else:
        X, y = make_classification(
            n_samples=2400,
            n_features=32,
            n_informative=20,
            n_redundant=4,
            n_classes=6,
            n_clusters_per_class=1,
            class_sep=1.2,
            random_state=seed,
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, np.asarray(y_train), np.asarray(y_test)


def class_incremental_indices(y: np.ndarray, n_experiences: int) -> list[np.ndarray]:
    """Partition sample indices by ordered class groups, so each class arrives once."""
    classes = np.unique(y)
    if n_experiences < 1 or n_experiences > len(classes):
        raise ValueError(f"n_experiences must be between 1 and {len(classes)} for this dataset.")
    class_groups = np.array_split(classes, n_experiences)
    streams = [np.flatnonzero(np.isin(y, group)) for group in class_groups]
    if any(len(indices) == 0 for indices in streams):
        raise ValueError("Every experience must contain at least one sample.")
    return streams
