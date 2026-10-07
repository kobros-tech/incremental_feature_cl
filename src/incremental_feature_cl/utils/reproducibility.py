# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Seeding, device resolution and provenance helpers."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(device: str = "auto") -> torch.device:
    """'auto' -> cuda if available else cpu; otherwise passed to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_git_commit(path: str | Path | None = None) -> str | None:
    """Return the current git commit hash of `path` (or cwd), if available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path) if path else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict[str, Any]:
    """Provenance record stored next to every result."""
    from .. import __version__

    versions = {
        "package_version": __version__,
        "git_commit": get_git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    for package, key in (("torchvision", "torchvision"), ("avalanche", "avalanche")):
        try:
            module = __import__(package)
            versions[key] = getattr(module, "__version__", "unknown")
        except (ImportError, OSError, RuntimeError):
            versions[key] = None
    return versions
