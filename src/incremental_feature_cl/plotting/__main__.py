# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""python -m incremental_feature_cl.plotting --results results/run/results.json"""

import argparse
from pathlib import Path

from ..evaluation.result_store import RunResult
from . import make_all_plots

p = argparse.ArgumentParser(description="Regenerate plots from a saved result (no retraining).")
p.add_argument("--results", required=True, help="results.json or its directory")
a = p.parse_args()
src = Path(a.results)
for f in make_all_plots(RunResult.load(src), src if src.is_dir() else src.parent):
    print(f)
