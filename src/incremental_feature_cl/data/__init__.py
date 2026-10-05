# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Dataset and stream utilities used by the research package."""
from .datasets import ArrayDataset, load_dataset, make_synthetic_dataset
from .streams import Experience, Stream, binary_labels, build_class_incremental_stream, build_target_vs_rest_stream
__all__ = ["ArrayDataset","Experience","Stream","binary_labels","build_class_incremental_stream","build_target_vs_rest_stream","load_dataset","make_synthetic_dataset"]
