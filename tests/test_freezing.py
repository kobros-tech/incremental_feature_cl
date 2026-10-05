# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone

def test_freeze_backbone_disables_backbone_gradients():
    model = IncrementalFeatureMapModel(build_backbone("smallconv"), num_outputs=2, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.backbone.parameters())

def test_freeze_old_blocks_only_freezes_previous_blocks():
    model = IncrementalFeatureMapModel(
        build_backbone("smallconv"), num_outputs=2, freeze_old_blocks=True
    )
    model.expand_feature_space(4)
    model.expand_feature_space(4)
    assert all(not p.requires_grad for p in model.feature_blocks[0].parameters())
    assert all(p.requires_grad for p in model.feature_blocks[1].parameters())
