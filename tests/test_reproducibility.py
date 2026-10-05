# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from incremental_feature_cl.utils.reproducibility import collect_environment


def test_collect_environment_records_core_versions_without_optional_dependencies():
    env = collect_environment()
    assert env["package_version"]
    assert env["python"]
    assert env["torch"]
    assert env["numpy"]
    assert "torchvision" in env
    assert "avalanche" in env
