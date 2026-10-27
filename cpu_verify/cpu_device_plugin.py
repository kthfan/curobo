# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
r"""Pytest plugin that makes ``DeviceCfg()`` default to the CPU.

Branch-local verification tool for machines without a GPU. Do not merge into ``dev/main``.

Tests that build ``DeviceCfg()`` at import time then run their torch tensors and Warp
kernels on the CPU. Only tests that do not need CUDA-specific features (CUDA graphs, custom
CUDA extensions) can pass this way.

Usage (from the repository root)::

    PYTHONPATH=. pytest -p cpu_verify.cpu_device_plugin \
        curobo/tests/_src/collision/test_robot_mesh_constraint.py
"""

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg

_CPU = torch.device("cpu")


def _patch_device_default() -> None:
    """Replace the CUDA default of the generated ``DeviceCfg.__init__`` with the CPU."""
    defaults = DeviceCfg.__init__.__defaults__
    DeviceCfg.__init__.__defaults__ = tuple(
        _CPU if isinstance(d, torch.device) else d for d in defaults
    )


# Patch at import, which ``-p`` does before any conftest or test module is imported.
_patch_device_default()
assert DeviceCfg().device == _CPU
