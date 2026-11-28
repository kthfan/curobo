# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Check the update steps of ``test_robot_mesh_constraint_cuda_graph.py`` on the CPU.

Branch-local verification tool. Do not merge into ``dev/main``.

The CUDA graph test can only reveal a stale graph at steps that change the check's result.
This runs the same steps eagerly on the CPU and reports the steps that change nothing.

Usage (from the repository root)::

    PYTHONPATH=. python cpu_verify/verify_cuda_graph_steps.py
"""

# Patch DeviceCfg to the CPU before any other CuRobo import.
import cpu_verify.cpu_device_plugin  # noqa: F401, I001

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo.tests._src.collision.test_robot_mesh_constraint_cuda_graph import run_update_steps


def main() -> int:
    """Run the steps for both margins; return 1 if a step changes nothing."""
    init_warp(device_cfg=DeviceCfg())
    failed = False
    for margin in (0.0, 0.02):
        unchanged = run_update_steps(use_cuda_graph=False, margin=margin)
        print(f"margin={margin}: steps that change nothing: {unchanged or 'none'}")
        failed |= bool(unchanged)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
