# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate a crash or hang in the tool pose bound tests, one step per subprocess.

NOT for main: see ``cpu_verify/README.md``.

Each step runs in a fresh ``python -X faulthandler`` subprocess with a timeout, so a segfault,
abort, or deadlock in one step is reported (signal / TIMEOUT plus the faulthandler stack) instead
of killing the run. Run from the repository root and send the full output::

    python cpu_verify/eef_bound/diagnose_crash.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

STEPS = [
    (
        "versions",
        """
        import platform, torch, warp, numpy
        print("python", platform.python_version(), "torch", torch.__version__,
              "cuda", torch.version.cuda, "warp", warp.__version__, "numpy", numpy.__version__)
        print("torch threads", torch.get_num_threads(), "cuda available", torch.cuda.is_available())
        print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
        """,
    ),
    (
        "torch CPU ops used by tool_pose_bound",
        """
        import torch
        a = torch.tensor([[0.1, 0.2, 0.3]]); b = torch.tensor([[0.3, -0.2, 0.5]])
        print(torch.cross(a, b, dim=-1), torch.atan2(a, b), torch.where(a > 0.15, a, b))
        """,
    ),
    (
        "import curobo tool_pose_bound",
        """
        import curobo._src.cost.tool_pose_bound as m
        print("ok", m.__file__)
        """,
    ),
    (
        "swing_twist_components (TestSwingTwist)",
        """
        import torch
        from curobo._src.cost.tool_pose_bound import swing_twist_components
        print(swing_twist_components(torch.tensor([0.9, 0.1, 0.2, 0.3])))
        """,
    ),
    (
        "tool_pose_bound_violation (TestViolation)",
        """
        import torch
        from curobo._src.cost.tool_pose_bound import ToolPoseBound, tool_pose_bound_violation
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        print(tool_pose_bound_violation(ToolPoseBound.plane("z", anchor="base"),
              torch.tensor([[0.3, -0.2, 0.03]]), q, torch.zeros(1, 3), q))
        """,
    ),
    (
        "import test modules (collection)",
        """
        import importlib
        for name in ["curobo.tests._src.cost.test_tool_pose_bound",
                     "curobo.tests._src.cost.test_cost_eef_bound"]:
            importlib.import_module(name); print("imported", name)
        """,
    ),
    (
        "warp init and devices",
        """
        from curobo._src.util.warp import init_warp
        import warp as wp
        init_warp(quiet=False)
        print([str(d) for d in wp.get_devices()], "kernel cache", wp.config.kernel_cache_dir)
        """,
    ),
    (
        "warp trivial CPU kernel",
        """
        import warp as wp
        from curobo._src.util.warp import init_warp
        init_warp()
        @wp.kernel
        def add_one(x: wp.array(dtype=float)):
            x[wp.tid()] += 1.0
        a = wp.zeros(4, dtype=float, device="cpu")
        wp.launch(add_one, dim=4, inputs=[a], device="cpu"); wp.synchronize()
        print(a.numpy())
        """,
    ),
    (
        "EEFBoundCost on cpu",
        """
        import torch
        from curobo._src.util.warp import init_warp
        from curobo._src.cost.cost_eef_bound import EEFBoundCost
        from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
        from curobo._src.cost.tool_pose_bound import ToolPoseBound
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.types.tool_pose import ToolPose
        init_warp()
        cfg = EEFBoundCostCfg(weight=1.0, tool_frames=["ee"],
                              device_cfg=DeviceCfg(device=torch.device("cpu")))
        cost = EEFBoundCost(cfg)
        cost.update_tool_pose_bounds({"ee": ToolPoseBound.plane("z", anchor="base")})
        q = torch.zeros(1, 2, 1, 4); q[..., 0] = 1.0
        print(cost.forward(ToolPose(["ee"], torch.full((1, 2, 1, 3), 0.1), q)).view(-1))
        """,
    ),
    (
        "EEFBoundCost on cuda:0",
        """
        import torch
        if not torch.cuda.is_available():
            print("skipped: no CUDA"); raise SystemExit(0)
        from curobo._src.util.warp import init_warp
        from curobo._src.cost.cost_eef_bound import EEFBoundCost
        from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
        from curobo._src.cost.tool_pose_bound import ToolPoseBound
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.types.tool_pose import ToolPose
        init_warp()
        dev = torch.device("cuda:0")
        cfg = EEFBoundCostCfg(weight=1.0, tool_frames=["ee"], device_cfg=DeviceCfg(device=dev))
        cost = EEFBoundCost(cfg)
        cost.update_tool_pose_bounds({"ee": ToolPoseBound.plane("z", anchor="base")})
        q = torch.zeros(1, 2, 1, 4, device=dev); q[..., 0] = 1.0
        out = cost.forward(ToolPose(["ee"], torch.full((1, 2, 1, 3), 0.1, device=dev), q))
        torch.cuda.synchronize(); print(out.view(-1))
        """,
    ),
]

PRELUDE = "import os, sys\nsys.path.insert(0, os.getcwd())\n"


def main() -> int:
    timeout = float(os.environ.get("DIAG_TIMEOUT", "180"))
    failed = 0
    for name, code in STEPS:
        source = PRELUDE + textwrap.dedent(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-c", source],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                status = "OK"
            elif proc.returncode < 0:
                status = f"CRASH signal {signal.Signals(-proc.returncode).name}"
            else:
                status = f"ERROR exit {proc.returncode}"
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            status = f"TIMEOUT after {timeout:.0f}s (hang)"
            output = (e.stdout or "") + (e.stderr or "")
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
        failed += status != "OK"
        print(f"=== {status:30s} {name}")
        tail = "\n".join(output.strip().splitlines()[-25:])
        if tail:
            print(textwrap.indent(tail, "    "))
    print(f"\n{failed} step(s) not OK")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
