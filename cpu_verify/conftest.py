# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Force the CPU-only verification harness onto CPU before torch/warp load.

NOT for main: see ``cpu_verify/README.md``.
"""

import os
import sys

# Hide any GPU so the checks behave the same on a workstation and on a CPU-only runner.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _path in (_REPO_ROOT, os.path.join(_HERE, "eef_bound")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
