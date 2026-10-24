# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""De-duplication of IK solutions in joint space.

The multi-waypoint planner generates many IK solutions per waypoint.  Many of
these are near-identical joint configurations that would explode the number of
edge pairs evaluated by trajectory optimization without adding any meaningful
diversity.  :func:`dedup_joint_solutions` greedily keeps the first solution and
prunes any later solution that is within ``tol`` of an already-kept one.
"""

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.util.logging import log_and_raise


def dedup_joint_solutions(
    q: torch.Tensor,
    tol: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """De-duplicate joint configurations by a joint-space tolerance.

    Greedy keep-first: iterate over rows of ``q`` in order, keeping a solution
    only if it is at least ``tol`` away (weighted L-infinity distance) from every
    already-kept solution.

    Args:
        q: Joint configurations of shape ``[N, dof]``.
        tol: Distance threshold.  Solutions closer than ``tol`` to a kept
            solution are treated as duplicates and dropped.
        weights: Optional per-joint weights of shape ``[dof]`` applied before
            taking the L-infinity norm.  ``None`` weights all joints equally.

    Returns:
        Unique joint configurations of shape ``[M, dof]`` with ``M <= N``,
        preserving the original ordering of kept solutions.
    """
    if q.ndim != 2:
        log_and_raise(f"dedup_joint_solutions expects a 2D [N, dof] tensor, got {q.shape}")

    n = q.shape[0]
    if n <= 1:
        return q.clone()

    kept = q[0:1].clone()
    for i in range(1, n):
        diff = torch.abs(q[i : i + 1] - kept)
        if weights is not None:
            diff = diff * weights.view(1, -1)
        # L-infinity distance to each kept solution.
        dist = diff.max(dim=1).values
        if bool((dist >= tol).all()):
            kept = torch.cat([kept, q[i : i + 1]], dim=0)
    return kept
