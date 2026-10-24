# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result of a multi-waypoint motion planning operation.

Unlike a single-problem solve (which returns one ``TrajOptSolverResult``), the
multi-waypoint planner produces a multi-segment result: the chosen IK node per
waypoint, the raw per-segment solver outputs, and the stitched trajectory ready
for execution.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState


@dataclass
class MultiWaypointPlanResult:
    """Result of a multi-waypoint motion planning operation."""

    #: Overall success flag for the whole multi-waypoint plan.
    success: Optional[torch.Tensor] = None

    #: Human-readable status / failure reason (e.g. "No IK at waypoint 2",
    #: "No surviving edges for segment 1").
    status: Optional[str] = None

    #: Chosen IK node index per waypoint (one entry per waypoint), indexing
    #: into each waypoint's de-duplicated IK solution set.
    chosen_ik_indices: Optional[List[int]] = None

    #: Raw per-segment trajectory-optimization results for the chosen path.
    #: Length equals ``num_waypoints - 1`` on success.
    segment_results: Optional[List[TrajOptSolverResult]] = None

    #: Stitched trajectory across all segments, ready for execution.
    composed_trajectory: Optional[JointState] = None

    #: Number of whole-plan attempts consumed.
    attempts: int = 0

    #: Wall-clock planning time in seconds.
    planning_time: float = 0.0
