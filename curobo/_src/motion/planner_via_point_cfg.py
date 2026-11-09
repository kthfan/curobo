# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and result of the via-point motion planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch

from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


@dataclass
class ViaPointMotionPlannerCfg:
    """Configuration for :class:`~curobo._src.motion.planner_via_point.ViaPointMotionPlanner`."""

    #: Motion planner whose IK and fixed-spline trajectory optimizer the planner uses.
    motion_planner_cfg: MotionPlannerCfg

    #: Whole-plan retries; each one solves the waypoint IK again with new seeds.
    max_attempts: int = 3

    #: IK solutions requested per waypoint, before de-duplication.
    ik_return_seeds: int = 16

    #: Joint-space distance (weighted L-infinity, rad or m) under which two IK solutions of a
    #: waypoint count as one.
    ik_dedup_joint_tol: float = 0.05

    #: Default tolerance (m, rad) for reaching every waypoint whose criteria set no
    #: ``terminal_tolerance``.
    waypoint_tolerance: Tuple[float, float] = (0.005, 0.05)

    #: Fewest native reference intervals given to one segment. Segments get more in proportion
    #: to the joint-space distance they cover.
    min_segment_intervals: int = 2

    def __post_init__(self):
        if self.max_attempts < 1 or self.ik_return_seeds < 1 or self.min_segment_intervals < 1:
            log_and_raise("max_attempts, ik_return_seeds and min_segment_intervals must be >= 1")
        if self.ik_dedup_joint_tol < 0.0:
            log_and_raise("ik_dedup_joint_tol must be >= 0")

    @classmethod
    def create(
        cls,
        robot: Any,
        *,
        max_attempts: int = 3,
        ik_return_seeds: int = 16,
        ik_dedup_joint_tol: float = 0.05,
        waypoint_tolerance: Tuple[float, float] = (0.005, 0.05),
        min_segment_intervals: int = 2,
        **motion_planner_kwargs: Any,
    ) -> ViaPointMotionPlannerCfg:
        """Build a configuration from a robot and via-point planning parameters.

        Args:
            robot: Robot configuration: YAML path, dict, or ``RobotCfg``.
            max_attempts: Whole-plan retries.
            ik_return_seeds: IK solutions requested per waypoint.
            ik_dedup_joint_tol: Joint-space de-duplication tolerance of IK solutions.
            waypoint_tolerance: Default reach tolerance (m, rad).
            min_segment_intervals: Fewest native reference intervals per segment.
            **motion_planner_kwargs: Forwarded to :meth:`MotionPlannerCfg.create`, for example
                ``scene_model``, ``num_knots`` or ``num_trajopt_seeds``.

        Returns:
            The configuration.
        """
        return cls(
            motion_planner_cfg=MotionPlannerCfg.create(robot, **motion_planner_kwargs),
            max_attempts=max_attempts,
            ik_return_seeds=ik_return_seeds,
            ik_dedup_joint_tol=ik_dedup_joint_tol,
            waypoint_tolerance=waypoint_tolerance,
            min_segment_intervals=min_segment_intervals,
        )


@dataclass
class ViaPointPlanResult:
    """Result of :meth:`ViaPointMotionPlanner.plan_waypoints`."""

    #: Whether a trajectory passes every waypoint and satisfies every segment, shape ``[1]``.
    success: torch.Tensor

    #: Why planning failed, or ``"Success."``.
    status: str

    #: Trajectory optimization result of the best attempt; ``None`` when no attempt reached
    #: trajectory optimization.
    trajopt_result: Optional[TrajOptSolverResult] = None

    #: Native sample index of every waypoint on the reference grid.
    waypoint_indices: List[int] = field(default_factory=list)

    #: Joint configurations of the returned trajectory at the waypoints ``[num_waypoints, dof]``.
    waypoint_configs: Optional[torch.Tensor] = None

    #: Position error (m) at every waypoint ``[num_waypoints, links]``.
    waypoint_position_error: Optional[torch.Tensor] = None

    #: Rotation error (rad) at every waypoint ``[num_waypoints, links]``.
    waypoint_rotation_error: Optional[torch.Tensor] = None

    #: Whole-plan attempts consumed.
    attempts: int = 0

    #: Wall-clock planning time (s), including the waypoint IK.
    planning_time: float = 0.0

    def get_interpolated_plan(self) -> JointState:
        """Return the executable trajectory, trimmed to its valid samples."""
        if self.trajopt_result is None:
            log_and_raise(f"no trajectory was planned: {self.status}")
        return self.trajopt_result.get_interpolated_plan()
