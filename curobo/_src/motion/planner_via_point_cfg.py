# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and result of the via-point motion planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch

from curobo._src.motion.ik_chain import ChainMetric, ChainMetricLike
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

    #: IK solutions requested per waypoint, before de-duplication; at most the IK solver's
    #: seeds, of which only the successful ones are kept.
    ik_return_seeds: int = 64

    #: Joint-space distance (weighted L-infinity, rad or m) under which two IK solutions of a
    #: waypoint count as one.
    ik_dedup_joint_tol: float = 0.05

    #: Default tolerance (m, rad) for reaching every waypoint whose criteria set no
    #: ``terminal_tolerance``.
    waypoint_tolerance: Tuple[float, float] = (0.005, 0.05)

    #: Fewest native reference intervals given to one segment (4 per spline control by
    #: default). Segments get more in proportion to the joint-space distance they cover; a
    #: segment that is too short cannot slow down after one waypoint to pass the next.
    min_segment_intervals: int = 8

    #: Scale of the seed's time step for the geometric optimization. TimeOpt then sets the
    #: fastest timing along the optimized path, so a slow geometric pass can pass sharp corners
    #: at the waypoints precisely without making the motion slower. The scaled time step is
    #: limited by the optimizer's ``maximum_trajectory_dt``.
    geometry_dt_scale: float = 10.0

    #: Add the full-turn variants of every IK solution (``q ± 2π`` on revolute joints whose
    #: limits span more than one turn) to the waypoint's IK solutions, so the IK chain can
    #: reach a waypoint "the other way round" when that makes the whole path shorter.
    expand_revolutions: bool = True

    #: Seed the IK of every waypoint with the current state and the IK solutions of the
    #: previous waypoint, besides random seeds, so the IK branches found at a waypoint have
    #: their nearest continuation at the next one.
    seed_ik_from_previous: bool = True

    #: How the length of an IK chain is measured when the planner looks for the shortest one:
    #: Euclidean joint-space distance, total joint rotation, time at the joint velocity limits,
    #: or the largest motion of one joint. See :class:`~curobo._src.motion.ik_chain.ChainMetric`.
    #: Segments also get native reference samples in proportion to their length in this metric.
    chain_metric: ChainMetricLike = ChainMetric.EUCLIDEAN

    #: The first attempt seeds the optimizer only with IK chains at most this fraction longer
    #: (in joint space) than the shortest chain, and returns the fastest trajectory among them.
    #: ``None`` seeds the shortest chain to every IK solution of the last waypoint instead, as
    #: retries do: the fastest trajectory then may take a much longer way in joint space.
    chain_length_slack: Optional[float] = 0.1

    #: Largest joint-space offset (rad or m) at the middle of every segment for the seeds that
    #: repeat an IK chain. The first attempt seeds at most half as many chains as seeds, so every
    #: chain has a straight seed and at least one bent one: the optimizer can cut the corner at
    #: a waypoint from a straight seed and pass it from a bent one. Zero repeats the straight
    #: seed.
    seed_bulge: float = 0.4

    #: Fraction of every joint's range that IK solutions keep from the joint limits, when a
    #: waypoint has other solutions. It matches the range in which the trajectory optimizer's
    #: joint limit cost is active: the optimizer does not reach a solution closer to a limit.
    joint_limit_margin: float = 0.01

    def __post_init__(self):
        if self.max_attempts < 1 or self.ik_return_seeds < 1 or self.min_segment_intervals < 1:
            log_and_raise("max_attempts, ik_return_seeds and min_segment_intervals must be >= 1")
        if self.geometry_dt_scale <= 0.0:
            log_and_raise("geometry_dt_scale must be positive")
        if self.ik_dedup_joint_tol < 0.0:
            log_and_raise("ik_dedup_joint_tol must be >= 0")
        self.chain_metric = ChainMetric(self.chain_metric)
        if self.chain_length_slack is not None and self.chain_length_slack < 0.0:
            log_and_raise("chain_length_slack must be >= 0 or None")
        if self.seed_bulge < 0.0:
            log_and_raise("seed_bulge must be >= 0")
        if not 0.0 <= self.joint_limit_margin < 0.5:
            log_and_raise("joint_limit_margin must be in [0, 0.5)")

    @classmethod
    def create(
        cls,
        robot: Any,
        *,
        max_attempts: int = 3,
        ik_return_seeds: int = 64,
        ik_dedup_joint_tol: float = 0.05,
        waypoint_tolerance: Tuple[float, float] = (0.005, 0.05),
        min_segment_intervals: int = 8,
        geometry_dt_scale: float = 10.0,
        expand_revolutions: bool = True,
        seed_ik_from_previous: bool = True,
        chain_metric: ChainMetricLike = ChainMetric.EUCLIDEAN,
        chain_length_slack: Optional[float] = 0.1,
        seed_bulge: float = 0.4,
        joint_limit_margin: float = 0.01,
        num_knots: int = 64,
        num_trajopt_seeds: int = 8,
        num_ik_seeds: int = 64,
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
            geometry_dt_scale: Time step scale of the geometric optimization.
            expand_revolutions: Add the full-turn variants of the IK solutions.
            seed_ik_from_previous: Seed every waypoint's IK with the previous waypoint's
                solutions.
            chain_metric: How IK chain length is measured: ``"euclidean"``,
                ``"total_rotation"``, ``"time"`` or ``"max_norm"``.
            chain_length_slack: Longest IK chain seeded on the first attempt, as a fraction
                above the shortest chain; ``None`` for no limit.
            seed_bulge: Largest mid-segment offset of the seeds that repeat a chain.
            joint_limit_margin: Fraction of every joint's range kept from the joint limits.
            num_knots: Spline controls of the trajectory optimizer. More controls follow sharp
                corners at the waypoints more closely and hold more waypoints; fewer give
                somewhat shorter motions and miss more corners.
            num_trajopt_seeds: Trajectory optimization seeds, one per IK chain.
            num_ik_seeds: IK seeds per waypoint. More seeds find more of a waypoint's IK
                solutions, so the shortest chain through them is closer to the shortest
                possible one.
            **motion_planner_kwargs: Forwarded to :meth:`MotionPlannerCfg.create`, for example
                ``scene_model``.

        Returns:
            The configuration.
        """
        return cls(
            motion_planner_cfg=MotionPlannerCfg.create(
                robot,
                num_knots=num_knots,
                num_trajopt_seeds=num_trajopt_seeds,
                num_ik_seeds=num_ik_seeds,
                **motion_planner_kwargs,
            ),
            max_attempts=max_attempts,
            ik_return_seeds=ik_return_seeds,
            ik_dedup_joint_tol=ik_dedup_joint_tol,
            waypoint_tolerance=waypoint_tolerance,
            min_segment_intervals=min_segment_intervals,
            geometry_dt_scale=geometry_dt_scale,
            expand_revolutions=expand_revolutions,
            seed_ik_from_previous=seed_ik_from_previous,
            chain_metric=chain_metric,
            chain_length_slack=chain_length_slack,
            seed_bulge=seed_bulge,
            joint_limit_margin=joint_limit_margin,
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
