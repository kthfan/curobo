# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the multi-waypoint motion planner.

The multi-waypoint planner builds a layered DAG of de-duplicated IK solutions
per waypoint, runs trajectory optimization on the edges between consecutive
layers, prunes failed edges, and composes a surviving path into a single
stitched trajectory.  These dataclasses hold the orchestration parameters
without overloading :class:`MotionPlannerCfg`.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg


@dataclass
class SegmentPlanCfg:
    """Per-segment trajectory-optimization constraints.

    A segment is the edge between two consecutive waypoints ``(Wi, Wi+1)``.
    Each segment may apply different tool-pose criteria (e.g. a linear-motion
    constraint along ``z`` for an approach segment, free motion for another).
    """

    #: Tool-pose criteria applied while planning this segment, keyed by tool
    #: frame name.  Passed to ``update_tool_pose_criteria`` before solving the
    #: edges of this segment.
    tool_pose_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)

    #: Optional human-readable label used in failure status messages.
    name: Optional[str] = None


@dataclass
class MultiWaypointMotionPlannerCfg:
    """Configuration for :class:`MultiWaypointMotionPlanner`.

    Wraps a single-problem :class:`MotionPlannerCfg` (which provides the IK and
    trajopt solvers) and adds the multi-waypoint orchestration parameters.
    """

    #: Underlying single-problem motion-planner config providing the IK and
    #: trajopt solvers reused for per-waypoint IK and per-edge trajopt.
    motion_planner_cfg: MotionPlannerCfg

    #: Per-segment trajectory-optimization constraints.  Length must equal
    #: ``num_waypoints - 1``.  May be overridden at call time by passing
    #: ``segment_tool_pose_criteria`` to ``plan_waypoints``.
    segment_constraints: List[SegmentPlanCfg] = field(default_factory=list)

    #: Whole-plan retries.  Mirrors ``plan_pose(max_attempts=...)``; on each
    #: attempt the underlying solver seeds are reset before re-planning.
    max_attempts: int = 4

    #: Number of IK solutions requested per waypoint (before de-duplication).
    ik_return_seeds: int = 20

    #: Joint-space tolerance for de-duplicating IK solutions at a waypoint.
    #: Solutions closer than this (see ``ik_dedup.py``) are treated as
    #: duplicates and pruned to limit IK-pair explosion.
    ik_dedup_joint_tol: float = 0.05

    #: Optional cap on the number of edges generated per layer pair.  When set,
    #: only the ``knn_edges`` nearest (in joint space) target nodes are
    #: considered for each source node.  ``None`` means full bipartite edges.
    knn_edges: Optional[int] = None

    #: Optional hard cap on total edges evaluated per layer pair after edge
    #: generation.  ``None`` means no cap.
    max_edges_per_layer_pair: Optional[int] = None

    #: How shared-waypoint criteria are merged from the two adjacent segments.
    #: ``"max_weights_min_tols"`` combines axis weight factors by ``max`` and
    #: convergence tolerances by ``min`` so the waypoint satisfies both
    #: segments' constraints.
    waypoint_constraint_merge_mode: str = "max_weights_min_tols"

    #: Add the full-turn variants of every IK solution (``q ± 2π`` on revolute joints whose
    #: limits span more than one turn), so a waypoint can be reached "the other way round"
    #: when that makes the whole path shorter. Use it with ``max_layer_nodes``: the variants
    #: multiply the IK solutions of a waypoint.
    expand_revolutions: bool = True

    #: Most IK solutions kept per waypoint for the edge trajectory optimizations: the ones on
    #: the shortest joint-space chains from the start state through all waypoints. ``None``
    #: keeps every solution.
    max_layer_nodes: Optional[int] = 12

    #: Count the way from the start state to the first waypoint's IK solution (as the time the
    #: joints need at their velocity limits) in the path search, so the trajectory starts at a
    #: solution near the start state instead of at any solution of the first waypoint.
    start_state_cost: bool = True

    @classmethod
    def create(
        cls,
        robot: Any,
        segment_constraints: Optional[List[SegmentPlanCfg]] = None,
        *,
        max_attempts: int = 4,
        ik_return_seeds: int = 20,
        ik_dedup_joint_tol: float = 0.05,
        knn_edges: Optional[int] = None,
        max_edges_per_layer_pair: Optional[int] = None,
        waypoint_constraint_merge_mode: str = "max_weights_min_tols",
        expand_revolutions: bool = True,
        max_layer_nodes: Optional[int] = 12,
        start_state_cost: bool = True,
        **motion_planner_kwargs: Any,
    ) -> "MultiWaypointMotionPlannerCfg":
        """Build a config from a robot spec and multi-waypoint parameters.

        Forwards ``robot`` and any additional keyword arguments to
        :meth:`MotionPlannerCfg.create` to construct the underlying solvers.

        Args:
            robot: Robot configuration - path to YAML, dict, or RobotCfg object.
            segment_constraints: Per-segment constraints; length must equal
                ``num_waypoints - 1`` when planning.  Defaults to an empty list.
            max_attempts: Whole-plan retries.
            ik_return_seeds: IK solutions requested per waypoint.
            ik_dedup_joint_tol: Joint-space de-duplication tolerance.
            knn_edges: Optional per-source nearest-neighbour edge cap.
            max_edges_per_layer_pair: Optional total edge cap per layer pair.
            waypoint_constraint_merge_mode: Shared-waypoint merge strategy.
            expand_revolutions: Add the full-turn variants of the IK solutions.
            max_layer_nodes: Most IK solutions kept per waypoint, ``None`` for all.
            start_state_cost: Count the way from the start state in the path search.
            **motion_planner_kwargs: Forwarded to ``MotionPlannerCfg.create``.

        Returns:
            Configured :class:`MultiWaypointMotionPlannerCfg` instance.
        """
        motion_planner_cfg = MotionPlannerCfg.create(robot, **motion_planner_kwargs)
        return cls(
            motion_planner_cfg=motion_planner_cfg,
            segment_constraints=segment_constraints if segment_constraints is not None else [],
            max_attempts=max_attempts,
            ik_return_seeds=ik_return_seeds,
            ik_dedup_joint_tol=ik_dedup_joint_tol,
            knn_edges=knn_edges,
            max_edges_per_layer_pair=max_edges_per_layer_pair,
            waypoint_constraint_merge_mode=waypoint_constraint_merge_mode,
            expand_revolutions=expand_revolutions,
            max_layer_nodes=max_layer_nodes,
            start_state_cost=start_state_cost,
        )
