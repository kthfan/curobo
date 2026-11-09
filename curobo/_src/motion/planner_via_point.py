# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan one continuous trajectory through ordered Cartesian via points.

:class:`ViaPointMotionPlanner` offers the per-segment flexibility of
:class:`~curobo._src.motion.multi_waypoint_planner.MultiWaypointMotionPlanner` (criteria and
bounds per segment, a choice among IK branches at every waypoint) with the structure of
:meth:`MotionPlanner.plan_pose_waypoints
<curobo._src.motion.motion_planner.MotionPlanner.plan_pose_waypoints>`: one fixed-spline
trajectory optimization over the whole path instead of one optimization per DAG edge.

A plan runs these steps:

1. Solve and de-duplicate IK at every waypoint, keeping solutions that satisfy the bounds of
   the adjacent segments.
2. Pick IK chains (one IK solution per waypoint) through the layers by dynamic programming on
   joint-space distance; each trajectory seed follows one chain, ending on a different IK
   solution of the last waypoint.
3. Give every segment native reference intervals in proportion to the joint-space distance it
   covers, and interpolate a Cartesian reference between the waypoints on that grid.
4. Write each segment's criteria and bounds into the segment tables of the optimizer's costs and
   run one trajectory optimization; the waypoints are enforced at their native samples and the
   segment criteria inside the segments, on native and dense validation samples.

The trajectory does not stop at intermediate waypoints. It starts at rest at the current state
and ends at rest at the last waypoint.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from curobo._src.cost.cost_eef_bound_eval import tool_pose_bounds_violation
from curobo._src.cost.cost_tool_pose_segment import ToolPoseSegmentTable
from curobo._src.cost.tool_pose_bound import ToolPoseBound, ToolPoseBoundAnchor
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.ik_dedup import dedup_joint_solutions
from curobo._src.motion.motion_planner import (
    MotionPlanner,
    _initial_iteration_budget,
    _waypoint_start_state,
)
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg, ViaPointPlanResult
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.trajectory_pose_reference import _slerp, pose_errors
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

SegmentCriteria = Optional[Dict[str, ToolPoseCriteria]]


def allocate_waypoint_indices(
    distances: Sequence[float], intervals: int, padding: int, min_gap: int
) -> List[int]:
    """Spread the native reference intervals over the segments by the distance they cover.

    Args:
        distances: Joint-space distance covered by every segment (non-negative).
        intervals: Native reference intervals between the stationary boundary spans.
        padding: Native index of the start sample.
        min_gap: Fewest intervals per segment; lowered when the segments do not all fit.

    Returns:
        Strictly increasing native index of every waypoint; the last one is
        ``padding + intervals``.
    """
    n = len(distances)
    if n < 1 or n > intervals:
        log_and_raise(f"{n} waypoints do not fit {intervals} native reference intervals")
    gap = max(1, min(min_gap, intervals // n))
    extra = intervals - n * gap
    distances = [max(0.0, float(d)) for d in distances]
    total = sum(distances)
    weights = [d / total for d in distances] if total > 1e-9 else [1.0 / n] * n
    raw = [extra * w for w in weights]
    gaps = [gap + int(math.floor(r)) for r in raw]
    remainder = intervals - sum(gaps)
    for k in sorted(range(n), key=lambda k: raw[k] - math.floor(raw[k]), reverse=True)[
        :remainder
    ]:
        gaps[k] += 1
    indices, index = [], padding
    for g in gaps:
        index += g
        indices.append(index)
    return indices


def interpolate_via_point_reference(
    start: ToolPose,
    waypoints: GoalToolPose,
    waypoint_indices: Sequence[int],
    horizon: int,
    padding: int,
) -> GoalToolPose:
    """Interpolate a native reference through the waypoints at the given sample indices.

    Positions are interpolated linearly and rotations with SLERP between consecutive targets;
    the samples up to ``padding`` hold the start pose and the samples after the last waypoint
    hold the last waypoint.

    Args:
        start: Current tool poses ``[B, 1, links, 3/4]``.
        waypoints: Targets ``[B, N, links, 1, 3/4]``.
        waypoint_indices: Native sample of every target, strictly increasing, after ``padding``.
        horizon: Number of native samples.
        padding: Native index of the start sample.

    Returns:
        Reference ``[B, horizon, links, 1, 3/4]``.
    """
    if waypoints.num_goalset != 1 or len(waypoint_indices) != waypoints.horizon:
        log_and_raise("waypoints need goalset size one and one native index each")
    device = waypoints.position.device
    position = torch.cat((start.position.unsqueeze(3), waypoints.position), dim=1)
    quaternion = torch.cat((start.quaternion.unsqueeze(3), waypoints.quaternion), dim=1)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    signs = torch.where(
        (quaternion[:, 1:] * quaternion[:, :-1]).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0
    ).cumprod(dim=1)
    quaternion = torch.cat((quaternion[:, :1], quaternion[:, 1:] * signs), dim=1)
    nodes = torch.tensor([padding, *waypoint_indices], device=device, dtype=position.dtype)
    query = torch.arange(horizon, device=device, dtype=position.dtype).clamp(
        float(nodes[0]), float(nodes[-1])
    )
    segment = (torch.searchsorted(nodes, query, right=True) - 1).clamp(0, nodes.numel() - 2)
    t = ((query - nodes[segment]) / (nodes[segment + 1] - nodes[segment])).clamp(0.0, 1.0)
    t = t.view(1, horizon, 1, 1, 1)
    reference_position = (1 - t) * position[:, segment] + t * position[:, segment + 1]
    reference_quaternion = _slerp(quaternion[:, segment], quaternion[:, segment + 1], t)
    return GoalToolPose(
        tool_frames=list(waypoints.tool_frames),
        position=reference_position.contiguous(),
        quaternion=reference_quaternion.contiguous(),
    )


def select_ik_chains(
    start: torch.Tensor, layers: Sequence[torch.Tensor], num_chains: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick IK chains through the waypoint layers by shortest joint-space path.

    Every chain is the shortest path from the start to one IK solution of the last waypoint;
    the ``num_chains`` chains end on the solutions with the shortest paths, repeated when the
    last waypoint has fewer solutions.

    Args:
        start: Start configuration ``[dof]``.
        layers: IK solutions of every waypoint, ``[M_j, dof]`` each.
        num_chains: Number of chains to return.

    Returns:
        Chains ``[num_chains, N, dof]`` and their joint-space path lengths ``[num_chains]``.
    """
    cost = (layers[0] - start).norm(dim=-1)
    parents = []
    for previous, layer in zip(layers[:-1], layers[1:]):
        total = cost.unsqueeze(1) + torch.cdist(previous, layer)
        cost, parent = total.min(dim=0)
        parents.append(parent)
    order = torch.argsort(cost)
    picks = order[torch.arange(num_chains, device=order.device) % order.numel()]
    nodes = [picks]
    for parent in reversed(parents):
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    chains = torch.stack([layer[index] for layer, index in zip(layers, nodes)], dim=1)
    return chains, cost[picks]


def knot_space_seeds(
    start: torch.Tensor,
    chains: torch.Tensor,
    waypoint_indices: Sequence[int],
    padding: int,
    intervals: int,
    num_knots: int,
) -> torch.Tensor:
    """Spline control seeds that follow each IK chain at its waypoints' reference progress.

    Args:
        start: Start configuration ``[dof]``.
        chains: IK chains ``[S, N, dof]``.
        waypoint_indices: Native sample of every waypoint.
        padding: Native index of the start sample.
        intervals: Native reference intervals.
        num_knots: Number of spline controls.

    Returns:
        Seeds ``[S, num_knots, dof]``; the last control is the chain's last configuration.
    """
    seeds, n, dof = chains.shape
    configs = torch.cat((start.view(1, 1, dof).expand(seeds, 1, dof), chains), dim=1)
    progress = torch.tensor(
        [0.0] + [(index - padding) / intervals for index in waypoint_indices],
        device=chains.device,
        dtype=chains.dtype,
    )
    knots = torch.linspace(0.0, 1.0, num_knots, device=chains.device, dtype=chains.dtype)
    segment = (torch.searchsorted(progress, knots, right=True) - 1).clamp(0, n - 1)
    t = ((knots - progress[segment]) / (progress[segment + 1] - progress[segment])).clamp(0, 1)
    t = t.view(1, num_knots, 1)
    return (1 - t) * configs[:, segment] + t * configs[:, segment + 1]


def base_anchored_segment_bounds(
    segment_criteria: Sequence[Dict[str, ToolPoseCriteria]],
    start: ToolPose,
    waypoints: GoalToolPose,
) -> List[Dict[str, List[ToolPoseBound]]]:
    """Return every segment's bounds anchored at the robot base.

    A goal anchor of segment ``k`` is waypoint ``k``. A start anchor is the current pose for the
    first segment and waypoint ``k - 1`` for the others: the robot passes that waypoint within
    its reach tolerance, so this is the waypoint target rather than the exact pose reached.

    Args:
        segment_criteria: Criteria of every segment, for every tool frame.
        start: Current tool poses ``[1, 1, links, 3/4]``.
        waypoints: Targets ``[1, N, links, 1, 3/4]``.

    Returns:
        Per segment, the base-anchored bounds of every tool frame that has bounds.
    """
    frames = list(waypoints.tool_frames)
    start_position = start.position[0, 0].tolist()
    start_quaternion = start.quaternion[0, 0].tolist()
    target_position = waypoints.position[0, :, :, 0].tolist()
    target_quaternion = waypoints.quaternion[0, :, :, 0].tolist()
    result = []
    for k, criteria in enumerate(segment_criteria):
        segment = {}
        for frame, c in criteria.items():
            if not c.bounds:
                continue
            li = frames.index(frame)
            if k == 0:
                start_pose = (start_position[li], start_quaternion[li])
            else:
                start_pose = (target_position[k - 1][li], target_quaternion[k - 1][li])
            anchors = {
                ToolPoseBoundAnchor.GOAL: (target_position[k][li], target_quaternion[k][li]),
                ToolPoseBoundAnchor.START: start_pose,
            }
            segment[frame] = [
                b
                if b.anchor == ToolPoseBoundAnchor.BASE
                else b.anchored_at_base(*anchors[b.anchor])
                for b in c.bounds
            ]
        result.append(segment)
    return result


class ViaPointMotionPlanner:
    """Plan one continuous trajectory through ordered Cartesian via points.

    Segment ``k`` ends at waypoint ``k`` and starts at waypoint ``k - 1``, or at the current
    state for ``k = 0``. Its :class:`ToolPoseCriteria` sets how waypoint ``k`` is reached
    (terminal fields, ``terminal_tolerance``) and how the path inside the segment is constrained
    (path fields, ``path_tolerance`` with ``enforce_path_tolerance``), relative to the line and
    SLERP between the segment's ends. Its ``bounds`` hold inside the segment. The default
    criteria reach the waypoint's full pose and leave the path free.

    The planner owns its :class:`MotionPlanner`: the optimizer's tool-pose and bound costs read
    the per-segment tables of this planner, so that motion planner is not exposed for other
    planning calls.
    """

    def __init__(self, config: ViaPointMotionPlannerCfg):
        """Create the motion planner and bind its optimizer costs to the segment tables."""
        self.config = config
        self._destroyed = True
        self._motion_planner = MotionPlanner(config.motion_planner_cfg)
        self._destroyed = False
        trajopt = self._motion_planner.trajopt_solver
        trajopt._validate_waypoint_layout()
        if not trajopt.config.enable_timeopt:
            log_and_raise("ViaPointMotionPlanner needs TrajOpt with enable_timeopt=True")
        layout = self._motion_planner.config.spline_layout
        self._table = ToolPoseSegmentTable(
            self.tool_frames,
            capacity=min(256, layout.reference_intervals),
            horizon=self._motion_planner.native_horizon,
            padding=layout.padding,
            device_cfg=self._motion_planner.device_cfg,
        )
        self._bound_costs = []
        rollouts = [(r, False) for r in trajopt.get_all_rollout_instances()]
        rollouts += [
            (r, name == "interpolated_rollout")
            for name, r in trajopt.additional_metrics_rollouts.items()
        ]
        for rollout, dense in rollouts:
            for cost in rollout.get_cost_component_by_name("tool_pose"):
                if cost is not None:
                    cost.attach_segment_table(self._table, dense)
            for cost in rollout.get_cost_component_by_name("eef_bound"):
                if cost is not None:
                    cost.attach_segment_table(self._table, dense)
                    self._bound_costs.append(cost)
        self._baseline_criteria = {
            frame: ToolPoseCriteria(device_cfg=self._motion_planner.device_cfg)
            for frame in self.tool_frames
        }

    # -- Lifecycle --

    def destroy(self) -> None:
        """Release the CUDA graph resources of the motion planner."""
        if getattr(self, "_destroyed", True):
            return
        self._destroyed = True
        self._motion_planner.destroy()

    def __del__(self):
        self.destroy()

    def __enter__(self) -> ViaPointMotionPlanner:
        return self

    def __exit__(self, *exc) -> bool:
        self.destroy()
        return False

    # -- Robot and scene --

    @property
    def tool_frames(self) -> List[str]:
        """Ordered tool frames that waypoints and criteria refer to."""
        return self._motion_planner.tool_frames

    @property
    def joint_names(self) -> List[str]:
        """Ordered active joint names of the planned trajectory."""
        return self._motion_planner.joint_names

    @property
    def default_joint_state(self) -> JointState:
        """Default joint state of the robot."""
        return self._motion_planner.default_joint_state

    @property
    def max_waypoints(self) -> int:
        """Most waypoints one plan can hold: the native reference intervals, at most 256."""
        return self._table.capacity

    def compute_kinematics(self, state: JointState):
        """Forward kinematics of joint states."""
        return self._motion_planner.compute_kinematics(state)

    def update_world(self, scene_cfg) -> None:
        """Replace the collision scene."""
        self._motion_planner.update_world(scene_cfg)

    def reset_seed(self) -> None:
        """Reset the random seeds of the solvers."""
        self._motion_planner.reset_seed()

    def warmup(self, num_waypoints: int = 2, num_warmup_iterations: int = 2) -> None:
        """Plan small motions from the default state to compile kernels and capture graphs.

        Args:
            num_waypoints: Waypoints of the warmup plan.
            num_warmup_iterations: Number of warmup plans.
        """
        state = JointState.from_position(
            self.default_joint_state.position.unsqueeze(0), joint_names=self.joint_names
        )
        pose = self.compute_kinematics(state).tool_poses
        offset = torch.zeros_like(pose.position)
        offset[..., 2] = -0.02
        position = torch.cat([pose.position + (k + 1) * offset for k in range(num_waypoints)], 1)
        quaternion = pose.quaternion.expand(-1, num_waypoints, -1, -1)
        waypoints = GoalToolPose(
            self.tool_frames,
            position.unsqueeze(3).contiguous(),
            quaternion.unsqueeze(3).contiguous(),
        )
        for _ in range(num_warmup_iterations):
            self.plan_waypoints(waypoints, state, max_attempts=1)

    # -- Planning --

    def plan_waypoints(
        self,
        waypoints: GoalToolPose,
        current_state: JointState,
        segment_criteria: Optional[Sequence[SegmentCriteria]] = None,
        *,
        waypoint_tolerance: Optional[Tuple[float, float]] = None,
        max_attempts: Optional[int] = None,
        finetune_attempts: int = 1,
        outer_iterations: int = 4,
    ) -> ViaPointPlanResult:
        """Plan one trajectory from the current state through every waypoint, in order.

        Args:
            waypoints: Targets ``[1, N, links, 1, 3/4]`` for the planner's tool frames, without
                the current pose. ``N`` is at most :attr:`max_waypoints`.
            current_state: Current joint state ``[1, dof]``, at rest.
            segment_criteria: ``N`` entries; entry ``k`` maps tool frames to the criteria of
                segment ``k``. Missing entries and tool frames use the default criteria.
            waypoint_tolerance: Reach tolerance (m, rad) for waypoints whose criteria set no
                ``terminal_tolerance``; defaults to the configured one. The tolerance of the
                last waypoint also bounds the endpoint of the final validation.
            max_attempts: Whole-plan retries; defaults to the configured number.
            finetune_attempts: Time-optimal refinement passes after the initial optimization.
            outer_iterations: Initial solve/check passes of the optimization.

        Returns:
            The plan; ``success`` is false and ``status`` says why when it failed.
        """
        started = time.time()
        if self._destroyed:
            log_and_raise("ViaPointMotionPlanner was destroyed")
        planner = self._motion_planner
        device = planner.device_cfg.device
        if waypoints.tool_frames != self.tool_frames:
            log_and_raise(f"waypoints must use the tool frames {self.tool_frames}")
        if waypoints.batch_size != 1 or waypoints.num_goalset != 1:
            log_and_raise("waypoints must have batch size one and goalset size one")
        n = waypoints.horizon
        if not 1 <= n <= self.max_waypoints:
            log_and_raise(f"plan_waypoints takes 1 to {self.max_waypoints} waypoints, got {n}")
        tolerance = waypoint_tolerance or self.config.waypoint_tolerance
        if tolerance[0] <= 0.0 or not 0.0 < tolerance[1] <= math.pi:
            log_and_raise("waypoint_tolerance must be positive meters and radians in (0, pi]")
        criteria = self._resolve_criteria(segment_criteria, n)
        reach = [self._with_reach_tolerance(c, tolerance) for c in criteria]
        state = _waypoint_start_state(planner, current_state)
        start_pose = planner.compute_kinematics(state).tool_poses
        segment_bounds = base_anchored_segment_bounds(criteria, start_pose, waypoints)
        initial_iters = _initial_iteration_budget(planner.trajopt_solver, outer_iterations)

        result = ViaPointPlanResult(
            success=torch.zeros(1, dtype=torch.bool, device=device), status="Not planned."
        )
        try:
            for attempt in range(max_attempts or self.config.max_attempts):
                result.attempts = attempt + 1
                if attempt > 0:
                    self.reset_seed()
                layers, status = self._ik_layers(waypoints, state, criteria, segment_bounds)
                if layers is None:
                    result.status = status
                    continue
                trajopt = self._plan_through(
                    waypoints,
                    state,
                    start_pose,
                    layers,
                    criteria,
                    reach,
                    segment_bounds,
                    tolerance,
                    finetune_attempts,
                    initial_iters,
                )
                indices, trajopt_result = trajopt
                result.waypoint_indices = indices
                result.trajopt_result = trajopt_result.clone()
                if bool(trajopt_result.success.any()):
                    result.success = torch.ones(1, dtype=torch.bool, device=device)
                    result.status = "Success."
                    break
                result.status = "Trajectory optimization found no valid trajectory."
        finally:
            planner.ik_solver.update_tool_pose_criteria(self._baseline_criteria)
        if result.trajopt_result is not None:
            self._attach_waypoint_errors(result, waypoints)
        result.planning_time = time.time() - started
        return result

    # -- Internal steps --

    def _resolve_criteria(
        self, segment_criteria: Optional[Sequence[SegmentCriteria]], n: int
    ) -> List[Dict[str, ToolPoseCriteria]]:
        """Return the criteria of every segment for every tool frame."""
        if segment_criteria is None:
            segment_criteria = [None] * n
        if len(segment_criteria) != n:
            log_and_raise(
                f"expected {n} segment criteria, one per waypoint, got {len(segment_criteria)}"
            )
        device_cfg = self._motion_planner.device_cfg
        resolved = []
        for k, criteria in enumerate(segment_criteria):
            criteria = dict(criteria or {})
            unknown = set(criteria) - set(self.tool_frames)
            if unknown:
                log_and_raise(f"segment {k} criteria name unknown tool frames {sorted(unknown)}")
            for frame in self.tool_frames:
                c = criteria.get(frame, self._baseline_criteria[frame])
                criteria[frame] = c.to(device_cfg)
            resolved.append(criteria)
        return resolved

    @staticmethod
    def _with_reach_tolerance(
        criteria: Dict[str, ToolPoseCriteria], tolerance: Tuple[float, float]
    ) -> Dict[str, ToolPoseCriteria]:
        """Fill unset terminal tolerances with ``tolerance`` (m, rad) in metric units."""
        out = {}
        for frame, c in criteria.items():
            if bool((c.terminal_tolerance == 0).all()):
                c = c.clone()
                c.terminal_tolerance.copy_(
                    c.terminal_tolerance.new_tensor(
                        [tolerance[0], 2.0 * math.sin(tolerance[1] / 2.0)]
                    )
                )
            out[frame] = c
        return out

    def _ik_layers(
        self,
        waypoints: GoalToolPose,
        state: JointState,
        criteria: List[Dict[str, ToolPoseCriteria]],
        segment_bounds: List[Dict[str, List[ToolPoseBound]]],
    ) -> Tuple[Optional[List[torch.Tensor]], str]:
        """Solve, bound-filter and de-duplicate IK at every waypoint."""
        ik_solver = self._motion_planner.ik_solver
        return_seeds = min(self.config.ik_return_seeds, ik_solver.config.num_seeds)
        layers = []
        for j in range(waypoints.horizon):
            ik_solver.update_tool_pose_criteria(_without_bounds(criteria[j]))
            target = GoalToolPose(
                waypoints.tool_frames,
                waypoints.position[:, j : j + 1].contiguous(),
                waypoints.quaternion[:, j : j + 1].contiguous(),
            )
            ik = ik_solver.solve_pose(target, current_state=state, return_seeds=return_seeds)
            solutions = ik.solution.reshape(-1, ik.solution.shape[-1])[ik.success.view(-1)]
            if solutions.shape[0] == 0:
                return None, f"No IK solution at waypoint {j}."
            bounds = segment_bounds[j : j + 2]
            solutions = self._filter_by_bounds(solutions, bounds)
            if solutions.shape[0] == 0:
                return None, f"No IK solution at waypoint {j} satisfies its segments' bounds."
            layers.append(dedup_joint_solutions(solutions, self.config.ik_dedup_joint_tol))
        return layers, ""

    def _filter_by_bounds(
        self, solutions: torch.Tensor, segment_bounds: List[Dict[str, List[ToolPoseBound]]]
    ) -> torch.Tensor:
        """Keep the configurations whose tool poses satisfy every given base-anchored bound."""
        if not any(segment_bounds):
            return solutions
        poses = self.compute_kinematics(
            JointState.from_position(solutions, joint_names=self.joint_names)
        ).tool_poses
        keep = torch.ones(solutions.shape[0], dtype=torch.bool, device=solutions.device)
        for bounds in segment_bounds:
            for frame, frame_bounds in bounds.items():
                li = poses.tool_frames.index(frame)
                violation = tool_pose_bounds_violation(
                    frame_bounds,
                    poses.position[:, 0, li],
                    poses.quaternion[:, 0, li],
                    None,
                    None,
                    None,
                    None,
                )
                keep &= violation.amax(dim=(1, 2)) == 0.0
        return solutions[keep]

    def _plan_through(
        self,
        waypoints: GoalToolPose,
        state: JointState,
        start_pose: ToolPose,
        layers: List[torch.Tensor],
        criteria: List[Dict[str, ToolPoseCriteria]],
        reach: List[Dict[str, ToolPoseCriteria]],
        segment_bounds: List[Dict[str, List[ToolPoseBound]]],
        tolerance: Tuple[float, float],
        finetune_attempts: int,
        initial_iters: int,
    ):
        """Seed and run one trajectory optimization through the IK layers."""
        planner = self._motion_planner
        solver = planner.trajopt_solver
        layout = planner.config.spline_layout
        start = state.position[0]
        chains, _ = select_ik_chains(start, layers, solver.config.num_seeds)
        best = torch.cat((start.unsqueeze(0), chains[0]), dim=0)
        distances = (best[1:] - best[:-1]).norm(dim=-1).tolist()
        indices = allocate_waypoint_indices(
            distances,
            layout.reference_intervals,
            layout.padding,
            self.config.min_segment_intervals,
        )
        reference = interpolate_via_point_reference(
            start_pose, waypoints, indices, planner.native_horizon, layout.padding
        )
        self._table.set_plan(criteria, reach, indices)
        for cost in self._bound_costs:
            cost.update_segment_tool_pose_bounds(segment_bounds)
        seed_traj = knot_space_seeds(
            start,
            chains,
            indices,
            layout.padding,
            layout.reference_intervals,
            solver.action_horizon,
        )
        result = solver.solve_pose(
            reference,
            state,
            seed_config=chains[:, -1].unsqueeze(0).contiguous(),
            seed_traj=seed_traj.unsqueeze(0).contiguous(),
            use_implicit_goal=True,
            finetune_attempts=finetune_attempts,
            initial_iters=initial_iters,
            initial_dt_scale=self.config.geometry_dt_scale,
            track_intermediate=True,
            path_tolerance=tolerance,
            final_tolerance=tolerance,
            track_orientation=True,
        )
        return indices, result

    def _attach_waypoint_errors(self, result: ViaPointPlanResult, waypoints: GoalToolPose) -> None:
        """Record the configurations and pose errors of the trajectory at the waypoints."""
        solution = result.trajopt_result.js_solution
        position = solution.position.reshape(-1, *solution.position.shape[-2:])[0]
        configs = self._motion_planner.trajopt_solver.get_active_js(
            JointState.from_position(
                position[result.waypoint_indices].contiguous(),
                joint_names=solution.joint_names,
            )
        ).position
        poses = self.compute_kinematics(
            JointState.from_position(configs, joint_names=self.joint_names)
        ).tool_poses
        result.waypoint_configs = configs
        result.waypoint_position_error, result.waypoint_rotation_error = pose_errors(
            poses.position[:, 0],
            poses.quaternion[:, 0],
            waypoints.position[0, :, :, 0],
            waypoints.quaternion[0, :, :, 0],
        )


def _without_bounds(criteria: Dict[str, ToolPoseCriteria]) -> Dict[str, ToolPoseCriteria]:
    """Return ``criteria`` without tool pose bounds, for the waypoint IK."""
    out = {}
    for frame, c in criteria.items():
        if c.bounds:
            c = c.clone()
            c.bounds = []
        out[frame] = c
    return out
