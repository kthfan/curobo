# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-waypoint motion planner: IK layers + edge trajopt + prune + compose.

For an ordered list of Cartesian waypoints this planner:

1. Generates a set of IK solutions per waypoint and de-duplicates them.
2. Builds a layered DAG with edges between IK nodes of consecutive waypoints.
3. Runs configuration-space trajectory optimization on each edge (segment
   constraints may differ) and prunes failed edges.
4. Finds the cheapest surviving path through the layers via dynamic programming.
5. Re-solves the chosen edges and stitches their trajectories into one.

The planner reuses the existing :class:`MotionPlanner` solvers (IK + trajopt)
and the existing :class:`ToolPoseCriteria` constraint switching; the new logic
is orchestration and bookkeeping.
"""

from __future__ import annotations

# Standard Library
import time
from typing import Dict, List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.ik_dedup import dedup_joint_solutions
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.multi_waypoint_planner_cfg import MultiWaypointMotionPlannerCfg
from curobo._src.motion.multi_waypoint_planner_result import MultiWaypointPlanResult
from curobo._src.motion.trajectory_stitching import stitch_jointstate_trajectories
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise

_INF = float("inf")


class MultiWaypointMotionPlanner:
    """Plan a single trajectory through an ordered list of Cartesian waypoints.

    Layers correspond to the waypoints; segment ``s`` is the edge between
    waypoint ``s`` and waypoint ``s + 1``.  There are ``num_waypoints - 1``
    segments.  The robot ``start_state`` is used as the IK seed.
    """

    def __init__(self, config: MultiWaypointMotionPlannerCfg):
        self.config = config
        self.motion_planner = MotionPlanner(config.motion_planner_cfg)
        self.ik_solver = self.motion_planner.ik_solver
        self.trajopt_solver = self.motion_planner.trajopt_solver
        self._destroyed = False

        # Baseline (track-everything) criteria restored after planning so the
        # planner is never left in a modified constraint state (mirrors how
        # ``plan_grasp`` restores ``standard_criteria``).
        self._baseline_criteria = {k: ToolPoseCriteria() for k in self.tool_frames}

    # -- Lifecycle --

    def destroy(self):
        """Release CUDA graph resources held by the underlying solvers."""
        if self._destroyed:
            return
        self._destroyed = True
        self.motion_planner.destroy()

    def __del__(self):
        self.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
        return False

    # -- Properties (delegated) --

    @property
    def joint_names(self) -> List[str]:
        return self.motion_planner.joint_names

    @property
    def action_dim(self) -> int:
        return self.motion_planner.action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self.motion_planner.tool_frames

    @property
    def kinematics(self):
        return self.motion_planner.kinematics

    @property
    def default_joint_state(self) -> JointState:
        return self.motion_planner.default_joint_state

    def compute_kinematics(self, state: JointState):
        return self.motion_planner.compute_kinematics(state)

    # -- Delegated solver controls --

    def reset_seed(self):
        self.motion_planner.reset_seed()

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        self.motion_planner.update_tool_pose_criteria(tool_pose_criteria)

    def update_world(self, scene_cfg):
        self.motion_planner.update_world(scene_cfg)

    # -- Constraint merging (shared-waypoint rule) --

    def _merge_two_criteria(
        self, a: ToolPoseCriteria, b: ToolPoseCriteria
    ) -> ToolPoseCriteria:
        """Merge two criteria so the shared waypoint satisfies both segments.

        Axis weight factors are combined by ``max`` (the stronger constraint
        wins) and convergence tolerances by ``min`` ignoring zeros (zero means
        "unset", so a non-zero tolerance from either side is preserved and the
        stricter of two non-zero tolerances is kept).
        """

        def min_tol(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x_or_y = torch.where(x > 0, x, y)
            y_or_x = torch.where(y > 0, y, x)
            return torch.minimum(x_or_y, y_or_x)

        proj = (
            a.project_distance_to_goal.bool() | b.project_distance_to_goal.bool()
        ).to(torch.uint8)
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=torch.maximum(
                a.terminal_pose_axes_weight_factor, b.terminal_pose_axes_weight_factor
            ),
            non_terminal_pose_axes_weight_factor=torch.maximum(
                a.non_terminal_pose_axes_weight_factor,
                b.non_terminal_pose_axes_weight_factor,
            ),
            terminal_pose_convergence_tolerance=min_tol(
                a.terminal_pose_convergence_tolerance,
                b.terminal_pose_convergence_tolerance,
            ),
            non_terminal_pose_convergence_tolerance=min_tol(
                a.non_terminal_pose_convergence_tolerance,
                b.non_terminal_pose_convergence_tolerance,
            ),
            project_distance_to_goal=proj,
            device_cfg=a.device_cfg,
        )

    def _merge_criteria_dicts(
        self,
        prev: Dict[str, ToolPoseCriteria],
        nxt: Dict[str, ToolPoseCriteria],
    ) -> Dict[str, ToolPoseCriteria]:
        merged: Dict[str, ToolPoseCriteria] = {}
        for frame in set(prev) | set(nxt):
            if frame in prev and frame in nxt:
                merged[frame] = self._merge_two_criteria(prev[frame], nxt[frame])
            elif frame in prev:
                merged[frame] = prev[frame]
            else:
                merged[frame] = nxt[frame]
        return merged

    def _build_waypoint_criteria(
        self, segment_criteria: List[Dict[str, ToolPoseCriteria]]
    ) -> List[Dict[str, ToolPoseCriteria]]:
        """Per-waypoint criteria using the shared-waypoint merge rule.

        Waypoint 0 uses segment 0's criteria, the last waypoint uses the last
        segment's criteria, and every interior waypoint merges its two adjacent
        segments' criteria.
        """
        if self.config.waypoint_constraint_merge_mode != "max_weights_min_tols":
            log_and_raise(
                "Unsupported waypoint_constraint_merge_mode: "
                f"{self.config.waypoint_constraint_merge_mode}"
            )
        num_waypoints = len(segment_criteria) + 1
        waypoint_criteria: List[Dict[str, ToolPoseCriteria]] = []
        for i in range(num_waypoints):
            if i == 0:
                waypoint_criteria.append(segment_criteria[0])
            elif i == num_waypoints - 1:
                waypoint_criteria.append(segment_criteria[-1])
            else:
                waypoint_criteria.append(
                    self._merge_criteria_dicts(
                        segment_criteria[i - 1], segment_criteria[i]
                    )
                )
        return waypoint_criteria

    # -- Edge construction --

    def _build_edge_indices(self, src: torch.Tensor, tgt: torch.Tensor):
        """Generate candidate (source, target) node-index pairs for one segment.

        Full bipartite by default; restricted to ``knn_edges`` nearest targets
        per source when set, and capped to ``max_edges_per_layer_pair`` total
        (keeping the joint-space-nearest pairs) when set.
        """
        ms, mt = src.shape[0], tgt.shape[0]
        device = src.device
        knn = self.config.knn_edges
        if knn is not None and knn < mt:
            dist = torch.cdist(src, tgt)
            nn = torch.topk(dist, knn, dim=1, largest=False).indices
            src_idx = torch.arange(ms, device=device).view(ms, 1).expand(ms, knn).reshape(-1)
            tgt_idx = nn.reshape(-1)
        else:
            src_idx = torch.arange(ms, device=device).view(ms, 1).expand(ms, mt).reshape(-1)
            tgt_idx = torch.arange(mt, device=device).view(1, mt).expand(ms, mt).reshape(-1)

        cap = self.config.max_edges_per_layer_pair
        if cap is not None and src_idx.shape[0] > cap:
            dist = torch.cdist(src, tgt)
            pair_dist = dist[src_idx, tgt_idx]
            order = torch.argsort(pair_dist)[:cap]
            src_idx, tgt_idx = src_idx[order], tgt_idx[order]
        return src_idx, tgt_idx

    def _solve_segment_edges(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Run trajopt on every candidate edge; return a cost matrix.

        Returns:
            A ``[Ms, Mt]`` tensor of edge costs.  Failed or non-existent edges
            are ``inf``.
        """
        ms, mt = src.shape[0], tgt.shape[0]
        device = src.device
        cost = torch.full((ms, mt), _INF, device=device)
        src_idx, tgt_idx = self._build_edge_indices(src, tgt)
        num_edges = src_idx.shape[0]
        max_batch = self.trajopt_solver.config.max_batch_size

        for start in range(0, num_edges, max_batch):
            chunk = slice(start, min(start + max_batch, num_edges))
            si = src_idx[chunk]
            ti = tgt_idx[chunk]
            current_state = JointState.from_position(src[si], self.joint_names)
            goal_state = JointState.from_position(tgt[ti], self.joint_names)
            result = self.trajopt_solver.solve_cspace(goal_state, current_state)

            batch = si.shape[0]
            success = result.success.view(batch, -1).any(dim=1)
            if result.seed_cost is not None:
                edge_cost = result.seed_cost.view(batch, -1).min(dim=1).values
            else:
                edge_cost = torch.ones(batch, device=device)
            edge_cost = torch.where(
                success, edge_cost, torch.full_like(edge_cost, _INF)
            )
            cost[si, ti] = edge_cost
        return cost

    # -- Graph search --

    def _shortest_path(self, layer_costs: List[torch.Tensor]) -> Optional[List[int]]:
        """Forward DP over the layered DAG; return one node index per layer.

        Returns ``None`` if no finite-cost path reaches the last layer.
        """
        dist = torch.zeros(layer_costs[0].shape[0], device=layer_costs[0].device)
        parents: List[torch.Tensor] = []
        for cost_matrix in layer_costs:
            total = dist.view(-1, 1) + cost_matrix  # [Ms, Mt]
            new_dist, arg = total.min(dim=0)
            parents.append(arg)
            dist = new_dist

        if not bool(torch.isfinite(dist).any()):
            return None
        end_node = int(torch.argmin(dist).item())
        if not bool(torch.isfinite(dist[end_node])):
            return None

        path = [end_node]
        node = end_node
        for arg in reversed(parents):
            node = int(arg[node].item())
            path.append(node)
        path.reverse()
        return path

    # -- Trajectory extraction --

    def _extract_segment_trajectory(self, result: TrajOptSolverResult) -> JointState:
        """Pull the (trimmed) single-trajectory joint state from a solve."""
        trajectory = result.interpolated_trajectory
        if trajectory is None:
            trajectory = result.js_solution
        elif result.interpolated_last_tstep is not None:
            last = int(result.interpolated_last_tstep.view(-1)[0].item())
            trajectory = trim_joint_state_trajectory(trajectory, 0, last + 1)
        return trajectory[0]

    # -- Public API --

    def plan_waypoints(
        self,
        waypoints: List[GoalToolPose],
        start_state: JointState,
        segment_tool_pose_criteria: Optional[List[Dict[str, ToolPoseCriteria]]] = None,
        max_attempts: Optional[int] = None,
    ) -> MultiWaypointPlanResult:
        """Plan a single trajectory through an ordered list of waypoints.

        Args:
            waypoints: Ordered Cartesian goals, one :class:`GoalToolPose` per
                waypoint.  At least two are required.
            start_state: Robot start joint state ``[1, dof]``; used as the IK
                seed.
            segment_tool_pose_criteria: Per-segment tool-pose criteria, length
                ``num_waypoints - 1``.  When ``None``, the criteria stored in
                ``config.segment_constraints`` are used.
            max_attempts: Whole-plan retries; defaults to ``config.max_attempts``.

        Returns:
            A :class:`MultiWaypointPlanResult`.  On failure ``success`` is
            ``False`` and ``status`` explains why.
        """
        t0 = time.time()
        device = start_state.device

        if len(waypoints) < 2:
            return MultiWaypointPlanResult(
                success=torch.tensor([False], device=device),
                status="At least two waypoints are required.",
                planning_time=time.time() - t0,
            )

        if start_state.position.ndim == 1:
            start_state = start_state.unsqueeze(0)

        num_segments = len(waypoints) - 1
        segment_criteria = self._resolve_segment_criteria(
            segment_tool_pose_criteria, num_segments
        )
        if max_attempts is None:
            max_attempts = self.config.max_attempts

        result = MultiWaypointPlanResult(
            success=torch.tensor([False], device=device),
            status="Planning not started.",
        )

        try:
            waypoint_criteria = self._build_waypoint_criteria(segment_criteria)
            for attempt in range(max_attempts):
                result.attempts = attempt + 1
                if attempt > 0:
                    self.reset_seed()

                ik_sets, status = self._generate_ik_layers(
                    waypoints, waypoint_criteria, start_state
                )
                if ik_sets is None:
                    result.status = status
                    continue

                layer_costs, status = self._plan_all_segments(
                    ik_sets, segment_criteria
                )
                if layer_costs is None:
                    result.status = status
                    continue

                path = self._shortest_path(layer_costs)
                if path is None:
                    result.status = "No surviving path through waypoint layers."
                    continue

                segment_results, segment_trajectories, status = self._resolve_path(
                    ik_sets, path, segment_criteria
                )
                if segment_results is None:
                    result.status = status
                    continue

                result.success = torch.tensor([True], device=device)
                result.status = "Success."
                result.chosen_ik_indices = path
                result.segment_results = segment_results
                result.composed_trajectory = stitch_jointstate_trajectories(
                    segment_trajectories
                )
                result.planning_time = time.time() - t0
                return result
        finally:
            self.update_tool_pose_criteria(self._baseline_criteria)

        result.planning_time = time.time() - t0
        return result

    # -- Internal staged helpers --

    def _resolve_segment_criteria(
        self,
        segment_tool_pose_criteria: Optional[List[Dict[str, ToolPoseCriteria]]],
        num_segments: int,
    ) -> List[Dict[str, ToolPoseCriteria]]:
        if segment_tool_pose_criteria is None:
            if not self.config.segment_constraints:
                log_and_raise(
                    "No segment criteria provided and config.segment_constraints is empty."
                )
            segment_criteria = [
                sc.tool_pose_criteria for sc in self.config.segment_constraints
            ]
        else:
            segment_criteria = segment_tool_pose_criteria

        if len(segment_criteria) != num_segments:
            log_and_raise(
                f"Expected {num_segments} segment criteria, got {len(segment_criteria)}."
            )
        # Empty criteria dicts fall back to baseline (track-everything) so a
        # segment never silently reuses the previous segment's constraints.
        return [crit if crit else self._baseline_criteria for crit in segment_criteria]

    def _generate_ik_layers(
        self,
        waypoints: List[GoalToolPose],
        waypoint_criteria: List[Dict[str, ToolPoseCriteria]],
        start_state: JointState,
    ):
        """Solve and de-duplicate IK per waypoint. Returns (ik_sets, status)."""
        dof = self.action_dim
        ik_sets: List[torch.Tensor] = []
        for i, waypoint in enumerate(waypoints):
            self.update_tool_pose_criteria(waypoint_criteria[i])
            ik_result = self.ik_solver.solve_pose(
                waypoint,
                return_seeds=self.config.ik_return_seeds,
                current_state=start_state,
            )
            success = ik_result.success.view(-1)
            solutions = ik_result.solution.reshape(-1, dof)
            good = solutions[success]
            if good.shape[0] == 0:
                return None, f"No IK solution at waypoint {i}."
            ik_sets.append(
                dedup_joint_solutions(good, self.config.ik_dedup_joint_tol)
            )
        return ik_sets, None

    def _plan_all_segments(
        self,
        ik_sets: List[torch.Tensor],
        segment_criteria: List[Dict[str, ToolPoseCriteria]],
    ):
        """Run edge trajopt for every segment. Returns (layer_costs, status)."""
        layer_costs: List[torch.Tensor] = []
        for s in range(len(ik_sets) - 1):
            self.update_tool_pose_criteria(segment_criteria[s])
            cost = self._solve_segment_edges(ik_sets[s], ik_sets[s + 1])
            if not bool(torch.isfinite(cost).any()):
                return None, f"No surviving edges for segment {s}."
            layer_costs.append(cost)
        return layer_costs, None

    def _resolve_path(
        self,
        ik_sets: List[torch.Tensor],
        path: List[int],
        segment_criteria: List[Dict[str, ToolPoseCriteria]],
    ):
        """Re-solve the chosen edges to obtain trajectories for stitching.

        Returns ``(segment_results, segment_trajectories, status)``;
        ``(None, None, status)`` if any chosen edge fails to re-solve.
        """
        segment_results: List[TrajOptSolverResult] = []
        segment_trajectories: List[JointState] = []
        for s in range(len(ik_sets) - 1):
            self.update_tool_pose_criteria(segment_criteria[s])
            current_q = ik_sets[s][path[s]].view(1, -1)
            goal_q = ik_sets[s + 1][path[s + 1]].view(1, -1)
            current_state = JointState.from_position(current_q, self.joint_names)
            goal_state = JointState.from_position(goal_q, self.joint_names)
            result = self.trajopt_solver.solve_cspace(goal_state, current_state)
            if not bool(result.success.view(-1).any()):
                return None, None, f"Re-solve of chosen edge for segment {s} failed."
            segment_results.append(result)
            segment_trajectories.append(self._extract_segment_trajectory(result))
        return segment_results, segment_trajectories, None
