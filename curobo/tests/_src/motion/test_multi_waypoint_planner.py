# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the multi-waypoint motion planner.

Logic-only tests (de-duplication, stitching, criteria merging, edge generation,
shortest-path DP) run on CPU.  Integration tests that exercise the real IK and
trajopt solvers require CUDA and are skipped otherwise.
"""

# Standard Library
from types import SimpleNamespace

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.motion.ik_dedup import dedup_joint_solutions
from curobo._src.motion.multi_waypoint_planner import MultiWaypointMotionPlanner
from curobo._src.motion.multi_waypoint_planner_cfg import (
    MultiWaypointMotionPlannerCfg,
    SegmentPlanCfg,
)
from curobo._src.motion.multi_waypoint_planner_result import MultiWaypointPlanResult
from curobo._src.motion.trajectory_stitching import stitch_jointstate_trajectories
from curobo._src.cost.tool_pose_bound import ToolPoseBound
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose

CPU_CFG = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)


def _cpu_criteria(terminal_weight, terminal_tol=(0.0, 0.0), project=False):
    """Build a ToolPoseCriteria on CPU (factories default to CUDA)."""
    return ToolPoseCriteria(
        terminal_pose_axes_weight_factor=list(terminal_weight),
        non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        terminal_pose_convergence_tolerance=list(terminal_tol),
        non_terminal_pose_convergence_tolerance=[0.0, 0.0],
        project_distance_to_goal=project,
        device_cfg=CPU_CFG,
    )


def _stub_planner(**config_fields) -> MultiWaypointMotionPlanner:
    """Create a planner instance without running __init__ (no solvers/CUDA).

    Used to exercise the pure-logic methods that only read ``self.config``.
    """
    planner = MultiWaypointMotionPlanner.__new__(MultiWaypointMotionPlanner)
    planner.config = SimpleNamespace(**config_fields)
    return planner


# --------------------------------------------------------------------------- #
# Public API / exports
# --------------------------------------------------------------------------- #


class TestPublicExports:
    def test_planner_class_exists(self):
        assert MultiWaypointMotionPlanner is not None
        assert hasattr(MultiWaypointMotionPlanner, "plan_waypoints")

    def test_public_module_reexports(self):
        from curobo.multi_waypoint_planner import (
            MultiWaypointMotionPlanner as PubPlanner,
            MultiWaypointMotionPlannerCfg as PubCfg,
            MultiWaypointPlanResult as PubResult,
            SegmentPlanCfg as PubSegment,
        )

        assert PubPlanner is MultiWaypointMotionPlanner
        assert PubCfg is MultiWaypointMotionPlannerCfg
        assert PubResult is MultiWaypointPlanResult
        assert PubSegment is SegmentPlanCfg

    def test_segment_plan_cfg_defaults(self):
        seg = SegmentPlanCfg()
        assert seg.tool_pose_criteria == {}
        assert seg.name is None


# --------------------------------------------------------------------------- #
# IK de-duplication
# --------------------------------------------------------------------------- #


class TestDedup:
    def test_dedup_removes_near_duplicates(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 1.0, 1.0])
        q = torch.stack([a, a + 1e-4, b, b + 1e-4])
        out = dedup_joint_solutions(q, tol=0.05)
        assert out.shape[0] == 2
        assert torch.allclose(out[0], a)
        assert torch.allclose(out[1], b)

    def test_dedup_keeps_distinct(self):
        q = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        out = dedup_joint_solutions(q, tol=0.5)
        assert out.shape[0] == 3

    def test_dedup_single_row(self):
        q = torch.tensor([[0.3, 0.2]])
        out = dedup_joint_solutions(q, tol=0.1)
        assert out.shape[0] == 1

    def test_dedup_empty(self):
        q = torch.empty((0, 4))
        out = dedup_joint_solutions(q, tol=0.1)
        assert out.shape[0] == 0


# --------------------------------------------------------------------------- #
# Trajectory stitching
# --------------------------------------------------------------------------- #


class TestStitch:
    def test_stitch_drops_boundary_sample(self):
        seg0 = JointState.from_position(torch.zeros(5, 2))
        seg1 = JointState.from_position(torch.ones(4, 2))
        stitched = stitch_jointstate_trajectories([seg0, seg1])
        # 5 + (4 - 1) = 8 after dropping the duplicate boundary sample.
        assert stitched.position.shape == (8, 2)

    def test_stitch_keeps_boundary_when_disabled(self):
        seg0 = JointState.from_position(torch.zeros(5, 2))
        seg1 = JointState.from_position(torch.ones(4, 2))
        stitched = stitch_jointstate_trajectories(
            [seg0, seg1], drop_duplicate_boundary=False
        )
        assert stitched.position.shape == (9, 2)

    def test_stitch_single_segment(self):
        seg0 = JointState.from_position(torch.zeros(6, 3))
        stitched = stitch_jointstate_trajectories([seg0])
        assert stitched.position.shape == (6, 3)

    def test_stitch_empty_raises(self):
        with pytest.raises(Exception):
            stitch_jointstate_trajectories([])

    @pytest.mark.parametrize(
        "shape, last_tstep, expected",
        [((1, 1, 42, 9), 30, (31, 9)), ((1, 42, 9), None, (42, 9)), ((42, 9), None, (42, 9))],
    )
    def test_extract_segment_trajectory_is_2d(self, shape, last_tstep, expected):
        """Segments handed to stitching are [horizon, dof] for any result layout."""
        result = SimpleNamespace(
            interpolated_trajectory=JointState.from_position(torch.zeros(*shape)),
            js_solution=None,
            interpolated_last_tstep=None if last_tstep is None else torch.tensor([last_tstep]),
        )
        segment = MultiWaypointMotionPlanner._extract_segment_trajectory(None, result)
        assert segment.position.shape == expected


# --------------------------------------------------------------------------- #
# Shared-waypoint criteria merging
# --------------------------------------------------------------------------- #


class TestCriteriaMerge:
    def test_merge_max_weights_min_tols(self):
        planner = _stub_planner(waypoint_constraint_merge_mode="max_weights_min_tols")
        a = _cpu_criteria([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], terminal_tol=(0.01, 0.0))
        b = _cpu_criteria([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], terminal_tol=(0.0, 0.05), project=True)

        merged = planner._merge_two_criteria(a, b)

        assert merged.terminal_pose_axes_weight_factor.tolist() == [1, 1, 1, 1, 1, 1]
        # min-ignoring-zero keeps the non-zero tolerance from each side.
        assert merged.terminal_pose_convergence_tolerance.tolist() == pytest.approx([0.01, 0.05])
        assert bool(merged.project_distance_to_goal.bool().item()) is True

    def test_merge_min_tol_picks_stricter(self):
        planner = _stub_planner(waypoint_constraint_merge_mode="max_weights_min_tols")
        a = _cpu_criteria([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], terminal_tol=(0.02, 0.03))
        b = _cpu_criteria([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], terminal_tol=(0.01, 0.05))
        merged = planner._merge_two_criteria(a, b)
        assert merged.terminal_pose_convergence_tolerance.tolist() == pytest.approx([0.01, 0.03])


class TestBuildWaypointCriteria:
    def test_endpoints_use_adjacent_segments_interior_merges(self):
        planner = _stub_planner(waypoint_constraint_merge_mode="max_weights_min_tols")
        c0 = {"ee": _cpu_criteria([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        c1 = {"ee": _cpu_criteria([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])}
        c2 = {"ee": _cpu_criteria([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
        segment_criteria = [c0, c1, c2]  # 3 segments -> 4 waypoints

        wp = planner._build_waypoint_criteria(segment_criteria)

        assert len(wp) == 4
        assert wp[0] is c0
        assert wp[3] is c2
        # Interior waypoint 1 merges segments 0 and 1.
        assert wp[1]["ee"].terminal_pose_axes_weight_factor.tolist()[:3] == [1, 1, 0]
        # Interior waypoint 2 merges segments 1 and 2.
        assert wp[2]["ee"].terminal_pose_axes_weight_factor.tolist()[:3] == [0, 1, 1]

    def test_unsupported_merge_mode_raises(self):
        planner = _stub_planner(waypoint_constraint_merge_mode="bogus")
        with pytest.raises(Exception):
            planner._build_waypoint_criteria([{"ee": _cpu_criteria([1, 0, 0, 0, 0, 0])}])


# --------------------------------------------------------------------------- #
# Tool pose bounds
# --------------------------------------------------------------------------- #


def _bounded(bound: ToolPoseBound) -> ToolPoseCriteria:
    criteria = _cpu_criteria([1.0] * 6)
    criteria.bound = bound
    return criteria


def _x_band(half_width: float, **kwargs) -> ToolPoseBound:
    return ToolPoseBound(
        position_lower=[-half_width, None, None], position_upper=[half_width, None, None], **kwargs
    )


def _waypoint(*xs: float) -> GoalToolPose:
    """Goalset waypoint for ``ee`` at ``(x, 0, 0.5)`` per entry, identity orientation."""
    g = len(xs)
    position = torch.zeros(1, 1, 1, g, 3)
    position[..., 0] = torch.tensor(xs)
    position[..., 2] = 0.5
    quaternion = torch.zeros(1, 1, 1, g, 4)
    quaternion[..., 0] = 1.0
    return GoalToolPose(tool_frames=["ee"], position=position, quaternion=quaternion)


class _RecordingMotionPlanner:
    """Fake motion planner: FK maps joint ``q`` to tool position ``(q, 0, 0.5)``."""

    joint_names = ["j0"]
    tool_frames = ["ee"]

    def __init__(self):
        self.bound_updates = []

    def update_tool_pose_bounds(self, bounds):
        self.bound_updates.append(dict(bounds))

    def compute_kinematics(self, state: JointState):
        n = state.position.shape[0]
        position = torch.zeros(n, 1, 1, 3)
        position[:, 0, 0, 0] = state.position[:, 0]
        position[..., 2] = 0.5
        quaternion = torch.zeros(n, 1, 1, 4)
        quaternion[..., 0] = 1.0
        tool_poses = SimpleNamespace(tool_frames=["ee"], position=position, quaternion=quaternion)
        return SimpleNamespace(tool_poses=tool_poses)


def _bound_planner(**config_fields) -> MultiWaypointMotionPlanner:
    planner = _stub_planner(**config_fields)
    planner.motion_planner = _RecordingMotionPlanner()
    planner._baseline_criteria = {"ee": _cpu_criteria([1.0] * 6)}
    planner._default_tool_pose_bounds = {}
    return planner


class TestToolPoseBounds:
    def test_waypoint_criteria_drop_bounds(self):
        planner = _stub_planner(waypoint_constraint_merge_mode="max_weights_min_tols")
        c0 = {"ee": _bounded(_x_band(0.1))}
        c1 = {"ee": _bounded(_x_band(0.2))}

        wp = planner._build_waypoint_criteria([c0, c1])

        assert all(criteria["ee"].bound is None for criteria in wp)
        # The segment criteria themselves keep their bounds.
        assert c0["ee"].bound is not None and c1["ee"].bound is not None

    def test_default_bound_fills_unbounded_segments(self):
        planner = _bound_planner(segment_constraints=[])
        default = _x_band(0.3)
        planner.update_tool_pose_bounds({"ee": default})
        own = _x_band(0.1)

        segments = planner._resolve_segment_criteria(
            [{"ee": _bounded(own)}, {"ee": _cpu_criteria([1.0] * 6)}, {}], 3
        )

        assert segments[0]["ee"].bound is own
        assert segments[1]["ee"].bound is default
        assert segments[2]["ee"].bound is default
        assert planner.motion_planner.bound_updates == [{"ee": default}]

    def test_default_bound_fills_missing_frame(self):
        planner = _bound_planner(segment_constraints=[])
        default = _x_band(0.3)
        planner.update_tool_pose_bounds({"ee": default})

        segments = planner._resolve_segment_criteria([{"other": _cpu_criteria([1.0] * 6)}], 1)

        assert segments[0]["ee"].bound is default
        assert segments[0]["other"].bound is None

    def test_clear_default_bound(self):
        planner = _bound_planner(segment_constraints=[])
        planner.update_tool_pose_bounds({"ee": _x_band(0.3)})
        planner.update_tool_pose_bounds({"ee": None})

        segments = planner._resolve_segment_criteria([{}], 1)

        assert segments[0]["ee"].bound is None
        assert planner._baseline_criteria["ee"].bound is None

    def test_unknown_frame_raises(self):
        planner = _bound_planner()
        with pytest.raises(Exception):
            planner.update_tool_pose_bounds({"missing": _x_band(0.1)})

    def test_filter_outgoing_goal_anchor(self):
        planner = _bound_planner()
        segments = [{"ee": _bounded(_x_band(0.1))}]
        q = torch.tensor([[-0.5], [0.05], [0.3]])

        kept = planner._filter_by_segment_bounds(q, 0, [_waypoint(9.0), _waypoint(0.0)], segments)

        assert kept.view(-1).tolist() == pytest.approx([0.05])

    def test_filter_incoming_goal_anchor_uses_goalset(self):
        planner = _bound_planner()
        segments = [{"ee": _bounded(_x_band(0.1))}]
        q = torch.tensor([[0.05], [0.5], [0.95]])

        kept = planner._filter_by_segment_bounds(
            q, 1, [_waypoint(9.0), _waypoint(0.0, 1.0)], segments
        )

        assert kept.view(-1).tolist() == pytest.approx([0.05, 0.95])

    def test_filter_skips_incoming_start_anchor(self):
        planner = _bound_planner()
        segments = [{"ee": _bounded(_x_band(0.1, anchor="start"))}]
        q = torch.tensor([[-0.5], [0.5]])

        kept = planner._filter_by_segment_bounds(q, 1, [_waypoint(0.0), _waypoint(0.0)], segments)

        assert kept.shape[0] == 2

    def test_filter_outgoing_start_anchor_with_offset(self):
        planner = _bound_planner()
        # The reference sits 0.2 m ahead of the segment start, outside the 5 cm band.
        bound = _x_band(0.05, anchor="start", offset_position=[0.2, None, None])
        segments = [{"ee": _bounded(bound)}]
        q = torch.tensor([[0.0], [0.5]])

        kept = planner._filter_by_segment_bounds(q, 0, [_waypoint(0.0), _waypoint(0.0)], segments)

        assert kept.shape[0] == 0

    def test_filter_base_anchor_on_both_segments(self):
        planner = _bound_planner()
        segments = [
            {"ee": _bounded(_x_band(0.4, anchor="base"))},
            {"ee": _bounded(ToolPoseBound(anchor="base", position_lower=[-0.1, None, None]))},
        ]
        q = torch.tensor([[-0.2], [0.0], [0.3], [0.5]])
        waypoints = [_waypoint(0.0), _waypoint(0.0), _waypoint(0.0)]

        kept = planner._filter_by_segment_bounds(q, 1, waypoints, segments)

        assert kept.view(-1).tolist() == pytest.approx([0.0, 0.3])

    def test_filter_without_bounds_returns_input(self):
        planner = _bound_planner()
        q = torch.tensor([[0.0], [1.0]])
        segments = [{"ee": _cpu_criteria([1.0] * 6)}]

        kept = planner._filter_by_segment_bounds(q, 0, [_waypoint(0.0), _waypoint(0.0)], segments)

        assert kept is q


# --------------------------------------------------------------------------- #
# Edge generation
# --------------------------------------------------------------------------- #


class TestEdgeIndices:
    def test_full_bipartite(self):
        planner = _stub_planner(knn_edges=None, max_edges_per_layer_pair=None)
        src = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        tgt = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
        si, ti = planner._build_edge_indices(src, tgt)
        assert si.shape[0] == 6
        pairs = set(zip(si.tolist(), ti.tolist()))
        assert pairs == {(s, t) for s in range(3) for t in range(2)}

    def test_knn_edges(self):
        planner = _stub_planner(knn_edges=1, max_edges_per_layer_pair=None)
        src = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        tgt = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]])
        si, ti = planner._build_edge_indices(src, tgt)
        assert si.tolist() == [0, 1]
        assert ti.tolist() == [0, 2]

    def test_max_edges_cap(self):
        planner = _stub_planner(knn_edges=None, max_edges_per_layer_pair=2)
        src = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        tgt = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        si, ti = planner._build_edge_indices(src, tgt)
        assert si.shape[0] == 2
        assert ti.shape[0] == 2


# --------------------------------------------------------------------------- #
# Shortest-path dynamic programming
# --------------------------------------------------------------------------- #


class TestShortestPath:
    def test_finds_min_cost_path(self):
        planner = _stub_planner()
        cost0 = torch.tensor([[1.0, 5.0], [5.0, 1.0]])
        cost1 = torch.tensor([[1.0, 5.0], [5.0, 1.0]])
        path = planner._shortest_path([cost0, cost1])
        assert path == [0, 0, 0]

    def test_prefers_cheaper_branch(self):
        planner = _stub_planner()
        # Only the s0 -> t1 -> u0 path is cheap.
        cost0 = torch.tensor([[10.0, 0.1], [10.0, 10.0]])
        cost1 = torch.tensor([[10.0, 10.0], [0.1, 10.0]])
        path = planner._shortest_path([cost0, cost1])
        assert path == [0, 1, 0]

    def test_no_path_returns_none(self):
        planner = _stub_planner()
        inf = float("inf")
        cost0 = torch.full((2, 2), inf)
        path = planner._shortest_path([cost0])
        assert path is None


# --------------------------------------------------------------------------- #
# Integration tests (require CUDA + real solvers)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def planner(cuda_device_cfg):
    config = MultiWaypointMotionPlannerCfg.create(
        robot="franka.yml",
        ik_return_seeds=10,
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = MultiWaypointMotionPlanner(config)
    yield planner
    planner.destroy()


def _waypoint_from_joint_config(planner, q):
    js = JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)
    pose_dict = planner.compute_kinematics(js).tool_poses.to_dict()
    return GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=planner.tool_frames, num_goalset=1
    )


def _unreachable_waypoint(planner, device):
    frames = planner.tool_frames
    n = len(frames)
    position = torch.full((1, 1, n, 1, 3), 10.0, device=device, dtype=torch.float32)
    quaternion = torch.zeros((1, 1, n, 1, 4), device=device, dtype=torch.float32)
    quaternion[..., 0] = 1.0
    return GoalToolPose(tool_frames=frames, position=position, quaternion=quaternion)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPlanWaypointsIntegration:
    def _start_state(self, planner):
        q = planner.default_joint_state.position
        return JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)

    def _segment_criteria(self, planner, num_segments):
        return [
            {f: ToolPoseCriteria.track_position_and_orientation() for f in planner.tool_frames}
            for _ in range(num_segments)
        ]

    def test_three_waypoints_free_space_succeeds(self, planner):
        start = self._start_state(planner)
        q0 = planner.default_joint_state.position
        offsets = [
            torch.tensor([0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], device=q0.device),
            torch.tensor([0.0, 0.3, 0.0, -0.3, 0.0, 0.3, 0.0], device=q0.device),
            torch.tensor([-0.3, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0], device=q0.device),
        ]
        waypoints = [_waypoint_from_joint_config(planner, q0 + o) for o in offsets]

        result = planner.plan_waypoints(
            waypoints,
            start,
            segment_tool_pose_criteria=self._segment_criteria(planner, 2),
            max_attempts=5,
        )

        assert bool(result.success.item()) is True
        assert result.composed_trajectory is not None
        assert result.composed_trajectory.position.shape[0] > 0
        assert len(result.segment_results) == 2
        assert len(result.chosen_ik_indices) == 3

    def test_unreachable_waypoint_fails_with_status(self, planner):
        start = self._start_state(planner)
        q0 = planner.default_joint_state.position
        reachable = _waypoint_from_joint_config(
            planner,
            q0 + torch.tensor([0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0], device=q0.device),
        )
        unreachable = _unreachable_waypoint(planner, q0.device)

        result = planner.plan_waypoints(
            [reachable, unreachable],
            start,
            segment_tool_pose_criteria=self._segment_criteria(planner, 1),
            max_attempts=2,
        )

        assert bool(result.success.item()) is False
        assert "waypoint" in result.status.lower()

    def _planar_waypoints(self, planner, dz: float):
        """Start waypoint and a goal 15 cm away in x (``dz`` above), same orientation."""
        q0 = planner.default_joint_state.position
        w0 = _waypoint_from_joint_config(planner, q0)
        w1 = GoalToolPose(
            tool_frames=w0.tool_frames,
            position=w0.position.clone(),
            quaternion=w0.quaternion.clone(),
        )
        w1.position[..., 0] += 0.15
        w1.position[..., 2] += dz
        return w0, w1

    def test_segment_bound_keeps_trajectory_on_plane(self, planner):
        start = self._start_state(planner)
        w0, w1 = self._planar_waypoints(planner, 0.0)
        frame = planner.tool_frames[0]
        z0 = w0.position[0, 0, 0, 0, 2].item()
        criteria = self._segment_criteria(planner, 1)
        criteria[0][frame].bound = ToolPoseBound.plane(
            "z", anchor="base", offset_position=[None, None, z0]
        )

        result = planner.plan_waypoints(
            [w0, w1], start, segment_tool_pose_criteria=criteria, max_attempts=5
        )

        assert bool(result.success.item()) is True
        traj = result.composed_trajectory
        names = traj.joint_names or planner.joint_names
        idx = [names.index(n) for n in planner.joint_names]
        positions = traj.position.view(-1, traj.position.shape[-1])[:, idx]
        tool = planner.compute_kinematics(
            JointState.from_position(positions, joint_names=planner.joint_names)
        ).tool_poses
        dz = (tool.position[:, 0, 0, 2] - z0).abs().max().item()
        # 2 mm feasibility tolerance plus slack for interpolation between optimized steps.
        assert dz < 0.003

    def test_default_bound_rejects_off_plane_waypoint(self, planner):
        start = self._start_state(planner)
        w0, w1 = self._planar_waypoints(planner, 0.05)
        frame = planner.tool_frames[0]
        z0 = w0.position[0, 0, 0, 0, 2].item()
        planner.update_tool_pose_bounds(
            {frame: ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, z0])}
        )
        try:
            result = planner.plan_waypoints(
                [w0, w1],
                start,
                segment_tool_pose_criteria=self._segment_criteria(planner, 1),
                max_attempts=1,
            )
        finally:
            planner.update_tool_pose_bounds({frame: None})

        assert bool(result.success.item()) is False
        assert "tool pose bounds" in result.status

    def test_too_few_waypoints_fails(self, planner):
        start = self._start_state(planner)
        single = _waypoint_from_joint_config(planner, planner.default_joint_state.position)
        result = planner.plan_waypoints(
            [single], start, segment_tool_pose_criteria=[]
        )
        assert bool(result.success.item()) is False
        assert "two waypoints" in result.status.lower()
