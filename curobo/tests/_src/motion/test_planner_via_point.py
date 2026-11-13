# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ViaPointMotionPlanner and its seeding and reference helpers."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from curobo._src.cost.tool_pose_bound import ToolPoseBound
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.planner_via_point import (
    ViaPointMotionPlanner,
    allocate_waypoint_indices,
    base_anchored_segment_bounds,
    interpolate_via_point_reference,
    knot_space_seeds,
)
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

CPU = DeviceCfg(device=torch.device("cpu"))
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _goal(positions, quaternion=IDENTITY):
    n = len(positions)
    return GoalToolPose(
        ["ee"],
        torch.tensor(positions, dtype=torch.float32).view(1, n, 1, 1, 3),
        torch.tensor([quaternion] * n, dtype=torch.float32).view(1, n, 1, 1, 4),
    )


def _start(position, quaternion=IDENTITY):
    return ToolPose(
        ["ee"],
        torch.tensor(position, dtype=torch.float32).view(1, 1, 1, 3),
        torch.tensor(quaternion, dtype=torch.float32).view(1, 1, 1, 4),
    )


# --------------------------------------------------------------------------- #
# Helpers (CPU)
# --------------------------------------------------------------------------- #


def test_allocate_waypoint_indices_follows_distance_and_keeps_a_minimum():
    indices = allocate_waypoint_indices([1.0, 3.0, 0.0], intervals=20, padding=4, min_gap=2)
    gaps = [b - a for a, b in zip([4] + indices[:-1], indices)]
    assert indices[-1] == 4 + 20
    assert min(gaps) >= 2 and gaps[1] > gaps[0] > gaps[2]
    # Zero total distance spreads the intervals evenly.
    assert allocate_waypoint_indices([0.0, 0.0], 10, 4, 2) == [9, 14]
    # More waypoints than the minimum allows lower the minimum to fit.
    assert allocate_waypoint_indices([1.0] * 5, 5, 0, 3) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        allocate_waypoint_indices([1.0] * 6, 5, 0, 1)


def test_reference_passes_the_waypoints_at_their_indices():
    waypoints = _goal([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    reference = interpolate_via_point_reference(_start([0.0, 0.0, 0.0]), waypoints, [4, 8], 11, 2)
    position = reference.position[0, :, 0, 0]
    assert reference.position.shape == (1, 11, 1, 1, 3)
    torch.testing.assert_close(position[:3], torch.zeros(3, 3))  # start padding and start
    torch.testing.assert_close(position[3], torch.tensor([0.5, 0.0, 0.0]))
    torch.testing.assert_close(position[4], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(position[6], torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(position[8:], torch.tensor([[1.0, 2.0, 0.0]] * 3))


def test_knot_space_seeds_pass_the_chain_at_its_progress():
    start = torch.zeros(2)
    chains = torch.tensor([[[2.0, 0.0], [2.0, 4.0]]])
    # Waypoints at native 6 and 10 of 10 intervals from 0: progress 0.6 and 1.0.
    seeds = knot_space_seeds(start, chains, [6, 10], 0, 10, num_knots=6)
    torch.testing.assert_close(seeds[0, 0], start)
    torch.testing.assert_close(seeds[0, 3], torch.tensor([2.0, 0.0]))  # knot progress 0.6
    torch.testing.assert_close(seeds[0, -1], chains[0, -1])


def test_knot_space_seeds_bulge_between_the_waypoints_only():
    start = torch.zeros(2)
    chains = torch.tensor([[[2.0, 0.0], [2.0, 4.0]]])
    bulge = torch.tensor([[[0.0, 1.0], [0.5, 0.0]]])
    straight = knot_space_seeds(start, chains, [5, 10], 0, 10, num_knots=5)
    seeds = knot_space_seeds(start, chains, [5, 10], 0, 10, num_knots=5, bulge=bulge)
    # Knots at progress 0, 0.25, 0.5, 0.75, 1: the waypoints are at 0.5 and 1.
    torch.testing.assert_close(seeds[0, [0, 2, 4]], straight[0, [0, 2, 4]], atol=1e-6, rtol=0)
    torch.testing.assert_close(seeds[0, 1] - straight[0, 1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(seeds[0, 3] - straight[0, 3], torch.tensor([0.5, 0.0]))


def test_segment_bounds_are_anchored_at_the_segment_ends():
    waypoints = _goal([[0.0, 0.0, 0.5], [0.0, 0.0, 0.2]])
    start = _start([0.0, 0.0, 0.9])
    below_goal = ToolPoseBound(anchor="goal", z=(None, 0.1))
    below_start = ToolPoseBound(anchor="start", z=(None, 0.0))
    criteria = [
        {"ee": ToolPoseCriteria(bounds=[below_start], device_cfg=CPU)},
        {"ee": ToolPoseCriteria(bounds=[below_goal, below_start], device_cfg=CPU)},
    ]
    bounds = base_anchored_segment_bounds(criteria, start, waypoints)
    upper = [bounds[0]["ee"][0].z, bounds[1]["ee"][0].z, bounds[1]["ee"][1].z]
    assert all(z[0] is None for z in upper)
    # Segment 0 starts at the current pose; segment 1 ends at waypoint 1 and starts at waypoint 0.
    assert [z[1] for z in upper] == pytest.approx([0.9, 0.3, 0.5], abs=1e-6)


# --------------------------------------------------------------------------- #
# Planner (GPU)
# --------------------------------------------------------------------------- #

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def planner():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = ViaPointMotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_table.yml",
        num_ik_seeds=32,
        use_cuda_graph=True,
    )
    planner = ViaPointMotionPlanner(config)
    planner.warmup(num_warmup_iterations=1)
    yield planner
    planner.destroy()


def _home(planner):
    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    return start, planner.compute_kinematics(start).tool_poses


def _offsets(planner, offsets):
    """Waypoints at offsets (m) from the home tool position, with the home orientation."""
    start, home = _home(planner)
    offsets = torch.tensor(offsets, dtype=torch.float32, device=home.position.device)
    n = offsets.shape[0]
    position = home.position[:, 0:1] + offsets.view(1, n, 1, 3)
    quaternion = home.quaternion[:, 0:1].expand(-1, n, -1, -1)
    waypoints = GoalToolPose(
        planner.tool_frames,
        position.unsqueeze(3).contiguous(),
        quaternion.unsqueeze(3).contiguous(),
    )
    return start, waypoints


def _native_tool_positions(planner, result):
    position = result.trajopt_result.js_solution.position
    solution = JointState.from_position(
        position.reshape(-1, *position.shape[-2:])[0].contiguous(),
        joint_names=result.trajopt_result.js_solution.joint_names,
    )
    active = planner._motion_planner.trajopt_solver.get_active_js(solution)
    return planner.compute_kinematics(active).tool_poses.position[:, 0, 0]


@cuda
def test_plan_reaches_every_waypoint_without_stopping(planner):
    start, waypoints = _offsets(planner, [[0.15, 0.0, 0.0], [0.15, 0.2, 0.0], [0.0, 0.2, -0.1]])
    result = planner.plan_waypoints(waypoints, start)
    assert result.success.item(), result.status
    assert result.waypoint_position_error.max() <= 0.005 + 1e-4
    assert result.waypoint_rotation_error.max() <= 0.05 + 1e-3
    # The robot keeps moving through the intermediate waypoints.
    velocity = result.trajopt_result.js_solution.velocity
    velocity = velocity.reshape(-1, *velocity.shape[-2:])[0]
    assert velocity[result.waypoint_indices[0]].abs().max() > 1e-3
    trajectory = result.get_interpolated_plan()
    assert trajectory.position.shape[-1] == len(trajectory.joint_names)
    assert set(planner.joint_names) <= set(trajectory.joint_names)


@cuda
def test_segment_criteria_constrain_only_their_segment(planner):
    """A straight-line segment stays on its line; the free segment before it does not need to."""
    start, home = _home(planner)
    # The line runs 15 cm along the home tool axis, which linear_motion("z") keeps free.
    w, x, y, z = home.quaternion[0, 0, 0].tolist()
    axis = torch.tensor(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        device=home.position.device,
    )
    first = home.position[0, 0, 0] + torch.tensor([0.15, 0.2, 0.05], device=axis.device)
    position = torch.stack([first, first + 0.15 * axis]).view(1, 2, 1, 1, 3)
    quaternion = home.quaternion[:, 0:1].expand(-1, 2, -1, -1).unsqueeze(3)
    waypoints = GoalToolPose(planner.tool_frames, position, quaternion.contiguous())
    straight = replace(
        ToolPoseCriteria.linear_motion("z", device_cfg=planner._motion_planner.device_cfg),
        path_tolerance=(0.005, 2.0 * math.sin(0.05 / 2.0)),
        enforce_path_tolerance=True,
    )
    frame = planner.tool_frames[0]
    result = planner.plan_waypoints(waypoints, start, [None, {frame: straight}])
    assert result.success.item(), result.status
    positions = _native_tool_positions(planner, result)
    begin, end = result.waypoint_indices
    offset = positions[begin + 1 : end] - first
    off_line = offset - (offset @ axis).unsqueeze(-1) * axis
    # Path samples stay within the path tolerance of the line through the reference.
    assert off_line.norm(dim=-1).max() <= 0.005 + 1e-3


@cuda
def test_segment_bounds_hold_inside_their_segment(planner):
    start, waypoints = _offsets(planner, [[0.15, 0.0, -0.1], [0.15, 0.25, -0.1]])
    _, home = _home(planner)
    height = float(home.position[0, 0, 0, 2]) - 0.1
    frame = planner.tool_frames[0]
    # Segment 1 must stay within 1 cm of the waypoint height while it moves sideways.
    flat = ToolPoseCriteria(
        bounds=[ToolPoseBound(anchor="base", z=(height - 0.01, height + 0.01))],
        device_cfg=planner._motion_planner.device_cfg,
    )
    result = planner.plan_waypoints(waypoints, start, [None, {frame: flat}])
    assert result.success.item(), result.status
    positions = _native_tool_positions(planner, result)
    first, last = result.waypoint_indices
    z = positions[first : last + 1, 2]
    assert (z >= height - 0.011).all() and (z <= height + 0.011).all()


@cuda
def test_a_bound_no_waypoint_ik_satisfies_fails_before_optimization(planner):
    start, waypoints = _offsets(planner, [[0.15, 0.0, 0.0]])
    frame = planner.tool_frames[0]
    unreachable = ToolPoseCriteria(
        bounds=[ToolPoseBound(anchor="base", z=(None, -1.0))],
        device_cfg=planner._motion_planner.device_cfg,
    )
    result = planner.plan_waypoints(waypoints, start, [{frame: unreachable}], max_attempts=1)
    assert not result.success.item()
    assert "bounds" in result.status and result.trajopt_result is None
    # The planner is not left with the failed plan's bounds.
    result = planner.plan_waypoints(waypoints, start)
    assert result.success.item(), result.status


@cuda
def test_back_and_forth_path_reaches_every_waypoint(planner):
    """Large swings miss the first waypoints at the default geometry time step; a retry with a
    slower geometric optimization passes them."""
    offsets = [[0.076, -0.214, 0.073], [0.144, 0.222, -0.004], [0.05, -0.235, -0.08]]
    start, waypoints = _offsets(planner, offsets + [[0.073, -0.228, 0.218]])
    result = planner.plan_waypoints(waypoints, start)
    assert result.success.item(), result.status
    assert result.waypoint_position_error.max() <= 0.005 + 1e-4


# --------------------------------------------------------------------------- #
# Joints with more than one turn of range (UR10e)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ur_planner():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = ViaPointMotionPlannerCfg.create(
        robot="ur10e.yml", scene_model="collision_table.yml", use_cuda_graph=True
    )
    planner = ViaPointMotionPlanner(config)
    planner.warmup(num_warmup_iterations=1)
    yield planner
    planner.destroy()


@cuda
def test_tool_roll_beyond_the_joint_limit_does_not_unwind(ur_planner):
    """The tool rolls on by 90, 260 and 360 deg from wrist 3 at 0, but 360 deg is beyond the
    joint limit. Turning on to 260 deg forces 260 deg back to 0 (520 deg in total); the planner
    reaches the second waypoint the other way round, at -100 deg (380 deg in total)."""
    planner = ur_planner
    wrist = planner.joint_names.index("wrist_3_joint")
    home = planner.default_joint_state.position.view(-1).clone()
    home[wrist] = 0.0
    start = JointState.from_position(home.view(1, -1), joint_names=planner.joint_names)
    configs = home.repeat(3, 1)
    configs[:, wrist] = torch.tensor([90.0, 260.0, 360.0], device=home.device) * math.pi / 180.0
    poses = planner.compute_kinematics(
        JointState.from_position(configs, joint_names=planner.joint_names)
    ).tool_poses
    waypoints = GoalToolPose(
        planner.tool_frames,
        poses.position.view(1, 3, 1, 1, 3).contiguous(),
        poses.quaternion.view(1, 3, 1, 1, 4).contiguous(),
    )
    result = planner.plan_waypoints(waypoints, start)
    assert result.success.item(), result.status
    chain = torch.cat((home.view(1, -1), result.waypoint_configs))
    length = float((chain[1:] - chain[:-1]).norm(dim=-1).sum())
    assert length < 1.15 * math.radians(380.0)
