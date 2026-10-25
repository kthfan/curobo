# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ToolPoseBound and its PyTorch evaluation helpers."""

import math

import pytest
import torch

from curobo._src.cost.tool_pose_bound import (
    UNBOUNDED,
    ToolPoseBound,
    ToolPoseBoundAnchor,
    _quat_mul,
    find_tool_pose_bound_violations,
    swing_twist_components,
    tool_pose_bound_violation,
)
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.tool_pose import GoalToolPose, ToolPose


def axis_angle(axis, angle):
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    return torch.cat([torch.tensor([math.cos(angle / 2)]), a * math.sin(angle / 2)])


IDENTITY = axis_angle([1.0, 0.0, 0.0], 0.0)


class TestToolPoseBoundValidation:
    def test_anchor_from_string(self):
        assert ToolPoseBound(anchor="base").anchor == ToolPoseBoundAnchor.BASE

    def test_unknown_anchor_raises(self):
        with pytest.raises(ValueError):
            ToolPoseBound(anchor="tool")

    def test_reference_outside_position_interval_raises(self):
        with pytest.raises(ValueError, match="does not contain the reference"):
            ToolPoseBound(position_lower=[None, None, 0.01])

    def test_reference_outside_rotation_interval_raises(self):
        with pytest.raises(ValueError, match="does not contain the reference"):
            ToolPoseBound(rotation_upper=[None, None, -0.1])

    def test_lower_above_upper_raises(self):
        with pytest.raises(ValueError):
            ToolPoseBound(position_lower=[-0.1, None, None], position_upper=[-0.2, None, None])

    def test_rotation_bound_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ToolPoseBound(rotation_upper=[None, None, 4.0])

    def test_unfilled_axes_are_unchecked(self):
        bound = ToolPoseBound(position_lower=[None, -0.1, None], position_upper=[None, 0.2, None])
        _, _, lo, hi, rot_lo, rot_hi, max_swing = bound.stacked_values(use_tolerance=False)
        assert lo == [-UNBOUNDED, -0.1, -UNBOUNDED]
        assert hi == [UNBOUNDED, 0.2, UNBOUNDED]
        assert rot_lo == [-UNBOUNDED] * 3 and rot_hi == [UNBOUNDED] * 3
        assert max_swing == [UNBOUNDED]
        assert not bound.uses_rotation

    def test_tolerance_widens_only_filled_bounds(self):
        bound = ToolPoseBound(
            position_lower=[0.0, None, None],
            position_upper=[0.0, None, None],
            max_swing=0.1,
            position_tolerance=0.002,
            rotation_tolerance=0.01,
        )
        _, _, lo, hi, _, _, max_swing = bound.stacked_values(use_tolerance=True)
        assert lo[0] == pytest.approx(-0.002) and hi[0] == pytest.approx(0.002)
        assert lo[1] == -UNBOUNDED
        assert max_swing[0] == pytest.approx(0.11)

    def test_offset_defaults(self):
        offset_p, offset_q, *_ = ToolPoseBound(offset_position=[None, 0.1, None]).stacked_values(
            use_tolerance=False
        )
        assert offset_p == [0.0, 0.1, 0.0]
        assert offset_q == [1.0, 0.0, 0.0, 0.0]

    def test_plane_and_line_factories(self):
        plane = ToolPoseBound.plane("z", tolerance=0.001)
        assert plane.position_lower == [None, None, 0.0]
        line = ToolPoseBound.line("z", lower=-0.15, upper=0.0)
        assert line.position_lower == [0.0, 0.0, -0.15]
        assert line.position_upper == [0.0, 0.0, 0.0]

    def test_criteria_copy_clears_bound(self, cpu_device_cfg):
        criteria = ToolPoseCriteria(bound=ToolPoseBound.plane(), device_cfg=cpu_device_cfg)
        criteria.copy_(ToolPoseCriteria(device_cfg=cpu_device_cfg))
        assert criteria.bound is None
        bounded = ToolPoseCriteria(bound=ToolPoseBound.plane(), device_cfg=cpu_device_cfg)
        assert bounded.clone().bound is not None


class TestSwingTwist:
    def test_separates_tilt_from_twist(self):
        # Tilt 10 degrees about x, then twist 170 degrees about the tool z axis.
        q = _quat_mul(
            axis_angle([1, 0, 0], math.radians(10)), axis_angle([0, 0, 1], math.radians(170))
        )
        components, angle = swing_twist_components(q)
        assert components[0].item() == pytest.approx(math.radians(10), abs=1e-5)
        assert components[1].item() == pytest.approx(0.0, abs=1e-5)
        assert components[2].item() == pytest.approx(math.radians(170), abs=1e-5)
        assert angle.item() == pytest.approx(math.radians(10), abs=1e-5)

    def test_sign_invariant(self):
        q = axis_angle([0.3, -0.2, 0.9], 1.1)
        a, _ = swing_twist_components(q)
        b, _ = swing_twist_components(-q)
        assert torch.allclose(a, b, atol=1e-6)

    def test_half_turn_swing_has_zero_twist(self):
        components, angle = swing_twist_components(axis_angle([1, 0, 0], math.pi))
        assert components[2].item() == 0.0
        assert angle.item() == pytest.approx(math.pi, abs=1e-5)


class TestViolation:
    def test_plane_violation_with_offset_reference(self):
        bound = ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, 0.8])
        position = torch.tensor([[0.3, -0.2, 0.83]])
        violation = tool_pose_bound_violation(
            bound, position, IDENTITY.unsqueeze(0), torch.zeros(1, 3), IDENTITY.unsqueeze(0)
        )
        assert violation[0, 2].item() == pytest.approx(0.03, abs=1e-6)
        assert violation[0, [0, 1, 3, 4, 5, 6]].abs().sum().item() == 0.0

    def test_reference_rotated_by_anchor(self):
        # Line along the anchor's x axis, anchor yawed by 90 degrees: world y is the line.
        bound = ToolPoseBound.line("x", lower=-1.0, upper=1.0)
        anchor_q = axis_angle([0, 0, 1], math.pi / 2).unsqueeze(0)
        on_line = tool_pose_bound_violation(
            bound,
            torch.tensor([[0.0, 0.5, 0.0]]),
            IDENTITY.unsqueeze(0),
            torch.zeros(1, 3),
            anchor_q,
        )
        off_line = tool_pose_bound_violation(
            bound,
            torch.tensor([[0.5, 0.0, 0.0]]),
            IDENTITY.unsqueeze(0),
            torch.zeros(1, 3),
            anchor_q,
        )
        assert on_line.sum().item() == pytest.approx(0.0, abs=1e-6)
        assert off_line[0, 1].item() == pytest.approx(0.5, abs=1e-6)

    def test_tolerance_accepts_small_violation(self):
        bound = ToolPoseBound.plane("z", anchor="base", tolerance=0.005)
        position = torch.tensor([[0.0, 0.0, 0.004]])
        args = (position, IDENTITY.unsqueeze(0), torch.zeros(1, 3), IDENTITY.unsqueeze(0))
        assert tool_pose_bound_violation(bound, *args, use_tolerance=True).sum().item() == 0.0
        assert tool_pose_bound_violation(bound, *args, use_tolerance=False)[0, 2].item() > 0.0


def _poses(start_p, goal_p, goal_q=None):
    start = ToolPose(
        tool_frames=["ee"],
        position=start_p.view(-1, 1, 1, 3),
        quaternion=IDENTITY.expand(start_p.shape[0], 1, 1, 4).contiguous(),
    )
    n, g = goal_p.shape[0], goal_p.shape[1]
    if goal_q is None:
        goal_q = IDENTITY.expand(n, 1, 1, g, 4).contiguous()
    goal = GoalToolPose(tool_frames=["ee"], position=goal_p.view(n, 1, 1, g, 3), quaternion=goal_q)
    return start, goal


class TestFindViolations:
    def test_all_satisfied(self):
        bound = ToolPoseBound.plane("z", anchor="base")
        start, goal = _poses(torch.tensor([[0.1, 0.0, 0.0]]), torch.tensor([[[0.5, 0.2, 0.0]]]))
        assert find_tool_pose_bound_violations({"ee": bound}, start, goal) == []

    def test_start_violation_reported_per_problem(self):
        bound = ToolPoseBound.plane("z", anchor="base", tolerance=0.001)
        start, goal = _poses(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
            torch.tensor([[[0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
        )
        messages = find_tool_pose_bound_violations({"ee": bound}, start, goal)
        assert len(messages) == 1
        assert "problem 1" in messages[0] and "start" in messages[0] and "z by" in messages[0]

    def test_goal_violation_against_start_anchor(self):
        bound = ToolPoseBound.plane("z", anchor="start")
        start, goal = _poses(torch.tensor([[0.0, 0.0, 0.2]]), torch.tensor([[[0.5, 0.0, 0.0]]]))
        messages = find_tool_pose_bound_violations({"ee": bound}, start, goal)
        assert len(messages) == 1 and "goal violates" in messages[0]

    def test_goalset_needs_one_feasible_goal(self):
        bound = ToolPoseBound.plane("z", anchor="base")
        start = torch.tensor([[0.0, 0.0, 0.0]])
        one_feasible = torch.tensor([[[0.5, 0.0, 0.3], [0.5, 0.0, 0.0]]])
        none_feasible = torch.tensor([[[0.5, 0.0, 0.3], [0.5, 0.0, 0.1]]])
        assert find_tool_pose_bound_violations({"ee": bound}, *_poses(start, one_feasible)) == []
        assert (
            len(find_tool_pose_bound_violations({"ee": bound}, *_poses(start, none_feasible))) == 1
        )

    def test_goal_anchor_start_feasible_for_any_goal(self):
        # Line along goal z: start lies on the line of the second goal only.
        bound = ToolPoseBound.line("z", lower=-0.2, upper=0.0)
        start = torch.tensor([[1.0, 0.0, -0.1]])
        goals = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        assert find_tool_pose_bound_violations({"ee": bound}, *_poses(start, goals)) == []
