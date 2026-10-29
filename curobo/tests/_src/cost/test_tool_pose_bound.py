# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ToolPoseBound and its PyTorch reference implementation."""

import math

import pytest
import torch

from curobo._src.cost.tool_pose_bound import (
    METRIC_EPSILON,
    UNBOUNDED,
    ToolPoseBound,
    ToolPoseBoundAnchor,
    _quat_mul,
    find_tool_pose_bound_violations,
    normalize_tool_pose_bounds,
    tool_pose_bound_components,
    tool_pose_bound_violation,
)
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])
ORIGIN = torch.zeros(3)
DEG = math.pi / 180.0


def rotation(axis, angle):
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    return torch.cat([torch.tensor([math.cos(angle / 2)]), a * math.sin(angle / 2)])


def components(bound, position=ORIGIN, quaternion=IDENTITY, anchor_p=ORIGIN, anchor_q=IDENTITY):
    c, parallel = tool_pose_bound_components(bound, position, quaternion, anchor_p, anchor_q)
    return c, parallel


class TestValidation:
    def test_anchor_from_string(self):
        assert ToolPoseBound(anchor="base", z=0.0).anchor == ToolPoseBoundAnchor.BASE

    def test_invalid_anchor_raises(self):
        with pytest.raises(ValueError, match="anchor"):
            ToolPoseBound(anchor="world", z=0.0)

    def test_invalid_reference_axes_raises(self):
        with pytest.raises(ValueError, match="reference_axes"):
            ToolPoseBound(reference_axes="world", z=0.0)

    def test_no_limits_raises(self):
        with pytest.raises(ValueError, match="no limits"):
            ToolPoseBound()

    def test_number_is_equality(self):
        assert ToolPoseBound(z=0.4).z == (0.4, 0.4)

    def test_open_ends(self):
        bound = ToolPoseBound(z=(0.0, None), x=(None, None))
        assert bound.z == (0.0, None)
        assert bound.x is None

    def test_lower_above_upper_raises(self):
        with pytest.raises(ValueError, match="lower limit"):
            ToolPoseBound(x=(0.1, -0.1))

    def test_tilt_outside_range_raises(self):
        with pytest.raises(ValueError, match="tilt"):
            ToolPoseBound(tilt=(None, 4.0))

    def test_yaw_needs_both_ends(self):
        with pytest.raises(ValueError, match="both"):
            ToolPoseBound(tilt=(1.0, 1.2), yaw=(0.0, None))

    def test_yaw_span_at_most_two_pi(self):
        with pytest.raises(ValueError, match="2\\*pi"):
            ToolPoseBound(tilt=(1.0, 1.2), roll=(-4.0, 4.0))

    @pytest.mark.parametrize("tilt", [None, (0.0, 1.0), (1.0, 3.1), (0.1, None)])
    def test_yaw_or_roll_needs_tilt_away_from_parallel(self, tilt):
        with pytest.raises(ValueError, match="parallel"):
            ToolPoseBound(tilt=tilt, roll=(-0.1, 0.1))

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="rotation_tolerance"):
            ToolPoseBound(tilt=(None, 0.1), rotation_tolerance=-0.1)

    def test_rotations_are_normalized(self):
        bound = ToolPoseBound(z=0.0, reference_rotation=[2.0, 0.0, 0.0, 0.0])
        assert bound.reference_rotation == [1.0, 0.0, 0.0, 0.0]
        assert bound.tool_rotation == [1.0, 0.0, 0.0, 0.0]

    def test_normalize_bounds(self):
        bound = ToolPoseBound(z=0.0)
        assert normalize_tool_pose_bounds(None) == []
        assert normalize_tool_pose_bounds(bound) == [bound]
        assert normalize_tool_pose_bounds((bound, bound)) == [bound, bound]
        with pytest.raises(ValueError):
            normalize_tool_pose_bounds([bound, "plane"])


class TestKernelValues:
    """Cost and feasibility intervals ``(center, lower, upper)`` per component."""

    def test_unlimited_component(self):
        center, lower, upper = ToolPoseBound(anchor="base", z=0.0).kernel_values(metric=True)
        assert (center[0], lower[0], upper[0]) == (0.0, -UNBOUNDED, UNBOUNDED)

    def test_inequality(self):
        bound = ToolPoseBound(anchor="base", z=(0.0, 0.1), position_tolerance=0.01)
        _, lower, upper = bound.kernel_values(metric=False)
        assert (lower[2], upper[2]) == pytest.approx((0.005, 0.095))
        _, lower, upper = bound.kernel_values(metric=True)
        assert (lower[2], upper[2]) == pytest.approx((-METRIC_EPSILON, 0.1 + METRIC_EPSILON))

    def test_equality(self):
        bound = ToolPoseBound(anchor="base", z=(0.2, 0.205), position_tolerance=0.01)
        _, lower, upper = bound.kernel_values(metric=False)
        assert lower[2] == upper[2] == pytest.approx(0.2025)
        _, lower, upper = bound.kernel_values(metric=True)
        assert (lower[2], upper[2]) == pytest.approx(
            (0.19 - METRIC_EPSILON, 0.215 + METRIC_EPSILON)
        )

    def test_one_sided(self):
        bound = ToolPoseBound(anchor="base", z=(0.0, None), position_tolerance=0.01)
        _, lower, upper = bound.kernel_values(metric=False)
        assert (lower[2], upper[2]) == (pytest.approx(0.005), UNBOUNDED)

    def test_trivial_tilt_sides_are_unlimited(self):
        bound = ToolPoseBound(tilt=(0.0, math.pi))
        _, lower, upper = bound.kernel_values(metric=True)
        assert (lower[3], upper[3]) == (-UNBOUNDED, UNBOUNDED)

    def test_wrapped_interval_is_centered(self):
        bound = ToolPoseBound(tilt=(1.0, 2.0), roll=(3.0, 3.4), rotation_tolerance=0.0)
        center, lower, upper = bound.kernel_values(metric=False)
        assert center[5] == pytest.approx(3.2)
        assert (lower[5], upper[5]) == pytest.approx((-0.2, 0.2))

    def test_goal_on_edge_keeps_zero_cost(self):
        bound = ToolPoseBound(anchor="goal", z=(-0.12, 0.0), position_tolerance=0.01)
        _, lower, upper = bound.kernel_values(metric=False)
        assert (lower[2], upper[2]) == pytest.approx((-0.115, 0.0))

    def test_base_anchor_shrinks_both_sides(self):
        bound = ToolPoseBound(anchor="base", z=(-0.12, 0.0), position_tolerance=0.01)
        _, lower, upper = bound.kernel_values(metric=False)
        assert (lower[2], upper[2]) == pytest.approx((-0.115, -0.005))


class TestAngles:
    """``R_FU = Rz(yaw) Ry(tilt) Rz(roll)`` on a tool whose z axis is pitched up 20 degrees."""

    def setup_method(self):
        self.bound = ToolPoseBound(anchor="base", tilt=(0.5, 1.5))
        # Tool z along base x, then pitched up 20 degrees: R_y(70 deg).
        self.pitched = rotation([0, 1, 0], 70 * DEG)

    def angles(self, quaternion):
        c, parallel = components(self.bound, quaternion=quaternion)
        assert not parallel
        return [a / DEG for a in c[3:].tolist()]

    def test_pitch(self):
        assert self.angles(self.pitched) == pytest.approx([70.0, 0.0, 0.0], abs=1e-3)

    def test_yaw_about_reference_axis(self):
        q = _quat_mul(rotation([0, 0, 1], 30 * DEG), self.pitched)
        assert self.angles(q) == pytest.approx([70.0, 30.0, 0.0], abs=1e-3)

    def test_roll_about_tool_axis(self):
        q = _quat_mul(self.pitched, rotation([0, 0, 1], 30 * DEG))
        assert self.angles(q) == pytest.approx([70.0, 0.0, 30.0], abs=1e-3)

    def test_yaw_and_roll_leave_tilt(self):
        q = _quat_mul(
            _quat_mul(rotation([0, 0, 1], 40 * DEG), self.pitched), rotation([0, 0, 1], -45 * DEG)
        )
        assert self.angles(q) == pytest.approx([70.0, 40.0, -45.0], abs=1e-3)

    def test_parallel_axes_are_flagged(self):
        _, parallel = components(self.bound, quaternion=rotation([0, 0, 1], 0.3))
        assert parallel

    def test_reference_axes_anchor_vs_base(self):
        anchor_q = rotation([1, 0, 0], 30 * DEG)
        follows = ToolPoseBound(anchor="goal", tilt=(None, 0.1))
        fixed = ToolPoseBound(anchor="goal", reference_axes="base", tilt=(None, 0.1))
        c, _ = components(follows, quaternion=anchor_q, anchor_q=anchor_q)
        assert c[3].item() == pytest.approx(0.0, abs=1e-4)
        c, _ = components(fixed, quaternion=anchor_q, anchor_q=anchor_q)
        assert c[3].item() == pytest.approx(30 * DEG, abs=1e-4)

    def test_tool_rotation_does_not_move_position(self):
        bound = ToolPoseBound(
            anchor="base", x=0.0, tool_rotation=rotation([0, 1, 0], 1.0).tolist()
        )
        c, _ = components(bound, position=torch.tensor([0.1, 0.2, 0.3]))
        assert c[:3].tolist() == pytest.approx([0.1, 0.2, 0.3])


class TestFactories:
    def test_plane_through_point(self):
        normal = [0.0, -math.sin(0.5), math.cos(0.5)]
        bound = ToolPoseBound.plane(normal, point=[0.5, 0.0, 0.4], anchor="base", tolerance=0.002)
        on_plane = torch.tensor([0.5, 0.0, 0.4]) + 0.3 * torch.tensor([1.0, 0.0, 0.0])
        off_plane = on_plane + 0.01 * torch.tensor(normal)
        assert tool_pose_bound_violation(bound, on_plane, IDENTITY, ORIGIN, IDENTITY).max() == 0.0
        violation = tool_pose_bound_violation(bound, off_plane, IDENTITY, ORIGIN, IDENTITY)
        assert violation[2].item() == pytest.approx(0.008, abs=1e-4)

    def test_plane_half_space(self):
        bound = ToolPoseBound.plane(max_distance=None, anchor="base")
        below = tool_pose_bound_violation(
            bound, torch.tensor([0.0, 0.0, -0.1]), IDENTITY, ORIGIN, IDENTITY
        )
        above = tool_pose_bound_violation(
            bound, torch.tensor([0.0, 0.0, 0.5]), IDENTITY, ORIGIN, IDENTITY
        )
        assert below[2].item() > 0.09 and above.max() == 0.0

    def test_line_segment(self):
        direction = torch.tensor([0.0, 1.0, 1.0]) / math.sqrt(2.0)
        point = torch.tensor([0.1, 0.0, 0.2])
        bound = ToolPoseBound.line(direction.tolist(), point.tolist(), -0.1, 0.1, anchor="base")
        inside = point + 0.05 * direction
        beyond = point + 0.2 * direction
        aside = inside + torch.tensor([0.01, 0.0, 0.0])
        violation = [
            tool_pose_bound_violation(bound, p, IDENTITY, ORIGIN, IDENTITY)
            for p in (inside, beyond, aside)
        ]
        assert violation[0].max() == 0.0
        assert violation[1][2].item() == pytest.approx(0.1, abs=1e-4)
        assert violation[2][:2].max().item() == pytest.approx(0.008, abs=1e-4)

    def test_axis_tilt_fixed_elevation(self):
        # Approach axis (tool z) 20 degrees above the horizon, in any heading and roll.
        bound = ToolPoseBound.axis_tilt(min_angle=70 * DEG, max_angle=70 * DEG, anchor="base")
        pitched = rotation([0, 1, 0], 70 * DEG)
        for q in (
            pitched,
            _quat_mul(rotation([0, 0, 1], 2.0), pitched),
            _quat_mul(pitched, rotation([0, 0, 1], -1.0)),
        ):
            assert tool_pose_bound_violation(bound, ORIGIN, q, ORIGIN, IDENTITY).max() == 0.0
        steeper = rotation([0, 1, 0], 60 * DEG)
        assert tool_pose_bound_violation(bound, ORIGIN, steeper, ORIGIN, IDENTITY)[3] > 0.0

    def test_axis_tilt_other_tool_axis(self):
        bound = ToolPoseBound.axis_tilt(
            (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), max_angle=0.1, anchor="base"
        )
        x_up = rotation([0, 1, 0], -90 * DEG)
        assert tool_pose_bound_violation(bound, ORIGIN, x_up, ORIGIN, IDENTITY).max() == 0.0
        assert tool_pose_bound_violation(bound, ORIGIN, IDENTITY, ORIGIN, IDENTITY)[3] > 1.0

    @pytest.mark.parametrize("axis, index", [([1, 0, 0], 5), ([0, 1, 0], 3), ([0, 0, 1], 4)])
    def test_orientation_box_axes(self, axis, index):
        """Rotations about the goal x, y, z axes map to roll, tilt - 90 deg and yaw."""
        bound = ToolPoseBound.orientation_box(0.1, 0.2, 0.3)
        goal = rotation([0.3, -0.2, 0.9], 0.8)
        q = _quat_mul(goal, rotation(axis, 0.05))
        c, parallel = components(bound, quaternion=q, anchor_q=goal)
        expected = [0.0, 0.0, 0.0]
        expected[{5: 2, 3: 0, 4: 1}[index]] = 0.05
        assert not parallel
        assert [c[3].item() - math.pi / 2, c[4].item(), c[5].item()] == pytest.approx(
            expected, abs=1e-4
        )
        assert tool_pose_bound_violation(bound, ORIGIN, q, ORIGIN, goal).max() == 0.0

    def test_orientation_box_limits(self):
        bound = ToolPoseBound.orientation_box(0.1, 0.2, 0.3, tolerance=0.0)
        for axis, limit in (([1, 0, 0], 0.1), ([0, 1, 0], 0.2), ([0, 0, 1], 0.3)):
            inside = rotation(axis, limit - 0.01)
            outside = rotation(axis, limit + 0.05)
            assert tool_pose_bound_violation(bound, ORIGIN, inside, ORIGIN, IDENTITY).max() == 0.0
            assert tool_pose_bound_violation(bound, ORIGIN, outside, ORIGIN, IDENTITY).max() > 0.04

    def test_orientation_box_pitch_limit(self):
        with pytest.raises(ValueError, match="max_pitch"):
            ToolPoseBound.orientation_box(max_pitch=1.5)
        assert ToolPoseBound.orientation_box(max_roll=0.1).tilt == pytest.approx(
            (10 * DEG, 170 * DEG)
        )


class TestFindViolations:
    def setup_method(self):
        self.start = ToolPose(
            tool_frames=["ee"],
            position=torch.zeros(1, 1, 1, 3),
            quaternion=IDENTITY.view(1, 1, 1, 4).clone(),
        )

    def goal(self, *zs):
        position = torch.zeros(1, 1, 1, len(zs), 3)
        position[..., 2] = torch.tensor(zs)
        quaternion = IDENTITY.view(1, 1, 1, 1, 4).repeat(1, 1, 1, len(zs), 1)
        return GoalToolPose(tool_frames=["ee"], position=position, quaternion=quaternion)

    def test_feasible(self):
        bounds = {
            "ee": [
                ToolPoseBound.plane(anchor="base"),
                ToolPoseBound.axis_tilt(max_angle=0.1, anchor="start"),
            ]
        }
        assert find_tool_pose_bound_violations(bounds, self.start, self.goal(0.0)) == []

    def test_goal_violation_names_bound_and_component(self):
        bounds = {
            "ee": [
                ToolPoseBound.axis_tilt(max_angle=0.1, anchor="base"),
                ToolPoseBound.plane(anchor="base"),
            ]
        }
        messages = find_tool_pose_bound_violations(bounds, self.start, self.goal(0.05))
        assert len(messages) == 1
        assert "goal violates bound 1: z by" in messages[0]

    def test_start_violation(self):
        bounds = {"ee": ToolPoseBound.plane(point=[0.0, 0.0, 0.1], anchor="base")}
        messages = find_tool_pose_bound_violations(bounds, self.start, self.goal(0.1))
        assert len(messages) == 1 and "start violates" in messages[0]

    def test_one_feasible_goal_in_set_is_enough(self):
        bounds = {"ee": ToolPoseBound.plane(anchor="base")}
        assert find_tool_pose_bound_violations(bounds, self.start, self.goal(0.3, 0.0)) == []

    def test_goal_anchored_start_uses_best_goal(self):
        # The start lies on the plane of the second goal only.
        bounds = {"ee": ToolPoseBound.plane()}
        assert find_tool_pose_bound_violations(bounds, self.start, self.goal(0.3, 0.0)) == []
        messages = find_tool_pose_bound_violations(bounds, self.start, self.goal(0.3))
        assert "start violates" in messages[0]
