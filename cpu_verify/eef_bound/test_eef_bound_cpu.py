# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for the defects found in review (see ``REVIEW.md``).

NOT for main: see ``cpu_verify/README.md``.

The redesign (tool pose bounds) fixed D1-D5; D2 and D3 went away with the margin API and the
PyTorch reference class. The remaining checks guard D1 and D4 in the frame the robot consumes.
The library's own CPU-capable unit tests live in ``curobo/tests/_src/cost``.

Run::

    python -m pytest cpu_verify
"""

from __future__ import annotations

import math

import pytest
import torch

from cpu_adapter import (
    IDENTITY,
    analytic_gradient,
    evaluate,
    finite_difference_world,
    init,
    make_cost,
    quat_axis_angle,
    random_quat,
)
from curobo._src.cost.tool_pose_bound import ToolPoseBound, _quat_mul
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

ORIGIN = torch.zeros(3)


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu():
    init()


@pytest.mark.parametrize("goal_yaw", [0.0, 0.5, 1.2, 2.5])
def test_d1_rotation_gradient_world_frame_rotated_goal(goal_yaw: float):
    """D1: the orientation gradient is in the base frame for any goal orientation."""
    bound = ToolPoseBound.orientation_box(0.02, 0.02, 0.02)
    cost = make_cost({"ee": bound})
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], goal_yaw)
    quaternion = _quat_mul(goal_q, quat_axis_angle([1.0, 0.4, 0.2], 0.6))

    def fn(p, q):
        return evaluate(cost, p, q, ORIGIN, goal_q)

    _, fd = finite_difference_world(fn, ORIGIN, quaternion)
    _, omega = analytic_gradient(cost, ORIGIN, quaternion, ORIGIN, goal_q)
    assert fd.norm() > 0.1
    assert torch.allclose(omega, fd, atol=5e-3, rtol=1e-2), (omega, fd)


def test_d1_large_rotation_gradient_is_exact():
    """The old rotvec corridor ignored the log-map Jacobian; the tilt gradient is exact."""
    bound = ToolPoseBound.axis_tilt(max_angle=0.1)
    cost = make_cost({"ee": bound})
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], 0.7)
    quaternion = _quat_mul(goal_q, quat_axis_angle([1.0, 1.0, 0.3], 2.6))

    def fn(p, q):
        return evaluate(cost, p, q, ORIGIN, goal_q)

    _, fd = finite_difference_world(fn, ORIGIN, quaternion)
    _, omega = analytic_gradient(cost, ORIGIN, quaternion, ORIGIN, goal_q)
    cosine = torch.nn.functional.cosine_similarity(omega, fd, dim=0).item()
    assert cosine > 0.999


def test_d4_goal_batch_smaller_than_rollout_batch():
    """D4: rollout rows look up their goal through idxs_goal (num_seeds > 1)."""
    gen = torch.Generator().manual_seed(2)
    cost = make_cost({"ee": ToolPoseBound.line(min_along=-0.3, max_along=0.0)})
    batch, horizon = 4, 5
    position = torch.randn(batch, horizon, 1, 3, generator=gen) * 0.3
    quaternion = random_quat(batch * horizon, gen).view(batch, horizon, 1, 4)
    goal_position = torch.randn(2, 1, 1, 1, 3, generator=gen) * 0.3
    goal_quaternion = random_quat(2, gen).view(2, 1, 1, 1, 4)
    idxs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    batched = cost.forward(
        ToolPose(["ee"], position, quaternion),
        GoalToolPose(["ee"], goal_position, goal_quaternion),
        idxs,
    ).clone()
    for b in range(batch):
        g = int(idxs[b])
        single = cost.forward(
            ToolPose(["ee"], position[b : b + 1], quaternion[b : b + 1]),
            GoalToolPose(["ee"], goal_position[g : g + 1], goal_quaternion[g : g + 1]),
            torch.zeros(1, dtype=torch.int32),
        )
        assert torch.allclose(batched[b : b + 1], single, atol=1e-6)


def test_plane_allows_motion_in_plane_only():
    """A plane bound (one constrained axis) leaves in-plane motion free."""
    cost = make_cost({"ee": ToolPoseBound.plane()})
    in_plane = evaluate(cost, torch.tensor([0.4, -0.3, 0.0]), IDENTITY, ORIGIN, IDENTITY)
    lifted = evaluate(cost, torch.tensor([0.4, -0.3, 0.03]), IDENTITY, ORIGIN, IDENTITY)
    assert in_plane == 0.0
    assert lifted == pytest.approx(0.5 * 0.03**2, rel=1e-4)


def test_tilt_ignores_rotation_about_the_tool_axis():
    """Rotating about the tool axis never costs tilt, even with a tight cone."""
    cost = make_cost({"ee": ToolPoseBound.axis_tilt(max_angle=math.radians(2))})
    twisted = quat_axis_angle([0.0, 0.0, 1.0], math.radians(170))
    tilted = quat_axis_angle([1.0, 0.0, 0.0], math.radians(5))
    assert evaluate(cost, ORIGIN, twisted, ORIGIN, IDENTITY) == 0.0
    assert evaluate(cost, ORIGIN, tilted, ORIGIN, IDENTITY) > 0.0
